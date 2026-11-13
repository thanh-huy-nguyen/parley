"""One clinician turn on the current mask. No input checking and no guards here: `ParleyEditor` adds those.

The engine holds the network, the frozen encoders, the features of the current image and, depending on the mode, a
listener and low-rank adapters.
Modes: base, listener, adapter, listener+adapter.
"""

import cv2
import numpy as np
import torch

from . import geometry as G
from . import model as M
from .adapters import LowRankAdapters
from .errors import UnsupportedCheckpointError
from .listener import Listener

MODES = ("base", "listener", "adapter", "listener+adapter")
# Parameters stored in the released checkpoints that inference never uses. They are dropped at load time.
UNUSED_PREFIXES = ("region_head.", "refine.", "film.", "orient.")


def load_network(checkpoint, device):
    """Build the network and load a checkpoint. Unknown or missing parameters raise UnsupportedCheckpointError."""
    sd = torch.load(checkpoint, map_location=device)
    net = M.ParleyNet().to(device).eval()
    kept = {k: v for k, v in sd.items() if not k.startswith(UNUSED_PREFIXES)}
    try:
        net.load_state_dict(kept, strict=True)
    except RuntimeError as e:
        raise UnsupportedCheckpointError(f"{checkpoint} does not match this architecture: {e}") from e
    return net


def describe_slot(hd, slot):
    """The parsed program of one slot, in plain fields. Amounts are in units of D."""
    t = M.TYPES[int(hd["typ"][0, slot].argmax())]
    p = dict(type=t)
    p["scope"] = M.SCOPE_DIRS[int(hd["scope"][0, slot].argmax())]
    p["place"] = M.PLACES[int(hd["place"][0, slot].argmax())]
    p["move_dir"] = M.MOVE_DIRS[int(hd["move_dir"][0, slot].argmax())]
    p["sign"] = "shrink" if int(hd["sign"][0, slot].argmax()) == 0 else "grow"
    p["size"] = M.SIZES[int(hd["size"][0, slot].argmax())]
    p["amount_D"] = round(float(hd["mag"][0, slot]), 5)
    p["move_D"] = [round(float(v), 5) for v in hd["move_vec"][0, slot]]
    p["confidence"] = round(float(torch.softmax(hd["typ"][0, slot], -1).max()), 5)
    return p


class Engine:
    def __init__(self, checkpoint, device="cuda", mode="base", text_encoder=None, image_encoder=None, recognizer=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.device, self.mode = device, mode
        self.net = (
            load_network(checkpoint, device)
            if isinstance(checkpoint, (str, bytes)) or hasattr(checkpoint, "__fspath__")
            else checkpoint
        )
        if text_encoder is None:
            from .frozen import TextEncoder

            text_encoder = TextEncoder(device)
        self.text = text_encoder
        self._image_encoder, self._recognizer = image_encoder, recognizer
        self.feat_tensor, self.image_tensor = None, None
        self._click, self._mshape = None, None
        self.listener = Listener(self) if "listener" in mode else None
        self.adapters = LowRankAdapters(self) if "adapter" in mode else None
        self.turns = []

    # ---- frozen models, loaded on first use
    @property
    def image_encoder(self):
        if self._image_encoder is None:
            from .frozen import ImageEncoder

            self._image_encoder = ImageEncoder(self.device)
        return self._image_encoder

    @property
    def recognizer(self):
        if self._recognizer is None:
            from .frozen import Recognizer

            self._recognizer = Recognizer(self.device)
        return self._recognizer

    # ---- protocol
    def begin_session(self):
        """Clear what was learned about the clinician: the listener returns to its prior and the adapters to a fresh
        initialization.
        """
        self.turns = []
        if self.adapters is not None:
            self.adapters.reset()
        if self.listener is not None:
            self.listener.begin_session()

    def set_image(self, rgb=None, features=None):
        """Encode an image once, or take precomputed (features (256, 64, 64) float16, image (3, 512, 512) float32)."""
        if features is None:
            self.feat_tensor, self.image_tensor = self.image_encoder(rgb)
        else:
            self.feat_tensor, self.image_tensor = (t.to(self.device) for t in features)

    def begin_image(self, start_mask, rgb=None, features=None):
        self.set_image(rgb, features)
        self.turns = []
        start_mask = np.asarray(start_mask).astype(bool)
        if self.listener is not None:
            return self.listener.begin_image(start_mask)
        return start_mask

    def step(self, mask, text, click=None, edit=True):
        """One turn. `click` is (x, y) in pixels of `mask` or None. Returns (new bool mask of the same shape, info)."""
        mask = np.asarray(mask).astype(bool)
        c = None if click is None else [float(click[0]), float(click[1])]
        new, info = self.forward_turn(mask, text, c)
        if self.listener is not None:
            self.listener.observe_turn(mask, info.get("parse"))
        if edit:
            self.turns.append((mask, text, c))
        return new, info

    def end_image(self, final_mask, reached, accept_text=None):
        """Close an image. An accepted image can teach the adapters and the listener. Returns the listener's summary, or
        None.
        """
        final_mask = np.asarray(final_mask).astype(bool)
        probe = None
        if reached and accept_text is not None:
            probe = self.forward_turn(final_mask, accept_text, None)[1].get("parse")
        if self.adapters is not None and reached:
            self.adapters.adapt_image(self.turns, final_mask)
        return self.listener.end_image(final_mask, probe) if self.listener is not None else None

    # ---- network
    def inputs(self, work, text, pad):
        D, phi, gx, gy, n = G.geom(work)

        def t(a):
            return torch.from_numpy(np.asarray(a, np.float32))[None].to(self.device)

        heat = np.zeros((G.WORK, G.WORK), np.float32)
        if self._click is not None:
            cx, cy = self._click[0] * G.WORK / self._mshape[1], self._click[1] * G.WORK / self._mshape[0]
            yy, xx = np.mgrid[: G.WORK, : G.WORK]
            heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.03 * D) ** 2)).astype(np.float32)
        return dict(
            image=self.image_tensor[None],
            heat=torch.from_numpy(heat)[None].to(self.device),
            flag=torch.tensor([float(self._click is not None)], device=self.device),
            feat=self.feat_tensor[None],
            mask=t(work),
            phi=t(phi),
            nx=t(gx / n),
            ny=t(gy / n),
            D=torch.tensor([D], dtype=torch.float32, device=self.device),
            text=text,
            text_pad=pad,
        )

    @torch.no_grad()
    def lesion_mask(self, mask):
        """The lesion head's mask for the current image given a mask, with no speech and no click. The listener's
        reference.
        """
        self._click, self._mshape = None, mask.shape
        text_f, pad = self.text([G.NEUTRAL])
        inp = self.inputs(G.to_work(mask), text_f, pad)
        _, _, les = self.net(inp)
        lg = cv2.resize(les[0].float().cpu().numpy(), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        return lg > 0

    @torch.no_grad()
    def forward_turn(self, mask, text, click):
        self._click, self._mshape = click, mask.shape
        text_f, pad = self.text([text])
        work = G.to_work(mask)
        inp = self.inputs(work, text_f, pad)
        hd, f, _ = self.net(inp)
        tp0 = self.net.plan(hd, 0)
        typ0 = M.TYPES[int(tp0["type"][0])]
        second = float(torch.sigmoid(hd["present"][0, 1])) > 0.5
        info = dict(types=[typ0], accept=typ0 == "accept")
        p0 = describe_slot(hd, 0)
        info["parse"] = dict(
            type=typ0,
            scope=p0["scope"],
            place=p0["place"],
            move_dir=p0["move_dir"],
            sign=p0["sign"],
            size=p0["size"],
            compound=second,
        )
        info["programs"] = [p0]
        logit = None
        if typ0 not in ("accept", "none"):
            logit = self.net.program(tp0, inp["phi"], inp["nx"], inp["ny"], inp["D"], inp["mask"])
            logit = self.net.refine_logit(f, logit, hd["fine"])
            if second:
                tp1 = self.net.plan(hd, 1)
                typ1 = M.TYPES[int(tp1["type"][0])]
                if typ1 not in ("accept", "none"):
                    new = (logit[0] > 0).float().cpu().numpy()
                    if new.any():
                        c1 = self.inputs(new, text_f, pad)
                        l1 = self.net.program(tp1, c1["phi"], c1["nx"], c1["ny"], c1["D"], c1["mask"])
                        logit = self.net.refine_logit(f, l1, hd["fine"])
                        info["types"].append(typ1)
                        info["programs"].append(describe_slot(hd, 1))
        if logit is None:
            return mask.copy(), info
        lg = cv2.resize(logit[0].float().cpu().numpy(), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        return lg > 0, info
