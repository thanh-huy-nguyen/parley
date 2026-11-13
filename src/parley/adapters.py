"""Low-rank adapters: a rank-4 correction on the attention of the two transformer decoders.

Up to 40 epochs per accepted image with early stopping, batch 1, focal plus Dice loss with weights 20:1, learning
rate 1e-4. The correction target is the accepted final mask, used
for every turn of that image; images that were not accepted teach nothing. The adapters are reinitialized at each
session boundary.
"""

import time

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from . import geometry as G
from . import model as M

RANK, EPOCHS, PATIENCE, LR, W_FOCAL, W_DICE = 4, 40, 3, 1e-4, 20.0, 1.0


class LowRankDelta(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(RANK, d))
        self.B = nn.Parameter(torch.zeros(3 * d, RANK))
        self.reset()

    def reset(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            self.B.zero_()

    def forward(self, W):
        return W + self.B @ self.A


def focal_dice(logit, y):
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p).clamp(1e-6, 1)
    focal = (-((1 - pt) ** 2) * torch.log(pt)).mean()
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return W_FOCAL * focal + W_DICE * dice


class LowRankAdapters:
    def __init__(self, engine):
        self.engine = engine
        self.seconds = 0.0
        self.n_fits = 0
        net = engine.net
        self.mods = []
        for dec in (net.text_dec, net.img_dec):
            for layer in dec.layers:
                for att in (layer.self_attn, layer.multihead_attn):
                    parametrize.register_parametrization(
                        att, "in_proj_weight", LowRankDelta(att.embed_dim).to(engine.device)
                    )
                    self.mods.append(att.parametrizations.in_proj_weight[0])
        for p in net.parameters():
            p.requires_grad_(False)
        self.params = [q for m in self.mods for q in (m.A, m.B)]
        for q in self.params:
            q.requires_grad_(True)

    def reset(self):
        self.n_fits = 0
        for m in self.mods:
            m.reset()

    def state_arrays(self):
        return {f"adapter_{i}": q.detach().cpu().numpy() for i, q in enumerate(self.params)}

    def load_arrays(self, arrays):
        with torch.no_grad():
            for i, q in enumerate(self.params):
                q.copy_(torch.from_numpy(arrays[f"adapter_{i}"]).to(q.device))

    def _logit(self, mask, text, click):
        """Differentiable forward of one turn, as the engine runs it."""
        eng, net = self.engine, self.engine.net
        eng._click, eng._mshape = click, mask.shape
        with torch.no_grad():
            text_f, pad = eng.text([text])
        inp = eng.inputs(G.to_work(mask), text_f, pad)
        hd, f, _ = net(inp)
        tp = net.plan(hd, 0)
        if M.TYPES[int(tp["type"][0])] in ("accept", "none", "split"):
            return None
        logit = net.refine_logit(
            f, net.program(tp, inp["phi"], inp["nx"], inp["ny"], inp["D"], inp["mask"]), hd["fine"]
        )
        if float(torch.sigmoid(hd["present"][0, 1])) > 0.5:
            tp1 = net.plan(hd, 1)
            if M.TYPES[int(tp1["type"][0])] not in ("accept", "none", "split"):
                new = (logit[0] > 0).float().detach().cpu().numpy()
                if new.any():
                    c1 = eng.inputs(new, text_f, pad)
                    logit = net.refine_logit(
                        f, net.program(tp1, c1["phi"], c1["nx"], c1["ny"], c1["D"], c1["mask"]), hd["fine"]
                    )
        return logit[0]

    def adapt_image(self, turns, final_mask):
        """Fit on all the turns of an accepted image, each with its own pre-turn mask, and the accepted mask as the
        target. `turns` holds (mask, text, click). Returns the epochs used.
        """
        if not turns:
            return 0
        t0 = time.time()
        opt = torch.optim.Adam(self.params, LR)
        target = torch.from_numpy(G.to_work(final_mask)).float().to(self.engine.device)
        best, bad, used = 1e9, 0, 0
        for _ in range(EPOCHS):
            tot, n = 0.0, 0
            for mask, text, click in turns:
                logit = self._logit(mask, text, click)
                if logit is None:
                    continue
                loss = focal_dice(logit, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                n += 1
            if n == 0:
                break
            used += 1
            if tot / n < best - 1e-3:
                best, bad = tot / n, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        self.seconds += time.time() - t0
        self.n_fits += 1
        return used
