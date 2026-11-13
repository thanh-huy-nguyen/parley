"""The network that reads one utterance on the current mask.

Inputs: frozen image features (256 x 64 x 64), the image at 512 x 512, the mask and its signed distance function at
512 x 512, a click
heat map, and frozen text token features. Output: up to two edit programs, each executed differentiably on the signed
distance function,
followed by a refinement near the new boundary. Lengths are fractions of D, the equivalent diameter of the current mask.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TYPES = ["add", "remove", "move", "shrink", "split", "accept", "none"]
T = {n: i for i, n in enumerate(TYPES)}
SCOPE_DIRS = ["global", "top", "bottom", "left", "right"]
MOVE_DIRS = ["right", "down and right", "down", "down and left", "left", "up and left", "up", "up and right"]
PLACES = ["top", "bottom", "left", "right"]
SIZES = ["small", "medium", "large"]
TAU = 1.5  # px, softness of the Heaviside


class MaskEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        c = 32
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(c, c * 2, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 2, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, d, 3, 1, 1),
        )

    def forward(self, x):  # (B, 3, 512, 512) -> (B, d, 64, 64)
        return self.net(x)


class ClickEnc(nn.Module):
    """Click heat map (B, 512, 512) -> (B, d, 64, 64).

    Bias-free, so an absent click (all zeros) adds exactly nothing.
    """

    def __init__(self, d):
        super().__init__()
        c = 32
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 5, 2, 2, bias=False),
            nn.GELU(),
            nn.Conv2d(c, c * 2, 3, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 2, 3, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(c * 2, d, 3, 1, 1, bias=False),
        )

    def forward(self, heat):
        return self.net(heat[:, None])


class RegionHead(nn.Module):
    """Add or remove region: image pixels + image features + current mask + type + side + window + click -> logits of
    the changed region.
    """

    def __init__(self, cond_dim=6):
        super().__init__()
        self.img1, self.img2, self.img3 = (
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.Conv2d(32, 32, 3, 2, 1),
        )
        self.proj = nn.Conv2d(256, 64, 1)
        self.body = nn.Sequential(
            nn.Conv2d(64 + 32 + 1 + cond_dim + 1 + 2, 96, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 2, dilation=2),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, 1, 4, dilation=4),
            nn.GELU(),
            nn.Conv2d(96, 48, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(48, 1, 1),
        )
        self.fine = nn.Sequential(nn.Conv2d(16 + 1 + 1 + 1, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, image, feat, mask, cond, win, heat, flag):
        fine = F.gelu(self.img1(image))
        mid = self.img3(F.gelu(self.img2(fine)))
        sz = mid.shape[-2:]
        lf = F.interpolate(self.proj(feat), size=sz, mode="bilinear", align_corners=False)
        m = F.interpolate(mask[:, None], size=sz, mode="bilinear", align_corners=False)
        w = F.interpolate(win[:, None], size=sz, mode="bilinear", align_corners=False)
        c = cond[:, :, None, None].expand(-1, -1, *sz)
        h = F.interpolate(heat[:, None], size=sz, mode="bilinear", align_corners=False)
        extra = [h, flag[:, None, None, None].expand(-1, 1, *sz)]
        logit = self.body(torch.cat([lf, mid, m, c, w] + extra, 1))
        up = F.interpolate(logit, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        return (up + self.fine(torch.cat([fine, up / 8.0, mask[:, None], heat[:, None]], 1)))[:, 0]


class ParleyNet(nn.Module):
    def __init__(self, d=256, text_dim=384, heads=8, layers=2):
        super().__init__()
        self.d = d
        self.click_enc = ClickEnc(d)
        self.rh = RegionHead()
        self.t_proj = nn.Linear(text_dim, d)
        self.img_proj = nn.Conv2d(256, d, 1)
        self.mask_enc = MaskEnc(d)
        self.fuse = nn.Sequential(nn.Conv2d(2 * d, d, 3, 1, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.pos = nn.Parameter(torch.randn(1, d, 64, 64) * 0.02)
        self.slots = nn.Parameter(torch.randn(2, d) * 0.02)
        self.text_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers
        )
        self.img_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True), layers
        )
        self.present = nn.Linear(d, 1)
        self.typ = nn.Linear(d, len(TYPES))
        self.scope = nn.Linear(d, len(SCOPE_DIRS))
        self.move_dir = nn.Linear(d, len(MOVE_DIRS))
        self.place = nn.Linear(d, len(PLACES))
        self.sign = nn.Linear(d, 2)
        self.size = nn.Linear(d, len(SIZES))
        self.mag = nn.Linear(d, 1)
        self.move_vec = nn.Linear(d, 2)
        self.refine_in = nn.Conv2d(d, d // 4, 1)
        self.img1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.img2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.img3 = nn.Conv2d(32, 32, 3, 2, 1)
        self.les_proj = nn.Conv2d(d, 64, 1)
        self.les = nn.Sequential(
            nn.Conv2d(64 + 32 + 1, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )
        self.les_fine = nn.Sequential(nn.Conv2d(16 + 1, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 1, 3, 1, 1))
        self.refine_v1 = nn.Sequential(nn.Conv2d(16 + d // 4 + 2, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 1, 3, 1, 1))

    def encode(self, feat, mask, phi_over_D, text, text_pad, heat):
        B = feat.shape[0]
        mx = torch.stack([mask, torch.clamp(phi_over_D * 4.0, -1, 1), torch.clamp(phi_over_D * 16.0, -1, 1)], 1)
        f = self.fuse(torch.cat([self.img_proj(feat.float()), self.mask_enc(mx)], 1)) + self.pos
        f = f + self.click_enc(heat)
        mem_img = f.flatten(2).transpose(1, 2)  # (B, 4096, d)
        mem_txt = self.t_proj(text)  # (B, L, d)
        q = self.slots[None].expand(B, -1, -1)
        q = self.text_dec(q, mem_txt, memory_key_padding_mask=text_pad)
        q = self.img_dec(q, mem_img)
        return q, f  # (B, 2, d), (B, d, 64, 64)

    def heads(self, q):
        return dict(
            present=self.present(q).squeeze(-1),
            typ=self.typ(q),
            scope=self.scope(q),
            move_dir=self.move_dir(q),
            place=self.place(q),
            sign=self.sign(q),
            size=self.size(q),
            mag=F.softplus(self.mag(q)).squeeze(-1) * 0.05,
            move_vec=self.move_vec(q) * 0.1,
        )

    @staticmethod
    def anchor_weight(nx, ny, scope_idx):
        """(B, H, W) weight for scope 0 global, 1 top, 2 bottom, 3 left, 4 right."""
        dirs = torch.tensor([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], device=nx.device)[
            scope_idx
        ]  # (B, 2)
        kappa = math.log(2.0) / (1.0 - math.cos(math.pi / 4))
        w = torch.exp(kappa * (nx * dirs[:, 0, None, None] + ny * dirs[:, 1, None, None] - 1.0))
        return torch.where((scope_idx == 0)[:, None, None], torch.ones_like(w), w)

    def program(self, tp, phi, nx, ny, D, mask):
        """Execute a program `tp` (dict of index tensors and magnitudes).

        Returns the logit (B, H, W) of the new mask.
        """
        t = tp["type"]
        B, H, W = phi.shape
        out = -phi / TAU
        mask_p = torch.sigmoid(-phi / TAU)
        # shrink / grow
        amt = tp["sign_sign"] * tp["mag"] * D  # px, positive shrinks
        w = self.anchor_weight(nx, ny, tp["scope"])
        g = torch.exp(-(phi**2) / (2 * (0.1 * D[:, None, None]) ** 2))
        g = torch.where((tp["scope"] == 0)[:, None, None], torch.ones_like(g), g)
        shrink = -(phi + amt[:, None, None] * w * g) / TAU
        # move: phi'(x) = phi(x - delta)
        dx, dy = tp["move_vec"][:, 0] * D, tp["move_vec"][:, 1] * D  # px
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=phi.device), torch.linspace(-1, 1, W, device=phi.device), indexing="ij"
        )
        grid = torch.stack([xs[None] - 2 * dx[:, None, None] / W, ys[None] - 2 * dy[:, None, None] / H], -1)
        move = (
            -F.grid_sample(phi[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0] / TAU
        )
        # add / remove through the anchored region
        win = self.window(tp["place"], H, W, mask)
        B_ = t.shape[0]
        cond = torch.zeros(B_, 6, device=phi.device)
        cond[torch.arange(B_), tp["place"].clamp(0, 3)] = 1.0
        cond[:, 4] = (t == T["add"]).float()
        cond[:, 5] = (t == T["remove"]).float()
        reg = torch.sigmoid(
            self.rh(
                self._ctx["image"], self._ctx["feat"].float(), mask, cond, win, self._ctx["heat"], self._ctx["flag"]
            )
        )
        add = torch.logit(torch.clamp(torch.maximum(mask_p, reg), 1e-4, 1 - 1e-4))
        rem = torch.logit(torch.clamp(mask_p * (1 - reg), 1e-4, 1 - 1e-4))
        res = out
        for name, val in (("add", add), ("remove", rem), ("move", move), ("shrink", shrink)):
            res = torch.where((t == T[name])[:, None, None], val, res)
        return res

    @staticmethod
    def window(place, H, W, mask):
        """Soft half-plane (top, bottom, left, right) relative to the mask's centroid, widened to a margin of 0.1 of the
        image.
        """
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=mask.device), torch.linspace(0, 1, W, device=mask.device), indexing="ij"
        )
        m = mask.flatten(1).sum(1).clamp(min=1)
        cx = (mask * xs[None]).flatten(1).sum(1) / m
        cy = (mask * ys[None]).flatten(1).sum(1) / m
        out = torch.ones_like(mask)
        sel = {
            0: -(ys[None] - cy[:, None, None]),
            1: (ys[None] - cy[:, None, None]),
            2: -(xs[None] - cx[:, None, None]),
            3: (xs[None] - cx[:, None, None]),
        }
        for k, v in sel.items():
            out = torch.where((place == k)[:, None, None], torch.sigmoid((v + 0.1) * 30.0), out)
        return out

    def forward(self, batch):
        """batch: image, feat (B, 256, 64, 64), mask, phi, nx, ny, D, heat, flag, text, text_pad. Returns (head outputs,
        fused features, lesion logits).
        """
        phi_over_D = batch["phi"] / batch["D"][:, None, None]
        q, f = self.encode(batch["feat"], batch["mask"], phi_over_D, batch["text"], batch["text_pad"], batch["heat"])
        hd = self.heads(q)
        self._ctx = dict(image=batch["image"], feat=batch["feat"], heat=batch["heat"], flag=batch["flag"])
        x = batch["image"]
        fine = F.gelu(self.img1(x))  # (B, 16, 512, 512)
        mid = self.img3(F.gelu(self.img2(fine)))  # (B, 32, 128, 128)
        lf = F.interpolate(self.les_proj(f), size=mid.shape[-2:], mode="bilinear", align_corners=False)
        m128 = F.interpolate(batch["mask"][:, None], size=mid.shape[-2:], mode="bilinear", align_corners=False)
        l128 = self.les(torch.cat([lf, mid, m128], 1))
        up = F.interpolate(l128, size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        lesion_logits = (up + self.les_fine(torch.cat([fine, up / 8.0], 1)))[:, 0]
        hd["fine"] = fine
        return hd, f, lesion_logits

    @staticmethod
    def plan(hd, slot):
        """Discrete choices of one slot: the argmax of each head."""
        return dict(
            type=hd["typ"][:, slot].argmax(-1),
            scope=hd["scope"][:, slot].argmax(-1),
            place=hd["place"][:, slot].argmax(-1),
            sign_sign=torch.where(hd["sign"][:, slot].argmax(-1) == 0, 1.0, -1.0),
            mag=hd["mag"][:, slot],
            move_vec=hd["move_vec"][:, slot],
        )

    def refine_logit(self, f, logit, fine):
        """Residual correction of the program output near the boundary, from the fused features and the full-resolution
        image features.
        """
        band = torch.exp(-((logit.detach() / 4.0) ** 2))
        r = F.interpolate(self.refine_in(f), size=logit.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([fine, r, logit[:, None] / 8.0, band[:, None]], 1)
        return logit + band * self.refine_v1(x)[:, 0]
