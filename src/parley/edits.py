"""Mask-space edits and boundary measurements. Numpy and scipy only.

An edit is a program on the signed distance function of the current mask. Lengths are fractions of D, the equivalent
diameter
of the mask. Coordinates: x to the right, y downward (image frame). These functions run at the resolution of the mask
they are given.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

CONNECTIVITY = np.ones((3, 3), int)  # 8-connectivity: pieces touching at a corner count as one
DIRS = {"top": (0.0, -1.0), "bottom": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
KAPPA = np.log(2.0) / (1.0 - np.cos(np.pi / 4))  # anchor weight falls to 0.5 at 45 degrees
ACCEPT_TOL = 0.01  # D, default half-width of the accept window
MIN_REGION_PX = 20  # px, interior holes below this size are not treated as boundary
PIXEL_FLOOR = 1.5  # px, pixel masks and resampling cannot meet a tighter bound on small lesions
SIZE_BINS = (0.04, 0.12)  # D, small / medium / large


@dataclass
class Edit:
    type: str  # add | remove | move | shrink | accept
    scope: str = "global"  # global | part
    direction: str | None = None  # top | bottom | left | right for part scope
    amount: float = 0.0  # D; shrink: signed (negative = grow)
    move: tuple | None = None  # (dx, dy) in D
    region: np.ndarray | None = field(default=None, repr=False)  # boolean mask for add, remove


def sdf(mask):
    m = mask.astype(bool)
    if not m.any():
        return np.full(m.shape, 1e3)
    if m.all():
        return np.full(m.shape, -1e3)
    pad = 64  # replicate the frame outward so the image edge is not scored as lesion boundary
    mp = np.pad(m, pad, mode="edge")
    phi = (ndi.distance_transform_edt(~mp) - ndi.distance_transform_edt(mp))[pad:-pad, pad:-pad]
    return phi - 0.5 * np.sign(phi)  # boundary halfway between pixel centers; inside boundary pixels sit at -0.5


def boundary_dist(cur, tgt):
    """Pixel coordinates of the current boundary and the signed distance (px) from each to the target boundary. Positive
    means outside the target.
    """
    ys, xs = np.where(boundary_pixels(cur))
    return ys, xs, sdf(tgt)[ys, xs] + 0.5


def equiv_diameter(mask):
    return float(np.sqrt(4.0 * mask.sum() / np.pi))


def _fill_small_holes(m):
    holes = ndi.binary_fill_holes(m) & ~m
    if not holes.any():
        return m
    lab, n = ndi.label(holes, structure=CONNECTIVITY)
    small = [j + 1 for j, v in enumerate(ndi.sum(holes, lab, range(1, n + 1))) if v < MIN_REGION_PX]
    return m | np.isin(lab, small)


def boundary_pixels(mask):
    m = mask.astype(bool)
    m = _fill_small_holes(m)
    return m & ~ndi.binary_erosion(m, border_value=1)


def normals(phi):
    gy, gx = np.gradient(ndi.gaussian_filter(phi, 1.0))
    n = np.sqrt(gx**2 + gy**2) + 1e-9
    return gx / n, gy / n


def anchor_weight(nx, ny, direction):
    dx, dy = DIRS[direction]
    return np.exp(KAPPA * (nx * dx + ny * dy - 1.0))


def sector_errors(cur, tgt):
    """Median signed distance, in D of `tgt`, from the current boundary to the target, per direction. Positive means the
    current mask is too big there.
    """
    D = equiv_diameter(tgt)
    phi_c = sdf(cur)
    nx, ny = normals(phi_c)
    ys, xs, dist = boundary_dist(cur, tgt)
    if len(ys) == 0:
        return {d: 0.0 for d in DIRS}
    dots = np.stack([nx[ys, xs] * v[0] + ny[ys, xs] * v[1] for v in DIRS.values()], 1)
    sector = dots.argmax(1)
    err = {}
    for k, d in enumerate(DIRS):
        sel = sector == k
        err[d] = float(np.median(dist[sel]) / D) if sel.any() else 0.0
    return err


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def apply(cur, edit):
    """Execute one edit on a boolean mask with no learned component.

    The listener's offset of a start mask uses this.
    """
    cur = cur.astype(bool)
    if edit.type == "accept":
        return cur
    if edit.type in ("remove", "add"):
        return (cur & ~edit.region) if edit.type == "remove" else (cur | edit.region)
    if not cur.any():
        return cur
    D = equiv_diameter(cur)
    if edit.type == "move":
        dx, dy = edit.move
        return ndi.shift(sdf(cur), (dy * D, dx * D), order=1, mode="nearest") <= 0
    if edit.type == "shrink":
        amt = edit.amount * D
        phi = sdf(cur)
        if edit.scope == "global":
            return (phi + amt) <= 0
        nx, ny = normals(phi)
        w = anchor_weight(nx, ny, edit.direction)
        g = np.exp(-(phi**2) / (2 * (0.1 * D) ** 2))
        return (phi + amt * w * g) <= 0
    raise ValueError(edit.type)
