"""Mask-space edits and measurements. CPU, no weights."""

import numpy as np
from scipy import ndimage as ndi

from parley import edits as E

H = 256


def disk(cx, cy, r):
    yy, xx = np.mgrid[:H, :H]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def test_equiv_diameter_disk():
    assert abs(E.equiv_diameter(disk(128, 128, 40)) - 80) < 1.0


def test_apply_on_empty_mask_does_not_divide_by_zero():
    empty = np.zeros((H, H), bool)
    assert not E.apply(empty, E.Edit("shrink", "part", direction="top", amount=0.05)).any()
    assert not E.apply(empty, E.Edit("move", "global", move=(0.1, 0.0))).any()


def test_shrink_at_the_frame_does_not_pull_the_mask_off_it():
    t = disk(20, 128, 40)
    for e in (E.Edit("shrink", "global", amount=0.08), E.Edit("shrink", "part", direction="left", amount=0.08)):
        c = E.apply(t, e)
        assert c[128, 0], e
        assert c.sum() < t.sum()


def test_grow_at_the_frame_stops_at_the_frame():
    t = disk(20, 128, 40)
    c = E.apply(t, E.Edit("shrink", "global", amount=-0.08))
    assert c.shape == t.shape and (c | t).sum() == c.sum()
    assert c[:, 0][t[:, 0]].all()


def test_global_shrink_offsets_the_boundary_by_the_amount():
    t = disk(128, 128, 40)
    c = E.apply(t, E.Edit("shrink", "global", amount=0.1))  # 0.1 D = 8 px
    assert abs(E.equiv_diameter(c) - 64) < 2.0


def test_sector_errors_sign_and_side():
    t = disk(128, 128, 40)
    c = E.apply(t, E.Edit("shrink", "part", direction="top", amount=-0.08))
    err = E.sector_errors(c, t)
    assert err["top"] == max(err.values()) and err["top"] > 0.03


def test_one_pixel_hole_is_not_boundary():
    t = disk(128, 128, 40)
    c = t.copy()
    c[128, 128] = False
    assert E.boundary_pixels(c).sum() == E.boundary_pixels(t).sum()


def test_diagonal_pixel_is_part_of_the_lesion():
    t = disk(128, 128, 40)
    ys, xs = np.where(t)
    y0, x0 = ys.min(), xs[ys == ys.min()].max()
    spur = t.copy()
    spur[y0 - 1, x0 + 1] = True
    assert ndi.label(spur)[1] == 2 and ndi.label(spur, structure=E.CONNECTIVITY)[1] == 1


def test_add_and_remove_respect_their_region():
    t = disk(128, 128, 40)
    region = disk(170, 128, 20)
    assert (E.apply(t, E.Edit("add", region=region)) & region).sum() == region.sum()
    assert not (E.apply(t, E.Edit("remove", region=region)) & region).any()
