"""The listener's likelihood and belief alone. CPU, no weights, and no network import."""

import numpy as np

from parley import edits as E
from parley import listener as L

H = 256


def disk(r, cx=128, cy=128):
    yy, xx = np.mgrid[:H, :H]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def record(events, ref=None):
    ref = disk(40) if ref is None else ref
    return dict(D=E.equiv_diameter(ref), area=int(ref.sum()), perim=int(E.boundary_pixels(ref).sum()), events=events)


def test_grid_and_prior_are_symmetric():
    assert len(L.GRID) == 801 and L.GRID[0] == -0.4 and L.GRID[-1] == 0.4
    b = L.MarginBelief()
    assert abs(b.mean()) < 1e-9
    lo, hi = b.interval90()
    assert lo == -hi


def test_a_grow_request_moves_the_belief_outward():
    """A clinician who asks to grow a mask that sits on the reference draws outside it."""
    b = L.MarginBelief()
    ev = dict(
        kind="edit",
        type="shrink",
        scope="global",
        direction=None,
        sign=-1,
        size="medium",
        g=0.0,
        s=dict(top=0, bottom=0, left=0, right=0),
    )
    b.add(record([ev]), 0.05)
    assert b.mean() > 0.01


def test_a_shrink_request_moves_the_belief_inward():
    b = L.MarginBelief()
    ev = dict(
        kind="edit",
        type="shrink",
        scope="global",
        direction=None,
        sign=1,
        size="medium",
        g=0.0,
        s=dict(top=0, bottom=0, left=0, right=0),
    )
    b.add(record([ev]), 0.05)
    assert b.mean() < -0.01


def test_a_part_edit_reads_its_own_sector():
    s = dict(top=0.05, bottom=0.0, left=0.0, right=0.0)
    ev = dict(kind="edit", type="shrink", scope="part", direction="top", sign=1, size="small", g=0.0, s=s)
    ev0 = dict(ev, s=dict(s, top=0.0))
    assert not np.array_equal(L.log_likelihood(record([ev]), 0.05), L.log_likelihood(record([ev0]), 0.05))


def test_an_accept_narrows_the_belief_around_the_mask_offset():
    b = L.MarginBelief()
    w0 = b.interval90()
    b.add(record([dict(kind="accept", g=0.03, s=dict(top=0, bottom=0, left=0, right=0))]), 0.02)
    lo, hi = b.interval90()
    assert lo < 0.03 < hi and (hi - lo) <= (w0[1] - w0[0])
    assert b.mean() > 0.0


def test_other_edit_types_carry_no_information():
    ev = dict(kind="edit", type="move", g=0.1, s=dict(top=0, bottom=0, left=0, right=0))
    assert (L.log_likelihood(record([ev]), 0.02) == 0).all()


def test_lapse_keeps_the_likelihood_finite_far_from_the_data():
    ev = dict(kind="accept", g=0.3, s=dict(top=0, bottom=0, left=0, right=0))
    assert np.isfinite(L.log_likelihood(record([ev]), 0.001)).all()


def test_accept_window_has_a_pixel_floor():
    small = disk(3)
    r = record([dict(kind="accept", g=0.0, s=dict(top=0, bottom=0, left=0, right=0))], small)
    assert np.array_equal(L.log_likelihood(r, 1e-6), L.log_likelihood(r, E.PIXEL_FLOOR / r["D"]))


def test_half_width_gives_iou_085():
    ref = disk(40)
    w = L.half_width_085(ref)
    assert abs(E.iou(E.apply(ref, E.Edit("shrink", "global", amount=w)), ref) - 0.85) < 0.01


def test_the_likelihood_does_not_mutate_module_state():
    keep = E.ACCEPT_TOL
    L.log_likelihood(record([dict(kind="accept", g=0.0, s={})]), 0.2)
    assert E.ACCEPT_TOL == keep
