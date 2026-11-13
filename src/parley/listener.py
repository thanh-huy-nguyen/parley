"""The listener: a per-clinician estimate of how tightly this clinician draws.

The unknown b is the clinician's margin in units of D, the equivalent diameter of the reference mask. Positive b
means the clinician's boundary lies outside
the lesion head's reference mask. The belief is an exact posterior on a grid of 801 points over -0.4 to 0.4 D, with a
Student-t prior (3 degrees of freedom).

Two kinds of event carry information: a shrink or grow program (its sign and size bin say where this clinician's
boundary lies relative to the reference, in the anchored
direction), and an accepted mask (its offset from the reference falls within a window). Every event carries a lapse
rate. The posterior mean offsets each new start mask once.

`log_likelihood` has no dependency on the network and can be tested alone.
"""

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t

from . import edits as E

GRID = np.linspace(-0.4, 0.4, 801)
PRIOR_DF = 3
PRIOR_SCALE = 0.02078  # D, fitted by empirical Bayes on the training generator
JUDGE_SD = 0.0025  # D
LAPSE = 0.10
SIZE_RANGES = {"small": (0.0, E.SIZE_BINS[0]), "medium": E.SIZE_BINS, "large": (E.SIZE_BINS[1], np.inf)}


def _interval(x, lo, hi, sd=JUDGE_SD):
    return ndtr((hi - x) / sd) - ndtr((lo - x) / sd)


def _diameter_ratio(area, perim, D):
    """D of the clinician's target over D of the reference, for each margin on the grid (area grown by perimeter * b *
    D).
    """
    a = np.clip(area + perim * GRID * D, 1.0, None)
    return 2 * np.sqrt(a / np.pi) / D


def log_likelihood(record, accept_tol=E.ACCEPT_TOL):
    """Log likelihood over GRID of the events in `record`.

    record: dict with `D`, `area` and `perim` of the reference mask, and `events`. An event has `kind` "accept" or
    "edit", `g` (median offset from the reference over the four sectors, in D)
    and `s` (offset per sector). An edit event also has `type`, `scope`, `direction`, `sign` (+1 shrink, -1 grow) and
    `size`.
    accept_tol: half-width of the accept window in D; floored at 1.5 px.
    """
    ratio = _diameter_ratio(record["area"], record["perim"], record["D"])
    w = max(accept_tol, E.PIXEL_FLOOR / record["D"])
    L = np.zeros_like(GRID)
    for e in record["events"]:
        if e["kind"] == "accept":
            delta = (e["g"] - GRID) / ratio
            L += np.log(LAPSE + (1 - LAPSE) * _interval(delta, -w, w))
            continue
        if e["type"] != "shrink":
            continue
        s = e["g"] if e["scope"] == "global" else e["s"][e["direction"]]
        a = (s - GRID) / ratio
        lo, hi = SIZE_RANGES[e["size"]]
        p = _interval(a, lo, hi) if e["sign"] > 0 else _interval(a, -hi, -lo)
        L += np.log(LAPSE + (1 - LAPSE) * p)
    return L


def offset_features(mask, reference):
    """Offset of `mask` from the reference, per sector and overall, in D of the reference."""
    err = E.sector_errors(mask, reference)
    return dict(s={d: float(v) for d, v in err.items()}, g=float(np.median(list(err.values()))))


def half_width_085(ref):
    """Uniform level-set offset w at which IoU(ref shrunk by w, ref) = 0.85, by bisection."""
    lo, hi = 0.0, 0.5
    for _ in range(28):
        mid = (lo + hi) / 2
        if E.iou(E.apply(ref, E.Edit("shrink", "global", amount=mid)), ref) >= 0.85:
            lo = mid
        else:
            hi = mid
    return lo


class MarginBelief:
    """Grid posterior of the margin. `logp` is the whole state."""

    def __init__(self):
        self.logp = student_t.logpdf(GRID / PRIOR_SCALE, PRIOR_DF)

    def posterior(self):
        p = np.exp(self.logp - self.logp.max())
        return p / p.sum()

    def mean(self):
        return float((GRID * self.posterior()).sum())

    def interval90(self):
        cdf = np.cumsum(self.posterior())
        return [float(GRID[np.searchsorted(cdf, 0.05)]), float(GRID[min(np.searchsorted(cdf, 0.95), len(GRID) - 1)])]

    def add(self, record, accept_tol):
        self.logp = self.logp + log_likelihood(record, accept_tol)


class Listener:
    def __init__(self, engine):
        self.engine = engine
        self.belief, self.ref, self.record0, self.w = None, None, None, None
        self.applied, self.start_offset, self.log = None, None, []
        self.n_images, self.n_events = 0, 0

    def begin_session(self):
        self.belief = MarginBelief()
        self.n_images, self.n_events = 0, 0

    def begin_image(self, start):
        """Set the reference from the lesion head on the start mask, and return the start mask offset by the posterior
        mean.
        """
        self.ref = self.engine.lesion_mask(start)
        if not self.ref.any():
            self.record0, self.w = None, None
        else:
            self.record0 = dict(
                D=float(E.equiv_diameter(self.ref)),
                area=int(self.ref.sum()),
                perim=int(E.boundary_pixels(self.ref).sum()),
            )
            self.w = half_width_085(self.ref)
        self.applied = self.belief.mean()
        self.log = []
        shifted = E.apply(start, E.Edit("shrink", "global", amount=-self.applied))
        self.start_offset = (
            round(float(np.median(list(E.sector_errors(shifted, self.ref).values()))), 4)
            if (self.record0 is not None and shifted.any())
            else None
        )
        return shifted

    def observe_turn(self, mask, parse):
        """Update on the first program of a turn when it is a shrink or grow, measured on the mask before the turn."""
        if parse and parse["type"] == "shrink" and self.record0 is not None and mask.any():
            f = offset_features(mask, self.ref)
            ev = dict(
                kind="edit",
                type="shrink",
                scope="global" if parse["scope"] == "global" else "part",
                direction=None if parse["scope"] == "global" else parse["scope"],
                sign=1 if parse["sign"] == "shrink" else -1,
                size=parse["size"],
                **f,
            )
            self.belief.add(dict(self.record0, events=[ev]), self.w)
            self.n_events += 1
            lo, hi = self.belief.interval90()
            self.log.append(
                dict(
                    parse=dict(scope=parse["scope"], sign=parse["sign"], size=parse["size"]),
                    g=round(f["g"], 4),
                    s={k: round(v, 4) for k, v in f["s"].items()},
                    mean=round(self.belief.mean(), 4),
                    lo=round(lo, 4),
                    hi=round(hi, 4),
                )
            )
        else:
            self.log.append(None)

    def end_image(self, final, accept_parse):
        if accept_parse and accept_parse.get("type") == "accept" and self.record0 is not None and final.any():
            ev = dict(kind="accept", **offset_features(final, self.ref))
            self.belief.add(dict(self.record0, events=[ev]), self.w)
            self.n_events += 1
        self.n_images += 1
        g_final = (
            float(np.median(list(E.sector_errors(final, self.ref).values())))
            if (self.ref is not None and self.ref.any() and final.any())
            else None
        )
        return dict(
            offset_applied=round(self.applied, 4),
            start_offset_measured=self.start_offset,
            mean_after=round(self.belief.mean(), 4),
            final_offset_from_reference=None if g_final is None else round(g_final, 4),
        )

    def summary(self):
        lo, hi = self.belief.interval90()
        return dict(
            mean=self.belief.mean(),
            interval90=[lo, hi],
            grid=GRID,
            posterior=self.belief.posterior(),
            n_images=self.n_images,
            n_events=self.n_events,
        )
