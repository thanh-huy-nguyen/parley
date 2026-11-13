"""EXAMPLE ONLY. A small simulated clinician that shows how to implement the interface.

Results from this simulator say nothing about clinical use or about how an editor generalizes: it has one short
phrase per situation, and it was not built to be independent of any editor.
It looks at the mask, measures the boundary error toward its target on four sides, and asks for the largest correction.
"""

import numpy as np
from scipy import ndimage as ndi

from .. import edits as E
from .clinician import SimulatedClinician

AMOUNT = [(0.04, "a touch"), (0.12, "a fair bit"), (np.inf, "a lot")]
SIDE = {"top": "upper edge", "bottom": "lower edge", "left": "left edge", "right": "right edge"}


class ToyClinician(SimulatedClinician):
    """EXAMPLE ONLY. `margin` is in units of D: positive means this clinician draws outside the reference."""

    def __init__(self, clinician_id="toy", margin=0.0, accept_iou=0.85, min_error=0.01):
        self.clinician_id, self.margin, self.accept_iou, self.min_error = clinician_id, margin, accept_iou, min_error
        self._targets = {}

    def target(self, case):
        key = (case.dataset, case.case_id)
        if key not in self._targets:
            self._targets[key] = E.apply(case.reference.astype(bool), E.Edit("shrink", "global", amount=-self.margin))
        return self._targets[key]

    def accept(self, obs):
        return E.iou(obs.mask, self.target(obs.case)) >= self.accept_iou

    def _missing_blob(self, mask, target):
        """The largest piece of the target far from the current mask, if there is one: (kind, region)."""
        for kind, region in (("add", target & ~mask), ("remove", mask & ~target)):
            lab, n = ndi.label(region, structure=E.CONNECTIVITY)
            if n:
                sizes = ndi.sum(region, lab, range(1, n + 1))
                k = int(np.argmax(sizes)) + 1
                if ndi.distance_transform_edt(lab == k).max() > 0.15 * E.equiv_diameter(target):
                    return kind, lab == k
        return None

    def next_utterance(self, obs):
        target = self.target(obs.case)
        if not target.any():
            return None
        if not obs.mask.any():
            return "there is a lesion here that is not marked"
        blob = self._missing_blob(obs.mask, target)
        if blob is not None:
            return "there is a piece missing here" if blob[0] == "add" else "take out this part"
        err = E.sector_errors(obs.mask, target)
        side = max(err, key=lambda s: abs(err[s]))
        if abs(err[side]) < self.min_error:
            return None
        words = next(w for limit, w in AMOUNT if abs(err[side]) < limit)
        return f"bring the {SIDE[side]} in {words}" if err[side] > 0 else f"push the {SIDE[side]} out {words}"

    def click(self, obs, utterance):
        if utterance.startswith("bring") or utterance.startswith("push"):
            return None
        blob = self._missing_blob(obs.mask, self.target(obs.case))
        if blob is None:
            return None
        y, x = ndi.center_of_mass(blob[1])
        return float(x), float(y)
