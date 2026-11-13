"""Outcomes of a dialogue and of a study.

reach: the IoU with the target reaches a threshold (default 0.85) at any of T turns (default 10). The mask the editor
shows first counts as turn 0, so a start that is already good has reached.
       Use `needs_correction=True` in `summarize` to report only dialogues whose given start mask was below the
       threshold. An editor that changes the start mask does not change this slice.
gain: IoU of the last mask with the target minus IoU of the given start mask with the target. A study reports the mean.
gain_stop: the same gain under a common stopping rule: the first mask at or above the threshold, else the last mask.
improved, worse: shares of dialogues with gain above and below zero. The remainder is unchanged.
simulated time: fixed costs per turn: seconds per word spoken, a recognizer delay on turns with speech, seconds per
click, and a review of the result.
       time_s is time spent: the turns taken, plus, for a clinician who gives up, the remaining turns up to T at that
       dialogue's mean cost per turn, so that an editor cannot look fast by driving clinicians away.
       time_reach is time to the first mask at or above the threshold. A dialogue that never reaches it, however it
       ended, is charged the remaining turns up to T at its mean cost per turn.

Not computed: agreement on the edit type (the simulator interface carries no intended type), exclusion of non-edit
turns from time, time on later cases only, and any floor at a clinician's mean cost per turn, since the kit charges
each dialogue on its own.
"""

from dataclasses import dataclass

import numpy as np

from .bootstrap import cluster_bootstrap


@dataclass(frozen=True)
class TimeCosts:
    per_word: float = 0.4
    recognizer: float = 0.5
    per_click: float = 2.0
    review: float = 4.0


def reach(rec, threshold=0.85, max_turns=10):
    return any(v >= threshold for v in rec.ious[: max_turns + 1])


def start_iou(rec):
    """IoU of the given start mask, which `ious[0]` need not be: an editor may change the mask before the first turn."""
    return rec.ious[0] if rec.start_iou is None else rec.start_iou


def gain(rec):
    return rec.ious[-1] - start_iou(rec)


def gain_stop(rec, threshold=0.85):
    return next((v for v in rec.ious if v >= threshold), rec.ious[-1]) - start_iou(rec)


def simulated_time(rec, costs=TimeCosts(), until_reach=None, max_turns=10):
    """Seconds for the turns taken, plus the remaining turns up to `max_turns` if the clinician gave up.

    With `until_reach` set to a threshold, a dialogue that reaches it counts only the turns up to the first that
    does, and one that does not is charged the remaining turns however it ended.
    A dialogue with no turns has no mean cost, so each charged turn costs the review alone. This is a lower bound:
    the kit has no floor at the clinician's mean cost per turn.
    """
    per_turn = [
        costs.per_word * w + (costs.recognizer if w else 0.0) + costs.per_click * c + costs.review
        for w, c in zip(rec.words, rec.clicks)
    ]
    if until_reach is not None:
        first = next((t for t, v in enumerate(rec.ious) if v >= until_reach), None)
        if first is not None:
            return sum(per_turn[:first])
    left = max(max_turns - len(per_turn), 0) if rec.gave_up or until_reach is not None else 0
    return sum(per_turn) + left * (sum(per_turn) / len(per_turn) if per_turn else costs.review)


def summarize(records, threshold=0.85, max_turns=10, costs=TimeCosts(), needs_correction=False, n_boot=2000, seed=0):
    """Mean and clustered 95% interval of each outcome. Returns a dict of {outcome: dict(mean, lo, hi)}, plus n,
    clinicians and images.
    """
    rows = [r for r in records if not needs_correction or start_iou(r) < threshold]
    if not rows:
        return dict(n=0, clinicians=0, images=0)
    clinicians = [r.clinician for r in rows]
    images = [f"{r.dataset}/{r.case}" for r in rows]
    values = dict(
        reach=[float(reach(r, threshold, max_turns)) for r in rows],
        gain=[gain(r) for r in rows],
        gain_stop=[gain_stop(r, threshold) for r in rows],
        improved=[float(gain(r) > 0) for r in rows],
        worse=[float(gain(r) < 0) for r in rows],
        time_s=[simulated_time(r, costs, max_turns=max_turns) for r in rows],
        time_reach=[simulated_time(r, costs, threshold, max_turns) for r in rows],
        turns=[float(len(r.words)) for r in rows],
    )
    out = dict(n=len(rows), clinicians=len(set(clinicians)), images=len(set(images)))
    for k, v in values.items():
        lo, hi = cluster_bootstrap(np.asarray(v, float), clinicians, images, n_boot=n_boot, seed=seed)
        out[k] = dict(mean=float(np.mean(v)), lo=lo, hi=hi)
    return out


COLUMNS = ("Arm", "n", "Clinicians", "Images", "Reach", "Gain", "Gain, stopping rule", "Improved", "Worse",
           "Time spent (s)", "Time to reach (s)")  # fmt: skip
CELLS = (("reach", ".3f"), ("gain", "+.3f"), ("gain_stop", "+.3f"), ("improved", ".3f"), ("worse", ".3f"),
         ("time_s", ".1f"), ("time_reach", ".1f"))  # fmt: skip


def _interval(s, key, fmt):
    v = s[key]
    return f"{v['mean']:{fmt}} [{v['lo']:{fmt}}, {v['hi']:{fmt}}]"


def summary_markdown(by_arm):
    """A reporting table from {arm: summarize(...)}. Each cell is the mean with its clustered 95% interval."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for arm, s in by_arm.items():
        if not s["n"]:
            cells = [arm, 0, 0, 0] + [""] * len(CELLS)
        else:
            cells = [arm, s["n"], s["clinicians"], s["images"]] + [_interval(s, k, f) for k, f in CELLS]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)
