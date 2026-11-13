"""Run dialogues. An editor under test needs four methods:

    begin_session(clinician_id, dataset)
    begin_image(case) -> start mask
    step(mask, text, click) -> (new mask, info dict)
    end_image(mask, reached, accept_text)

`ParleyEvalAdapter` provides them for `ParleyEditor`. Any other conversational editor can be wrapped the same way.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from ..edits import iou
from .clinician import Observation


@dataclass
class DialogueRecord:
    """One dialogue. `start_iou` is the IoU of the given start mask; `ious[0]` is that of the mask the editor shows
    first, which may differ; `ious[t]` follows turn t. `words` and `clicks` have one entry per turn.
    """

    clinician: str
    case: str
    dataset: str
    arm: str
    ious: list[float] = field(default_factory=list)
    words: list[int] = field(default_factory=list)
    clicks: list[int] = field(default_factory=list)
    utterances: list[str] = field(default_factory=list)
    programs: list[list] = field(default_factory=list)
    accepted: bool = False
    gave_up: bool = False
    start_iou: float | None = None

    def to_dict(self):
        return asdict(self)


def run_dialogue(editor, sim, case, arm="editor", max_turns=10):
    """One case, one clinician: up to `max_turns` turns, stopping when the clinician accepts or gives up."""
    target = sim.target(case).astype(bool)
    sim.begin_image(case)
    mask = np.asarray(editor.begin_image(case)).astype(bool)
    rec = DialogueRecord(
        sim.clinician_id,
        case.case_id,
        case.dataset,
        arm,
        ious=[iou(mask, target)],
        start_iou=iou(np.asarray(case.start_mask).astype(bool), target),
    )
    for t in range(max_turns + 1):
        obs = Observation(case, mask, t)
        if sim.accept(obs):
            rec.accepted = True
            break
        if t == max_turns:
            break
        utterance = sim.next_utterance(obs)
        if utterance is None:
            rec.gave_up = True
            break
        click = sim.click(obs, utterance)
        mask, info = editor.step(mask, utterance, click)
        mask = np.asarray(mask).astype(bool)
        rec.ious.append(iou(mask, target))
        rec.words.append(len(utterance.split()))
        rec.clicks.append(0 if click is None else 1)
        rec.utterances.append(utterance)
        rec.programs.append(info.get("programs", info.get("types", [])))
    editor.end_image(mask, rec.accepted, sim.accept_phrase if rec.accepted else None)
    return rec


def run_study(editor, clinicians, cases_by_dataset, arm="editor", max_turns=10):
    """Every clinician on every case of each dataset. One session per clinician and dataset. Returns a list of
    DialogueRecord.
    """
    out = []
    for sim in clinicians:
        for dataset, cases in cases_by_dataset.items():
            editor.begin_session(sim.clinician_id, dataset)
            for case in cases:
                out.append(run_dialogue(editor, sim, case, arm, max_turns))
    return out


class ParleyEvalAdapter:
    """Lets the runner drive a ParleyEditor. `wav_for(text)` may return an audio path to send the turn through the
    recognizer.
    """

    def __init__(self, editor, wav_for=None):
        self.editor, self.wav_for = editor, wav_for

    def begin_session(self, clinician_id, dataset):
        self.editor.begin_session(clinician_id, dataset)

    def begin_image(self, case):
        return self.editor.begin_image(case.image, case.start_mask)

    def step(self, mask, text, click):
        wav = self.wav_for(text) if self.wav_for else None
        r = self.editor.edit(mask, text=None if wav else text, wav=wav, click=click)
        return r.mask, dict(programs=r.programs, status=r.status, transcript=r.transcript)

    def end_image(self, mask, reached, accept_text):
        self.editor.end_image(mask, reached, accept_text)
