"""The public editor: input checking, guards, undo and session state around the engine."""

import json
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from . import io as IO
from .errors import InputError, ParleyInputWarning, SessionStateError

STATE_FORMAT = 1


MODE_ALIASES = {
    "margin": "listener",
    "margin+adapter": "listener+adapter",
}  # the paper calls the listener the margin model


@dataclass
class EditResult:
    """One turn's outcome.

    mask: bool (H, W), same shape as the input mask.
    status: edited | unchanged | accepted | no_input | low_confidence. The last two mean the model did not run an edit.
    transcript: what the program was read from.
    programs: parsed programs, at most two, each with type, scope, place, move_dir, sign, size, amount_D, move_D and
    confidence.
              Only some fields apply: scope, sign, size and amount_D to shrink; move_dir and move_D to move; place to
              add and remove.
    changed_fraction: share of pixels that changed.
    warnings: human-readable notes on the inputs.
    """

    mask: np.ndarray
    status: str
    transcript: str | None = None
    programs: list = field(default_factory=list)
    changed_fraction: float = 0.0
    warnings: list = field(default_factory=list)


def describe_program(p):
    """One short line for a parsed program."""
    t = p["type"]
    if t == "shrink":
        where = "whole contour" if p["scope"] == "global" else p["scope"]
        return f"{p['sign']} {where}, {p['size']} ({p['amount_D']:.3f} D)"
    if t == "move":
        return f"move {p['move_dir']} ({p['move_D'][0]:+.3f}, {p['move_D'][1]:+.3f} D)"
    if t in ("add", "remove"):
        return f"{t} region, {p['place']} side"
    return t


class ParleyEditor:
    """Speech-directed mask editor.

    mode: base, listener, adapter or listener+adapter. "margin" and "margin+adapter" are aliases of the two listener
    modes (the margin model).
    seed: which released checkpoint (0, 1 or 2).
    weights: path of a checkpoint or of a directory; see `parley.weights`.
    verify: check the checkpoint's SHA-256. Turn off only for weights you trained yourself.
    pass_empty_transcript: run the network on an empty transcript instead of returning no_input.
    min_confidence: if set, a turn whose first program has a lower type confidence is not executed (status
    low_confidence). Off by default; no threshold has been validated.
    """

    def __init__(
        self,
        mode="base",
        seed=0,
        device="cuda",
        weights=None,
        verify=True,
        pass_empty_transcript=False,
        min_confidence=None,
        engine=None,
    ):
        from . import weights as W
        from .engine import Engine

        mode = MODE_ALIASES.get(mode, mode)
        self.mode, self.seed, self.device = mode, seed, device
        self.pass_empty_transcript, self.min_confidence = pass_empty_transcript, min_confidence
        if engine is None:
            path, self.weights_sha256 = W.resolve(seed, weights, verify)
            engine = Engine(str(path), device=device, mode=mode)
        else:
            self.weights_sha256 = None
        self.engine = engine
        self.clinician_id, self.dataset = None, None
        self.current = None
        self._image_shape = None
        self._undo = []

    # ---- reporting
    def _warn(self, messages):
        for m in messages:
            warnings.warn(m, ParleyInputWarning, stacklevel=3)
        return list(messages)

    # ---- session and image
    def begin_session(self, clinician_id="anonymous", dataset="default"):
        """Start one clinician on one dataset. The listener returns to its prior and the adapters to a fresh
        initialization.

        `dataset` is only a label stored in the state file: the prior does not depend on it, and any name is accepted.
        """
        self.clinician_id, self.dataset = clinician_id, dataset
        self.engine.begin_session()

    def begin_image(self, image, start_mask, features=None):
        """`image`: array or path. `start_mask`: array or path. Returns the start mask, offset by the listener's mean in
        the listener modes.

        `features`: optional precomputed (features, image tensor), which skips the image encoder.
        """
        if self.clinician_id is None:
            self.begin_session()
        notes = []
        rgb = None
        if image is not None:
            raw = IO.load_image(image) if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
            rgb, w = IO.as_uint8_image(raw)
            notes += w
        elif features is None:
            raise InputError("pass an image or precomputed features")
        m = (
            IO.load_mask(start_mask)
            if isinstance(start_mask, (str, bytes)) or hasattr(start_mask, "__fspath__")
            else start_mask
        )
        m, w = IO.check_mask(m, None if rgb is None else rgb.shape)
        notes += w
        self.last_warnings = self._warn(notes)
        self._image_shape = m.shape
        self.current = self.engine.begin_image(m, rgb=rgb, features=features)
        self._undo = []
        return self.current

    # ---- one turn
    def edit(self, mask=None, text=None, wav=None, samples=None, sample_rate=None, click=None, is_edit=True):
        """One turn on `mask` (default: the current mask).

        Give `text`, or `wav` (a path), or `samples` with `sample_rate`. `text` wins if several are given. `click` is
        one (x, y) in mask pixels.
        `is_edit=False` marks a remark that the adapters must not learn from.
        """
        if self.current is None and mask is None:
            raise InputError("call begin_image first")
        m = self.current if mask is None else mask
        m, notes = IO.check_mask(m)
        click, w = IO.check_click(click, m.shape)
        notes += w
        transcript = text
        if transcript is None:
            if wav is None and samples is None:
                raise InputError("pass text, wav or samples")
            target, silent, w = IO.check_audio(path=wav, samples=samples, rate=sample_rate)
            notes += w
            if silent:
                notes.append("audio is silent: no edit")
                return EditResult(m.copy(), "no_input", None, warnings=self._warn(notes))
            transcript = self.engine.recognizer(target)
        if not transcript.strip() and not self.pass_empty_transcript:
            notes.append("empty transcript: no edit")
            return EditResult(m.copy(), "no_input", transcript, warnings=self._warn(notes))
        snapshot = self._snapshot(m)
        new, info = self.engine.step(m, transcript, click=click, edit=is_edit)
        programs = info["programs"]
        if self.min_confidence is not None and programs[0]["confidence"] < self.min_confidence:
            self._restore(snapshot)
            notes.append(f"type confidence {programs[0]['confidence']:.3f} below {self.min_confidence}: no edit")
            return EditResult(m.copy(), "low_confidence", transcript, programs, warnings=self._warn(notes))
        self._undo.append(snapshot)
        self.current = new
        changed = float((new != m).mean())
        status = "accepted" if info["accept"] else ("edited" if changed > 0 else "unchanged")
        return EditResult(new, status, transcript, programs, changed, self._warn(notes))

    def transcribe(self, wav=None, samples=None, sample_rate=None):
        """The recognizer alone. Returns (transcript or None when the audio is silent, warnings)."""
        target, silent, w = IO.check_audio(path=wav, samples=samples, rate=sample_rate)
        return (None, w) if silent else (self.engine.recognizer(target), w)

    def _snapshot(self, m):
        li = self.engine.listener
        return dict(
            mask=m.copy(),
            current=None if self.current is None else self.current.copy(),
            n_turns=len(self.engine.turns),
            logp=None if li is None else li.belief.logp.copy(),
            n_events=None if li is None else li.n_events,
            n_log=None if li is None else len(li.log),
        )

    def _restore(self, s):
        li = self.engine.listener
        del self.engine.turns[s["n_turns"] :]
        if li is not None:
            li.belief.logp, li.n_events = s["logp"], s["n_events"]
            del li.log[s["n_log"] :]

    def undo(self):
        """Step back one turn: the mask, the listener's belief and the turn list return to where they were. Returns the
        mask, or None if there is nothing to undo.
        """
        if not self._undo:
            return None
        s = self._undo.pop()
        self._restore(s)
        self.current = s["mask"]
        return self.current

    @property
    def can_undo(self):
        return bool(self._undo)

    def end_image(self, final_mask=None, accepted=False, accept_text=None):
        """Close the image. Only an accepted image teaches: the adapters fit on its turns, and `accept_text` (a phrase
        heard after the final mask) is read as an accept event by the listener. Undo history is cleared.
        """
        m = self.current if final_mask is None else final_mask
        out = self.engine.end_image(m, accepted, accept_text)
        self._undo = []
        return out

    def belief(self):
        """The listener's belief: mean, interval90 (both in units of D), grid, posterior, n_images, n_events. None
        outside the listener modes.
        """
        return None if self.engine.listener is None else self.engine.listener.summary()

    # ---- session state
    def save_state(self, path):
        """Write what the session has learned: the listener's log posterior and the adapter matrices. Does not hold
        images, masks or the current image's turns.
        """
        li, ad = self.engine.listener, self.engine.adapters
        meta = dict(
            format=STATE_FORMAT,
            package=__version__,
            mode=self.mode,
            seed=self.seed,
            weights_sha256=self.weights_sha256,
            clinician_id=self.clinician_id,
            dataset=self.dataset,
            n_images=None if li is None else li.n_images,
            n_events=None if li is None else li.n_events,
            n_fits=None if ad is None else ad.n_fits,
        )
        arrays = {}
        if li is not None:
            arrays["listener_logp"] = li.belief.logp
        if ad is not None:
            arrays.update(ad.state_arrays())
        with open(path, "wb") as f:
            np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
        return meta

    def load_state(self, path, dataset=None):
        """Restore a saved session into this editor. The mode must match, and for adapters so must the seed. If
        `dataset` is given and differs from the saved one, loading fails: a new dataset starts a new session.
        """
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError) as e:
            raise SessionStateError(f"cannot read session state {path}: {e}") from e
        if meta.get("format") != STATE_FORMAT:
            raise SessionStateError(f"state format {meta.get('format')} is not supported")
        if meta["mode"] != self.mode:
            raise SessionStateError(f"state was saved in mode {meta['mode']} and this editor is in mode {self.mode}")
        li, ad = self.engine.listener, self.engine.adapters
        if ad is not None and meta["seed"] != self.seed:
            raise SessionStateError(
                f"adapters were fitted on seed {meta['seed']} and this editor uses seed {self.seed}"
            )
        if dataset is not None and dataset != meta["dataset"]:
            raise SessionStateError(
                f"state belongs to dataset {meta['dataset']!r}, not {dataset!r}: a new dataset starts a new session"
            )
        self.begin_session(meta["clinician_id"], meta["dataset"])
        if li is not None:
            li.belief.logp = z["listener_logp"]
            li.n_images, li.n_events = meta["n_images"], meta["n_events"]
        if ad is not None:
            ad.load_arrays({k: z[k] for k in z.files if k.startswith("adapter_")})
            ad.n_fits = meta["n_fits"]
        self._undo = []
        return meta
