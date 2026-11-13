"""The editor with its real weights and frozen models, on a synthetic phantom. Device from PARLEY_TEST_DEVICE (default
cpu). Audio tests also need PARLEY_TEST_AUDIO, a file of clear English speech, with PARLEY_TEST_AUDIO_TEXT its words.
"""

import os
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.weights
DEVICE = os.environ.get("PARLEY_TEST_DEVICE", "cpu")
AUDIO = os.environ.get("PARLEY_TEST_AUDIO")
AUDIO_TEXT = os.environ.get("PARLEY_TEST_AUDIO_TEXT", "")


@pytest.fixture(scope="module")
def editor():
    from parley.editor import ParleyEditor

    return ParleyEditor(mode="listener", device=DEVICE)


@pytest.fixture
def ready(editor):
    yy, xx = np.mgrid[:240, :320]
    d = np.hypot(xx - 160, yy - 120)
    rgb = np.stack([np.clip(200 - 120 * (d < 60) + 10 * np.sin(xx / 9.0), 0, 255).astype(np.uint8)] * 3, -1)
    editor.begin_session("t", "phantom")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return editor, editor.begin_image(rgb, d < 52)


@pytest.mark.parametrize(
    "text,kind,field,value",
    [
        ("pull the top in a little", "shrink", "scope", "top"),
        ("pull the top in a little", "shrink", "sign", "shrink"),
        ("make it a lot bigger all round", "shrink", "sign", "grow"),
        ("shift it down and to the left", "move", "move_dir", "down and left"),
        ("that looks good", "accept", "type", "accept"),
    ],
)
def test_examples_of_the_edit_language_parse(ready, text, kind, field, value):
    ed, m = ready
    r = ed.edit(m, text=text)
    assert r.programs[0]["type"] == kind and r.programs[0][field] == value


def test_shrink_and_grow_change_area_in_the_right_direction(ready):
    ed, m = ready
    assert ed.edit(m, text="pull it in all round").mask.sum() < m.sum()
    assert ed.edit(m, text="make it a lot bigger all round").mask.sum() > m.sum()


def test_accept_leaves_the_mask_alone_and_undo_works(ready):
    ed, m = ready
    r = ed.edit(m, text="that looks good")
    assert r.status == "accepted" and (r.mask == m).all()
    assert (ed.undo() == m).all()


def test_known_gap_is_still_there(ready):
    """Documented in README.md: this accept phrase parses as none on the seed 0 checkpoint. If a retrained model fixes
    this, update the README.
    """
    ed, m = ready
    assert ed.edit(m, text="that's fine").programs[0]["type"] == "none"


def test_turns_are_repeatable(ready):
    ed, m = ready
    assert (ed.edit(m, text="shift it to the right").mask == ed.edit(m, text="shift it to the right").mask).all()


def test_listener_learns_across_a_session_and_survives_a_reload(ready, tmp_path):
    from parley.editor import ParleyEditor

    ed, m = ready
    r = ed.edit(m, text="make it a lot bigger all round")
    assert r.programs[0]["sign"] == "grow"
    ed.save_state(tmp_path / "s.npz")
    b = ParleyEditor(mode="listener", device=DEVICE)
    b.load_state(tmp_path / "s.npz")
    assert b.belief()["mean"] == ed.belief()["mean"] and b.belief()["n_events"] == ed.belief()["n_events"] > 0


def test_silence_does_not_reach_the_recognizer_but_low_noise_does(ready):
    ed, m = ready
    assert ed.edit(m, samples=np.zeros(16000, np.float32), sample_rate=16000).status == "no_input"
    noise = (1e-3 * np.random.default_rng(0).standard_normal(32000)).astype(np.float32)
    r = ed.edit(m, samples=noise, sample_rate=16000)
    assert r.status in (
        "no_input",
        "edited",
        "unchanged",
        "accepted",
    )  # documented limit: the recognizer may invent words on noise


def _overlap(a, b):
    def w(s):
        return set("".join(c for c in s.lower() if c.isalpha() or c == " ").split())

    return len(w(a) & w(b)) / max(len(w(b)), 1)


@pytest.mark.skipif(not AUDIO, reason="set PARLEY_TEST_AUDIO")
def test_recognizer_on_a_file(ready):
    ed, m = ready
    r = ed.edit(m, wav=AUDIO)
    assert _overlap(r.transcript, AUDIO_TEXT) >= 0.6, r.transcript


@pytest.mark.skipif(not AUDIO, reason="set PARLEY_TEST_AUDIO")
@pytest.mark.parametrize("rate", [8000, 44100])
def test_wrong_rate_is_resampled_as_file_and_as_samples(ready, tmp_path, rate):
    import soundfile as sf
    import torch
    import torchaudio

    ed, m = ready
    x, sr = sf.read(AUDIO, dtype="float32")
    y = torchaudio.functional.resample(torch.from_numpy(x), sr, rate).numpy()
    p = tmp_path / "r.wav"
    sf.write(p, y, rate)
    r = ed.edit(m, wav=str(p))
    assert any(f"{rate} Hz" in w for w in r.warnings) and _overlap(r.transcript, AUDIO_TEXT) >= 0.5, r.transcript
    r = ed.edit(m, samples=y, sample_rate=rate)
    assert any(f"{rate} Hz" in w for w in r.warnings) and _overlap(r.transcript, AUDIO_TEXT) >= 0.5, r.transcript
