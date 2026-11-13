"""The behavior table of README.md. Each input has a defined error, no-op or warning. CPU, no weights: the network has
random weights and the frozen encoders are stubs.
"""

import warnings

import numpy as np
import pytest
from PIL import Image

from parley import io as IO
from parley.errors import ImageTooLargeError, InputError, ParleyInputWarning


def note(result_warnings, fragment):
    assert any(fragment in w for w in result_warnings), result_warnings


@pytest.fixture
def loaded(stub_editor_factory, phantom):
    ed = stub_editor_factory()
    rgb, mask = phantom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ed.begin_image(rgb, mask)
    return ed, rgb, mask


def edit(ed, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ed.edit(*a, **k)


def test_empty_mask_warns_and_keeps_its_shape(loaded):
    ed, rgb, mask = loaded
    r = edit(ed, np.zeros_like(mask), text="make it a bit bigger")
    note(r.warnings, "empty mask")
    assert r.mask.shape == mask.shape and r.mask.dtype == bool


def test_full_mask_warns_and_keeps_its_shape(loaded):
    ed, rgb, mask = loaded
    r = edit(ed, np.ones_like(mask), text="pull the top in")
    note(r.warnings, "full-image mask")
    assert r.mask.shape == mask.shape


def test_one_pixel_mask_warns(loaded):
    ed, rgb, mask = loaded
    m = np.zeros_like(mask)
    m[100, 100] = True
    r = edit(ed, m, text="make it bigger")
    note(r.warnings, "very small mask")
    assert r.mask.shape == mask.shape


def test_multiple_components_warn(loaded):
    ed, rgb, mask = loaded
    m = np.zeros_like(mask)
    m[20:50, 20:50] = True
    m[150:200, 200:260] = True
    note(edit(ed, m, text="shift it left").warnings, "2 components")


def test_diagonal_contact_is_one_component(loaded):
    ed, rgb, mask = loaded
    m = np.zeros_like(mask)
    m[50:60, 50:60] = True
    m[60:70, 60:70] = True
    assert not any("components" in w for w in edit(ed, m, text="shift it left").warnings)


def test_empty_transcript_is_a_no_op(loaded):
    ed, rgb, mask = loaded
    for text in ("", "   "):
        r = edit(ed, mask, text=text)
        assert r.status == "no_input" and (r.mask == mask).all() and r.programs == []
    assert not ed.can_undo


def test_empty_transcript_can_be_passed_to_the_network(stub_editor_factory, phantom):
    rgb, mask = phantom
    ed = stub_editor_factory(pass_empty_transcript=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ed.begin_image(rgb, mask)
    assert edit(ed, mask, text="").status != "no_input"


def test_silent_audio_is_a_no_op_and_skips_the_recognizer(loaded):
    ed, rgb, mask = loaded
    ed.engine._recognizer = lambda a: pytest.fail("recognizer must not run on silence")
    r = edit(ed, mask, samples=np.zeros(16000, np.float32), sample_rate=16000)
    assert r.status == "no_input" and (r.mask == mask).all() and r.transcript is None


def test_silent_file_is_a_no_op(loaded, tmp_path):
    import soundfile as sf

    ed, rgb, mask = loaded
    ed.engine._recognizer = lambda a: pytest.fail("recognizer must not run on silence")
    p = tmp_path / "s.wav"
    sf.write(p, np.zeros(16000, np.float32), 16000)
    assert edit(ed, mask, wav=str(p)).status == "no_input"


def test_empty_audio_is_a_no_op(loaded):
    ed, rgb, mask = loaded
    assert edit(ed, mask, samples=np.zeros(0, np.float32), sample_rate=16000).status == "no_input"


def test_samples_without_a_rate_are_an_error(loaded):
    ed, rgb, mask = loaded
    with pytest.raises(InputError):
        edit(ed, mask, samples=np.ones(100, np.float32))


def test_wrong_sample_rate_warns_and_is_passed_on(loaded):
    ed, rgb, mask = loaded
    seen = {}
    ed.engine._recognizer = lambda a: seen.update(a) or "make it bigger"
    tone = 0.1 * np.sin(np.linspace(0, 800, 8000)).astype(np.float32)
    r = edit(ed, mask, samples=tone, sample_rate=8000)
    note(r.warnings, "8000 Hz resampled")
    assert seen["sampling_rate"] == 8000


def test_wrong_rate_file_warns(loaded, tmp_path):
    import soundfile as sf

    ed, rgb, mask = loaded
    p = tmp_path / "t.wav"
    sf.write(p, 0.1 * np.sin(np.linspace(0, 800, 44100)), 44100)
    note(edit(ed, mask, wav=str(p)).warnings, "44100 Hz resampled")


def test_stereo_audio_is_mixed_down_with_a_warning(loaded):
    ed, rgb, mask = loaded
    seen = {}
    ed.engine._recognizer = lambda a: seen.update(a) or "make it bigger"
    s = 0.1 * np.ones((16000, 2), np.float32)
    note(edit(ed, mask, samples=s, sample_rate=16000).warnings, "2 audio channels")
    assert seen["raw"].ndim == 1


def test_missing_audio_file_is_an_error(loaded):
    ed, rgb, mask = loaded
    with pytest.raises(InputError):
        edit(ed, mask, wav="/nonexistent.wav")


def test_no_input_at_all_is_an_error(loaded):
    ed, rgb, mask = loaded
    with pytest.raises(InputError):
        edit(ed, mask)


def test_off_image_click_warns_and_runs(loaded):
    ed, rgb, mask = loaded
    r = edit(ed, mask, text="take that out", click=(-50, 10))
    note(r.warnings, "outside the mask")
    assert r.mask.shape == mask.shape


def test_on_image_click_gives_no_warning(loaded):
    ed, rgb, mask = loaded
    assert not any("outside" in w for w in edit(ed, mask, text="take that out", click=(5.5, 10)).warnings)


@pytest.mark.parametrize("bad", [[(1, 2), (3, 4)], "ab", (1,), (1, 2, 3, 4), (float("nan"), 2)])
def test_bad_clicks_are_errors(loaded, bad):
    ed, rgb, mask = loaded
    with pytest.raises(InputError):
        edit(ed, mask, text="take that out", click=bad)


def test_grayscale_warns_and_matches_rgb():
    g = np.tile(np.arange(64, dtype=np.uint8)[None], (64, 1))
    out, w = IO.as_uint8_image(g)
    note(w, "grayscale")
    assert np.array_equal(np.asarray(Image.fromarray(out).convert("RGB")), np.stack([g] * 3, -1))


def test_rgba_warns_and_drops_alpha():
    a = np.zeros((8, 8, 4), np.uint8)
    a[..., 0], a[..., 3] = 200, 7
    out, w = IO.as_uint8_image(a)
    note(w, "RGBA")
    assert (np.asarray(Image.fromarray(out).convert("RGB"))[..., 0] == 200).all()


def test_rgb_uint8_passes_through_untouched():
    a = np.random.default_rng(0).integers(0, 255, (9, 9, 3), dtype=np.uint8)
    out, w = IO.as_uint8_image(a)
    assert out is a or np.array_equal(out, a)
    assert not w


def test_16_bit_image_is_rescaled_with_a_warning():
    a = np.array([[0, 1000], [2000, 4000]], np.uint16)
    out, w = IO.as_uint8_image(a)
    note(w, "rescaled")
    assert out.dtype == np.uint8 and out.max() == 255 and out.min() == 0


def test_16_bit_file_round_trip(tmp_path):
    p = tmp_path / "a16.png"
    Image.fromarray(np.tile(np.arange(0, 60000, 1000, dtype=np.uint16)[None], (10, 1))).save(p)
    out, w = IO.as_uint8_image(IO.load_image(p))
    note(w, "rescaled")
    assert out.max() == 255


def test_constant_16_bit_image_does_not_divide_by_zero():
    out, _ = IO.as_uint8_image(np.full((4, 4), 77, np.uint16))
    assert (out == 0).all()


def test_float_image_range_is_checked():
    with pytest.raises(InputError):
        IO.as_uint8_image(np.full((4, 4), 2.0))
    assert IO.as_uint8_image(np.full((4, 4), 1.0))[0].max() == 255


@pytest.mark.parametrize("shape", [(4,), (4, 4, 5), (2, 2, 2, 2)])
def test_bad_image_shapes_are_errors(shape):
    with pytest.raises(InputError):
        IO.as_uint8_image(np.zeros(shape, np.uint8))


def test_dicom_is_refused_with_a_clear_message():
    with pytest.raises(InputError, match="DICOM"):
        IO.load_image("scan.dcm")


def test_large_image_warns_and_too_large_is_an_error():
    m = np.zeros((5000, 10), bool)
    m[10:20, 2:8] = True
    note(IO.check_mask(m)[1], "above 4096")
    with pytest.raises(ImageTooLargeError):
        IO._check_size(20000, 20000)


def test_size_mismatch_warns():
    note(IO.check_mask(np.ones((10, 10), bool), (20, 20, 3))[1], "differ in size")


def test_mask_file_conventions(tmp_path):
    for top in (1, 255):
        p = tmp_path / f"m{top}.png"
        Image.fromarray((np.eye(16) * top).astype(np.uint8)).save(p)
        assert IO.load_mask(p).sum() == 16
    p = tmp_path / "m16.png"
    Image.fromarray((np.eye(16) * 65535).astype(np.uint16)).save(p)
    assert IO.load_mask(p).sum() == 16
    p = tmp_path / "zero.png"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(p)
    assert IO.load_mask(p).sum() == 0


def test_warnings_can_be_made_errors(loaded):
    ed, rgb, mask = loaded
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParleyInputWarning)
        with pytest.raises(ParleyInputWarning):
            ed.edit(np.zeros_like(mask), text="make it bigger")


def test_large_image_runs_end_to_end(stub_editor_factory):
    ed = stub_editor_factory()
    rgb = np.zeros((4200, 300, 3), np.uint8)
    m = np.zeros((4200, 300), bool)
    m[1000:2000, 100:200] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ed.begin_image(rgb, m)
        r = ed.edit(text="make it a bit bigger")
    assert r.mask.shape == (4200, 300)
