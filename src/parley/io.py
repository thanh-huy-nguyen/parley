"""Reading and checking images, masks, clicks and audio. Every function returns its warnings so the caller can report
them.

Nothing here runs a model. The rules are listed in docs/usage.md (behavior table) and tested in
tests/test_edge_cases.py.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ImageTooLargeError, InputError

MAX_PIXELS = 178_956_970  # the decompression-bomb limit of Pillow
LARGE_SIDE = 4096  # px, long side above which a warning is given
SILENCE_RMS = 1e-4  # root mean square of samples in -1..1, about -80 dBFS
EXPECTED_RATE = 16000


def _check_size(h, w):
    if h * w > MAX_PIXELS:
        raise ImageTooLargeError(f"{w} x {h} is above the limit of {MAX_PIXELS:,} pixels.")
    if h == 0 or w == 0:
        raise InputError("empty array")


def as_uint8_image(arr):
    """Any supported image array -> (uint8 array accepted by the engine, warnings). uint8 (H, W, 3) passes through
    untouched.
    """
    arr = np.asarray(arr)
    warns = []
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (2, 3, 4)):
        raise InputError(f"image must be (H, W) or (H, W, 1, 2, 3 or 4), got shape {arr.shape}")
    _check_size(*arr.shape[:2])
    if arr.dtype == np.uint8:
        pass
    elif arr.dtype in (np.uint16, np.int16, np.uint32, np.int32):
        lo, hi = float(arr.min()), float(arr.max())
        warns.append(f"{arr.dtype} image rescaled linearly from its own range [{lo:g}, {hi:g}] to 8 bits")
        arr = (
            np.zeros(arr.shape, np.uint8)
            if hi <= lo
            else np.round((arr.astype(np.float64) - lo) / (hi - lo) * 255.0).astype(np.uint8)
        )
    elif arr.dtype in (np.float32, np.float64):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise InputError("a floating-point image must be finite and in 0..1")
        warns.append("floating-point image scaled from 0..1 to 8 bits")
        arr = np.round(arr * 255.0).astype(np.uint8)
    else:
        raise InputError(f"unsupported image dtype {arr.dtype}")
    if arr.ndim == 2:
        warns.append("grayscale image repeated to three channels")
    elif arr.shape[2] == 2:
        warns.append("grayscale image with alpha: alpha dropped, gray repeated to three channels")
    elif arr.shape[2] == 4:
        warns.append("RGBA image: alpha dropped without compositing")
    return arr, warns


def load_image(path):
    """Read an image file into an array of its own bit depth. DICOM is not read."""
    p = Path(path)
    if p.suffix.lower() in (".dcm", ".dicom"):
        raise InputError("DICOM is not supported: export the frame to PNG first")
    try:
        with Image.open(p) as im:
            return np.array(im)
    except Image.DecompressionBombError as e:
        raise ImageTooLargeError(str(e)) from e
    except (OSError, ValueError) as e:
        raise InputError(f"cannot read image {p}: {e}") from e


def load_mask(path):
    """Read a mask file: foreground is above half of the file's maximum, so 0/1, 0/255 and 0/65535 files all work. An
    alpha or color mask uses its first channel.
    """
    arr = load_image(path)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    top = arr.max() if arr.size else 0
    return arr > top / 2 if top > 0 else np.zeros(arr.shape, bool)


def check_mask(mask, image_shape=None):
    """Array -> (bool (H, W) mask, warnings). Any nonzero value is foreground."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise InputError(f"mask must be (H, W), got shape {m.shape}")
    _check_size(*m.shape)
    m = m.astype(bool)
    warns = []
    if not m.any():
        warns.append("empty mask: shrink, grow and move cannot change it, and only an add program can")
    elif m.all():
        warns.append("full-image mask: shrink, grow and move cannot change it, and only a remove program can")
    else:
        area = int(m.sum())
        s = 512 * 512 / (m.shape[0] * m.shape[1])
        if area * s < 50:
            warns.append(
                f"very small mask ({area} px, about {area * s:.0f} px on the working grid): "
                "D is floored at 8 working pixels, and the mask may vanish on resampling"
            )
        from scipy import ndimage as ndi

        n = ndi.label(m, structure=np.ones((3, 3), int))[1]
        if n > 1:
            warns.append(
                f"mask has {n} components: D and the side windows use all of them together, "
                "and a move moves all of them"
            )
    if max(m.shape) > LARGE_SIDE:
        warns.append(
            f"long side {max(m.shape)} px is above {LARGE_SIDE}: the network works at 512 x 512, "
            "so the new boundary has only that resolution, and listener updates are slow"
        )
    if image_shape is not None and tuple(image_shape[:2]) != m.shape:
        warns.append(
            f"image {tuple(image_shape[:2])} and mask {m.shape} differ in size: "
            "both are resampled to 512 x 512, and clicks are read in mask pixels"
        )
    return m, warns


def check_click(click, shape):
    """(x, y) or (x, y, label) -> (click, warnings). One click per turn. Coordinates are in mask pixels; x to the right,
    y downward; floats allowed.
    """
    if click is None:
        return None, []
    if isinstance(click, (str, bytes)) or not hasattr(click, "__len__") or len(click) not in (2, 3):
        raise InputError("click must be one (x, y) pair; several clicks in one turn are not supported")
    try:
        c = tuple(float(v) for v in click)
    except (TypeError, ValueError) as e:
        raise InputError(
            "click must be one (x, y) pair of numbers; several clicks in one turn are not supported"
        ) from e
    if not all(np.isfinite(c)):
        raise InputError("click coordinates must be finite")
    warns = []
    h, w = shape
    if not (0 <= c[0] <= w - 1 and 0 <= c[1] <= h - 1):
        warns.append(
            f"click ({c[0]:g}, {c[1]:g}) is outside the mask ({w} x {h}); "
            "it is passed on as given, and its effect fades with distance"
        )
    return click, warns


def check_audio(path=None, samples=None, rate=None):
    """Inspect audio without changing it. Returns (input for the recognizer, is_silent, warnings).

    `path` goes to the recognizer as a file, which reads and resamples it through ffmpeg. `samples` must come with
    `rate`.
    """
    import soundfile as sf

    warns = []
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise InputError(f"no audio file at {p}")
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        except sf.LibsndfileError as e:
            raise InputError(f"cannot read audio file {p}: {e}") from e
        target = str(p)
    else:
        if rate is None:
            raise InputError("audio samples need a sample rate")
        data, sr = np.asarray(samples), int(rate)
        if data.ndim not in (1, 2):
            raise InputError(f"audio must be (n) or (n, channels), got shape {data.shape}")
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        data = data.astype(np.float32)
        if data.ndim == 1:
            data = data[:, None]
        target = None
    if data.size == 0:
        return target, True, warns + ["audio has no samples"]
    if data.shape[1] > 1:
        warns.append(f"{data.shape[1]} audio channels mixed down to mono")
    if sr != EXPECTED_RATE:
        warns.append(f"audio at {sr} Hz resampled to {EXPECTED_RATE} Hz")
    if target is None:
        target = dict(raw=data.mean(1), sampling_rate=sr)
    silent = float(np.sqrt(np.mean(data**2))) < SILENCE_RMS
    return target, silent, warns
