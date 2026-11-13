"""Command line: `parley edit`, `parley weights`, `parley demo`.

Exit codes: 0 success, 2 bad input, 3 weights problem.
"""

import argparse
import json
import os
import sys
import warnings


def set_process_title():
    """Rename the process when PARLEY_PROCTITLE is set and setproctitle is installed. Useful on shared GPU machines."""
    title = os.environ.get("PARLEY_PROCTITLE")
    if title:
        try:
            from setproctitle import setproctitle

            setproctitle(title)
        except ImportError:
            pass


def _add_model_args(p):
    p.add_argument(
        "--mode",
        default="base",
        choices=["base", "listener", "adapter", "listener+adapter", "margin", "margin+adapter"],
    )
    p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2], help="which released checkpoint")
    p.add_argument("--device", default=None, help="cuda, cuda:1 or cpu; default cuda when available")
    p.add_argument(
        "--weights",
        default=None,
        help="checkpoint file or directory; default PARLEY_WEIGHTS, then ~/.cache/parley/weights",
    )
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check, for weights you trained yourself")


def _device(arg):
    if arg:
        return arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _edit(a):
    import numpy as np
    from PIL import Image

    from .editor import ParleyEditor, describe_program

    if a.text is None and a.wav is None:
        raise SystemExit("parley edit: pass --text or --wav")
    ed = ParleyEditor(
        mode=a.mode,
        seed=a.seed,
        device=_device(a.device),
        weights=a.weights,
        verify=not a.no_verify,
        min_confidence=a.min_confidence,
    )
    if a.state and os.path.exists(a.state):
        ed.load_state(a.state)
    else:
        ed.begin_session(a.clinician, a.dataset)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ed.begin_image(a.image, a.mask)
        res = ed.edit(text=a.text, wav=a.wav, click=None if a.click is None else tuple(a.click))
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    Image.fromarray((res.mask.astype(np.uint8)) * 255).save(a.out)
    if a.state:
        ed.save_state(a.state)
    out = dict(
        transcript=res.transcript,
        status=res.status,
        programs=res.programs,
        changed_fraction=round(res.changed_fraction, 6),
        output=a.out,
    )
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"transcript: {res.transcript}")
        print(f"status: {res.status}")
        for k, p in enumerate(res.programs):
            print(f"program {k + 1}: {describe_program(p)} [type confidence {p['confidence']:.2f}]")
        print(f"changed: {100 * res.changed_fraction:.2f}% of pixels")
        print(f"written: {a.out}")
    return 0


def _weights(a):
    from . import weights as W

    path = W.locate(a.seed, a.weights)
    print(f"path: {path}")
    print(f"size: {path.stat().st_size} bytes")
    if a.no_verify:
        print(f"sha256: {W.sha256(path)} (not compared)")
    else:
        print(f"sha256: {W.verify(path, a.seed)} (matches the released value)")
    return 0


def _demo(a):
    from .demo import launch

    launch(mode=a.mode, seed=a.seed, device=_device(a.device), weights=a.weights, verify=not a.no_verify, port=a.port)
    return 0


def main(argv=None):
    from . import __version__
    from .errors import (
        InputError,
        SessionStateError,
        UnsupportedCheckpointError,
        WeightsHashMismatchError,
        WeightsNotFoundError,
    )

    set_process_title()
    ap = argparse.ArgumentParser(
        prog="parley", description="Speech-directed segmentation mask editor. Research prototype, not for clinical use."
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit", help="apply one spoken or written correction to a mask")
    e.add_argument("--image", required=True)
    e.add_argument("--mask", required=True)
    e.add_argument("--text", default=None, help="the correction as text")
    e.add_argument("--wav", default=None, help="the correction as an audio file, English")
    e.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"), default=None, help="one click in mask pixels")
    e.add_argument("--out", required=True, help="output PNG, foreground 255")
    e.add_argument("--state", default=None, help="session state file, read if present and written after the edit")
    e.add_argument("--clinician", default="anonymous")
    e.add_argument("--dataset", default="default")
    e.add_argument("--min-confidence", type=float, default=None, help="do not execute below this type confidence")
    e.add_argument("--json", action="store_true", help="print the result as JSON")
    _add_model_args(e)
    e.set_defaults(fn=_edit)
    w = sub.add_parser("weights", help="show which weights would be used and check them")
    w.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    w.add_argument("--weights", default=None)
    w.add_argument("--no-verify", action="store_true")
    w.set_defaults(fn=_weights)
    d = sub.add_parser("demo", help="local web demo on 127.0.0.1 (needs the demo extra)")
    _add_model_args(d)
    d.add_argument("--port", type=int, default=7860)
    d.set_defaults(fn=_demo)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (InputError, SessionStateError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (WeightsNotFoundError, WeightsHashMismatchError, UnsupportedCheckpointError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
