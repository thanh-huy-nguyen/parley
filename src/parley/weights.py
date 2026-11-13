"""Where the weights are and whether they are the released ones.

Lookup order for the editor checkpoint:
    1. the `weights` argument (a file, or a directory holding parley-editor-s{seed}.pt)
    2. the environment variable PARLEY_WEIGHTS (same forms)
    3. ~/.cache/parley/weights/parley-editor-s{seed}.pt

Verification compares the SHA-256 of the checkpoint with `weights_manifest.json` in this package. Pass
`verify=False`, or set
PARLEY_NO_VERIFY=1, or use `--no-verify` on the command line, to run weights you trained yourself.

The three frozen Hugging Face models are always loaded at the commit named in the manifest.
"""

import hashlib
import json
import os
from importlib import resources
from pathlib import Path

from .errors import WeightsHashMismatchError, WeightsNotFoundError

ENV_WEIGHTS = "PARLEY_WEIGHTS"
ENV_NO_VERIFY = "PARLEY_NO_VERIFY"
DEFAULT_DIR = Path.home() / ".cache" / "parley" / "weights"
SAM2_REPO = "facebook/sam2.1-hiera-tiny"
TEXT_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ASR_REPO = "openai/whisper-small.en"


def manifest():
    return json.loads(resources.files("parley").joinpath("weights_manifest.json").read_text())


def hf_revision(repo):
    return manifest()["hf"][repo]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_name(seed):
    return f"parley-editor-s{seed}.pt"


def locate(seed=0, weights=None):
    """Return the checkpoint path for `seed`, following the lookup order above. Raises WeightsNotFoundError listing what
    was tried.
    """
    tried = []
    for source, value in (("argument", weights), (ENV_WEIGHTS, os.environ.get(ENV_WEIGHTS)), ("default", DEFAULT_DIR)):
        if value is None or value == "":
            continue
        p = Path(value).expanduser()
        candidate = p / file_name(seed) if p.is_dir() else p
        tried.append(f"{source}: {candidate}")
        if candidate.is_file():
            return candidate
    raise WeightsNotFoundError(
        f"no weights for seed {seed}. Tried {'; '.join(tried)}. "
        f"Pass `weights=`, set {ENV_WEIGHTS}, or place {file_name(seed)} in {DEFAULT_DIR}."
    )


def verification_disabled(verify=True):
    return (not verify) or os.environ.get(ENV_NO_VERIFY, "") not in ("", "0")


def verify(path, seed=0):
    """Compare a checkpoint with the released SHA-256 of `seed`.

    Returns the digest, or raises WeightsHashMismatchError.
    """
    got = sha256(path)
    want = manifest()["weights"][str(seed)]["sha256"]
    if got != want:
        raise WeightsHashMismatchError(
            f"{path} has SHA-256 {got}, and the released weights for seed {seed} have {want}. "
            f"If you retrained the model, pass verify=False, set {ENV_NO_VERIFY}=1, or use --no-verify."
        )
    return got


def resolve(seed=0, weights=None, verify_hash=True):
    """Locate and, unless disabled, verify. Returns (path, digest or None)."""
    path = locate(seed, weights)
    if verification_disabled(verify_hash):
        return path, None
    return path, verify(path, seed)
