"""The three frozen models: the image encoder, the text encoder and the recognizer."""

import numpy as np
import torch

from . import geometry as G
from .weights import ASR_REPO, SAM2_REPO, TEXT_REPO, hf_revision


class TextEncoder:
    """Token features of the frozen sentence encoder."""

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer

        rev = hf_revision(TEXT_REPO)
        self.tok = AutoTokenizer.from_pretrained(TEXT_REPO, revision=rev)
        self.m = AutoModel.from_pretrained(TEXT_REPO, revision=rev).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts):
        e = self.tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        return self.m(**e).last_hidden_state, e["attention_mask"] == 0


class ImageEncoder:
    """Frozen SAM 2.1 tiny encoder at 512 x 512. Half precision on a GPU and single precision on the CPU; features are
    stored as float16 in both cases.
    """

    def __init__(self, device, half=None):
        from transformers import Sam2Model

        self.device = device
        self.half = not str(device).startswith("cpu") if half is None else half
        enc = Sam2Model.from_pretrained(SAM2_REPO, revision=hf_revision(SAM2_REPO)).vision_encoder.to(device).eval()
        self.enc = enc.half() if self.half else enc

    def __call__(self, rgb):
        """uint8 image array -> (features: float16 tensor (256, 64, 64), image: float32 tensor (3, 512, 512)), both on
        the device.
        """
        im = G.resample_image(rgb)
        chw = G.normalize_image(im)
        x = torch.from_numpy(chw)[None].to(self.device)
        x = x.half() if self.half else x
        with torch.no_grad():
            f = self.enc(x).fpn_hidden_states[1][0].float().cpu().numpy().astype(np.float16)
        return torch.from_numpy(f).to(self.device), torch.from_numpy(chw).to(self.device)


class Recognizer:
    """whisper-small.en, English only. Accepts a path to an audio file or a dict with `raw` (float32 mono) and
    `sampling_rate`; other rates are resampled to 16 kHz.
    """

    def __init__(self, device, dtype=None):
        from transformers import pipeline

        on_gpu = str(device).startswith("cuda")
        index = (int(str(device).split(":")[1]) if ":" in str(device) else 0) if on_gpu else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=ASR_REPO,
            revision=hf_revision(ASR_REPO),
            device=index,
            dtype=dtype or ("float16" if on_gpu else "float32"),
        )

    def __call__(self, audio):
        return self.pipe(audio)["text"].strip()
