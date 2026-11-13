"""A click-only comparator: SAM 2.1 tiny with the standard click protocol, every click of the image so far plus the
previous mask as a mask prompt. No speech.
"""

import numpy as np
import torch

from .weights import SAM2_REPO, hf_revision


class Sam2ClickEditor:
    def __init__(self, device="cuda"):
        from transformers import Sam2Model, Sam2Processor

        rev = hf_revision(SAM2_REPO)
        self.device = device
        self.proc = Sam2Processor.from_pretrained(SAM2_REPO, revision=rev)
        self.sam = Sam2Model.from_pretrained(SAM2_REPO, revision=rev).to(device).eval()
        self._pil, self._points = None, []

    def begin_image(self, rgb, start_mask):
        """`rgb`: uint8 image array. `start_mask`: bool (H, W). The image is resized to the mask."""
        from PIL import Image

        start_mask = np.asarray(start_mask).astype(bool)
        self._pil = Image.fromarray(rgb).convert("RGB").resize((start_mask.shape[1], start_mask.shape[0]))
        self._points = []
        return start_mask

    def step(self, mask, click):
        """One click on `mask`. Returns (bool mask, info).

        `click`: (x, y, label) in pixels of `mask`; label 1 is positive and 0 is negative.
        """
        mask = np.asarray(mask).astype(bool)
        self._points.append(([float(click[0]), float(click[1])], int(click[2])))
        pts, labs = [p for p, _ in self._points], [lab for _, lab in self._points]
        prev = torch.from_numpy(np.where(mask, 8.0, -8.0).astype(np.float32))[None, None]
        prev = torch.nn.functional.interpolate(prev, size=(256, 256), mode="bilinear", align_corners=False)
        inp = self.proc(images=self._pil, input_points=[[pts]], input_labels=[[labs]], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            o = self.sam(**inp, input_masks=prev.to(self.device), multimask_output=False)
        return self.proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0, 0].numpy() > 0, dict(
            types=["click"], accept=False
        )
