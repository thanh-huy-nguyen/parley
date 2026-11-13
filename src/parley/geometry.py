"""The 512 x 512 working grid and the signed distance inputs of the network.

The network works on a fixed 512 x 512 grid. Images and masks of any size are resampled to it (aspect ratio is not
kept), so
lengths in the working grid are anisotropic on non-square inputs. The output mask is resampled back to the size of
the input mask.
"""

import numpy as np
import torch
from scipy import ndimage as ndi

WORK = 512
NEUTRAL = "Okay, next."
IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_work(mask):
    """Nearest-neighbor resampling of a mask to the working grid. Returns float32 (512, 512)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=(WORK, WORK), mode="nearest")[0, 0].numpy()


def sdf_np(m):
    """Signed distance in pixels of the working grid, negative inside. An empty mask gives +512 and a full mask gives
    -512.
    """
    mb = m > 0.5
    if not mb.any():
        return np.full(m.shape, 512.0, np.float32)
    if mb.all():
        return np.full(m.shape, -512.0, np.float32)
    phi = ndi.distance_transform_edt(~mb) - ndi.distance_transform_edt(mb)
    return (phi - 0.5 * np.sign(phi)).astype(np.float32)


def geom(before):
    """Working-grid mask -> (D, signed distance, gradient x, gradient y, gradient norm).

    D is the equivalent diameter in working pixels, floored at 8 px. An empty mask has D = 128 px, so nothing divides
    by zero.
    """
    D = max(float(np.sqrt(4 * before.sum() / np.pi)), 8.0) if before.any() else 0.25 * WORK
    phi = sdf_np(before)
    gy, gx = np.gradient(ndi.gaussian_filter(phi, 1.0))
    return D, phi, gx, gy, np.sqrt(gx**2 + gy**2) + 1e-9


def resample_image(rgb):
    """uint8 image array -> uint8 (512, 512, 3), bilinear. Grayscale is replicated and alpha is dropped."""
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).convert("RGB").resize((WORK, WORK), Image.BILINEAR))


def normalize_image(im):
    """uint8 (512, 512, 3) -> float32 (3, 512, 512) with the ImageNet mean and standard deviation removed."""
    return ((im.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD).transpose(2, 0, 1).copy()
