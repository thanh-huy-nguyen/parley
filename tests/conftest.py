"""Shared fixtures. A small network with random weights and stub encoders lets the protocol tests run on the CPU with no
downloads.
"""

import numpy as np
import pytest
import torch


class StubText:
    """Stands in for the sentence encoder: token features derived from the characters of the text."""

    def __call__(self, texts):
        L = max(2, max(len(t) for t in texts) + 2)
        out = torch.zeros(len(texts), L, 384)
        for i, t in enumerate(texts):
            g = torch.Generator().manual_seed(sum(ord(c) * (k + 1) for k, c in enumerate(t)) + 1)
            out[i, : len(t) + 2] = torch.randn(len(t) + 2, 384, generator=g)
        pad = torch.ones(len(texts), L, dtype=torch.bool)
        for i, t in enumerate(texts):
            pad[i, : len(t) + 2] = False
        return out, pad


class StubImage:
    """Stands in for the image encoder: smooth features computed from the image."""

    def __call__(self, rgb):
        from parley import geometry as G

        im = G.resample_image(rgb)
        chw = torch.from_numpy(G.normalize_image(im))
        pooled = torch.nn.functional.adaptive_avg_pool2d(chw[None], 64)[0]
        return pooled.repeat(86, 1, 1)[:256].half(), chw


class StubRecognizer:
    def __call__(self, audio):
        return "make it a bit bigger"


@pytest.fixture
def stub_engine_factory():
    from parley.engine import Engine
    from parley.model import ParleyNet

    def make(mode="base", seed=0):
        torch.manual_seed(seed)
        net = ParleyNet().eval()
        return Engine(
            net,
            device="cpu",
            mode=mode,
            text_encoder=StubText(),
            image_encoder=StubImage(),
            recognizer=StubRecognizer(),
        )

    return make


@pytest.fixture
def stub_editor_factory(stub_engine_factory):
    from parley.editor import ParleyEditor

    def make(mode="base", seed=0, **kw):
        return ParleyEditor(mode=mode, seed=seed, device="cpu", engine=stub_engine_factory(mode, seed), **kw)

    return make


@pytest.fixture
def phantom():
    """A synthetic image and a disk mask."""
    yy, xx = np.mgrid[:240, :320]
    d = np.hypot(xx - 160, yy - 120)
    img = np.clip(200 - 120 * (d < 60) + 10 * np.sin(xx / 9.0), 0, 255).astype(np.uint8)
    return np.stack([img] * 3, -1), d < 52


def weights_available():
    from parley import weights as W

    try:
        W.locate(0)
        return True
    except W.WeightsNotFoundError:
        return False


def pytest_collection_modifyitems(config, items):
    has_gpu = torch.cuda.is_available()
    has_w = weights_available()
    for item in items:
        if "weights" in item.keywords and not has_w:
            item.add_marker(pytest.mark.skip(reason="no weights: set PARLEY_WEIGHTS"))
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="no GPU"))
