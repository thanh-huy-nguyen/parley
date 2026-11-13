"""The demo's helpers and its network binding. CPU, no weights: the models load only when a page event needs them. The
full browser path is exercised by hand with `parley demo`.
"""

import os
import tempfile
import urllib.request

import numpy as np
import pytest

os.environ.setdefault("GRADIO_TEMP_DIR", tempfile.mkdtemp(prefix="parley-demo-"))  # gradio reads this once, on import
pytest.importorskip("gradio")


def test_overlay_marks_the_boundary_and_keeps_the_shape():
    from parley.demo import CONTOUR_CORE, overlay

    rgb = np.full((100, 120, 3), 200, np.uint8)
    m = np.zeros((100, 120), bool)
    m[30:60, 40:80] = True
    out = overlay(rgb, m, click=(5, 5))
    assert (
        out.shape == (100, 120, 3)
        and tuple(out[30, 50]) == CONTOUR_CORE
        and tuple(out[5, 5]) != (200, 200, 200)
        and tuple(out[90, 90]) == (200, 200, 200)
    )


def test_belief_figure_with_and_without_a_listener():
    from parley.demo import belief_figure
    from parley.listener import GRID, MarginBelief

    b = MarginBelief()
    assert belief_figure(None) is not None
    assert (
        belief_figure(
            dict(mean=b.mean(), interval90=b.interval90(), grid=GRID, posterior=b.posterior(), n_images=0, n_events=0)
        )
        is not None
    )


def test_binds_to_localhost_with_no_share_link():
    import gradio

    from parley.demo import launch

    app, url, share = launch(port=7893, prevent_thread_lock=True, device="cpu")
    try:
        assert url.startswith("http://127.0.0.1:7893") and share is None
        page = urllib.request.urlopen(url, timeout=20)
        html = page.read().decode()
        assert (
            page.status == 200 and "cdnjs" not in html and "fonts.googleapis" not in html
        )  # the page fetches nothing from the internet
    finally:
        gradio.close_all()
