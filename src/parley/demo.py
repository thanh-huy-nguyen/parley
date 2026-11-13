"""Local web demo. Binds to 127.0.0.1 and never creates a share link. Needs the `demo` extra. Fetches nothing from the
internet at run time.

Upload an image and a start mask, say or type a correction, optionally click on the image, and see the parsed
program, the new mask and, in the margin-model modes, the margin belief. Undo steps back one turn.
"""

import os
import re
import tempfile
import warnings

import numpy as np

from .editor import ParleyEditor, describe_program

# The Parley palette. PARCHMENT and PARCHMENT_2 are the text colors on the dark surfaces.
INK, SURFACE, SURFACE_2 = "#151016", "#1F1820", "#2A2129"
AUBERGINE, AUBERGINE_2, AUBERGINE_3 = "#24121F", "#3D1F38", "#5A2F5A"
LILAC, TEAL, GOLD, GOLD_LIGHT = "#C7A6D9", "#6FB8AD", "#C9A45F", "#DDC08A"
PARCHMENT, PARCHMENT_2 = "#EBE1CC", "#B9AB93"
FONT_STACK = '"Inter", "Segoe UI", system-ui, sans-serif'
MONO_STACK = 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace'
# (label, value) pairs: the page says "margin model", the API keeps its mode strings.
MODE_CHOICES = [
    ("Base", "base"),
    ("Margin model", "listener"),
    ("Adapter", "adapter"),
    ("Margin model + adapter", "listener+adapter"),
]
HOST = os.environ.get(
    "PARLEY_DEMO_HOST", "127.0.0.1"
)  # only a container entry point should change this; see the Dockerfile


def _rgb(color):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


CONTOUR_CORE, CONTOUR_CASING, CLICK_DOT = _rgb(PARCHMENT), _rgb(INK), _rgb(GOLD)
MASK_TINT = _rgb(LILAC)


def overlay(image, mask, click=None):
    """Draw the mask boundary on the image, with a light fill and the pending click. The line is a light core on a dark
    casing, so that it reads on bright and on dark images. Same shape as the mask, so clicks stay in mask pixels.
    """
    from PIL import Image
    from scipy import ndimage as ndi

    rgb = np.array(Image.fromarray(image).convert("RGB").resize((mask.shape[1], mask.shape[0])))
    out = rgb.astype(np.float32)
    out[mask] = 0.84 * out[mask] + 0.16 * np.array(MASK_TINT, np.float32)
    if mask.any() and not mask.all():
        # Signed distance in pixels, positive inside. The core sits just inside the boundary; the casing surrounds it.
        s = ndi.distance_transform_edt(mask) - 0.5 - (ndi.distance_transform_edt(~mask) - 0.5)
        s = ndi.gaussian_filter(s.astype(np.float32), 0.6)
        width = float(max(2, max(mask.shape) // 180))
        for color, half in ((CONTOUR_CASING, width / 2 + 1.5), (CONTOUR_CORE, width / 2)):
            alpha = np.clip(half + 0.75 - np.abs(s - width / 2), 0, 1)[..., None]
            out = (1 - alpha) * out + alpha * np.array(color, np.float32)
    out = out.round().astype(np.uint8)
    if click is not None:
        x, y = int(round(click[0])), int(round(click[1]))
        r = max(3, max(mask.shape) // 80)
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        out[d2 <= (r + max(1, r // 3)) ** 2] = CONTOUR_CASING
        out[d2 <= r**2] = CLICK_DOT
    return out


def _plot_font():
    """Inter if matplotlib can find it on this machine, else its default sans-serif. Nothing is downloaded."""
    from matplotlib import font_manager as fm

    if not any(f.name == "Inter" for f in fm.fontManager.ttflist):
        for path in fm.findSystemFonts():
            if os.path.basename(path).startswith("Inter-"):
                fm.fontManager.addfont(path)
    return "Inter" if any(f.name == "Inter" for f in fm.fontManager.ttflist) else "DejaVu Sans"


def belief_figure(belief):
    from matplotlib.figure import Figure

    family = _plot_font()
    fig = Figure(figsize=(3.3, 1.6), dpi=400)
    fig.patch.set_alpha(0)
    ax = fig.subplots()
    ax.set_facecolor("none")
    if belief is None:
        ax.text(0.5, 0.5, "No margin model in this mode", ha="center", va="center", color=PARCHMENT_2, family=family)
        ax.axis("off")
    else:
        g, p = belief["grid"], belief["posterior"]
        keep = np.abs(g) <= 0.15
        ax.fill_between(g[keep], p[keep], color=AUBERGINE_3, alpha=0.65, lw=0)
        ax.plot(g[keep], p[keep], color=LILAC, lw=1.6)
        ax.axvline(belief["mean"], color=GOLD, ls="--", lw=1.3)
        lo, hi = belief["interval90"]
        ax.set_title(
            f"Margin belief: {belief['mean']:+.3f} D, events: {belief['n_events']}\n"
            f"90% interval [{lo:+.3f}, {hi:+.3f}]",
            fontsize=8,
            color=PARCHMENT,
            family=family,
            linespacing=1.3,
        )
        ax.set_xlabel("Margin (units of D)", fontsize=8, color=PARCHMENT, family=family)
        ax.set_yticks([])
        ax.tick_params(labelsize=8, colors=PARCHMENT, length=2.5)
        for t in ax.get_xticklabels():
            t.set_family(family)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(PARCHMENT_2)
    fig.tight_layout(pad=0.4)
    return fig


class StripExternalAssets:
    """ASGI middleware. Gradio's page template names a script on a CDN and preconnects to a font host. Take those out,
    so that the page asks nothing of the internet.
    """

    PATTERN = re.compile(
        rb"""<script[^>]*\ssrc=["']https?://[^>]*>\s*</script>|<link[^>]*\srel=["'](?:preconnect|dns-prefetch|stylesheet)["'][^>]*\shref=["']https?://[^>]*/?>|<link[^>]*\shref=["']https?://[^>]*\srel=["'](?:preconnect|dns-prefetch|stylesheet)["'][^>]*/?>""",
        re.S | re.I,
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, chunks = None, []

        async def wrapped(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = {k.lower(): v for k, v in message.get("headers", [])}
                if headers.get(b"content-type", b"").startswith(b"text/html") and b"content-encoding" not in headers:
                    start = message
                    return
            elif start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = self.PATTERN.sub(b"", b"".join(chunks))
                start["headers"] = [
                    (k, str(len(body)).encode() if k.lower() == b"content-length" else v)
                    for k, v in start.get("headers", [])
                ]
                await send(start)
                await send({"type": "http.response.body", "body": body})
                return
            await send(message)

        await self.app(scope, receive, wrapped)


def parley_theme():
    """A dark theme in the Parley palette, the same in the page's light and dark modes. Plain font names only: a Google
    font would make the page fetch it.
    """
    import gradio as gr

    lilac = gr.themes.Color(
        c50="#F7F1FA",
        c100="#EEE3F4",
        c200="#E1CFEB",
        c300="#D4BBE2",
        c400=LILAC,
        c500="#B48FCB",
        c600="#9B74B5",
        c700="#7D5A96",
        c800=AUBERGINE_3,
        c900=AUBERGINE_2,
        c950=AUBERGINE,
        name="parley_lilac",
    )
    gold = gr.themes.Color(
        c50="#F8F2E5",
        c100="#F0E5CD",
        c200="#E8D6AE",
        c300=GOLD_LIGHT,
        c400=GOLD,
        c500="#B28E4C",
        c600="#94753C",
        c700="#755C2F",
        c800="#574422",
        c900="#3A2D17",
        c950="#241C0E",
        name="parley_gold",
    )
    neutral = gr.themes.Color(
        c50="#F4EEDF",
        c100=PARCHMENT,
        c200="#D3C7B0",
        c300=PARCHMENT_2,
        c400="#A3968A",
        c500="#7E7274",
        c600="#5A4F58",
        c700="#3E343D",
        c800=SURFACE_2,
        c900=SURFACE,
        c950=INK,
        name="parley_neutral",
    )
    theme = gr.themes.Base(
        primary_hue=lilac,
        secondary_hue=gold,
        neutral_hue=neutral,
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_md,
        text_size=gr.themes.sizes.text_md,
        font=("Inter", "Segoe UI", "system-ui", "sans-serif"),
        font_mono=("ui-monospace", "Consolas", "monospace"),
    )
    # The dark values of Base already draw on the neutral scale above: use them for the light mode too.
    for name in [k[:-5] for k in vars(theme) if k.endswith("_dark")]:
        if getattr(theme, name + "_dark") is not None:
            setattr(theme, name, getattr(theme, name + "_dark"))
    primary = dict(
        button_primary_background_fill=LILAC,
        button_primary_background_fill_hover="#D4BBE2",
        button_primary_border_color=LILAC,
        button_primary_border_color_hover="#D4BBE2",
        button_primary_text_color=INK,
        button_primary_text_color_hover=INK,
    )
    secondary = dict(
        button_secondary_background_fill=AUBERGINE_2,
        button_secondary_background_fill_hover=AUBERGINE_3,
        button_secondary_border_color=AUBERGINE_3,
        button_secondary_border_color_hover=LILAC,
        button_secondary_text_color=PARCHMENT,
        button_secondary_text_color_hover=PARCHMENT,
    )
    cancel = dict(
        button_cancel_background_fill=AUBERGINE_2,
        button_cancel_background_fill_hover=AUBERGINE_3,
        button_cancel_border_color=AUBERGINE_3,
        button_cancel_border_color_hover=GOLD,
        button_cancel_text_color=PARCHMENT,
        button_cancel_text_color_hover=PARCHMENT,
    )
    surfaces = dict(
        body_background_fill=INK,
        background_fill_primary=SURFACE,
        background_fill_secondary=SURFACE_2,
        block_background_fill=SURFACE,
        panel_background_fill=SURFACE,
        code_background_fill=SURFACE_2,
        stat_background_fill=AUBERGINE_3,
        table_even_background_fill=SURFACE,
        table_odd_background_fill=SURFACE_2,
        table_row_focus=AUBERGINE_2,
    )
    lines = dict(
        border_color_primary=AUBERGINE_2,
        border_color_accent=GOLD,
        border_color_accent_subdued=AUBERGINE_3,
        block_border_color=AUBERGINE_2,
        block_label_border_color=AUBERGINE_3,
        block_title_border_color="none",
        panel_border_color=AUBERGINE_2,
        table_border_color=AUBERGINE_2,
        block_shadow="none",
        input_shadow="none",
        input_shadow_focus="none",
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_primary_shadow_active="none",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        button_secondary_shadow_active="none",
        button_cancel_shadow="none",
        button_cancel_shadow_hover="none",
        button_cancel_shadow_active="none",
    )
    text = dict(
        body_text_color=PARCHMENT,
        body_text_color_subdued=PARCHMENT_2,
        block_title_text_color=PARCHMENT_2,
        block_title_background_fill="none",
        block_label_text_color=PARCHMENT_2,
        block_label_background_fill=SURFACE,
        block_info_text_color=PARCHMENT_2,
        link_text_color=LILAC,
        link_text_color_hover=GOLD_LIGHT,
        link_text_color_active=GOLD_LIGHT,
        link_text_color_visited=LILAC,
        accordion_text_color=PARCHMENT,
        table_text_color=PARCHMENT,
        error_text_color=PARCHMENT,
        error_background_fill=AUBERGINE_2,
        error_border_color=GOLD,
        error_icon_color=GOLD,
    )
    inputs = dict(
        input_background_fill=SURFACE_2,
        input_background_fill_focus=SURFACE_2,
        input_background_fill_hover=SURFACE_2,
        input_border_color=AUBERGINE_3,
        input_border_color_hover=LILAC,
        input_border_color_focus=GOLD,
        input_placeholder_color=PARCHMENT_2,
        color_accent=LILAC,
        color_accent_soft=AUBERGINE_2,
        loader_color=GOLD,
        slider_color=LILAC,
        checkbox_background_color=SURFACE_2,
        checkbox_background_color_hover=SURFACE_2,
        checkbox_background_color_focus=SURFACE_2,
        checkbox_background_color_selected=LILAC,
        checkbox_border_color=AUBERGINE_3,
        checkbox_border_color_hover=LILAC,
        checkbox_border_color_focus=GOLD,
        checkbox_border_color_selected=LILAC,
        checkbox_label_background_fill=SURFACE_2,
        checkbox_label_background_fill_hover=AUBERGINE_2,
        checkbox_label_background_fill_selected=AUBERGINE_2,
        checkbox_label_border_color=AUBERGINE_3,
        checkbox_label_border_color_hover=LILAC,
        checkbox_label_border_color_selected=LILAC,
        checkbox_label_text_color=PARCHMENT,
        checkbox_label_text_color_selected=PARCHMENT,
    )
    for name, value in {**primary, **secondary, **cancel, **surfaces, **lines, **text, **inputs}.items():
        setattr(theme, name, value)
        setattr(theme, name + "_dark", value)
    theme.font = FONT_STACK
    theme.font_mono = MONO_STACK
    return theme


CSS = f"""
:root, .gradio-container {{ --font: {FONT_STACK}; --font-mono: {MONO_STACK}; }}
footer {{ display: none !important; }}
.gradio-container {{ max-width: none !important; padding: 0 !important; background: {INK} !important; }}
main.fillable, .gradio-container main {{ padding: 10px 16px 12px !important; }}
#parley-head {{
  background: linear-gradient(90deg, {AUBERGINE}, {AUBERGINE_2}); border: 0; border-bottom: 1px solid {GOLD};
  border-radius: 8px 8px 0 0; padding: 9px 16px 8px; display: flex; align-items: baseline; gap: 14px;
}}
#parley-head .name {{ color: {GOLD_LIGHT}; font-size: 20px; font-weight: 600; letter-spacing: 0.01em; }}
#parley-head .what {{ color: {PARCHMENT}; font-size: 14px; }}
#parley-head .note {{ color: {PARCHMENT_2}; font-size: 12.5px; margin-left: auto; }}
.block {{ border-color: {AUBERGINE_2} !important; }}
textarea, input {{ -webkit-text-fill-color: currentColor !important; opacity: 1 !important; }}
textarea:disabled {{ color: {PARCHMENT} !important; background: {SURFACE_2} !important; }}
textarea::placeholder, input::placeholder {{
  color: {PARCHMENT_2} !important; -webkit-text-fill-color: {PARCHMENT_2} !important; opacity: 1 !important;
}}
button {{ font-weight: 500 !important; }}
#image_in button[title="Fullscreen"], #mask_in button[title="Fullscreen"],
#image_in button[aria-label="Fullscreen"], #mask_in button[aria-label="Fullscreen"] {{ display: none !important; }}
#audio select:disabled {{ display: none !important; }}
#program textarea {{ font-size: 13px; line-height: 1.35; }}
#plot {{ background: {SURFACE}; border-color: {AUBERGINE_2}; }}
#notes textarea {{ font-size: 13px; }}
"""


class DemoState:
    """One browser session's editor and current case."""

    def __init__(self, mode, seed, device, weights, verify):
        self.kwargs = dict(seed=seed, device=device, weights=weights, verify=verify)
        self.mode, self.editors = mode, {}
        self.image, self.click = None, None
        self.history = []

    def editor(self, mode=None):
        mode = mode or self.mode
        if mode not in self.editors:
            self.editors[mode] = ParleyEditor(mode=mode, **self.kwargs)
            self.editors[mode].begin_session("demo", "demo")
        self.mode = mode
        return self.editors[mode]


def build(mode="base", seed=0, device="cuda", weights=None, verify=True):
    # A private upload folder, since /tmp/gradio may belong to another user on a shared machine, and no usage reports.
    os.environ.setdefault("GRADIO_TEMP_DIR", tempfile.mkdtemp(prefix="parley-demo-"))
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr

    S = DemoState(mode, seed, device, weights, verify)

    def view(note=""):
        ed = S.editor()
        belief = ed.belief()
        plot_view = gr.update(
            value=belief_figure(belief), visible=belief is not None
        )  # no empty box in the modes without a margin model
        note_view = gr.update(value=note, visible=bool(note))  # the Notes box shows only when there is a note
        if S.image is None or ed.current is None:
            return None, plot_view, note_view
        return overlay(S.image, ed.current, S.click), plot_view, note_view

    def load(image, mask, mode_choice):
        if image is None or mask is None:
            return *view("Upload an image and a start mask."), ""
        ed = S.editor(mode_choice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = np.asarray(mask)
            if m.ndim == 3:
                m = m[:, :, 0]
            top = m.max() if m.size else 0
            ed.begin_image(image, m > top / 2 if top > 0 else np.zeros(m.shape, bool))
        S.image, S.click = np.asarray(image), None
        if S.image.dtype != np.uint8:
            from .io import as_uint8_image

            S.image = as_uint8_image(S.image)[0]
        S.history = []
        notes = "; ".join(str(w.message) for w in caught)
        return *view(notes or "Ready. Say or type a correction."), ""

    def on_click(evt: gr.SelectData):
        S.click = (float(evt.index[0]), float(evt.index[1]))
        return view(f"Click stored at ({S.click[0]:.0f}, {S.click[1]:.0f}). It goes with the next correction.")

    def apply_edit(text, audio):
        ed = S.editor()
        if ed.current is None:
            return *view("Upload an image and a start mask first."), "", gr.update()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if text and text.strip():
                r = ed.edit(text=text, click=S.click)
            elif audio:
                r = ed.edit(wav=audio, click=S.click)
            else:
                r = ed.edit(text="", click=S.click)
        S.click = None
        lines = [
            f"Heard: {r.transcript}" if r.transcript is not None else "Heard: nothing",
            f"Result: {r.status}, {100 * r.changed_fraction:.2f}% of pixels changed",
        ]
        lines += [
            f"Program {k + 1}: {describe_program(p)} (type confidence {p['confidence']:.2f})"
            for k, p in enumerate(r.programs)
        ]
        S.history.append(lines[0])
        note = "; ".join(str(w.message) for w in caught)
        return *view(note), "\n".join(lines), ""

    def undo():
        ed = S.editor()
        m = ed.undo()
        return *view("Stepped back one turn." if m is not None else "Nothing to undo."), ""

    def accept():
        ed = S.editor()
        if ed.current is None:
            return view("Nothing to accept.")
        ed.end_image(accepted=True, accept_text="that looks good")
        return view("Accepted. The next image in this session starts from what was learned.")

    with gr.Blocks(title="Parley demo", analytics_enabled=False, theme=parley_theme(), css=CSS) as app:
        gr.HTML(
            padding=False,
            value=(
                '<div id="parley-head"><span class="name">Parley</span>'
                '<span class="what">A speech-directed mask editor</span>'
                '<span class="note">Research prototype. Not for clinical use. Runs on this machine only.</span></div>'
            ),
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=220):
                with gr.Row():
                    image = gr.Image(
                        label="Image",
                        type="numpy",
                        image_mode=None,
                        height=125,
                        sources=["upload"],
                        show_fullscreen_button=False,
                        show_download_button=False,
                        elem_id="image_in",
                    )
                    mask = gr.Image(
                        label="Start mask",
                        type="numpy",
                        image_mode="L",
                        height=125,
                        sources=["upload"],
                        show_fullscreen_button=False,
                        show_download_button=False,
                        elem_id="mask_in",
                    )
                mode_box = gr.Dropdown(MODE_CHOICES, value=mode, label="Mode", elem_id="mode")
                load_btn = gr.Button("Load image and mask", elem_id="load")
                plot = gr.Plot(show_label=False, format="png", elem_id="plot", visible=False)
            with gr.Column(scale=6, min_width=420):
                canvas = gr.Image(
                    label="Current mask (click to place a click)",
                    type="numpy",
                    interactive=False,
                    height=560,
                    elem_id="canvas",
                    show_fullscreen_button=False,
                    show_download_button=False,
                )
                status = gr.Textbox(label="Notes", lines=1, interactive=False, elem_id="notes", visible=False)
            with gr.Column(scale=3, min_width=280):
                text = gr.Textbox(
                    label="Correction (text)",
                    placeholder='For example, "pull the top in a little"',
                    lines=3,
                    elem_id="correction",
                )
                audio = gr.Audio(
                    label="Correction (microphone or file)",
                    sources=["microphone", "upload"],
                    type="filepath",
                    elem_id="audio",
                )
                go = gr.Button("Apply", variant="primary", elem_id="apply")
                with gr.Row():
                    undo_btn = gr.Button("Undo", elem_id="undo", min_width=60)
                    accept_btn = gr.Button("Accept", elem_id="accept", min_width=60)
                program = gr.Textbox(label="Parsed program", lines=8, interactive=False, elem_id="program")
        load_btn.click(load, [image, mask, mode_box], [canvas, plot, status, program])
        canvas.select(on_click, None, [canvas, plot, status])
        go.click(apply_edit, [text, audio], [canvas, plot, status, program, text])
        undo_btn.click(undo, None, [canvas, plot, status, program])
        accept_btn.click(accept, None, [canvas, plot, status])
    return app


def launch(mode="base", seed=0, device="cuda", weights=None, verify=True, port=7860, prevent_thread_lock=False):
    app = build(mode, seed, device, weights, verify)
    from starlette.middleware import Middleware

    return app.launch(
        server_name=HOST,
        server_port=port,
        share=False,
        inbrowser=False,
        show_api=False,
        prevent_thread_lock=prevent_thread_lock,
        app_kwargs=dict(middleware=[Middleware(StripExternalAssets)]),
    )
