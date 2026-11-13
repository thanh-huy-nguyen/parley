"""Time the stages of a Parley turn: image encoding, speech recognition, the turn from a transcript, and audio to mask.

    python scripts/benchmark.py --device cpu --threads 8 --turns 200 --out bench_cpu.json
    python scripts/benchmark.py --device cuda --images data/images --masks data/masks --wavs data/wavs --out
    bench_gpu.json

Without --images, the cases are synthetic (a textured background with an elliptical lesion and a perturbed start mask).
Images and masks are paired by file name. Recognition and audio-to-mask rows need --wavs, a folder of spoken corrections
in any format ffmpeg reads; without it those rows are skipped. Each row reports the median and the 95th percentile in
ms.

The paper's Table 3 used the same stages on validation images and on the validation generator's synthesized audio,
which is not distributed, so numbers from this script measure your machine and inputs, not a reproduction of that table.
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

CORRECTIONS = [
    "pull the top in a little",
    "open it up a lot all the way round",
    "the left side is too loose, bring it in",
    "shift it down and to the right",
    "there is a piece missing at the bottom",
    "get rid of the blob on the right",
    "tighten the whole thing a bit",
    "make the bottom edge a touch bigger",
    "move it up slightly",
    "trim the right side quite a bit",
    "that looks good",
    "grow it a little on the top",
]


def synthetic_cases(n, seed=0, size=(384, 448)):
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    for _ in range(n):
        cy, cx = rng.uniform(0.35, 0.65) * h, rng.uniform(0.35, 0.65) * w
        ry, rx = rng.uniform(0.12, 0.25) * h, rng.uniform(0.12, 0.25) * w
        lesion = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        tex = rng.normal(0, 18, (h, w)) + 30 * np.sin(xx / rng.uniform(6, 14))
        img = np.clip(150 + tex - 70 * lesion, 0, 255).astype(np.uint8)
        start = ((yy - cy) / (ry * rng.uniform(0.8, 1.25))) ** 2 + (
            (xx - cx - rng.normal(0, 6)) / (rx * rng.uniform(0.8, 1.25))
        ) ** 2 < 1
        yield np.stack([img] * 3, -1), start


def folder_cases(images, masks):
    names = sorted(
        p.name for p in Path(images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    if not names:
        raise SystemExit(f"no images in {images}")
    for name in names:
        mask_path = next((p for p in Path(masks).glob(Path(name).stem + ".*")), None)
        if mask_path is None:
            continue
        rgb = np.array(Image.open(Path(images) / name).convert("RGB"))
        yield rgb, np.array(Image.open(mask_path).convert("L")) > 127


def summarize(seconds):
    a = np.asarray(seconds, float) * 1000.0
    return (
        dict(median_ms=round(float(np.median(a)), 1), p95_ms=round(float(np.percentile(a, 95)), 1), n=int(a.size))
        if a.size
        else None
    )


class Clock:
    def __init__(self, device):
        import torch

        self.sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)

    def __call__(self, fn, *args, **kwargs):
        self.sync()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        self.sync()
        return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--seed", type=int, default=0, help="which released checkpoint")
    ap.add_argument("--mode", default="listener", help="base, listener, adapter or listener+adapter")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--per-case", type=int, default=4, help="turns per image")
    ap.add_argument("--warmup", type=int, default=2, help="images run first and not timed")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--wavs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from parley import ParleyEditor, __version__
    from parley.errors import ParleyInputWarning

    warnings.simplefilter("ignore", ParleyInputWarning)

    if a.threads:
        torch.set_num_threads(a.threads)
    editor = ParleyEditor(mode=a.mode, seed=a.seed, device=a.device)
    engine, clock = editor.engine, Clock(a.device)
    wavs = sorted(str(p) for p in Path(a.wavs).iterdir() if p.is_file()) if a.wavs else []

    n_images = a.warmup + int(np.ceil(a.turns / a.per_case))
    if a.images:
        cases = list(folder_cases(a.images, a.masks or a.images))
        cases = [cases[i % len(cases)] for i in range(n_images)]
    else:
        cases = list(synthetic_cases(n_images))

    rows = {"image_encoding": [], "recognizer": [], "turn_from_transcript": [], "audio_to_mask": []}
    editor.begin_session("benchmark", "benchmark")
    k = 0
    for i, (rgb, start) in enumerate(cases):
        timed = i >= a.warmup
        feats, t_enc = clock(engine.image_encoder, rgb)
        editor.begin_image(None, start, features=feats)
        if timed:
            rows["image_encoding"].append(t_enc)
        for j in range(a.per_case):
            text = CORRECTIONS[k % len(CORRECTIONS)]
            click = (
                tuple(float(v) for v in np.argwhere(start).mean(0)[::-1])
                if "blob" in text or "missing" in text
                else None
            )
            if wavs:
                transcript, t_asr = clock(editor.transcribe, wav=wavs[k % len(wavs)])
                text = transcript[0] or text
            _, t_turn = clock(editor.edit, text=text, click=click)
            if timed:
                rows["turn_from_transcript"].append(t_turn)
                if wavs:
                    rows["recognizer"].append(t_asr)
                    rows["audio_to_mask"].append(t_asr + t_turn)
            k += 1
        editor.end_image()

    report = dict(
        parley=__version__,
        mode=a.mode,
        seed=a.seed,
        device=a.device,
        torch=torch.__version__,
        torch_threads=torch.get_num_threads(),
        python=platform.python_version(),
        machine=platform.processor() or platform.machine(),
        gpu=torch.cuda.get_device_name() if str(a.device).startswith("cuda") else None,
        cases="folder" if a.images else "synthetic",
        turns=len(rows["turn_from_transcript"]),
        stages={name: summarize(v) for name, v in rows.items()},
    )
    Path(a.out).write_text(json.dumps(report, indent=1))
    for name, s in report["stages"].items():
        print(
            f"{name:<22} "
            + (
                "skipped (no --wavs)"
                if s is None
                else f"{s['median_ms']:>8.1f} ms median   {s['p95_ms']:>8.1f} ms p95   n={s['n']}"
            )
        )


if __name__ == "__main__":
    main()
