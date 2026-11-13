"""Run the toy simulated clinician through Parley and print the reporting table.

    python examples/evalkit_toy_run.py --device cpu --out toy_run.json
    python examples/evalkit_toy_run.py --images my/images --masks my/masks --starts my/starts --out toy_run.json

This shows how the evaluation kit is wired. It is not an evaluation of the editor: the toy clinician has one short
phrase per situation and was written next to the editor. To test an editor, write your own `SimulatedClinician`.

Without --images, the cases are synthetic. With folders, images, reference masks and start masks are paired by file
name; without --starts, the start mask is the reference mask grown by 8%.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from parley import ParleyEditor
from parley import edits as E
from parley.evalkit import Case, ParleyEvalAdapter, run_study, summarize, summary_markdown
from parley.evalkit.toy import ToyClinician


def synthetic_cases(n, seed=0, size=(320, 384)):
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    out = []
    for i in range(n):
        cy, cx = rng.uniform(0.4, 0.6) * h, rng.uniform(0.4, 0.6) * w
        ry, rx = rng.uniform(0.15, 0.25) * h, rng.uniform(0.15, 0.25) * w
        truth = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        img = np.clip(150 + rng.normal(0, 15, (h, w)) - 70 * truth, 0, 255).astype(np.uint8)
        start = E.apply(truth, E.Edit("shrink", "global", amount=float(rng.choice([-0.12, -0.06, 0.06, 0.12]))))
        out.append(Case(f"synthetic_{i:02d}", "synthetic", np.stack([img] * 3, -1), start, truth))
    return out


def folder_cases(images, masks, starts, dataset, n):
    def read_mask(folder, stem):
        path = next(Path(folder).glob(stem + ".*"), None)
        return None if path is None else np.array(Image.open(path).convert("L")) > 127

    out = []
    for path in sorted(Path(images).iterdir())[:n]:
        truth = read_mask(masks, path.stem)
        if truth is None:
            continue
        start = read_mask(starts, path.stem) if starts else None
        if start is None:
            start = E.apply(truth, E.Edit("shrink", "global", amount=-0.08))
        out.append(Case(path.stem, dataset, np.array(Image.open(path).convert("RGB")), start, truth))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--starts")
    ap.add_argument("--dataset", default="my_data")
    ap.add_argument("--n", type=int, default=6, help="number of images")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    warnings.simplefilter("ignore")
    if a.images:
        data = {a.dataset: folder_cases(a.images, a.masks, a.starts, a.dataset, a.n)}
    else:
        data = {"synthetic": synthetic_cases(a.n)}
    clinicians = [
        ToyClinician("tight", margin=-0.02),
        ToyClinician("neutral", margin=0.0),
        ToyClinician("loose", margin=0.02),
    ]

    tables, dialogues = {}, []
    for mode in ("base", "listener"):
        records = run_study(ParleyEvalAdapter(ParleyEditor(mode=mode, device=a.device)), clinicians, data, arm=mode)
        dialogues += [r.to_dict() for r in records]
        tables[mode] = summarize(records)

    table = summary_markdown(tables)
    note = "Toy clinician: shows that the kit runs, not how well the editor works."
    print(note + "\n\n" + table)
    Path(a.out).write_text(json.dumps(dict(note=note, summaries=tables, table=table, dialogues=dialogues), indent=1))


if __name__ == "__main__":
    main()
