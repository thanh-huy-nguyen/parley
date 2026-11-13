"""Print the licence metadata of every installed distribution that the package requires, directly or through its
dependencies, as a Markdown table. Reads the installed METADATA files.
"""

import importlib.metadata as IM
import re
import sys


def license_of(d):
    m = d.metadata
    expr = m.get("License-Expression")
    if expr:
        return expr
    cl = [c.split("::")[-1].strip() for c in m.get_all("Classifier", []) if c.startswith("License ::")]
    if cl:
        return "; ".join(cl)
    text = (m.get("License") or "").strip()
    return text.splitlines()[0][:70] if text else "not stated in metadata"


def closure(root, extras=()):
    seen, todo = {}, [(root, set(extras))]
    while todo:
        name, ex = todo.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        try:
            d = IM.distribution(name)
        except IM.PackageNotFoundError:
            continue
        seen[key] = d
        for r in d.requires or []:
            base, _, marker = r.partition(";")
            if "extra ==" in marker and not any(f"'{e}'" in marker or f'"{e}"' in marker for e in ex):
                continue
            todo.append((re.split(r"[\s<>=!~\[(]", base.strip())[0], set()))
    return seen


if __name__ == "__main__":
    extras = sys.argv[1:] or ["demo"]
    print("| Distribution | Version | Licence metadata |\n|---|---|---|")
    for k, d in sorted(closure("parley", extras).items()):
        if k != "parley":
            print(f"| {d.metadata['Name']} | {d.version} | {license_of(d)} |")
