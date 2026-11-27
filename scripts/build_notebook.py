#!/usr/bin/env python
"""Flatten the ``geobridge`` package into ONE notebook cell (Kaggle / Colab).

    python scripts/build_notebook.py                       # -> GeoBridgeFlow_single_cell.ipynb
    python scripts/build_notebook.py --py /tmp/cell.py     # also write the cell as a plain script

The notebook is generated, never edited by hand: the package stays the single source of truth.  The cell is

    notebook/header.py      settings + environment bootstrap (Kaggle / Colab paths, missing pip packages)
    <one import block>      the union of every module's top-level imports
    <the modules>           in dependency order, intra-package imports removed
    notebook/driver.py      runs the stages and shows the figures inline

Flattening is done on the syntax tree, not with regular expressions:
  * relative imports (anywhere, also inside functions) are deleted - the names are globals of the cell;
  * ``alias.name`` for a module alias (``ot.log_sinkhorn``, ``dsp.log_mel``, ...) becomes ``name``;
  * two modules defining the same top-level name is an error, so nothing can be silently shadowed.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES: List[Tuple[str, str]] = [
    ("geobridge/config.py", "CONFIG"),
    ("geobridge/utils.py", "UTILITIES"),
    ("geobridge/text_units.py", "TEXT UNITS (grapheme-byte tokenisation, no vocabulary)"),
    ("geobridge/audio_dsp.py", "AUDIO DSP (pitch, mel, MFCC, Griffin-Lim)"),
    ("geobridge/data/download.py", "DATA: download"),
    ("geobridge/data/synthworld.py", "DATA: SynthWorld (controlled benchmark)"),
    ("geobridge/data/flickr8k.py", "DATA: Flickr8k-Tri (real benchmark)"),
    ("geobridge/data/features.py", "DATA: frozen front-ends and the one-time feature cache"),
    ("geobridge/data/captioner.py", "DATA: extra captions written by a pretrained captioner"),
    ("geobridge/data/dataset.py", "DATA: dataset, persistent splits, loaders"),
    ("geobridge/models/layers.py", "MODEL: layers"),
    ("geobridge/models/ot.py", "MODEL: entropic optimal transport"),
    ("geobridge/models/text_codec.py", "MODEL: grapheme-byte text codec"),
    ("geobridge/models/audio_codec.py", "MODEL: tone-gated speech codec"),
    ("geobridge/models/hub.py", "MODEL: barycentric visual interlingua (hub), explainer, reader"),
    ("geobridge/models/flow.py", "MODEL: hub (+) private flow writers"),
    ("geobridge/models/pixel.py", "MODEL: preview pixel decoder"),
    ("geobridge/models/geobridge.py", "MODEL: GeoBridge (assembled)"),
    ("geobridge/losses.py", "LOSSES"),
    ("geobridge/cache.py", "LATENT CACHE (transport phase)"),
    ("geobridge/metrics.py", "METRICS"),
    ("geobridge/trainer.py", "TRAINER"),
    ("geobridge/inferencer.py", "INFERENCER (figures, internal views, saved outputs)"),
    ("geobridge/ablation.py", "ABLATION STUDY"),
    ("geobridge/cli.py", "PIPELINE STAGES"),
]
RULE = "# " + "=" * 118


def _top_names(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
    return names


def flatten(path: str, title: str, index: int, imports: Dict[str, Set], owners: Dict[str, str]) -> str:
    src = open(os.path.join(ROOT, path), encoding="utf8").read()
    tree = ast.parse(src)
    lines = src.split("\n")
    drop: Set[int] = set()            # 0-based line numbers to delete
    replace: Dict[int, str] = {}      # line -> replacement (a lone relative import in a block becomes ``pass``)
    aliases: Set[str] = set()         # local names that stand for a module of the package

    parent_body: Dict[ast.AST, List[ast.stmt]] = {}
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            for child in block if isinstance(block, list) else []:
                if isinstance(child, ast.stmt):
                    parent_body[child] = block
        for handler in getattr(node, "handlers", []):
            for child in handler.body:
                parent_body[child] = handler.body

    def span(node: ast.stmt) -> range:
        return range(node.lineno - 1, node.end_lineno)

    # ---- module docstring -> banner
    doc = ast.get_docstring(tree)
    if doc is not None:
        drop.update(span(tree.body[0]))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:           # intra-package import
            for a in node.names:
                if node.module is None or a.name in ("ot", "features", "audio_dsp"):
                    aliases.add(a.asname or a.name)
            if len(parent_body[node]) == 1:
                replace[node.lineno - 1] = " " * node.col_offset + "pass"
                drop.update(list(span(node))[1:])
            else:
                drop.update(span(node))
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            drop.update(span(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and node in tree.body:   # top-level absolute import
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports["import"].add((a.name, a.asname))
            else:
                for a in node.names:
                    imports["from"].add((node.module, a.name, a.asname))
            drop.update(span(node))

    # ---- alias.name -> name
    edits: List[Tuple[int, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            v = node.value
            edits.append((v.lineno - 1, v.col_offset, v.end_col_offset + 1))   # the name and the dot
    for ln, a, b in sorted(edits, reverse=True):
        assert lines[ln][a:b].endswith("."), (path, ln, lines[ln])
        lines[ln] = lines[ln][:a] + lines[ln][b:]

    # ---- no two modules may define the same global
    for name in sorted(_top_names(tree)):
        if name in owners:
            raise SystemExit(f"top-level name '{name}' is defined in both {owners[name]} and {path}")
        owners[name] = path

    body = [replace.get(i, l) for i, l in enumerate(lines) if i not in drop or i in replace]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    banner = [RULE, f"# {index:02d}. {title}", f"#     ({path})", RULE]
    if doc:
        banner += ["# " + l if l.strip() else "#" for l in doc.split("\n")] + [RULE]
    return "\n".join(banner + [""] + body) + "\n"


def import_block(imports: Dict[str, Set]) -> str:
    std = set(sys.stdlib_module_names)
    plain = sorted(imports["import"], key=lambda t: (t[0].split(".")[0] not in std, t[0]))
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for mod, name, asname in imports["from"]:
        groups.setdefault(mod, []).append((name, asname))
    out, third = [], []
    for name, asname in plain:
        (out if name.split(".")[0] in std else third).append(f"import {name}" + (f" as {asname}" if asname else ""))
    for mod in sorted(groups):
        names = ", ".join(n + (f" as {a}" if a else "") for n, a in sorted(set(groups[mod])))
        (out if mod.split(".")[0] in std else third).append(f"from {mod} import {names}")
    return "\n".join(out + [""] + third) + "\n"


def missing_modules() -> List[str]:
    """Modules of the package that the cell would leave out.  A module that is only imported inside a function
    (lazily) is never needed by the tests on the small benchmark, so a gap here shows up only on real data."""
    listed = {p for p, _ in MODULES}
    found = []
    for d, _, files in os.walk(os.path.join(ROOT, "geobridge")):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")
            if f.endswith(".py") and f != "__init__.py" and "__pycache__" not in rel and rel not in listed:
                found.append(rel)
    return sorted(found)


def build() -> str:
    gap = missing_modules()
    if gap:
        raise SystemExit(f"modules of the package missing from MODULES (the cell would not define their names): {gap}")
    imports: Dict[str, Set] = {"import": set(), "from": set()}
    owners: Dict[str, str] = {}
    sections = [flatten(p, t, i + 1, imports, owners) for i, (p, t) in enumerate(MODULES)]
    header = open(os.path.join(ROOT, "notebook", "header.py"), encoding="utf8").read().rstrip() + "\n"
    driver = open(os.path.join(ROOT, "notebook", "driver.py"), encoding="utf8").read().rstrip() + "\n"
    toc = "\n".join(f"#   {i + 1:02d}. {t}" for i, (_, t) in enumerate(MODULES))
    header = header.replace("#   <TABLE OF CONTENTS>", toc + f"\n#   {len(MODULES) + 1:02d}. RUN")
    imp = "\n".join([RULE, "# IMPORTS", RULE, import_block(imports)])
    run = "\n".join([RULE, f"# {len(MODULES) + 1:02d}. RUN", RULE, ""])
    cell = "\n\n".join([header, imp, *sections]) + "\n\n" + run + driver
    compile(cell, "<notebook cell>", "exec")   # must at least be valid Python
    return cell


def to_notebook(cell: str) -> Dict:
    lines = cell.split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "GeoBridgeFlow_single_cell.ipynb"))
    ap.add_argument("--py", default="", help="also write the cell as a plain Python file")
    args = ap.parse_args()
    cell = build()
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(to_notebook(cell), f, indent=1, ensure_ascii=False)
        f.write("\n")
    if args.py:
        with open(args.py, "w", encoding="utf8") as f:
            f.write(cell)
    print(f"{args.out}: 1 cell, {cell.count(chr(10))} lines, {len(cell) / 1024:.0f} kB")


if __name__ == "__main__":
    main()
