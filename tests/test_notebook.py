"""The single-cell notebook is generated from the package: build it and execute the cell end to end
(tiny controlled benchmark, CPU) exactly as Kaggle / Colab would - as one ``__main__`` script."""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_notebook  # noqa: E402


def _settings(cell: str, **values) -> str:
    for name, value in values.items():
        # (a setting may continue on indented lines: a wrapped tuple, the comment beside it)
        cell, n = re.subn(rf"(?m)^{name} = .*(?:\n[ \t]+.*)*$", lambda _: f"{name} = {value!r}", cell, count=1)
        assert n == 1, name
    return cell


def test_single_cell_runs_end_to_end(tmp_path):
    cell = build_notebook.build()
    nb = build_notebook.to_notebook(cell)
    assert len(nb["cells"]) == 1 and "".join(nb["cells"][0]["source"]) == cell
    work, data = str(tmp_path / "work"), str(tmp_path / "data")
    cell = _settings(cell, PRESET="synth_tiny", EXP_NAME="", STAGES=("prepare", "train", "eval", "infer"), ABLATION_PRESET="synth_tiny",
                     ABLATION_GROUPS=("private",), ABLATION_ONLY=(), WORK_DIR=work, DATA_DIR=data, SHOW_FIGURES=False,
                     OVERRIDES={"step_batches": 2, "num_workers": 0, "torch_threads": 1, "extra_images": 24, "extra_speech": 16, "extra_per_batch": 4},
                     ABLATION_OVERRIDES={"torch_threads": 1})
    script = tmp_path / "cell.py"
    script.write_text(cell, encoding="utf8")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    run = os.path.join(work, "runs", "synth_tiny")
    for rel in ("checkpoints/last.pt", "checkpoints/best.pt", "curves/losses_train_vs_valid.png", "eval_test.json",
                "inference/report.json", "history.json"):
        assert os.path.exists(os.path.join(run, rel)), rel
    assert not json.load(open(os.path.join(run, "inference", "report.json"))).get("failed")
    abl = os.path.join(work, "runs", "synth_tiny_ablation")
    assert os.path.exists(os.path.join(abl, "SUMMARY.md"))
    # the ablation of PRESET reuses the run above: no second full model, no second representation phase
    assert os.path.exists(os.path.join(abl, "_full", "full_s0", "results.json"))
    assert not os.path.isdir(os.path.join(abl, "_full", "full_s0", "checkpoints")) and not os.path.isdir(os.path.join(abl, "_phase_a"))
    assert "[time]" not in out.stdout + out.stderr
    # the full model's test scores are the ones the "eval" stage stored: the same state is not evaluated twice
    assert "test scores taken from" in out.stdout + out.stderr


def test_every_module_and_every_lazy_import_is_in_the_cell():
    """The end-to-end test above runs the small benchmark, which never reaches code imported lazily for real data
    (the captioner was once left out and the first real run stopped on a NameError).  So: every module of the
    package is in the cell, and every name that a module imports from inside the package is defined in it."""
    import ast

    assert build_notebook.missing_modules() == []
    cell = ast.parse(build_notebook.build())
    defined = {n.name for n in cell.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {x.id for n in cell.body if isinstance(n, ast.Assign) for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    defined |= {t.target.id for t in cell.body if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)}
    wanted = set()
    for path, _ in build_notebook.MODULES:
        for node in ast.walk(ast.parse(open(os.path.join(ROOT, path), encoding="utf8").read())):
            if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module is not None:   # from .x import name
                wanted.update(a.name for a in node.names if a.name not in ("ot", "features", "audio_dsp"))
    assert not (wanted - defined), sorted(wanted - defined)
