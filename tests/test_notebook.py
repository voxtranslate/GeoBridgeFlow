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
        cell, n = re.subn(rf"(?m)^{name} = .*$", f"{name} = {value!r}", cell, count=1)
        assert n == 1, name
    return cell


def test_single_cell_runs_end_to_end(tmp_path):
    cell = build_notebook.build()
    nb = build_notebook.to_notebook(cell)
    assert len(nb["cells"]) == 1 and "".join(nb["cells"][0]["source"]) == cell
    work, data = str(tmp_path / "work"), str(tmp_path / "data")
    cell = _settings(cell, PRESET="synth_tiny", EXP_NAME="", STAGES=("prepare", "train", "eval", "infer"), ABLATION_PRESET="synth_tiny",
                     ABLATION_GROUPS=("private",), WORK_DIR=work, DATA_DIR=data, SHOW_FIGURES=False,
                     OVERRIDES={"step_batches": 2, "num_workers": 0, "torch_threads": 1},
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
    assert os.path.exists(os.path.join(work, "runs", "synth_tiny_ablation", "SUMMARY.md"))
