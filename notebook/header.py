# ======================================================================================================================
#  GeoBridge-Flow  -  one-cell notebook for Kaggle (GPU T4 x2) and Google Colab
# ======================================================================================================================
#  Any-to-any conversion between images, text and speech for languages with little or no written data.
#
#    text   --grapheme-byte codec-->  z_txt --reader--\                              /--flow writer--> z_txt --> text
#    speech --tone-gated codec---->   z_aud --reader---+--> hub code z* (barycentres +---flow writer--> z_aud --> speech
#    image  --frozen vision tower-->  z_img --ODE-----/      of frozen image patches) \--flow writer--> z_img --> image
#
#  HOW TO RUN
#    Kaggle : Settings -> Accelerator "GPU T4 x2", Internet "On"  (Persistence "Files only" keeps the checkpoints
#             between sessions).  Paste this cell, run it.  Both GPUs are used: one batch per GPU in every
#             training step, and the frozen vision / speech encoders are run on both for feature extraction.
#    Colab  : Runtime -> Change runtime type -> T4 GPU.  Paste this cell, run it.  USE_DRIVE = True keeps the run
#             folder on Google Drive, so that a new session continues where the last one stopped.
#    Interrupted?  Run the cell again.  Downloads, the split file, the feature caches, checkpoints and finished
#             ablation variants are found on disk; training resumes at the exact step it had reached.
#    A first look: PRESET = "synth_fast", ABLATION_PRESET = "" runs the whole pipeline on the controlled benchmark
#             in minutes, with no download.
#
#  CONTENTS
#   00. SETTINGS AND ENVIRONMENT   (the only part meant to be edited)
#   <TABLE OF CONTENTS>
# ======================================================================================================================
from __future__ import annotations

# ======================================================================================================================
# 00. SETTINGS
# ======================================================================================================================
PRESET = "flickr8k_yo_t4"        # benchmark + model size.  Real data: "flickr8k_yo_t4" (2 x T4), "flickr8k_yo",
                                 # "flickr8k_yo_cpu".  Controlled benchmark: "synth", "synth_fast", "synth_tiny".
STAGES = ("prepare", "train", "eval", "infer")   # any of: prepare, train, eval, infer, ablate  (run in this order)
OVERRIDES = {}                   # any Config field, e.g. {"phase_a_steps": 8000, "bridge_frac": 0.05, "seed": 1}
EXP_NAME = ""                    # name of the run folder ("" = the preset name)
EVAL_SPLIT = "test"              # split scored by "eval" and drawn by "infer"

ABLATION_PRESET = "synth_fast"   # benchmark of the ablation study, run after the stages above ("" = no ablation).
                                 # "synth_fast" / "synth" are the controlled benchmark: every factor has a ground
                                 # truth and one variant trains in minutes.  Use PRESET's name to ablate on real data.
ABLATION_GROUPS = ()             # () = all groups; or e.g. ("hub", "island", "tone")
ABLATION_SEEDS = (0,)            # e.g. (0, 1, 2) for mean +- std in the tables
ABLATION_OVERRIDES = {}          # Config fields of the ablation runs

WORK_DIR = ""                    # runs/ (checkpoints, curves, figures, tables).  "" = /kaggle/working, /content or .
DATA_DIR = ""                    # downloads + feature caches (about 20 GB for the real benchmark).  "" = automatic
USE_DRIVE = False                # Colab only: put WORK_DIR on Google Drive
SHOW_FIGURES = True              # display the curves and the inferencer figures below the cell when done

# ======================================================================================================================
# 00. ENVIRONMENT  (Kaggle / Colab / anything else; installs only what is missing)
# ======================================================================================================================
import importlib.util as _ilu
import os as _os
import subprocess as _sp
import sys as _sys

IN_KAGGLE = _os.path.isdir("/kaggle/working")
IN_COLAB = "google.colab" in _sys.modules and not IN_KAGGLE


def _first_writable(*candidates: str) -> str:
    for c in candidates:
        try:
            _os.makedirs(c, exist_ok=True)
            return c
        except OSError:
            continue
    raise OSError(f"none of {candidates} is writable")


if not WORK_DIR:
    if IN_KAGGLE:
        WORK_DIR = "/kaggle/working/geobridge"            # kept as notebook output (20 GB limit: runs only)
    elif IN_COLAB and USE_DRIVE:
        from google.colab import drive as _drive

        _drive.mount("/content/drive")
        WORK_DIR = "/content/drive/MyDrive/geobridge"
    else:
        WORK_DIR = "/content/geobridge" if IN_COLAB else _os.path.abspath("geobridge_runs")
if not DATA_DIR:
    if IN_KAGGLE:
        DATA_DIR = _first_writable("/kaggle/temp/geobridge_data", "/tmp/geobridge_data")   # scratch disk, not output
    else:
        DATA_DIR = "/content/geobridge_data" if IN_COLAB else _os.path.abspath("geobridge_data")
_os.makedirs(WORK_DIR, exist_ok=True)
_os.makedirs(DATA_DIR, exist_ok=True)
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_REAL = not PRESET.startswith("synth") or not (ABLATION_PRESET or "synth").startswith("synth")
_NEEDED = {"regex": "regex", "sacrebleu": "sacrebleu", "tqdm": "tqdm", "matplotlib": "matplotlib", "sklearn": "scikit-learn"}
if _REAL:
    _NEEDED.update({"transformers": "transformers", "huggingface_hub": "huggingface_hub", "pyarrow": "pyarrow",
                    "soundfile": "soundfile", "PIL": "pillow"})
if "ablate" in STAGES or ABLATION_PRESET:
    _NEEDED["tokenizers"] = "tokenizers"
_missing = [pip for mod, pip in _NEEDED.items() if _ilu.find_spec(mod) is None]
if _missing:
    print(f"[setup] installing {_missing}")
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)
