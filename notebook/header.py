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
#    Session limit: the cell stops by itself TIME_BUDGET_HOURS after the session started, with the step it reached
#             saved, instead of being cut off (Kaggle ends a session after 12 hours).  Run it again to continue.
#    A first look: PRESET = "synth_fast" runs the whole pipeline on the controlled benchmark in minutes, with no
#             download.
#    Out of GPU memory?  A batch is 64 caption groups of the benchmark plus 64 extra ones per GPU, 13 of them
#             spoken.  OVERRIDES = {"extra_per_batch": 32} halves the extra part; use a new EXP_NAME with it.
#    What to watch: every validation of the transport phase prints one "[text]" line per direction - content F1,
#             BLEU and chrF++, each next to its chance level (the same outputs scored against the caption of another
#             image) - and "[sample]" lines (reference | output).  A score is evidence only by its distance from
#             chance: on Flickr8k a caption that ignores its source still gets about 1 BLEU and 15 chrF++.
#             Whether the model speaks is one line of "infer": "[infer] written speech heard by ..." gives what an
#             independent recogniser understands of the vocoder alone, the renderer alone, the speech codec's
#             reconstruction, speech written from text, and Yoruba speech translated into English speech.
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
                                 # The full real presets add to the training split about 23,700 Flickr30k photos
                                 # with English captions and 5,000 SpokenCOCO photos with 25,031 spoken captions
                                 # (OVERRIDES = {"extra_images": 0, "extra_speech": 0} trains on Flickr8k alone).
STAGES = ("prepare", "train", "eval", "infer")   # any of: prepare, train, eval, infer, ablate  (run in this order)
OVERRIDES = {}                   # any Config field, e.g. {"phase_a_steps": 8000, "bridge_frac": 0.05, "seed": 1}
EXP_NAME = "flickr8k_yo_t4_v6"   # name of the run folder ("" = the preset name).  A run folder is only ever resumed
                                 # with the settings it was started with: new settings need a new name.
EVAL_SPLIT = "test"              # split scored by "eval" and drawn by "infer"

ABLATION_PRESET = PRESET         # benchmark of the ablation study, run after the stages above ("" = none).
                                 # PRESET ablates the run above: that run is the full model and provides the
                                 # representation, so only variants are trained, cheapest first.  On 2 x T4 a
                                 # variant that keeps the representation costs about 2 h (groups read, decoder, island,
                                 # private, lora), one that changes it about 4.5 h (hub, tone, speech, data,
                                 # text_unit, bridge).
                                 # "synth_fast" / "synth": the controlled benchmark, minutes per variant.
ABLATION_GROUPS = ("island", "decoder")   # () = all groups
ABLATION_ONLY = ("bridge pairs only", "decoder frozen")   # () = every variant of those groups; else parts of names.
                                 # These two fit in the session after the main run.  In the last run the first
                                 # showed no effect of back-translation and the second was cut by the session
                                 # limit before it had test scores; both are worth measuring again now that the
                                 # Yoruba recordings are cleaned and the speech latent keeps the order of time.
ABLATION_SEEDS = (0,)            # e.g. (0, 1, 2) for mean +- std in the tables
ABLATION_OVERRIDES = {}          # Config fields of the ablation runs (on top of OVERRIDES when the preset is PRESET)

TIME_BUDGET_HOURS = 11.0         # stop cleanly this long after the session started (0 = never).  Leaves room for
                                 # the evaluation and the figures before Kaggle's 12-hour limit.

WORK_DIR = ""                    # runs/ (checkpoints, curves, figures, tables).  "" = /kaggle/working, /content or .
DATA_DIR = ""                    # downloads + feature caches (about 37 GB for the real benchmark).  "" = automatic
USE_DRIVE = False                # Colab only: put WORK_DIR on Google Drive
SHOW_FIGURES = True              # display the curves and the inferencer figures below the cell when done

# ======================================================================================================================
# 00. ENVIRONMENT  (Kaggle / Colab / anything else; installs only what is missing)
# ======================================================================================================================
import importlib.util as _ilu
import os as _os
import subprocess as _sp
import sys as _sys
import time as _time

IN_KAGGLE = _os.path.isdir("/kaggle/working")
IN_COLAB = "google.colab" in _sys.modules and not IN_KAGGLE


def _session_start() -> float:
    """When the session began, as well as it can be known: the hosted time limit counts from there, not from the
    moment this cell is run.  On Kaggle / Colab that is the start of the container; elsewhere, of this kernel."""
    now = _time.time()
    try:
        import psutil

        kernel = psutil.Process().create_time()
        box = psutil.Process(1).create_time() if (IN_KAGGLE or IN_COLAB) else kernel
        start = box if 0 <= now - box < 12 * 3600 else kernel
        return start if 0 <= now - start < 12 * 3600 else now
    except Exception:
        return now


SESSION_START = _session_start()


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

_NEEDS_HUB = not PRESET.startswith("synth") or not (ABLATION_PRESET or "synth").startswith("synth")
_NEEDED = {"regex": "regex", "sacrebleu": "sacrebleu", "tqdm": "tqdm", "matplotlib": "matplotlib", "sklearn": "scikit-learn"}
if _NEEDS_HUB:
    _NEEDED.update({"transformers": "transformers", "huggingface_hub": "huggingface_hub", "pyarrow": "pyarrow",
                    "soundfile": "soundfile", "PIL": "pillow"})
if "ablate" in STAGES or ABLATION_PRESET:
    _NEEDED["tokenizers"] = "tokenizers"
_missing = [pip for mod, pip in _NEEDED.items() if _ilu.find_spec(mod) is None]
if _missing:
    print(f"[setup] installing {_missing}")
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)
