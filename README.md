# GeoBridge-Flow

Any-to-any conversion between **text, speech and images** for languages that have little or no text, built
around a hub that contains no text at all.

```
 text  ──grapheme-byte codec──▶ z_txt ◀──text spoke──┐
 speech ─tone-gated codec─────▶ z_aud ◀─speech spoke─┼──▶  hub z*  =  K barycentres of frozen
 image ──frozen vision tower──▶ z_img ◀─image spoke──┘               vision patch tokens
```

* The hub code of an utterance is a set of attention-weighted **barycentres of frozen vision patch tokens**: it is
  text-free, language-agnostic, sentence-level, and can be drawn on the image.
* Each modality is tied to the hub by a **bidirectional flow** (conditional flow matching) whose source is
  `[hub tokens ; private Gaussian tokens]`, with a rank-`r` LoRA per language on the velocity field.
* Lexical **tone** reaches the hub through a learned per-language gate; pitch register and timbre bypass it.
* A language with **speech only** is attached through a small number of image-prompted utterances (bridge pairs).
  Its remaining unpaired speech can additionally be **docked** onto the hub by an annealed, anchor-pinned fused
  Gromov–Wasserstein coupling; in the experiments run so far this has not improved on the bridge pairs alone.
* At test time nothing but the source is needed: read it into the hub with one ODE, write the target with another.

`docs/LITERATURE_AUDIT.md` records which of the roadmap's citations and novelty claims survive a check against
the sources. `docs/METHOD.md` gives the equations and lists every change made to the roadmap and why.
`docs/PAPER_PLAN.md` maps claims to the tables this code produces. `docs/RESULTS.md` holds the numbers
measured so far, with their limits.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

The controlled benchmark needs no downloads and trains on a CPU:

```bash
python run.py all --preset synth_fast
```

The real benchmark (Flickr8k images, English text + speech, Yorùbá speech, Hindi text; about 7.5 GB, downloaded
automatically, no account needed):

```bash
python run.py prepare --preset flickr8k_yo
```

```bash
python run.py train --preset flickr8k_yo
```

```bash
python run.py eval --preset flickr8k_yo
```

```bash
python run.py infer --preset flickr8k_yo
```

```bash
python run.py ablate --preset flickr8k_yo --groups anchor docking bridge tone --seeds 0 1 2
```

Any field of `geobridge/config.py:Config` can be overridden, for example
`--set bridge_frac=0.01 vision_encoder=facebook/dinov2-base seed=1`.

| Preset | Data | Meant for |
|---|---|---|
| `flickr8k_yo` | Flickr8k-Tri, all 8,091 images | the paper's main results (GPU) |
| `flickr8k_yo_cpu` | 1,200-image subset, small model | pipeline check on a laptop |
| `synth` | SynthWorld, 2,400 scenes | controlled study |
| `synth_fast` | SynthWorld, 1,200 scenes, small model | ablations on a CPU |
| `synth_tiny` | 240 scenes | unit tests (seconds) |

## What each command does

**`prepare`** downloads the pinned dataset revisions, builds the manifest, creates the image-level
train/valid/test split **once** and stores it in `data/cache/<dataset>/splits.json` (later runs load it; the
official YFACC split is used where it exists), then runs the frozen vision and speech encoders, the pitch
tracker and the mel extractor once and stores memory-mapped arrays. Extraction is chunked and resumes after an
interruption.

**`train`** runs both phases in one job with a tqdm bar showing every loss. It writes
`runs/<exp>/checkpoints/step_XXXXXXXX.pt` every `save_every` steps, plus `last.pt`, `phase_a.pt` and `best.pt`.
Re-running the command resumes from `last.pt`; `--resume_from <any step checkpoint>` resumes from that step.
A checkpoint holds the model, the EMA, both optimisers, the AMP scaler, RNG states, the full history and the
position inside the epoch, and the epoch order is a function of (seed, epoch), so the data stream continues where
it stopped. Train-versus-valid curves are saved to `runs/<exp>/curves/` at every validation.

**`eval`** scores every direction on the test split and writes `runs/<exp>/eval_test.json` with metrics and
sample outputs.

**`infer`** writes translations, synthesised speech (`.wav`), and fourteen interpretability figures to
`runs/<exp>/inference/`: slot attention on the image, ODE trajectories, the hub manifold, transport plans against
the true partners, tonality gates and pitch-scaling response, frame- and grapheme-level attribution, retrieval
galleries, quality against ODE steps, script fertility, hub agreement, per-language LoRA magnitudes, probes of
hub versus private tokens, and codec reconstructions (see the docstring of `geobridge/inferencer.py`).

**`ablate`** trains and evaluates every variant of every ablation group, sharing the representation phase
between variants that do not change it, and writes Markdown, CSV and LaTeX tables and bar charts to
`runs/<exp>_ablation/`. Finished cells are skipped on re-run; `--list` prints the groups.

## Repository map

| Path | Content |
|---|---|
| `geobridge/config.py` | `Config` dataclass, presets |
| `geobridge/data/download.py` | pinned Hugging Face sources of the real benchmark |
| `geobridge/data/flickr8k.py`, `synthworld.py` | benchmark builders (manifest, media access) |
| `geobridge/data/features.py` | frozen front-ends, one-time feature cache, anchor projection |
| `geobridge/data/dataset.py` | persistent splits, view visibility rules, collation, loaders |
| `geobridge/text_units.py`, `audio_dsp.py` | grapheme / byte / BPE units; YIN pitch, tone features, mel, Griffin-Lim |
| `geobridge/models/` | `text_codec`, `audio_codec`, `hub`, `flow`, `ot`, `geobridge` |
| `geobridge/losses.py` | `GeoBridgeLoss`: every objective of both phases |
| `geobridge/trainer.py` | `Trainer`: schedule, checkpoints, resume, validation, curves |
| `geobridge/metrics.py` | `Evaluator`: BLEU/chrF/WER, retrieval, speech round trip, tone metrics |
| `geobridge/inferencer.py` | `Inferencer`: conversion API, saved outputs, figures |
| `geobridge/ablation.py` | `AblationRunner`: groups, tables, plots |
| `tests/` | fast checks on SynthWorld (`pytest -q tests`) |
| `scripts/paper_experiments.sh` | the complete GPU study (main table, DINOv2 anchor, all ablations, 3 seeds) |
| `scripts/cpu_ablation_queue.sh` | the reduced single-seed ablation that produced `docs/RESULTS.md` |
| `scripts/make_results.py` | rebuilds `docs/RESULTS.md` from the JSON files under `runs/` |

## Using a trained model

```python
from geobridge.config import Config
from geobridge.inferencer import Inferencer

cfg = Config.load("runs/flickr8k_yo/config.json")
inf = Inferencer(cfg)                                    # loads best.pt with EMA weights

inf.translate_text(["a dog runs on the beach"], "en", "hi")
yo = [u for u, x in enumerate(inf.data.utts) if x["lang"] == "yo"][:4]
inf.speech_to_text(yo, "en")                             # Yorùbá speech -> English text
hub = inf.text_to_hub(["two children play football"], "en")
wav, mel, f0 = inf.hub_to_wave(hub, "yo", ref_utt=yo[0])  # English text -> Yorùbá speech (voice of ref_utt)
```

## Data protocol

* Yorùbá is treated as unwritten: its transcripts are in the manifest but are only used for analysis.
* **Strict non-parallel training** (default): a caption is visible in one language only, so no sentence of a
  non-hub language has its translation in the training set, and no loss pairs two languages.
* `bridge_frac` sets the share of island utterances that keep their image; the rest are fed as unpaired speech.
  The budgets are nested subsets stored in `splits.json`.

Sources and caveats are listed in `geobridge/data/download.py`. Two things to check before publishing numbers: the
two speech mirrors on the Hub carry no licence metadata, and the YFACC mirror stores every clip as exactly 8.0 s.
