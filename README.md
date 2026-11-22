# GeoBridge-Flow

Any-to-any conversion between **text, speech and images** for languages that have little or no text, built
around a hub that contains no text at all.

```
 read  (one code per sentence)                        write (many realisations per code)
 text   ─ grapheme-byte codec ─▶ z_txt ─ reader ─┐    ┌─ flow writer ─▶ z_txt ─▶ text
 speech ─ tone-gated codec ────▶ z_aud ─ reader ─┼─▶ hub z* ─┼─ flow writer ─▶ z_aud ─▶ speech
 image  ─ frozen vision tower ─▶ z_img ─ ODE ────┘    └─ flow writer ─▶ z_img ─▶ image
                                   hub z* = K barycentres of frozen vision patch tokens
```

* The hub code of an utterance is a set of attention-weighted **barycentres of frozen vision patch tokens**: it is
  text-free, language-agnostic, sentence-level, and can be drawn on the image.
* **Mutual sufficiency** makes the code say what the sentence says and nothing else: an explainer must
  reconstruct the sentence from the code, an image-free reader must predict the code from the sentence, and
  **counterfactual patch credit** points every slot at the patch that would explain more.
* **Reading is a function, writing is a distribution.** The reader turns a source into its hub code in one pass,
  without an image and without an ODE. Writers are conditional flows from `[hub tokens ; private Gaussian tokens]`
  with a rank-`r` LoRA per language.
* Lexical **tone** reaches the hub through a learned per-language gate; pitch register and timbre bypass it.
* A language with **speech only** is attached through a few image-prompted utterances (bridge pairs); its
  unpaired speech is used by **hub-space back-translation** with an optimal-transport rectification of the pseudo
  codes and a relay to the text writers.
* No loss ever pairs two languages, and a caption is visible in one language only during training.

| Document | Content |
|---|---|
| `GeoBridgeFlow_single_cell.ipynb` | the whole project in **one notebook cell** for Kaggle (GPU T4 x2) and Colab |
| `docs/RESULTS.md` | what has been measured so far (small CPU runs, one seed), what works and what does not |
| `docs/METHOD.md` | equations, and every change made to the roadmap with the reason |
| `docs/LITERATURE_AUDIT.md` | which of the roadmap's citations and novelty claims survive a check against the sources |
| `docs/PAPER_PLAN.md` | claims mapped to the tables this code produces, controls reviewers will ask for, risks |

## Notebook (Kaggle / Colab)

`GeoBridgeFlow_single_cell.ipynb` holds one code cell: settings at the top, then the complete implementation,
then the run. Paste the cell into a notebook and run it.

* **Kaggle**: Accelerator *GPU T4 x2*, Internet *On*. Both GPUs are used: every training step runs one batch per
  GPU, and the frozen vision and speech encoders run on both during feature extraction. Runs go to
  `/kaggle/working/geobridge`, downloads and feature caches to the scratch disk.
* **Colab**: a T4 runtime. `USE_DRIVE = True` keeps the run folder on Google Drive.
* Running the cell again continues: downloads, the split file, feature caches, checkpoints and finished ablation
  variants are found on disk, and training resumes at the step it had reached.

The notebook is generated from the package and tested by executing the cell end to end on the tiny benchmark:

```bash
python scripts/build_notebook.py
```

```bash
pytest -q tests/test_notebook.py
```

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
python run.py ablate --preset synth --groups hub island tone --seeds 0 1 2
```

Any field of `geobridge/config.py:Config` can be overridden, for example
`--set bridge_frac=0.01 vision_encoder=facebook/dinov2-base seed=1`.

| Preset | Data | Meant for |
|---|---|---|
| `flickr8k_yo` | Flickr8k-Tri, all 8,091 images | the main results (one large GPU) |
| `flickr8k_yo_t4` | same | two 16 GB GPUs (Kaggle T4 x2) or one (Colab) |
| `flickr8k_yo_cpu` | 1,200-image subset, small model | pipeline check on a laptop |
| `synth` | SynthWorld, 2,400 scenes | controlled study, ablations on a GPU |
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
it stopped. Train-versus-valid curves are saved to `runs/<exp>/curves/` at every validation. With several GPUs
every step uses one batch per GPU (`n_gpu`, `step_batches`).

**`eval`** scores every direction on the test split and writes `runs/<exp>/eval_test.json` with metrics and
sample outputs.

**`infer`** writes translations, synthesised speech (`.wav`), generated image previews and sixteen figures to
`runs/<exp>/inference/`: slot attention on the image, ODE trajectories, the hub manifold, alignment of unpaired
island speech, tonality gates and pitch-scaling response, frame- and grapheme-level attribution, retrieval
galleries, quality against ODE steps, script fertility, hub agreement, per-language LoRA magnitudes, probes of
hub versus private tokens, codec reconstructions, generated images, and read / write modes (see the docstring of
`geobridge/inferencer.py`).

**`ablate`** trains and evaluates every variant of every ablation group, sharing the representation phase
between variants that do not change it, and writes Markdown, CSV and LaTeX tables and bar charts to
`runs/<exp>_ablation/`. Finished cells are skipped on re-run; `--list` prints the groups.

## Repository map

| Path | Content |
|---|---|
| `geobridge/config.py` | `Config` dataclass, presets |
| `geobridge/data/download.py` | pinned Hugging Face sources of the real benchmark |
| `geobridge/data/flickr8k.py`, `synthworld.py` | benchmark builders (manifest, media access) |
| `geobridge/data/features.py` | frozen front-ends (on all GPUs), one-time feature cache, anchor projection |
| `geobridge/data/dataset.py` | persistent splits, view visibility rules, collation, loaders |
| `geobridge/text_units.py`, `audio_dsp.py` | grapheme / byte / BPE units; YIN pitch, tone features, mel, Griffin-Lim |
| `geobridge/models/` | `text_codec`, `audio_codec`, `hub` (grounder, explainer, reader), `flow`, `pixel`, `ot`, `geobridge` |
| `geobridge/losses.py` | `GeoBridgeLoss`: every objective of both phases |
| `geobridge/cache.py` | `LatentCache`: latents and hub codes for the transport phase |
| `geobridge/trainer.py` | `Trainer`: schedule, multi-GPU step, checkpoints, resume, validation, curves |
| `geobridge/metrics.py` | `Evaluator`: BLEU/chrF/WER, retrieval, speech round trip, tone metrics |
| `geobridge/inferencer.py` | `Inferencer`: conversion API, saved outputs, figures |
| `geobridge/ablation.py` | `AblationRunner`: groups, tables, plots |
| `geobridge/cli.py`, `run.py` | pipeline stages and the command line |
| `notebook/`, `scripts/build_notebook.py` | settings header, driver and generator of the single-cell notebook |
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
