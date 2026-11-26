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
  unpaired speech is used by **hub-space back-translation** with a relay to the text writers.
* No loss ever pairs two languages, and a caption is visible in one language only during training.
* The hub language may bring **extra data** of its own, for training only and in no other language: image–text
  pairs (Flickr30k captions for English) and image–text–speech triplets (the SpokenCOCO test part: 25,031 captions
  read aloud by 1,766 speakers).

| Document | Content |
|---|---|
| `GeoBridgeFlow_single_cell.ipynb` | the whole project in **one notebook cell** for Kaggle (GPU T4 x2) and Colab |
| `docs/RESULTS.md` | what has been measured so far (two Flickr8k runs on 2 × T4, small CPU runs; one seed), what works and what does not |
| `docs/METHOD.md` | equations, and every change made to the roadmap with the reason |
| `docs/LITERATURE_AUDIT.md` | which of the roadmap's citations and novelty claims survive a check against the sources |
| `docs/PAPER_PLAN.md` | claims mapped to the tables this code produces, controls reviewers will ask for, risks |

## Notebook (Kaggle / Colab)

`GeoBridgeFlow_single_cell.ipynb` holds one code cell: settings at the top, then the complete implementation,
then the run. Paste the cell into a notebook and run it.

* **Kaggle**: Accelerator *GPU T4 x2*, Internet *On*. Both GPUs are used: every training step runs one batch per
  GPU, and the frozen vision and speech encoders run on both during feature extraction. Runs go to
  `/kaggle/working/geobridge`, downloads and feature caches to the scratch disk (a new session downloads and
  extracts them again, about 40 minutes).
* **Colab**: a T4 runtime. `USE_DRIVE = True` keeps the run folder on Google Drive.
* Running the cell again continues: downloads, the split file, feature caches, checkpoints and finished ablation
  variants are found on disk, and training resumes at the step it had reached. A run folder is only resumed with
  the settings it was started with; new settings need a new `EXP_NAME`.
* Every validation of the transport phase prints one `[text]` line per direction with content F1, BLEU and chrF++
  **next to their chance level** (the same outputs scored against the caption of another image), `[sample]` lines
  (reference | output) and a `[speed]` line. On Flickr8k a caption that ignores its source still scores about
  1 BLEU and 15 chrF++; only the distance from chance is evidence.
* `TIME_BUDGET_HOURS` (11 by default) stops the cell by itself, on a saved step, before the session limit; running
  it again continues. A session that is cut off instead can lose everything written after the last checkpoint.
* `ABLATION_PRESET = PRESET` ablates the run of the cell: that run is the full model and provides the
  representation, so only the variants are trained, cheapest first (`ABLATION_GROUPS`, `ABLATION_ONLY` choose
  which). The study starts by itself after training and inference. Variants that only change how the trained
  model is read and written at test time are scored from its checkpoint with no training; a variant that trains
  a transport phase costs about two hours on 2 × T4, one that changes the representation about five, so a study
  spans several sessions. The cell's defaults: the test-time variants (one candidate, eight candidates, the
  inverted flow), then image-sibling transport, then the two decoders without exposure to written latents.

**Status on real data** (`docs/RESULTS.md`, section 0; one seed). The second full Flickr8k run on 2 × T4 writes
fluent captions that are right about the kind of scene and wrong about most details: English text → hub → English
text names the right kind of subject in 91 % of the stored outputs (22 % by chance), Hindi text → English text
with no parallel data in 79 %, Yorùbá speech → English text, from 985 image pairs and no Yorùbá text, in 63 %
(18 % by chance). BLEU is 9.2, 6.1 and 3.6, against about 1.1 for a caption that ignores its source. Word error
rates are near 80 %: this is not transcription. The tonality gate has no support on real data, the effect of the
counterfactual credit is mixed on validation data, and the effect of back-translation on real data is not known.

That run trained on 6,847 images and memorised them within 15 epochs. The presets now add about 23,700 Flickr30k
photos with English captions (text only, training only, no Flickr8k photo among them) and the SpokenCOCO test
part as speech. On the controlled benchmark the same kind of extra data lifts the language that received none:
island speech → hub text from 60.8 % to 73.8 % exact content.

**The third run** used that data. The text side improved (the text codec is exact, a caption finds its image
twice as often: 23 % against 12 % R@1; Hindi → English BLEU 7.4, Yorùbá speech → English text 4.2) and the island
did not (Yorùbá speech finds its image in 1.2 % of cases). Three things it established:

* Back-translation of the unpaired island speech shows **no effect** on real data (content F1 11.5 with it,
  11.2 without, chance 6.5); the decoder's exposure to written latents is what makes written text readable.
* **Nothing that passes through the speech codec is intelligible.** An independent recogniser hears the vocoder
  alone word for word (4 % word error rate) and "Mmm" in the codec's reconstruction, in text-to-speech and in
  speech-to-speech: the codec's 16 free latent tokens held the utterance's mean and no order of time.
* **The Yorùbá recordings were half rumble.** With it the trimmer trimmed nothing (7.3 s where 4.0 s is speech)
  and the pitch tracker found pitch in 9 % of the frames, so nothing the first three runs say about lexical
  tone means anything.

The recordings are repaired in the presets (`highpass_hz`: 4.0 s and 35 % of the frames voiced, measured on the
recordings). The codec has a speaking path beside it (`audio_anchor = extra`), which on real features
reconstructs speech with its order where the codec reconstructed a constant; a recogniser understood such
reconstructions in a short CPU run once they explained about half of the frame variance, which the speaking
path of the presets had not reached when its short run ended, and speech *written* by the flow is not solved
(`docs/RESULTS.md`). Neither has been in a full run.

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

The real benchmark (Flickr8k images, English text + speech, Yorùbá speech, Hindi text; about 7.5 GB, plus extra
training data for the hub language: 4.4 GB of Flickr30k photos with English captions and 4.3 GB of SpokenCOCO
photos with spoken captions; downloaded automatically, no account needed):

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
| `flickr8k_yo` | Flickr8k-Tri, all 8,091 images, + about 23,700 Flickr30k photos with English captions + 5,000 SpokenCOCO photos with 25,031 spoken captions | one large GPU |
| `flickr8k_yo_t4` | same | two 16 GB GPUs (Kaggle T4 x2) or one (Colab); 8,000 + 8,000 steps |
| `flickr8k_yo_cpu` | 1,200-image subset, small model, no extra data | pipeline check on a laptop |
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

**`eval`** scores every direction on the test split and writes `runs/<exp>/eval_test.json` with metrics, their
chance levels and sample outputs. The evaluated state is `best.pt`: the validation with the highest content F1
above chance, averaged over the text directions.

**`infer`** writes translations, synthesised speech (`.wav`), generated image previews and sixteen figures to
`runs/<exp>/inference/`. On the real benchmark it also lets a recogniser the model has never met (Whisper)
transcribe the speech the model writes, next to the same recogniser on vocoded real speech and on the speech
codec's reconstruction (`outputs/speech_asr.json`). The figures: slot attention on the image, ODE trajectories, the hub manifold, alignment of unpaired
island speech, tonality gates and pitch-scaling response, frame- and grapheme-level attribution, retrieval
galleries, quality against ODE steps, script fertility, hub agreement, per-language LoRA magnitudes, probes of
hub versus private tokens, codec reconstructions, generated images, and read / write modes (see the docstring of
`geobridge/inferencer.py`).

**`ablate`** trains and evaluates every variant of every ablation group and writes Markdown, CSV and LaTeX tables
and bar charts to `runs/<exp>_ablation/`. Nothing is trained twice: the full model is the run `runs/<exp>` itself
(trained there if it does not exist), variants that do not change the representation start from its phase-A
checkpoint, and finished cells are skipped on re-run. Variants run cheapest first, and each is scored at the state
its own validation selected. `--list` prints the groups, `--only` selects variants.

`--hours H` on any command stops training cleanly after H hours (exit code 75); the same command continues.

## Repository map

| Path | Content |
|---|---|
| `geobridge/config.py` | `Config` dataclass, presets |
| `geobridge/data/download.py` | pinned Hugging Face sources of the real benchmark and of the extra data |
| `geobridge/data/flickr8k.py`, `synthworld.py` | benchmark builders (manifest, media access, extra image–text pairs) |
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
| `scripts/refit_preview.py` | refits only the preview pixel decoder of a finished run (changes no metric) |
| `scripts/chance_levels.py` | scores the stored outputs of any `eval_*.json` against chance (references of other images) |

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
* **Extra hub-language data** (`extra_images`, on in the full real presets): Flickr30k photos with their five
  English captions, as text, in the training split only. Flickr30k *contains* the Flickr8k photos with the same
  captions; all 8,091 are removed by Flickr photo id, whatever their split, so no test image and no caption that
  the strict protocol gives to Yorùbá or Hindi comes back as English training text. About 23,700 photos remain.
  As many extra caption groups as own ones join every batch; validation and test are untouched.
* **Extra hub-language speech** (`extra_speech`, on in the full real presets): the test part of SpokenCOCO, 5,000
  COCO photos whose 25,031 captions were read aloud by 1,766 speakers (Flickr8k's speech has 183). The text of
  every clip comes from the COCO annotations by the caption id in the clip's name (checked with a recogniser:
  6.5 % word error against the joined caption on a sample). Training split only; the Flickr8k photo-id rule
  applies (none of the 5,000 is a Flickr8k photo); a fifth of the extra places of a batch go to spoken captions
  (`extra_speech_share`). On the controlled benchmark that share costs the island language nothing, a third
  already did, and speech in every extra place halves its translation; `listen_only` (off) keeps extra speech
  out of everything that shapes the hub, which repairs most of that damage but is worse than plain grounding
  at a fifth (`docs/RESULTS.md`, fourth round).
* **Recordings are high-passed at 70 Hz** before trimming, pitch tracking and feature extraction (`highpass_hz`):
  half of the energy of the YFACC Yorùbá recordings is rumble below 80 Hz, which kept the silence trimmer from
  trimming and the pitch tracker from finding pitch.
* **A speaking path beside the speech codec** (`audio_anchor`, `audio_private_slots`): the free content tokens
  every latent has stay exactly as they were and are all the hub reads; tokens laid out in time (token k
  summarises the k-th span of the utterance) and a second decoder are added for saying it again, with no
  gradient back into the codec. With free tokens alone the speech codec reconstructed a constant and the model
  could not speak; with every token laid out in time the hub could no longer find the content.
* **Every island utterance keeps its image** in the real presets (`bridge_frac = 1`, about 5,000 pairs; 20 % is
  the scarce-pairs study and the only setting with unpaired speech for back-translation).
* **Captions by a pretrained captioner** (`captioner`, `gen_captions`; full real presets): five sampled English
  captions per training image, written once by GIT and cached. Extra text views of images the benchmark already
  has; training images only.
* **Test-time choices** that need no training: read-back selection (`select_k`: write several candidates, keep
  the one that reads back closest to its hub code), the reader or the inverted flow (`read_mode`), integration
  steps (`ode_steps`). The ablation group `decode` scores them from the trained model.
* **Options measured on the controlled benchmark and left off**: image-sibling transport (`w_sib`), listening
  tokens (`listen_tokens`), every speech token laid out in time (`audio_anchor = all`). `docs/RESULTS.md`, fifth
  round, says what each did; the first two are ablation variants for real data.
* **Strict non-parallel training** (default): a caption is visible in one language only, so no sentence of a
  non-hub language has its translation in the training set, and no loss pairs two languages.
* `bridge_frac` sets the share of island utterances that keep their image; the rest are fed as unpaired speech.
  The budgets are nested subsets stored in `splits.json`.

Sources and caveats are listed in `geobridge/data/download.py`. Two things to check before publishing numbers: the
two speech mirrors on the Hub carry no licence metadata, and the YFACC mirror stores every clip as exactly 8.0 s.
