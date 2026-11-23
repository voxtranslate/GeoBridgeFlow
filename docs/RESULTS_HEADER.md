# Results measured so far

Everything below was produced by the code in this repository on a machine with **4 CPU cores and no GPU**.
The runs are therefore small: 3,000 optimisation steps (1,500 representation + 1,500 transport), a model of 5.9 M
parameters, **one seed**, and a test set of 240 caption groups (60 images). With 240 utterances a difference of
fewer than about 8 points between two rows is within noise; the two-object columns rest on 92 captions and are
noisier still. The numbers show that the pipeline runs end to end and which mechanisms have an effect in a
controlled setting. They are **not** the numbers of a paper: those need the GPU schedule in
`scripts/paper_experiments.sh` or the notebook (full data, three seeds, significance tests).

Sections 1–4 are generated from the JSON files under `runs/` by

```bash
python scripts/make_results.py --synth runs/sf_ablation/_full/full_s0 --ablation runs/sf_ablation --real_check runs/real_smoke
```

## 0. What the runs show

**Reference run (SynthWorld, full model, test split; section 1).** `sa` is the hub language, `sb` a tonal language
with speech only, `sc` a Devanagari-script language with text only. Training sees 205 image–speech bridge pairs
of `sb`, 875 unpaired `sb` utterances, no `sb` text and no sentence in two languages.

- *Zero-pair speech translation of an unwritten, tonal language.* `sb` speech → `sa` text: BLEU 48.1, chrF++ 77.1,
  exact content 56.7 % (75.7 % on one-object sentences, 26.1 % on two-object sentences). The previous design of
  this repository reached BLEU 22.4 and 28 % on the same split.
- *Tone survives, and only where it is lexical.* Tone-contrastive (colour) words are right in 87.5 % of island
  utterances and in 50.8 % when the pitch contour is removed from the input; the learned gate is 0.87 for the
  tonal language and 0.01 for the non-tonal one, whose colour accuracy (98.3 %) is identical with and without
  pitch.
- *Hub-language speech recognition through the hub.* WER 13.5 %, exact content 75.4 % (100 % on one-object
  sentences, 35.9 % on two-object sentences).
- *Zero-pair text translation across scripts.* `sa` → `sc` 76.7 % exact content (BLEU 66.8), `sc` → `sa` 85.0 %
  (BLEU 79.9); island speech → `sc` text 51.3 %.
- *What the hub code keeps of a sentence.* Reading a sentence and writing it again in the same language is exact
  for 87.5 % (67.4 % of two-object sentences). With the image in hand, the grounder covers both objects of a
  two-object caption in 92 % of text captions, 63 % of hub-language speech and 60 % of island speech; 7–12 % of
  the slots sit on an object the sentence does not mention.
- *Retrieval in hub coordinates* (60 images; chance R@1 1.7 %): text → image R@1 55.4 %, island speech → image
  42.1 %, hub-language speech → image 50.0 %.

**What was learned on the way to this configuration** (validation split, single runs; the first two are also
cells of the `hub` ablation).

- *Counterfactual patch credit is what recovers the second object.* Without it both objects of a two-object text
  caption were covered in 68 % of cases in one run (speech 25 %, island speech 30 %) and 36 % in the ablation
  cell; with it 92–94 % in three runs (speech 59–67 %, island speech 58–68 %). Without the credit the outcome
  depends on the run; with it it does not.
- *Stricter versions of it are worse.* A veto on moves that verifiably explain less gave 75 % / 30 % / 51 %.
  Requiring a verified gain before pulling switched the mechanism off (6 %): the explainer has to co-adapt before
  a newly covered object pays off. The latter variant is not kept in the code.
- *"Text teaches speech" in the grounder is harmful.* Making the text code a fixed target for speech dropped the
  coverage of two-object text captions to 22 % and 12 % (two seeds) from 68 % with mutual agreement.
- *The reader beats the backward ODE as a read path for speech* (hub-language speech → text 77 % vs 52 % exact
  content, island 61.5 % vs 41 % on 200 validation captions), and starting the writer from the explainer's draft
  lowers two-object accuracy, so it is off by default.

**What the ablations say** (section 2; test split, one seed, 1,500 + 1,500 steps per cell; "ours" is the
reference run above).

- *Counterfactual patch credit.* Without it, two-object sentences collapse: read-and-rewrite of text 23.9 % (ours
  67.4 %), island speech 4.3 % (26.1 %), both objects covered for 36 % of text captions (92 %) and 9 % of island
  captions (60 %); hub-language recognition WER 31.3 % (13.5 %), text translation 66.7 % / 71.2 % (76.7 % /
  85.0 %). Its price: 7.3 % of slots sit on an unmentioned object (0.2 % without). On one-object sentences it
  changes nothing, so overall island content moves only from 53.8 % to 56.7 %.
- *Minimality.* Without it every object in the picture gets a slot (both covered 100 %, but 20 % of slots on
  unmentioned objects) and the code stops being what the sentence says: island content 37.5 % (56.7 %),
  read-and-rewrite of two-object text 42.4 % (67.4 %).
- *Sentence-level visual hub vs scene-level visual pivot.* The pivot loses on every translation column: island
  BLEU 21.9 (48.1), hub-language WER 45.4 % (13.5 %), text translation 50.8 % / 58.3 % (76.7 % / 85.0 %).
- *Visual hub vs text anchor.* The text anchor ties on island → hub-language text (56.7 % content, BLEU 34.8 vs
  48.1) and on recognition, is **better** at read-and-rewrite of two-object text (83.7 % vs 67.4 %), and is worse
  for the third language (60.0 % / 58.3 % vs 76.7 % / 85.0 %) and useless for images (retrieval at chance). The
  claim that survives is "balanced across languages and modalities", not "better everywhere".
- *Hub-space back-translation.* Bridge pairs alone give 40.4 % island content (BLEU 35.5); adding
  back-translation of the 875 unpaired utterances gives 56.7 % (48.1). Without the relay to the text writers it
  is 48.8 % (42.5): the relay is about half of the gain, at the edge of the noise.
- *Transport rectification of the pseudo codes: no measurable effect* (58.8 % without it vs 56.7 %). It stays in
  the code as an option and should not be claimed.
- *Read-hub exposure and hub jitter: no measurable effect* (55.4 % with neither, 52.5 % without jitter, vs
  56.7 %).
- *Reader vs backward ODE.* For speech the one-pass reader is far better: island content 56.7 % vs 32.5 %,
  hub-language WER 13.5 % vs 26.1 %. For text the backward ODE is better: read-and-rewrite of two-object text
  91.3 % vs 67.4 %, text translation 81.2 % / 88.3 % vs 76.7 % / 85.0 %. A per-modality read mode would take the
  better of each; it was not trained.
- *Tone stream.* Without it, tone-contrastive words on the island fall from 87.5 % to 73.3 %, island content from
  56.7 % to 43.3 %, island speech → image R@1 from 42.1 % to 32.5 %; the non-tonal hub language moves within
  noise (75.4 % → 71.7 %).
- *Not run under this design:* the bridge-budget sweep, private tokens, LoRA rank, text units, the protocol
  group, the veto and "text teaches speech" cells of `hub`, Gromov–Wasserstein docking, the free anchor, and the
  canonical-code cell of `read`. The numbers quoted for the veto and for "text teaches speech" above are
  validation-split diagnostics of single runs.

**Does not work yet.**

- *The second thing a sentence says, from speech.* Two-object sentences are translated correctly from island
  speech in 26 % of cases and transcribed from hub-language speech in 36 %, against 67 % for text. The grounder
  covers both objects for 60–63 % of spoken captions; the rest of the gap is in the speech reader.
- *Image captioning and image writing.* Image → text reaches 9.6 % exact content: an image does not say which of
  its objects a caption is about, and a single read of the image latent does not either. Written image latents
  retrieve the right image at R@1 13.3 % (text) and 10.8 % (island speech); R@10 35 % and 32 % (chance 16.7 %).
  The preview (`15_generated_images.png`) shows blobs of mostly the right colours in the wrong places: a caption
  does not say where things are. The preview decoder of the reference run was refitted alone after training
  (`scripts/refit_preview.py`): with the plain L1 loss it had been trained with, it drew only background. It
  shares no gradient with the rest of the model, so no number in this document depends on it; the ablation cells
  were trained before that fix.
- *The speaker is not kept out of the hub.* A linear probe finds the speaker in the reader's hub code with
  31.7 % accuracy for the hub language (chance 20 %; 41.7 % in the backward-ODE read-out) and 80.8 % for the
  island (chance 50 %; 84.2 %). The reader leaks less than the ODE, not nothing.
- *Speech into the island language.* Text → island speech → (re-read) text recovers the content in 36.7 % of
  cases (50.4 % for hub-language speech); this is a feature-level round trip, not a listening test.

**Real data: the first full run failed** (Flickr8k-Tri, Kaggle 2 × T4, preset `flickr8k_yo_t4` as it was then:
16,000 + 30,000 steps, 74 M parameters, one seed; numbers read off the log of that run, which trained to the end;
nothing from it is stored in this repository).

| Test split, best checkpoint (step 30,000) | |
|---|---|
| English text → hub → English text, BLEU | 2.05 |
| English speech → English text, BLEU | 1.22 |
| Yorùbá speech → English text, BLEU | 0.91 |
| Hindi text → English text / English text → Hindi text, BLEU | 1.43 / 0.92 |
| Yorùbá speech → Hindi text, BLEU | 0.49 |
| Image → English text, BLEU | 0.42 |
| Retrieval of the image among 624, R@1 (chance 0.16 %): English text / Hindi text / English speech / Yorùbá speech | 11.76 / 6.09 / 4.01 / 1.16 % |
| Same, with the image latent written by the flow | 12.02 / 5.90 / 3.69 / 1.54 % |

The first row is the diagnosis: the source is text and the target is the same language, so nothing has to be
recognised or translated, and the output is still unreadable. The failure is in writing text from a hub code.

- *The machinery works.* Download, split, feature extraction (45,981 utterances, 50.4 h of speech, 42 minutes in
  all), both GPUs (`cuda x2 (DataParallel)`, one batch of 64 per GPU and step, mixed precision), checkpoints,
  validation: no error. Speed: 1.15 steps/s in the representation phase, 3.1 in the transport phase, 1.7 once
  back-translation starts.
- *Text written through the hub is unreadable.* English speech → English text: BLEU 1.3–1.6, WER 92–95 %;
  Yorùbá speech → English text: BLEU 0.9–1.5; image → text: BLEU 0.3–0.6. Retrieval says the hub itself carries
  content (English text → image R@1 17–19 % among 620 images, chance 0.16 %), so the failure is in turning a hub
  code back into text. On real captions the hub explains about 28 % of the text latent's variance on held-out
  data (86 % on SynthWorld); the writer has to invent the rest, and the byte decoder had only ever been trained
  on exact encoder latents.
- *The representation phase overfits.* Training grounding accuracy reached 99 %, validation peaked at 63 % after
  4,000 steps and fell to 53 %; validation unexplained variance rose from 0.55 to 0.72; the reader's validation
  error rose throughout the transport phase (0.176 → 0.187 text, 0.244 → 0.256 speech) while the training error
  fell.
- *The credit weight is high.* Its mean rose from 0.25 to 0.78 after the release and the training loss rose with
  it (4.0 → 5.4): with 49 diverse patches and half of the latent unexplained, the best of many first-order
  estimates is usually large. On SynthWorld the mean is 0.4–0.5 where the credit works; whether it helps or
  disturbs on real photographs cannot be read off this log.
- *The tonality gate did the opposite of the claim.* English 0.55, Yorùbá 0.00. Every language in a batch paid the
  same price for its gate, while only image-paired utterances can show that pitch is useful; Yorùbá has 985 of
  them, English 30,000.
- *Yorùbá speech → image retrieval is close to chance* (1 of 67 validation utterances at R@1; 1.16 % on test).
- *The transport phase overfits as well.* Validation split, measured every 2,000 steps: the flow-matching error
  of the image writer was lowest after 4,000 transport steps (0.380) and rose to 0.632 by the end; that of the
  speech writer was lowest after 10,000 (0.411) and ended at 0.456; that of the text writer was lowest after
  14,000 (0.374) and ended at 0.387; the reader's error rose throughout (text 0.176 → 0.205, speech 0.244 →
  0.264); English text → image R@1 fell from 18.8 % to about 15 %. The writers are trained on a fixed set of
  cached latents. The validation score that selects the checkpoint peaked after 14,000 transport steps, with text
  scores too low to mean anything.
- *The session ended inside the ablation.* Preparation took 42 min, training ended 9 h 07 min after the start of
  the cell, the test evaluation took 11 min and the sixteen figures 2 min. The ablation then trained the
  representation phase again from scratch, although the main run held exactly that representation, and Kaggle's
  12-hour limit cut it at step 10,533 of 16,000: 2 h 40 min that produced nothing.
- *Two runs of the same representation phase separate.* That repeated phase had the same settings, seed and data
  order as the first, and its logged losses agree with the first to three decimals for 2,000 steps. After that
  the speech codec's semantic reconstruction error falls more slowly on the training data (0.57 against 0.47 at
  step 10,000) and, on held-out speech, does not fall at all: 0.85 at step 10,000 against 0.67. Text
  reconstruction, the mel and pixel losses and grounding accuracy stay within a few hundredths of the first run,
  and the reader's held-out error on speech is no worse (0.340 against 0.362 at step 8,000). Whether the speech
  codec learns a code that generalises was decided by GPU non-determinism in this pair of runs. One seed cannot
  carry a claim about speech output on real data.

**What was changed in response, and what was tried and rejected** (SynthWorld, single runs; the two-object
coverage is the validation-split diagnostic used throughout).

- *Kept: write-latent exposure of the byte decoder* (`w_dec`; section 2, group `decoder`). Same phase-A state,
  test split: read-and-rewrite of text 87.5 % → 95.8 % (two-object sentences 67.4 % → 89.1 %), hub-language
  recognition WER 13.5 % → 8.6 %, zero-pair text translation 76.7 % / 85.0 % → 92.1 % / 93.8 %; island speech
  unchanged (55.0 % vs 56.7 %).
- *Kept:* a shorter schedule (8,000 + 14,000 steps); the transport phase starts from the phase-A state with the
  best validation hub losses (`phase_a_select`); noise on the latents the reader is trained on (`read_noise`); a
  gate price paid per grounded utterance (`gate_price = evidence`); sample outputs and a data-versus-compute
  timing line in the log at every validation; resuming a run folder under different settings is refused.
- *Rejected: dropout.* Residual dropout 0.1 in the codecs, grounder, explainer and reader destroys the two-object
  coverage the credit loss buys: 13 % of text captions against 93 %. It is the standard answer to the overfitting
  seen on Flickr8k and it is not in the presets for this reason.
- *Rejected: a scale-free credit rule* (gain relative to the utterance's unexplained variance): coverage 49 %.
  With the veto on top and all regularisers: 4 %. The variant was removed from the code.
- *Not adopted: speech time masking + weight decay 0.05* (with the evidence gate price, no dropout): coverage
  72 %, not separated from run-to-run noise.
- With exactly the option set of the real-data presets switched on, SynthWorld gives: island speech → text 60.8 % exact content (reference 56.7 %), hub-language
  recognition WER 10.1 % (13.5 %), read-and-rewrite of text 96.3 % (87.5 %), zero-pair text translation 89.6 % /
  94.2 % (76.7 % / 85.0 %), both objects covered for 97 % of two-object text captions (92 %), gates 0.00 / 0.92.
  Trained from scratch (1,500 + 1,500 steps), test split; nothing got worse
  (`docs/results/synth_with_real_preset_options_eval_test.json`).

- *Kept, after the complete log:* the ablation never trains what the main run has (the main run is its full
  model, and its phase-A checkpoint is used by every variant that leaves the representation alone); the cell and
  the command line take a time budget and stop on a saved step before a hosted session is cut; the transport
  phase is 14,000 steps, where that run's validation score peaked.
- *Open: the transport phase has no regulariser of its own* beyond the jitter on hub codes, the noise on the
  reader's inputs and the checkpoint selection. Dropout inside the writers would not touch the credit loss, which
  acts in the representation phase; it has not been tried.

**None of this has been run on real data at scale.** The changes answer what the log showed; whether they are
enough is not known until the next GPU run.

**Not verified.**

- *Real data, again.* No real-data number in this repository supports any claim.
- *Seeds.* Everything is one seed.
