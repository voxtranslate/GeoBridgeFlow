# Results measured so far

Everything below was produced by the code in this repository on a machine with **4 CPU cores and no GPU**, except
the two full Flickr8k runs on Kaggle (2 × T4) reported in section 0.
The CPU runs are small: 3,000 optimisation steps (1,500 representation + 1,500 transport), a model of 5.9 M
parameters, **one seed**, and a test set of 240 caption groups (60 images). With 240 utterances a difference of
fewer than about 8 points between two rows is within noise; the two-object columns rest on 92 captions and are
noisier still. The numbers show that the pipeline runs end to end and which mechanisms have an effect in a
controlled setting. They are **not** the numbers of a paper: those need the GPU schedule in
`scripts/paper_experiments.sh` or the notebook (full data, three seeds, significance tests).

Sections 1–4 are generated from the JSON files under `runs/` and `docs/results/` by

```bash
python scripts/make_results.py --synth runs/sf_ablation/_full/full_s0 --ablation runs/sf_ablation --real_eval docs/results/real_run2/eval_test_step22000.json
```

## 0. What the runs show

**In short.** On the controlled benchmark the mechanisms do what they were designed to do, in single small runs.
On Flickr8k (second full run, one seed) the model converts between text, speech and languages at the level of
*what is in the picture*: its captions name the right kind of subject in 91 % of cases from English text, 79 % from
Hindi text without parallel data, and 63 % from Yorùbá speech learned from 985 image pairs and no text, against
about 20 % by chance; wording is not preserved (word error rate near 80 %). The tonality gate has no support on
real data, and whether counterfactual credit and back-translation help there is not established.

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

**Real data: the second full run writes readable text, right about the kind of scene and wrong about the
details** (Flickr8k-Tri, Kaggle 2 × T4, 8,000 + 14,000 steps, 74 M parameters, one seed; the settings above. Its
scores, sample outputs, curves and log lines are in `docs/results/real_run2/`.)

Test split (624 images, 3,120 captions; 519 Yorùbá utterances, counted from the step of the retrieval scores). The run's checkpoint selection picked
step 9,000; the last state is step 22,000.

| | first run | step 9,000 | step 22,000 |
|---|---|---|---|
| English text → hub → English text, BLEU | 2.05 | 7.37 | 9.22 |
| English speech → English text, BLEU | 1.22 | 5.65 | 5.86 |
| Yorùbá speech → English text, BLEU | 0.91 | 4.12 | 3.59 |
| Hindi text → English text / English text → Hindi text, BLEU | 1.43 / 0.92 | 5.93 / 4.89 | 6.09 / 5.10 |
| Yorùbá speech → Hindi text, BLEU | 0.49 | 2.22 | 2.27 |
| Image → English text, BLEU | 0.42 | 2.60 | 3.06 |
| Image retrieval R@1 (chance 0.16 %): English text / Hindi text / English speech / Yorùbá speech | 11.8 / 6.1 / 4.0 / 1.2 % | 14.7 / 8.6 / 5.2 / 1.5 % | 15.9 / 8.4 / 5.1 / 1.0 % |

- *What a BLEU of 4 means here.* A caption model that ignores its source still writes fluent captions. Scoring
  the stored outputs (60 per direction, step 22,000) against the reference of another image gives the chance
  level: about 1.1 BLEU, 15.5 chrF++. The first run was at chance in every text direction; this one is not
  (`python scripts/chance_levels.py docs/results/real_run2/eval_test_step22000.json`; chance ± spread over 200
  re-pairings; "subject" = the output names the right kind of subject, e.g. dog, man, woman, child):

  | 60 outputs each | content F1 (chance) | BLEU (chance) | right subject (chance) |
  |---|---|---|---|
  | English text → English text | 22.6 (7.1 ± 0.8) | 10.2 (1.2 ± 0.5) | 91 % (22 ± 5) |
  | English speech → English text | 16.2 (6.8 ± 0.7) | 5.2 (1.2 ± 0.5) | 82 % (19 ± 5) |
  | Hindi text → English text, no parallel data | 16.2 (7.2 ± 0.8) | 5.6 (1.2 ± 0.6) | 79 % (21 ± 5) |
  | Image → English text | 13.8 (6.0 ± 0.6) | 5.5 (1.0 ± 0.4) | 70 % (22 ± 5) |
  | **Yorùbá speech → English text**, 985 image pairs, no Yorùbá text | 10.3 (6.0 ± 0.7) | 3.0 (1.2 ± 0.5) | 63 % (18 ± 5) |
  | Yorùbá speech → Hindi text | 9.6 (6.1 ± 0.7) | 1.4 (0.9 ± 0.3) | – |
  | English text → written English speech → read again | 7.1 (6.6 ± 0.6) | 1.0 (1.0 ± 0.4) | 25 % (18 ± 3) |
  | English text → written Yorùbá speech → read again | 11.1 (5.8 ± 0.7) | 3.0 (1.1 ± 0.5) | 55 % (16 ± 5) |

  Typical outputs: "A black lab with tags frolicks in the water" → "A black dog with a red collar splashes in the
  water"; Yorùbá speech for "a black dog jumping to catch a rope toy" → "A black dog is chasing a ball while
  another dog in a field"; Yorùbá speech for "A blond woman in a blue shirt appears to wait for a ride" → "Two
  kids sit on a statue of a large plastic box".
- *This is not transcription.* Word error rate is 78 % for text → text and 85 % for English speech → text. The
  hub keeps what can be pointed at in a picture: on held-out captions it explains 47 % of the text latent's
  variance; wording, counts and attributes are filled in by the writer. A hub of visual barycentres cannot be the
  channel of a verbatim recogniser or translator, and the numbers should not be presented as ASR or MT.
- *The text codec is itself lossy.* Encoded and decoded without the hub, held-out English captions come back
  with a character error rate of 28.5 % (Hindi 44.4 %) at the last state, although the decoder's byte accuracy
  under teacher forcing is 97 %. Whether that is free-running error accumulation or the decoder having moved
  towards written latents during the transport phase was not separated. Nothing that passes through the hub can
  be more exact than this.
- *The island is attached, weakly.* Yorùbá speech finds its image among 624 at R@1 1.0 %, R@10 10.6 % (chance
  0.16 % and 1.6 %), and its English text names the right subject in 63 % of 57 outputs against 18 % by chance.
  English speech, with twenty times more image pairs, reaches 82 % and R@1 5.1 %: the speech side as a whole is
  weak, and the island is not far below what the hub language's speech achieves.
- *The checkpoint was selected by noise.* The selection score was the island BLEU on the validation subset used
  during training: 400 random caption groups, of which 67 are Yorùbá utterances (the validation split holds about
  500). Its BLEU was highest at the first validation
  of the transport phase (5.3 after 1,000 steps) and the test evaluation and all figures were made from that
  state. Text → text was still improving: 7.4 BLEU there, 9.2 at the end.
- *Both phases overfit, and the transport phase is finished early.* Representation phase: validation grounding
  accuracy 63 % after 4,000 steps, 58 % after 8,000 (training 95 %), in this run and in the first whatever the
  schedule; the transport phase started from the state of step 5,000. Transport phase, validation every 1,000
  steps: image → text best after 2,000 of its steps, English speech → text and text → text after 8,000–9,000,
  then all decline; the decoder's validation loss is lowest after 3,000 steps and rises from 0.42 to 0.67 while
  its training loss falls to 0.17; the reader's validation error rises from the first step (speech 0.255 → 0.275,
  training 0.16 → 0.12).
- *Yorùbá gets worse during the transport phase.* Validation BLEU 5.3, 5.0, 5.2 in the first 3,000 steps, then
  3.9 falling to 1.7 at the end; against all captions of the image 12.9 falling to 4.9. Back-translation starts
  at step 2,800 of the phase, but the multi-reference score falls before that too, the sample is 67 utterances,
  and on the test split the difference between the two states is 4.1 against 3.6. The variant without
  back-translation was not reached, so whether back-translation helps or hurts on real data is not known.
- *Counterfactual credit on real data: mixed, validation only.* The one ablation variant that trained (no
  credit; interrupted 3,700 steps before its end, so no test scores) differs from the full model as follows, at
  the same steps, one run each. With credit the hub explains more: unexplained variance of the text latent on
  held-out captions 0.53 against 0.59 at the selected state (0.53 against 0.62 at the end of the phase), about
  92 % of the slots in use against 83 %, lower flow-matching error of the
  text writer at all ten validations (0.296 against 0.321 on average), lower word error rate of text → text at
  nine of ten (78.6 against 80.6), better image → text at nine of ten. Without credit the code is easier to
  read without the image: reader error 0.16 against 0.20 for text and 0.19 against 0.26 for speech, and the
  island's validation BLEU is higher at seven of ten (4.3 against 3.9, 67 utterances). BLEU of text → text,
  English speech → text and retrieval do not differ.
- *Speech is the weak side.* The speech codec's held-out reconstruction error of the semantic stream stayed at
  0.83 in both representation phases of this session (training 0.62); so far it has generalised in one phase-A
  run of four.
- *Tone: no support.* The gates were still closing when the representation phase ended (English 0.69, Yorùbá
  0.38 at step 8,000, both falling) and the values reported with the selected state (0.83, 0.61) are a snapshot
  1,000 steps after their release. Yorùbá's gate closes faster than English's under either price rule; removing
  the pitch contour does not lower Yorùbá's score (BLEU 3.96 flat, 3.59 with pitch) and moves its hub code less
  than English's (0.06 against 0.29 of the distance between two utterances).
- *Written speech.* Read back by the model's own reader, written English speech carries no measurable content
  (25 % right subject, chance 18 ± 3) while written Yorùbá speech does (55 %). Back-translation trains the reader
  on written Yorùbá speech and nothing trains it on written English speech, so this compares the reader's
  exposure, not the two writers. No recogniser has listened to the audio.
- *The ablation gave one comparison of two identical rows.* Under the real-data preset both variants of the
  `decoder` group had the base settings and were the same run. The study started with the most expensive group:
  the first variant needed a representation phase of its own (2 h 12 min) and was cut by the time budget after
  10,259 of its 14,000 transport steps.
- *The machinery.* Preparation 52 min (speech features 34, mel 12), representation phase 2 h 08 min at 1.05
  steps/s, transport phase 3 h 04 min (1.9 steps/s, 1.25 once back-translation runs), test evaluation 10 min,
  figures 2 min. The main run was recognised as the full model of its ablation and the cell stopped by itself
  10 h 57 min after its start, with the interrupted step saved.

**What was changed after the second run.**

- *Every text score is reported with its chance level*, and a content measure was added (content F1: word
  overlap weighted by inverse document frequency; `docs/METHOD.md`, section 8). The log prints one line per
  direction at every validation.
- *Checkpoint selection* uses content F1 above chance averaged over the text directions, and the validation
  subset gives up to half of its places to island captions (600 groups in the T4 preset: about 300 Yorùbá
  utterances instead of 67).
- *Schedule:* 8,000 + 10,000 steps.
- *Ablation:* every group has exactly one variant with the base settings, listed first in its table; variants run
  cheapest first (those that reuse an existing representation before those that need their own); each variant is
  scored at the state its own validation selected, and the main run's stored test scores are reused. The
  notebook's defaults run the two variants that answer what this run left open: no back-translation, and the
  decoder frozen.
- *Plots:* the gate panel showed the price paid for the gates as a language called "pen" and a gate for Hindi,
  which has no speech; both are gone.
- Decoder exposure is now the default everywhere.
- *Tried and rejected: dropout confined to the encoders* (`enc_dropout = 0.1`; SynthWorld, same settings as the
  run with the real-preset options above, test split). The idea was that the overfitting of the representation
  phase could be treated upstream of the latent without disturbing the grounder and explainer, whose gradient is
  the credit signal. It collapses the second object all the same: both objects covered for 15 % of two-object
  text captions against 97 %, read-and-rewrite of text 73.8 % against 96.3 % (two-object sentences 31.5 % against
  90.2 %), island two-object sentences 4.3 % against 29.3 %, hub-language recognition WER 27.0 % against 10.1 %;
  one-object island sentences improve (93.9 % against 80.4 %). With speech time masking on top: the same picture
  (22 % coverage). Files: `docs/results/synth_encoder_dropout*_eval_test.json`. Whatever makes a hub code cover
  everything a sentence says does not survive noise in the latent it has to explain. The representation phase's
  overfitting on Flickr8k therefore has no remedy in this repository beyond selecting an early state.

**Third round: more data, and the remaining problems of the analysis** (first run on a GPU: the third run, below).

- *More data for the hub language.* The full real presets now add the Flickr30k photos with their English
  captions to the training split, as text (`docs/METHOD.md`, section 1). Flickr30k contains the Flickr8k photos
  with the same captions: in the first of its nine files 1,045 of 3,532 photos are Flickr8k photos. All of them
  are removed by Flickr photo id, whatever their split; a test fails if one gets through. About 23,700 photos
  with 118,000 captions remain, against 6,847 training images before. Checked on real files with a 1,200-image
  subset and 300 extra photos (131 Flickr8k photos dropped, none kept, features on the same scale as the
  benchmark's own), and trained end to end on a 60-image subset.
- *The same thing on SynthWorld, where it can be measured before a GPU is spent:* 2,400 extra scenes with 9,600
  hub-language captions as text (the benchmark's own training split has 1,080 scenes), same settings and number
  of steps as the run with the real-preset options, test split, one run each
  (`docs/results/synth_extra_hub_data_eval_test.json`). The island language received nothing new and improves
  with the rest:

  | | benchmark data only | + extra hub-language image–text pairs |
  |---|---|---|
  | Island speech → hub text, exact content (one-object / two-object sentences) | 60.8 % (80.4 / 29.3) | **73.8 %** (93.9 / 41.3) |
  | Island speech → text of the third language | 65.0 % | 73.8 % |
  | Hub-language speech → text, WER / exact content | 10.1 % / 81.3 % | **2.1 % / 93.3 %** |
  | Text read and written again (two-object sentences) | 96.3 % (90.2) | 100 % (100) |
  | Text translation, both directions | 89.6 % / 94.2 % | 98.8 % / 98.8 % |
  | Island speech → hub-language speech, read back | 32.9 % | 55.0 % |
  | Both objects covered: text / hub speech / island speech | 97 % / 62 % / 57 % | 100 % / 90 % / 63 % |
  | Retrieval of the image, R@1: text / hub speech / island speech | 55.4 / 54.6 / 45.0 % | 57.9 / 53.3 / 42.1 % |

  Two runs of the same settings differ by about five points on these columns, so every row but the last is a
  real difference. Retrieval does not move: the hub was already as discriminative as the grounder could make
  it; what the extra captions buy is a text side that is exact, and with it the codes that speech is taught to
  agree with. Batches are also twice as large with the extra groups, which this run does not separate from the
  data itself.
- *A recogniser listens to the written speech* (`infer`; Whisper). Three rungs: the real utterance's log-mel
  through the vocoder, the speech codec's reconstruction, and speech written from text or from Yorùbá speech.
  On the 60-image check model, which is untrained, the first rung is transcribed almost word for word (six
  utterances: word error rate 5 %, content F1 94 against 6 by chance; "Lady was straw hat covering face and head
  standing on a rock in front of the ocean" for "Lady with straw hat …"), so the mel features and the vocoder
  are right; the other rungs are noise there, as they must be. What a trained model scores is not known.
- *The reader is trained on the speech the writer writes, in the hub language too* (`w_reread`, on by default).
  On Flickr8k written Yorùbá speech could be read back and written English speech could not, because
  back-translation shows the reader written island speech and nothing showed it written hub-language speech.
  Now real hub codes are written as hub-language speech and read back at every transport step. SynthWorld,
  same settings as the run with the real-preset options, test split, one run each
  (`docs/results/synth_reader_exposure_eval_test.json`): island speech → hub-language speech, read back, 43.8 %
  exact content against 32.9 %; text → island speech 48.3 % against 43.3 %; text → hub-language speech
  unchanged (51.3 % against 51.7 %: it was readable there already). Nothing else moves by more than two runs of
  the same settings differ (island speech → text 65.8 % against 60.8 %, recognition WER 6.2 % against 10.1 %,
  read-and-rewrite of text 95.4 % against 96.3 %). Whether it repairs the round trip on Flickr8k is not known.
- *The text codec alone is scored at every validation.* On SynthWorld it is exact (character error rate 0.0 %),
  so the 28.5 % measured on Flickr8k is what the codec had learned from 30,000 captions, or what the decoder
  lost while it was trained on written latents; the validation log will now separate the two, and the extra
  captions are also text for the codec.
- *Back-translation's relay draws from the benchmark's own captions,* so the extra English text does not crowd
  the Hindi writer out of it.
- *The tonality gates are logged at the selected state and at the end of the representation phase,* with a
  warning when they are still moving. This fixes the reporting, not the result: on real data the claim has no
  support.
- *Preparation:* the mel pass runs in worker processes; the cache it writes is identical to the old one, bit for
  bit (checked on a 60-image subset). A GPU that runs out of memory now says which setting to lower.

**Fourth round: speech** (first run on a GPU: the third run, below).

- *More real speech for the hub language.* The test part of SpokenCOCO joins the training split: 5,000 COCO
  photos whose 25,031 captions were read aloud by 1,766 speakers (the Flickr8k recordings have 183). The mirror
  holds audio and images; the text of a clip comes from the COCO annotations by the caption id in the clip's
  name. That join was checked, not assumed: on 22 clips a recogniser's word error rate against the joined
  caption is 6.5 %, against the caption of another clip 103 %. None of the 5,000 photos is a Flickr8k photo
  (by Flickr address; the rule that drops them is in place and tested). Prepared on real files with a 40-image
  subset (200 clips, 130 speakers) and trained end to end on a small subset.
- *What extra hub-language speech does, measured on SynthWorld before a GPU is spent.* 1,200 more scenes whose
  4,800 hub-language captions are written *and* spoken, with and without the 2,400 text-only scenes of the
  third round; 48 extra caption groups in every batch, same settings and steps as the reference, test split,
  one run each. The cached speech of the extra scenes was recomputed for a sample and is identical, and every
  extra caption matches its image. SynthWorld's extra speech comes from the benchmark's own six speakers, so
  these runs measure what more hub-language speech in a batch costs or buys, not the benefit of more voices,
  which only real data can show.

  *The recipe as shipped* (reader exposure and augmentation on; `docs/results/synth_shipped_recipe_*`):

  | | benchmark only¹ | + text | + text + speech (a fifth of the places) | … and that speech only listens |
  |---|---|---|---|---|
  | Island speech → hub text, exact content (two-object sentences) | 60.8 % (29.3) | 73.8 % (40.2) | **77.1 %** (50.0) | 68.3 % (26.1) |
  | Island speech → text of the third language | 65.0 % | 73.3 % | 77.9 % | 68.3 % |
  | Hub-language speech → text, WER / exact content | 10.1 % / 81.3 % | 2.6 % / 91.7 % | 2.0 % / 93.8 % | 4.9 % / 85.0 % |
  | Text → island speech, read back / island speech → hub speech, read back | 43.3 % / 32.9 % | 47.9 % / 56.7 % | 53.8 % / 56.3 % | 46.7 % / 45.0 % |
  | Both objects covered: text / hub speech / island speech | 97 / 62 / 57 % | 99 / 97 / 62 % | 98 / 95 / 77 % | 98 / 90 / 67 % |
  | Retrieval of the image, R@1: text / hub speech / island speech | 55.4 / 54.6 / 45.0 % | 57.1 / 54.6 / 45.8 % | 58.8 / 57.1 / 49.2 % | 58.3 / 49.6 / 47.9 % |

  ¹ the earlier reference, without the reader's exposure and augmentation.

  At a fifth of the extra places the extra speech costs nothing that can be measured: no row is worse than
  with the extra text alone by more than two runs of the same settings differ (about five points), and the rows
  that are better by more than that are the island's (two-object sentences 50.0 % against 40.2 %, both objects
  covered 77 % against 62 %, text → island speech 53.8 % against 47.9 %). This is the default of the full real
  presets: 13 of the 64 extra caption groups of a batch are spoken ones.

  *More of it is worse* (no reader exposure or augmentation in these four; `docs/results/synth_extra_hub_*`):

  | | + text | + text + speech (a third of the places) | + speech (every extra place) | … and that speech only listens |
  |---|---|---|---|---|
  | Island speech → hub text, exact content | 73.8 % | 62.5 % | 33.3 % | 57.9 % |
  | Hub-language speech → text, WER / exact content | 2.1 % / 93.3 % | 4.5 % / 92.1 % | 16.7 % / 74.2 % | 5.2 % / 90.4 % |
  | Text read and written again | 100 % | 99.6 % | 85.0 % | 98.8 % |
  | Island speech → hub-language speech, read back | 55.0 % | 43.8 % | 27.5 % | 42.1 % |
  | Both objects covered: text / hub speech / island speech | 100 / 90 / 63 % | 99 / 99 / 80 % | 77 / 78 / 14 % | 100 / 96 / 60 % |
  | Retrieval of the image, R@1: hub speech / island speech | 53.3 / 42.1 % | 56.7 / 45.4 % | 45.4 / 26.7 % | 57.9 / 34.6 % |

  Extra speech that takes every extra place is destructive: the island language loses half of its translation,
  text gets worse although the same captions are there as text, and even recognition of the hub language's own
  speech gets worse with four times as much of it. At a third of the places the island translated worse than
  with the extra text alone (62.5 % against 73.8 %). That run differs from the one at a fifth in the share *and*
  in the reader's exposure and augmentation, and the two were not separated; with the extra text alone the
  reader's two changes make no difference (73.8 % with and without them), so the share is the likelier cause.
- *Extra speech that only listens* (`listen_only`, off; an ablation variant). The speech views of extra spoken
  captions go through the codec like all speech and are then only *read*: the reader must turn them into the hub
  code that the text of the same caption and the image define. They are left out of every term that shapes the
  hub (grounding, sufficiency, credit, minimality, text–speech agreement) and pay for no tone gate; a test
  checks this on gradients (whatever the listening speech says, the gradient of everything in the hub but the
  reader is the same). The last column of each table is what it does. Where speech floods the batch it undoes
  most of the damage: text and hub-language speech come out better than on the benchmark alone and the island
  is back at the reference (57.9 % against 60.8 %), so the damage was done through the terms that shape the
  hub. But the island gets none of what the same captions give as text (73.8 %) and finds its image less often
  than on the benchmark alone (34.6 % against 45.0 %). At a fifth of the places, where grounded speech does no
  damage, listening is the worse of the two (68.3 % against 77.1 %, hub-language speech 85.0 % against 93.8 %)
  and below the extra text alone by about what two runs differ by. A likely reason, not tested: listening
  speech still trains the speech codec, by reconstruction only, so from it the encoder learns to rebuild
  hub-language speech and not to be read. It stays as an option for shares well above a fifth.
- *Giving every language the same say in the losses does not help either* (`lang_balance`, off; an ablation
  variant). Alone: island speech → text 55.8 % against 60.8 %, both objects of an island caption covered in 33 %
  of cases against 57 %. With both kinds of extra data at a third: 67.5 % against 62.5 % for the island, inside
  the noise, at the price of hub-language speech (83.8 % exact content against 92.1 %) and of island coverage
  (64 % against 80 %) (`docs/results/synth_language_balance_eval_test.json`,
  `docs/results/synth_extra_hub_text_and_speech_language_balance_eval_test.json`). The few island views of a
  batch become a few very heavy, noisy ones.
- *The reader is trained on time-masked encodings of speech* (`read_aug`, on by default). The reader's error on
  held-out speech rose all through the transport phase of the Flickr8k run while its training error fell; the
  latents it reads are cached and never change. Now two more encodings of every spoken training utterance are
  cached, each with two spans of up to 15 % of the utterance hidden, and the reader reads a random one of the
  three and must give the same hub code. Nothing upstream of the cache moves, so the representation phase, and
  with it the coverage that dropout destroyed, cannot be affected. SynthWorld, transport phase only, started
  from the same representation, test split (`docs/results/synth_reader_aug_{off,on}_eval_test.json`):

  | | clean latents only | + time-masked encodings |
  |---|---|---|
  | Island speech → hub text, exact content (one-object sentences) | 61.7 % (83.8) | 66.3 % (90.5) |
  | Island speech → text of the third language | 64.2 % | 68.3 % |
  | Text → island speech, read back / island speech → hub speech, read back | 48.3 % / 32.1 % | 52.5 % / 39.2 % |
  | Hub-language speech → text, WER / exact content | 7.5 % / 81.7 % | 8.2 % / 80.4 % |
  | Reader error on held-out speech (training) | 0.267 (0.178) | 0.264 (0.249) |

  One run each. The two share their representation phase, so they differ by the transport phase alone. A
  second seed was not run, and four to five points on the island rows are at the edge of what two full runs of
  the same settings differ by.
  The reader's error on held-out speech does not fall; what changes is that training and held-out error meet,
  and the island, which has the fewest image pairs to read from, gains. Hub-language speech is unchanged.
  With the extra text in place neither this nor the reader's exposure to written speech moves the island's
  translation (73.8 % with both, 73.8 % with neither). Both stay on: they cost nothing there, and each answers
  something seen on Flickr8k that SynthWorld does not reproduce.

**Real data: the third full run. Text is better, the island is where it was, and the model does not speak**
(Flickr8k-Tri plus Flickr30k text and SpokenCOCO speech, Kaggle 2 × T4, run `flickr8k_yo_t4_v5`: 8,000 + 10,000
steps, 2.4 h + 1.7 h; test scores at the selected state, step 17,000; files in `docs/results/real_run3/`).

| Test split | second run | third run |
|---|---|---|
| Yorùbá speech → English text, BLEU / chrF++ | 3.6 / 18.7 | 4.2 / 19.6 |
| ... content F1 (chance level) | – | 11.5 (6.7) |
| English speech → text, word error rate | 85.1 % | 81.5 % |
| Captioning, multi-reference BLEU | 8.6 | 9.3 |
| English → Hindi / Hindi → English, BLEU | 5.1 / 6.1 | 5.6 / 7.4 |
| Text → written image, R@1 | 11.8 % | 23.2 % |
| Yorùbá speech → image, R@1 | 1.0 % | 1.2 % |
| Text codec alone, character error rate (validation) | 28.5 % | 2.0 % |

- *The extra text did what text can do.* The text codec is now exact, a caption finds its image twice as often,
  and every output is a fluent caption. What passes through the hub is still the kind of scene and not the
  sentence: text read and written again has 73 % word error rate (validation) while the codec alone has 2 % character
  error rate, so the loss is in the hub, as designed and as before.
- *Back-translation does nothing that can be measured on real data.* With the unpaired Yorùbá speech left out of
  the transport phase, Yorùbá speech → English text has content F1 11.2 (chance 6.2) against 11.5 (chance 6.7),
  BLEU 3.8 against 4.2, and every other direction is equal or slightly better without it (text read and written
  again 26.6 against 25.3, captioning 10.9 against 9.3). One seed; the central claim about the island has no
  support on Flickr8k.
- *The decoder's exposure to written latents is what makes the text readable.* The variant with the decoder
  frozen after the representation phase was cut by the session limit at step 15,000 and has no test scores,
  but its validation curve is flat from the start at about 10 content F1 for text read and written again,
  8 for English speech and 5 to 6 for Yorùbá speech, against 26, 19 and 11 to 12 with exposure. This is the
  first component with clear support on real data.
- *Reading in one pass is better than inverting the flow:* Yorùbá speech → text 11.9 against 10.3 content F1,
  English speech 19.8 against 14.4 (inference sample).
- *The model does not speak.* A recogniser that was not trained with the model hears the vocoder alone almost
  word for word (4 % word error rate, content F1 95), and nothing at all in anything that has been through the
  speech codec: "Mmm", "Shh shh shh" (content F1 0.1 for the codec's own reconstruction of a real utterance,
  0.2 for speech written from text, 0.1 for Yorùbá speech translated into English speech). The codec's
  reconstruction of the segmental stream is constant in time (`codec_reconstruction.png`): the 16 latent
  tokens hold the utterance's mean and nothing about its order. Every score that involved written speech in
  the first three runs measured that hum.
- *The same fact explains the listening side.* All eight hub slots of a spoken caption look at the same
  patches, where those of the written caption differ; held-out speech codec error has not moved since step
  4,000 (0.82 of the feature variance, training 0.63); the reader's error on held-out speech rises through
  the transport phase (0.223 to 0.235) while the training error falls, augmentation or not. A speech latent
  that is one vector can carry the kind of scene and cannot carry more.
- *The Yorùbá recordings were half noise, and their pitch was never seen.* Half of the energy of the YFACC
  recordings is rumble below 80 Hz (English: 0.2 %). It hid the pauses from the silence trimmer, so a Yorùbá
  utterance was 7.3 s long where 4.0 s of it is speech, and it hid the periodicity from the pitch tracker,
  which marked 9 % of the Yorùbá frames as voiced (English 38 %; three Yorùbá utterances in four had almost no
  pitch at all). Measured on the recordings themselves (30 utterances per language), not inferred from the
  run. So the tonality results of all three runs are void: the gate of the tonal language closed further than
  the English one (0.38 against 0.56), flattening the pitch changed nothing (10.6 against 10.8 content F1) and
  pitch had 0.2 % of the attribution, because there was no pitch to use.
- *Unexplained:* written with 16 integration steps and read by the inverted flow the score drops (8.6 against
  11.5 with 2 to 8 steps). The default path (reader, 4 steps) is not affected; not investigated.
- *Still no docking* (2 to 10 % of the island utterances matched, chance 2 %), *still one seed*, and the island
  still sits apart in the hub (centroid gap 0.21 where the other views are 0.07 to 0.10 from each other).

**What was changed after the third run.**

- *Recordings are high-passed at 70 Hz before anything is computed from them* (`highpass_hz`, real presets).
  On the same 30 + 30 utterances: Yorùbá mean length 7.3 s → 4.0 s, voiced frames 9 % → 35 %, utterances with
  almost no pitch 63 % → 10 %; English 3.4 s → 3.2 s and 38 % → 39 %. The features have a new cache name, so
  they are extracted again.
  In the pipeline's own feature cache of a 60-image subset: Yorùbá 7.7 s → 4.5 s per utterance, voiced frames
  6 % → 38 %, utterances with almost no pitch 85 % → 0 %; English 3.5 s and 43 % before and after.
- *The speech codec gets a speaking path: tokens laid out in time, and a decoder of their own*
  (`audio_anchor = extra`, `audio_private_slots`; real presets). Speech is reconstructed by regression, and a
  regression has a resting place: every latent token the utterance mean, every frame decoded to it. A set of
  free queries stays there on real speech. A token that is laid out in time starts from the mean of its own
  span of the utterance, and every frame of the decoder starts from the ordered tokens around its own
  position; attention only adds detail. Codec alone, trained on the real features of a 1,200-image subset
  (5,778 utterances, mHuBERT, the T4 preset's codec; a CPU, so few steps), scored on 192 held-out utterances as
  the share of the frame variance that the reconstruction explains *beyond the utterance mean*, English /
  Yorùbá:

  | speech latent | step 300 | 600 | 900 | 1,500 |
  |---|---|---|---|---|
  | 16 free tokens (as in the three runs) | 0.00 / 0.08 | 0.03 / 0.12 | – | – |
  | 16 tokens, all laid out in time | 0.23 / 0.24 | 0.38 / 0.32 | 0.44 / 0.34 | 0.48 / 0.36 |
  | 32 tokens, all laid out in time | 0.37 / 0.30 | 0.49 / 0.39 | 0.51 / 0.41 | – |
  | **16 free content tokens + a speaking path with 16 ordered ones (the presets)** | 0.21 / 0.22 | 0.29 / 0.28 | – | – |

  The free latent reproduces what the run showed, on a CPU, in 300 steps: exactly the utterance mean. A
  recogniser that hears the vocoder at 5 % word error rate and the renderer of these short runs at 20 to 24 %
  hears content in the reconstructions of 24 held-out English utterances: content F1 30 with 16 ordered tokens
  after 1,500 steps ("a little boy singing in the middle of hers and on the sidewalk" for "The little boy stands
  in the middle of birds who have gathered on the sidewalk"), and 17 with 32 after 900. The
  speaking path of the presets, slower to start, had reached 29 % explained when its 600 steps ended, and there
  the recogniser finds nothing yet (content F1 0.0): the reconstructions that were understood explained 48 %
  and more. Whether 8,000 steps on ten times the data take it there is what the next run's ladder will say.
  The third run's codec, after 8,000 steps on 55,000 utterances, scored 0.1 and explained nothing.
- *It is added beside the codec, not put into it, and that was not the first idea.* Three layouts were run end
  to end on SynthWorld (same recipe and data as the reference with extra text and speech, whose island speech
  → text is 77.1 % exact content and whose validation grounding loss is 0.93 at step 1,000):
  - *every speech token laid out in time, read by the hub:* the codec reconstructs better (held-out error 0.10
    against 0.24) and the island is lost. Island speech → text 4.6 %, hub-language speech 73.8 % against 93.8 %,
    text read and written again 82.5 % against 100 %, grounding 1.70
    (`docs/results/synth_speech_latent_all_tokens_time_ordered_eval_test.json`). Free queries had been putting
    each thing an utterance is about into the same token wherever it is said, and the hub was reading it there.
  - *ordered tokens added to the free ones in the same codec, hidden from the hub:* grounding 1.86 at step 1,000
    (stopped there). The ordered tokens then do the reconstruction and nothing puts the content into the free
    ones.
  - *a speaking path beside the codec* (what the presets use): the free content tokens, their decoder and their
    losses are exactly what they were, and they are all the hub reads, grounds and explains. Time-ordered
    tokens come from a small encoder of their own and are decoded, together with the content tokens
    (stop-gradient), by a second decoder; the flow writes both kinds of token. A test checks that the speaking
    objective sends no gradient into the codec and that changing the ordered tokens changes nothing the hub
    computes. On SynthWorld the listening side is where it was (island speech → text
    76.7 % against 77.1 %, hub-language speech 89.6 % against 93.8 %, retrieval 60 / 55 / 45 % against
    59 / 57 / 49 %; grounding 0.97 against 0.93 at step 1,000; `docs/results/synth_speaking_path_eval_test.json`).
    **The speaking side is not solved by it.** Speech that the flow *writes* and the speaking decoder renders is
    read back far worse than speech rendered by the content decoder (text → hub-language speech → text 12.5 %
    against 65.8 %): the realisation tokens are easy to reconstruct from and hard to write, and the speaking
    decoder has never seen written ones. On SynthWorld, where the content decoder is enough, the speaking path
    therefore stays off. On real speech the content decoder renders a hum, so there is nothing to lose; the
    next step is the one that made written text readable, exposure of the decoder to written latents, which
    for speech needs the frames in the transport phase.
- *Checked end to end:* the test suite's shared run uses the speaking path (training, resuming, the latent
  cache, back-translation, evaluation, every figure), and a 60-image real-data run through the cleaned
  recordings completes.
- *The transport phase of the two-T4 preset is 8,000 steps* (10,000 before): the validation scores of the third
  run were flat after 5,000, and both ablation variants now fit in the session after the main run.
- *The recogniser's ladder has a rung for the renderer alone* (the real streams rendered and vocoded), so the
  next run says whether what is lost is lost in the latent or in the renderer.

**Fifth round: what the published systems have that this one did not** (literature scan of 26 November 2026,
`docs/LITERATURE.md`; nothing of this has run on a GPU yet).

The published Yorùbá speech → English text result that uses only images (Oneata & Kamper 2024) is BLEU 6.65
with one reference and 15.82 with five; the third run has 4.2 and 8.4. Their system trains on every image
pair of Yorùbá, on five sampled captions per image from a pretrained captioner, and directly on speech →
caption. This one trained on a fifth of the pairs, on the human captions only, and tied the island to the hub
language by nothing but a shared hub code. Each of those differences is removed, inside the hub and without
giving up the zero-parallel protocol:

- *Every island utterance keeps its image* (`bridge_frac = 1` in the real presets): about 5,000 pairs instead
  of 985. The 20 % setting stays as the scarce-pairs study and is the only one with unpaired speech, so
  back-translation is inactive in the headline configuration.
- *Captions from a pretrained captioner* (`captioner = microsoft/git-large-coco`, five sampled captions per
  training image; `data/captioner.py`). Checked on real images with the base model: 150 captions for the 50
  training images of a 60-image subset, none dropped ("A fluffy dog running outside on a grassy field ."), and
  a model trained end to end with them.
- *Image-sibling transport* (`w_sib`): from the code read out of an island or written-only view, the text
  writer writes another caption of the same image, the decoder is exposed to what it writes, and the reader
  learns from it. Never the caption itself. **On SynthWorld it is harmful, and it is off.** Transport phase alone, same
  representation, island speech → text / text read and written again / hub-language speech → text, exact
  content (`docs/results/synth_image_sibling_transport_*`):

  | | island speech → text | text → text | hub speech → text |
  |---|---|---|---|
  | without | **77.1 %** | 100 % | 93.8 % |
  | weight 1, the reader learns too | 52.1 % | 86.3 % | 74.2 % |
  | weight 1, writer and decoder only | 61.7 % | 74.6 % | 67.1 % |
  | weight 0.3, writer and decoder only | 74.2 % | 98.3 % | 88.8 % |

  On SynthWorld the captions of one scene name different objects, so "write another caption of this picture"
  teaches the writer to write about the scene and not about what was said, also for the hub language, whose
  codes look the same to it. The captions of a real photo paraphrase each other, the model's own fidelity to a
  sentence is no better on Flickr8k than that of a sibling caption (text read and written again scores 11.8
  BLEU against its own caption; one annotator scores 8.3 against another), and this supervision is what the
  published number was obtained with. So it may well help there, and that is what the first transport-phase
  ablation of the notebook measures (weight 0.3, writer and decoder only: the form that costs least here).
- *Listening tokens* (`listen_tokens`): the hub reads the speech encoder's states span by span, beside the
  content tokens. SynthWorld and real speech disagree, and the presets follow each. On
  SynthWorld, whose utterances are two to six syllables, they slow grounding down (validation loss 1.83 against
  0.87 at step 1,250; run stopped), like every other attempt to let the hub read speech tokens that are laid
  out in time; they stay off there. On real speech, two representation phases on the 1,200-image subset (600
  steps, batch 24, no speaking path; `docs/results/real_subset_retrieval_*`), held-out retrieval among 91 or 92
  images, R@1 / R@5 / R@10:

  | | content tokens only | + 24 listening tokens |
  |---|---|---|
  | English speech → image, validation (450 queries) | 1.6 / 10.2 / 18.7 % | 2.9 / 13.6 / **29.6 %** |
  | English speech → image, test (460 queries) | 2.4 / 12.0 / 23.5 % | 2.8 / 14.1 / **30.7 %** |
  | Yorùbá speech → image, validation (77 queries) | 1.3 / 15.6 / 26.0 % | 0.0 / 15.6 / 24.7 % |
  | Yorùbá speech → image, test (78 queries) | 1.3 / 7.7 / 11.5 % | 1.3 / 10.3 / 21.8 % |
  | English text → image, test (460 queries) | 2.0 / 8.7 / 15.4 % | 3.0 / 11.1 / 19.8 % |

  Chance is 1.1 / 5.5 / 11 %. Both models have barely begun to train, so R@1 says nothing yet; in the top ten,
  English speech gains 7 to 11 points, where the sampling error of 450 queries is about two, and Yorùbá is
  within its own error of five. One seed, a seventh of the images, a thirteenth of the steps. A real utterance
  is long and its free summary is an average (third run); read span by span it is found more often. The real
  presets use 32 listening tokens.
- *The speaking decoder is trained on written latents* (`w_dec_aud`), written from the utterance's own private
  code so that the latent still belongs to the recording's frames. **On SynthWorld it changes nothing**: text →
  hub-language speech → text through the speaking decoder is 13.3 % exact content with it and 13.3 % without
  (12.5 % before either existed), against 65.8 % through the content decoder
  (`docs/results/synth_speaking_path_own_encoder_*`). What limits written speech is what the flow writes into
  the time-ordered tokens, not what the decoder has seen. It stays on in the real presets, where the content
  decoder's output is constant in time and is no alternative, and the notebook's `decoder` ablation measures
  it there.
- *How far two runs of one configuration differ on SynthWorld: about 20 points.* The two runs just mentioned
  scored 53.8 % and 57.1 % on island speech → text where the reference has 77.1 %. The speaking path is not
  the cause, and three checks say so. Its loss changes no gradient outside its own parameters (same batch and
  random state, with and without it: the largest difference is exactly 0 over 549 tensors). With the same
  listening weights, a model with it and one without give the same value of every loss term of the
  representation phase and the same gradient on every shared parameter (now a test). And clipping its
  gradient apart from the rest (`clip_apart`, now the default) leaves the validation curve where it was (1.42
  against 1.43 at step 500). A run with a speaking path is the reference model with other random numbers; so
  the reference configuration was run again with another seed (which also draws another world), and the
  speaking configuration with it:

  | same settings, other random numbers | validation grounding, end of the representation phase | island speech → text | two-object sentences | hub speech → text | text → text |
  |---|---|---|---|---|---|
  | reference, seed 0 | 0.87 | 77.1 % | 50.0 % | 93.8 % | 100 % |
  | reference, seed 1 | 0.85 | 59.6 % | 21.7 % | 84.2 % | 99.2 % |
  | first speaking path, seed 0 | 0.89 | 76.7 % | 42.4 % | 89.6 % | 97.9 % |
  | speaking path, seed 0 | 1.13 | 53.8 % | 4.3 % | 75.0 % | 85.0 % |

  (The speaking configuration with seed 1 was stopped during its representation phase: validation grounding
  1.29 / 1.51 / 1.05 at steps 500 / 750 / 1,000, where the reference with that seed has 1.11 / 1.26 / 0.97.)

  One configuration covers 54 % to 77 % on this benchmark's headline number (two-object island sentences 4 % to
  50 %, both objects covered for island speech 24 % to 77 %), and a good validation loss of the representation
  phase does not predict where a run lands (0.85 gave 59.6 %). What is stable across these runs is the text
  side: both objects are covered for 97 to 99 % of text captions in all of them. What follows for every
  comparison in this document between single SynthWorld runs *with different representation phases*: on
  island-speech scores a gap is evidence only when it is far larger than 20 points (time-ordered tokens read
  by the hub: 1 to 5 %; extra speech alone: 33 %). Gaps inside it, read above as gains or as costs, are
  unproven either way: more hub-language data (61.7 % → 73.8 % → 77.1 %), listening-only extras (68.3 %), the
  share of spoken extras (62.5 to 67.5 %), and the credit loss's effect on island two-object sentences (4.3 %
  against 26.1 %; its effect on text coverage, 36 % without it and 97 to 100 % with it in every run since, is
  not in question). Comparisons that share one representation phase are a different matter: the two
  decoder-exposure runs above differ by 3 points with nothing but the random numbers of the transport phase
  between them, and that is the scale against which image-sibling transport (15 to 25 points at weight 1, 3 at
  weight 0.3), read-back selection (under 1) and the transport-phase rows of section 2 are to be read.
- *The image space is SigLIP 2 on an 8 × 8 grid* (`vision_encoder = google/siglip2-base-patch16-256`,
  `vision_grid = 8`; SigLIP at 7 × 7 before). SigLIP 2 adds dense objectives to SigLIP's training and loads
  through the same classes; the hub's barycentres are then barycentres of 64 patches instead of 49. The same
  two short representation phases as for the listening tokens (1,200-image subset, 600 steps, batch 24, no
  speaking path, no listening tokens), held-out retrieval among 91 or 92 images, R@1 / R@5 / R@10
  (`docs/results/real_subset_retrieval_*`):

  | | SigLIP, 7 × 7 | SigLIP 2, 8 × 8 |
  |---|---|---|
  | English speech → image, validation (450 queries) | 1.6 / 10.2 / 18.7 % | 5.1 / 24.9 / **44.7 %** |
  | English speech → image, test (460 queries) | 2.4 / 12.0 / 23.5 % | 4.3 / 19.3 / **39.3 %** |
  | Yorùbá speech → image, validation (77 queries) | 1.3 / 15.6 / 26.0 % | 5.2 / 20.8 / 29.9 % |
  | Yorùbá speech → image, test (78 queries) | 1.3 / 7.7 / 11.5 % | 3.8 / 16.7 / **34.6 %** |
  | English text → image, validation (455 queries) | 2.0 / 9.0 / 14.9 % | 3.1 / 11.9 / 23.5 % |
  | English text → image, test (460 queries) | 2.0 / 8.7 / 15.4 % | 2.2 / 12.0 / 23.9 % |
  | Hindi text → image, test (460 queries) | 1.5 / 6.3 / 13.3 % | 1.5 / 6.1 / 12.0 % |

  English speech finds its image in the top ten about twice as often (sampling error about two points);
  Yorùbá gains 23 points on the test images and 4 on the validation images, with an error of five; Hindi
  text, which has one caption per image and the least to learn from, does not move. Tower and grid were
  changed together, so which of the two it is has not been separated. One seed, a seventh of the images, a
  thirteenth of the steps; on real data two runs of one representation phase have differed visibly before,
  though not by a factor of two.

  **The two gains did not add.** A fourth arm with both (SigLIP 2 and 24 listening tokens) has English speech
  → image R@10 25.1 % on validation and 24.3 % on test, Yorùbá 19.5 % and 17.9 %, English text 22.2 % and
  21.3 %: the text side keeps what SigLIP 2 gave it, the speech side has lost it. Either listening tokens
  hurt on the finer grid, or one of the two speech gains above is the luck of a short single run; the four
  arms are being repeated with another training seed on the same split to tell (below, when done).
- *Read-back selection* (`select_k = 4` in the real presets): several candidates are written and the one that
  reads back closest to its hub code is kept. On a SynthWorld checkpoint, no training involved: island speech →
  text 77.1 % → 77.5 % (four candidates) → 77.9 % (eight), two-object sentences 50.0 → 51.1 → 52.2 %, text →
  hub-language speech → text 65.8 → 69.6 → 68.8 %; one candidate reproduces the stored evaluation exactly.
  Small, never negative on text.
- *BLEU as that literature computes it* (`bleu_r1` … `bleu_r5`): lower-cased, punctuation removed, one to five
  references with the source's own caption among them, so the next run's numbers can be set beside theirs.
- *Ablations run by themselves after training and inference*, cheapest first: variants that only change how
  the trained model is read and written are scored from the main run's checkpoint with no training (group
  `decode`), then the transport-phase variants (`sibling`, `decoder`).
- *Not adopted, and why:* the large frozen towers of the retrieval systems (HuBERT-Large, ViT-L/14) and of the
  baseline (XLS-R 2B, GPT-2). The speech encoder in use, a mid layer of mHuBERT-147, is what the low-resource
  word-level literature now recommends over the large models; the rest would change what this model is. So the
  published retrieval numbers (speech → image R@1 39 to 44 %) are not a target.

**What is not fixed, and why.**

- *Wording is not carried across.* A hub of image barycentres passes what can be pointed at. Between two
  languages that share no sentence nothing else *can* be aligned; within one language a shared code for wording
  could be learned from paired speech and text, but that would make English speech recognition better and
  leave every claim of this work where it is. More data should raise how much of a caption the hub explains
  (47 % on held-out captions so far); it will not make this a transcription system.
- *The listening side of speech is where it was.* What the hub reads from an utterance is still a latent that
  holds its gist (the content tokens are the old codec, left alone on purpose, because every way of putting
  the order of time into them cost the hub its content on SynthWorld). English speech → text will stay near
  80 % word error rate; what the cleaned recordings and the pitch they now carry do for Yorùbá is for the next
  run to say. The Yorùbá side has no new data: its 5,500 recordings are all there is with images.
- *Speech that is written, as opposed to reconstructed.* The speaking path reconstructs real speech with its
  order; whether the flow can write its realisation tokens well enough to be understood is not known on real
  data and is poor on SynthWorld (text → speech → text 13 % through the speaking decoder against 65.8 %
  through the content decoder). Training the speaking decoder on written latents, the fix that worked for
  text, is built and changes nothing there. What is missing is upstream: the flow has to produce, from four
  hub slots and noise, tokens that say *when* each thing is said, and nothing in its objective rewards that
  beyond matching the latents. On real data intelligible speech output should not be expected from the next
  run; the recogniser ladder will say at which rung it is lost.
- *How much extra speech is right on real data.* The share (a fifth) was chosen on SynthWorld, where a third
  already cost the island language and where the extra speech has no new voices to offer. On Flickr8k the
  right share may be another; `extra_speech_share` and the `data` ablation group are there to find out.
- *Tonality gate and counterfactual credit on real data:* measurements that have not been made (the gate's
  three runs were made without the pitch). Back-translation has been measured once and showed nothing; the
  notebook's defaults measure it again, and the decoder variant, after the main run. Credit and the gate need
  a representation phase each.
- *One seed, no baseline on the real split.*

**Not verified.**

- *Seeds.* Everything on real data is one seed. On SynthWorld a second seed of the reference configuration now
  exists and lands 17 points below the first on island speech → text (fifth round); the tables of section 2
  are single runs and have to be read with that in mind. Two runs of the same representation phase have
  differed visibly on real data too.
- *Credit, minimality, the tonality gate on real data.* One ablation variant has finished on Flickr8k (without
  back-translation: no difference); the decoder variant has validation curves only.
- *Baselines on real data.* No cascade, no supervised topline, no published system has been run on this split.
- *Speech output.* A recogniser found no content in it in the third run. The speaking path has been heard
  only as a codec on a CPU; no trained model with it, no listener.
- *Everything changed after the third run* (recordings high-passed, the speaking path, the shorter transport
  phase, and all of the fifth round: every Yorùbá pair, generated captions, listening tokens, SigLIP 2,
  read-back selection, the literature's BLEU) has run on SynthWorld, on small real subsets on a CPU and in the
  tests, not on a GPU and not at full size: the time and memory of the larger model are estimates, and no
  BLEU of the new configuration exists. Whether it reaches the published 6.65 / 15.82 is what the next run
  measures; nothing here shows that it does.
