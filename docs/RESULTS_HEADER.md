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

**Not verified.**

- *Seeds.* Everything is one seed, on both benchmarks. Two runs of the same representation phase have differed
  visibly on real data.
- *Back-translation, decoder exposure, credit, minimality on real data.* No ablation variant has finished on
  Flickr8k; the comparisons above are validation curves of one unfinished variant.
- *Baselines on real data.* No cascade, no supervised topline, no published system has been run on this split.
- *Speech output.* Scored only by the model's own reader; no recogniser, no listener.
- *The changes listed above* have run on SynthWorld and in the tests, not on a GPU.
