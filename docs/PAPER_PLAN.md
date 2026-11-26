# Paper plan

Working title: **GeoBridge-Flow: Speaking Through Pictures — A Sentence-Level Visual Interlingua for Unwritten,
Tonal Languages**

## 1. What to claim (and what not to)

Three contributions carry the paper. Each has one figure and one ablation group that isolate it. The numbers that
currently stand behind each claim are in `RESULTS.md`; a claim is only as strong as that file says.

| | Contribution | Claim, stated so it can be falsified | Evidence produced by this repository |
|---|---|---|---|
| C1 | **Barycentric visual interlingua with mutual sufficiency** | A hub code made of utterance-selected barycentres of frozen patch tokens, trained to be sufficient for the sentence and predictable from it, carries *what was said* (all of it) and serves every direction; a scene-level visual pivot does not. Counterfactual patch credit is what makes the second and later things a sentence says reach the code. Against a text anchor the claim is narrower: the visual hub is not centred on one written language. | `ablate --groups hub` (no credit, no minimality, global pivot, text anchor, free anchor); `01_slot_attention.png`; pointing game; share of two-object captions with both objects covered |
| C2 | **Lexical-tonality gate** | A per-language gate on the speaker-normalised F0 contour (i) stays open for tonal languages and closes for non-tonal ones without tone labels, and (ii) tone-contrastive words survive speech → hub → text only when pitch is available. Supported on SynthWorld; **not yet shown on real data**. | `ablate --groups tone`; `05_tone.png`; tone-word accuracy with pitch removed; twin-confusion rate |
| C3 | **Asymmetric read / write with hub-space back-translation** | For speech, reading with a one-pass, image-free reader is far better than inverting the flow (for text the inversion is better, so the read mode should be per modality); unpaired speech of an unwritten language improves its translation through back-translation with the hub as pivot and a relay to the text writers, and the gain should grow as bridge pairs become scarce (the budget sweep has not been run). | `ablate --groups read island bridge`; `16_read_write_modes.png`; `13_hub_vs_private.png` (speaker probe on reader hub vs ODE hub) |

Measured and **not** supported at this scale, so not to be claimed: the optimal-transport rectification of pseudo
codes, read-hub exposure and hub jitter (no effect within noise), and "the reader keeps the speaker out of the
hub" (it leaks less than the ODE read-out, not nothing).

**Where the claims stand after three full Flickr8k runs** (one seed; `RESULTS.md`, section 0):

- C1, the hub: works at the level of "what is in the picture" (right kind of subject in 91 % of text → text
  outputs, 79 % for Hindi → English without parallel data; chance 22 %) and not at the level of wording (word
  error rate 78 %). The paper has to be written about meaning-level conversion and scored with content measures
  next to their chance levels. Counterfactual credit, the specific novelty, is mixed on validation data: the hub
  explains more with it and is harder to read; there is no test number yet.
- C2, the tonality gate: **not tested on real data so far.** In all three runs the pitch tracker found pitch in
  9 % of the Yorùbá frames, because the recordings carry rumble that hides it; the gate closed for a language
  whose pitch it never saw. With the recordings high-passed the next run is the first test. Until then it is
  a SynthWorld-only result.
- C3, attaching the island: Yorùbá speech → English text has content F1 11.5 against 6.7 by chance (BLEU 4.2)
  from 985 image pairs and no Yorùbá text. **Back-translation contributes nothing measurable** (11.2 without
  it), and the island sits apart in the hub. The claim as written has no support; what the island gets, it gets
  from the bridge pairs and the shared speech backbone. To be measured again once the island's audio is clean
  and the speech latent carries more than the utterance's mean.
- The system-level claim "including speech into that language" **fails as measured**: a recogniser finds no
  content in any speech the model writes, nor in the speech codec's reconstruction of a real utterance. The
  cause is located (a latent of free queries that rests at the utterance mean) and addressed on real features
  (content tokens for the hub plus a speaking path with time-ordered realisation tokens, `audio_anchor = extra`):
  reconstruction with the order of time, yes; speech written by the flow, not yet (on SynthWorld it is read
  back worse through the speaking decoder, 13 % against 66 %, and training that decoder on written latents
  does not change it: the limit is what the flow writes into the ordered tokens). Whether either becomes
  intelligible at full scale is for the next run's recogniser ladder; it should not be expected. If it does, "what is
  said in free slots, how it is said in ordered ones, two objectives" is a component worth its own ablation
  row: the obvious alternatives fail, each on one side (free only: no speech out; ordered only, or both in one
  codec: no content in).
- Supported on real data, one seed each: the decoder's exposure to written latents (text content F1 about 26
  with it, 10 without, validation), and reading in one pass over inverting the flow (English speech 19.8
  against 14.4).
- Missing for any submission: seeds and the baselines of section 3 on real data.
- The benchmark as trained from now on is **Flickr8k-Tri plus hub-language data**: about 23,700 Flickr30k photos
  with English captions (text) and the 5,000 SpokenCOCO test photos with 25,031 spoken captions (text and
  speech) join the training split (no other language, no Flickr8k photo; validation and test unchanged). This has to be stated wherever a number is compared with work trained on Flickr8k alone
  (Oneata & Kamper 2024 among them), and the `data` ablation (`extra_images = 0`) is the like-for-like row.

Negative result worth a paragraph: entropic Gromov–Wasserstein docking, the roadmap's central proposal, does not
dock an unpaired speech island (pure GW drives it to chance; semi-relaxed collapses; balanced fused GW ties with
no docking). It stays in the `island` ablation.

A second one, SynthWorld only so far: more speech in the high-resource language is not free for the low-resource
one. Extra hub-language text raises island speech → text from 61 % to 74 %; the same captions with their speech,
filling the batch, lower it to 33 %; at a fifth of the extra places the speech costs nothing (77 %). Keeping the
extra speech out of the terms that shape the hub (`listen_only`) undoes most of the damage of the flooded batch,
which says where the damage is done, and is worse than plain grounding at a fifth. One run each; the real-data
version of this is the `data` ablation group.

System-level claim: an any-to-any (text ↔ speech ↔ image) model for an unwritten language that uses no text of
that language and no parallel sentences, including speech *into* that language. "First" needs a final literature
check on the submission date.

Supporting components, **cited, not claimed**: hub ⊕ private writers (MUNI, MUNITE, CrossFlow), read-hub exposure
(AlignFlow and DDIB for why a plain ODE cycle is vacuous), back-translation itself (Sennrich et al.; BT4ST;
Translatotron 3), balanced Sinkhorn assignment (SwAV), sequential slots with inhibition (Houba 2026 for evidence
depletion), grapheme–byte compressor (LD4LG, MYTE, MAGNET), per-language LoRA (LoRA-Whisper, Zipper-LoRA). See
`docs/LITERATURE_AUDIT.md`, including the addendum on the components added in the redesign.

Do not claim: "first to couple GW with flows" (GENOT), "cycle-consistency loss" as a contribution, back-translation
or optimal transport as such, script-fair tokenisation as a contribution, or that contrastive alignment erases
tone (the documented loss is at quantisation).

## 1b. Where the literature stands (scan of 26 November 2026) and what has to be beaten

The field around this exact task is small and has barely moved since 2024. Sources are linked in
`docs/LITERATURE.md`; numbers are as reported there, protocols differ and are given.

| Task | Best published | Protocol | Ours so far (third run) |
|---|---|---|---|
| Yorùbá speech → English text, images only (YFACC test, 500 utterances) | BLEU 6.65 (1 reference, beam captions) to 15.82 (5 references, diverse captions); Oneata & Kamper 2024 | sacrebleu, lower-cased, punctuation removed, own caption always a reference | 4.2 (1 reference, another tokenisation), 8.4 (5 references, validation) |
| ... with real translations (their topline) | 15.23 / 22.01 | same | – |
| English speech → English text, images only (FACC) | 6.56 / 17.46 | same | 7.5 / 15.4 (validation; trained with transcripts, so not the same task) |
| English speech → image, Flickr8k 1k test, frozen encoders | R@1 43.8 (Zhou & Li 2024), 43.1 (SpeechCLIP+), 39.1 (SpeechCLIP large), 26.7 (SpeechCLIP base) | 5 spoken captions per image | about 5 (second run) |
| Yorùbá speech ↔ image on YFACC | none published | – | 1.2 |
| Image → English speech captioning, Flickr8k | BLEU-4 21.1 (TMT), 20.6 (Kim et al. 2024) | recogniser, then caption metrics | not intelligible |
| Speech into Yorùbá without Yorùbá text; any-to-any on Flickr8k/YFACC | none published | – | not intelligible |

What the systems that hold these numbers have that this model did not, and what was done about each (`RESULTS.md`,
fifth round):

- *All image pairs of the low-resource language, and a model warm-started on the hub language.* The real presets
  now use every pair; the hub language and the island share every module from the first step.
- *Captions from a pretrained captioner as targets, several sampled ones per image* (GIT; fewer than three hurt).
  Now generated for the training images and used as text views and sibling targets.
- *Direct speech → caption supervision through the image.* Built inside the hub as image-sibling transport. On
  SynthWorld, whose captions of one scene name different objects, it is harmful and it is off; whether it helps
  on real captions, which paraphrase each other, is the first trained ablation after the next run.
- *Speech read at frame rate with attention* (every retrieval system; mean pooling costs 15 to 20 points of top-1
  in Krahn & Fosler-Lussier 2026). Built as listening tokens and left off: the gain of a first short run on
  real speech was not there with a second seed, and with the SigLIP 2 image space they cost 13 points of R@10.
- *Patch features trained to be dense.* The image space is now SigLIP 2 on an 8 × 8 grid: on the same real
  subset and short schedule, two seeds, held-out English speech finds its image in the top ten in 38 % of
  cases against 22 % with SigLIP at 7 × 7, Yorùbá speech in 31 % against 20 %.
- *A mid layer of a base-size multilingual speech model* is what the low-resource word-level work recommends
  (mHuBERT-147 layer 8 beats HuBERT-Large and WavLM-Large in Pirlogeanu et al. 2026). That is the encoder
  already in use; no change.
- *Large frozen towers* (HuBERT-Large with CLIP or BLIP-2 ViT-L/14, XLS-R 2B with GPT-2). Not adopted: the model
  stays at tens of millions of trained parameters on two T4s, and that is part of what it is for. It also
  means the retrieval numbers above are out of reach by design and are not a target.

Nearest neighbours of the method, to cite and to be distinguished from: **FlowBind** (ICLR 2026: a learned shared
latent with one invertible flow per modality, partially paired data; no speech, no languages, no grounding of
the hub), **CrossFlow** / **FlowTok** (flow matching directly between two modality latents), **OmniFlow**. What
none of them has: a hub that is made of the patches of an image (barycentric, inspectable), languages as
modalities with zero parallel data, and an asymmetric read and write. No 2025–2026 paper was found that uses a
flow or a bridge as a hub across languages or for speech.

## 2. Experiments

### 2.1 Benchmarks

| Benchmark | Purpose | Languages | Status in this repository |
|---|---|---|---|
| **Flickr8k-Tri** (`flickr8k_yo`) | Headline. Same data and task as Oneata & Kamper (2024): Yorùbá speech → English text with no parallel data. | en (text + speech, 183 speakers), yo (speech, 1 speaker; orthography held out), hi (text, machine-translated) | Automatic download, splits, caching, training, evaluation all implemented and run end-to-end on a subset. |
| **SynthWorld** (`synth`) | Controlled: tonal minimal pairs, exact content accuracy, grounding ground truth, known translations. | sa (hub), sb (tonal island), sc (Devanagari text) | Implemented; full ablation runs on a CPU. |
| WAXAL / Vaani image-prompted speech | Breadth (more tonal languages). | 19–100 languages | **Not implemented**: the public WAXAL release (`google/WaxalNLP`) ships no images, Vaani is gated. Add a builder with the same three methods as `Flickr8kTri` once images are available. |
| FLEURS / CoVoST 2 | Non-visual speech translation. | — | Out of scope for a visual hub (no images); see limits in `METHOD.md`. |

### 2.2 Main table (Flickr8k-Tri, test split, 3 seeds)

Rows: GeoBridge-Flow at bridge budgets 1 %, 5 %, 20 %, 100 %; anchor baselines (`global`, `free`, `text`); the
written upper bound (`use_island_text=True`); Oneata & Kamper (2024) as reported (BLEU-4 14.2 with four references;
their setup trains on generated captions of the paired image and uses all pairs, so compare at 100 % and report
the multi-reference BLEU column).

Columns: yo-speech → en-text BLEU / chrF++ (parallel reference) and BLEU (all captions of the image);
yo-speech → image R@1/5/10; en ASR WER; image → en-text BLEU; en-text → yo-speech round-trip BLEU; en ↔ hi BLEU;
trainable parameters; per-language parameters.

### 2.3 Ablations (`python run.py ablate`)

`hub`, `read`, `island`, `bridge`, `tone`, `text_unit`, `private`, `lora`, `protocol` — see
`geobridge/ablation.py:GROUPS`. Run every group on SynthWorld with 3 seeds and `hub`, `read`, `island`, `bridge`,
`tone` on Flickr8k-Tri.

### 2.4 Analyses (`python run.py infer`)

Slot attention across languages; hub manifold; island speech against the hub codes of its translations; gate
values and pitch-scaling response; frame- and grapheme-level attribution; quality against ODE steps; fertility per
script; speaker probes on reader hub vs ODE read-out; read / write modes; generated image previews.

## 3. Controls reviewers will ask for

1. **Functional load of tone.** The gate could be reacting to pitch range, not lexicality. Controls: SynthWorld
   gives `sa` and `sb` the same pitch range (only `sb` is lexical); on real data add a second non-tonal and a
   second tonal language, and report the gate against a typological tone-inventory measure (cf. Liang & Levow).
2. **Text supervision inside the anchor.** SigLIP was trained with text. Report the main table with
   `vision_encoder=facebook/dinov2-base` as the strict zero-text setting.
3. **Implicit parallelism.** `ablate --groups protocol` quantifies how much the numbers inflate when captions are
   visible in every language.
4. **Single-speaker island.** YFACC has one speaker; the speech spoke may learn speaker, not language. Report
   hub-space retrieval between yo and en speech and, if possible, add multi-speaker island data.
5. **Round-trip speech metric.** It uses the model's own recogniser. Add an external ASR for English output and
   an MOS or a learned quality score for a camera-ready.
6. **Seeds and significance.** Three seeds minimum; paired bootstrap on BLEU for the main comparisons. This is
   not a formality here: on SynthWorld two seeds of the reference configuration differ by 17 points on island
   speech → text (77.1 % and 59.6 %), so no single-run ablation gap below about 20 points on an island-speech
   score can go into a table without its seeds.
7. **Compute fairness.** Baseline anchors share codecs, flows, data and steps; only the hub differs.

## 4. Risks

| Risk | How it would show | Mitigation |
|---|---|---|
| Barycentric codes carry too little on real photos (PCA of patch tokens to `d` dims loses semantics) | `anchor_explained_var` low; text → image R@1 near chance | larger `latent_dim`; DINOv2 with registers; more slots |
| Back-translation does not help once 5 % bridges exist | `island` / `bridge` tables flat | report honestly; the claim then narrows to the ≤ 1 % regime |
| Counterfactual credit depends on the explainer being right | slots on unmentioned objects rise; reader error rises | the verify step; lower `w_seek`; report the share of slots on unmentioned objects next to coverage |
| Gate does not separate on real data (SSL features already encode tone) | `tone/gate_yo ≈ tone/gate_en`; no drop with pitch removed | this is itself a finding (consistent with Osakuade & King); use discrete units for the segmental stream, where tone is known to be lost |
| The YFACC mirror stores every clip as exactly 8.0 s | long utterances may be truncated | verify against the original release (Olaleye et al., 2022) before final numbers |
| Hindi captions are machine-translated | en ↔ hi scores measure agreement with an MT system | describe as silver; use only for the script-robustness analysis |

## 5. Venues

The honest positioning is a low-resource speech/multimodal paper with a controlled-benchmark component:
ACL / EMNLP (Speech and Multimodality; Multilinguality), Interspeech, or TACL. Workshops for early feedback:
AfricaNLP, MMLow, IWSLT (low-resource track), SIGTYP for the tonality-gate analysis. Acceptance cannot be
guaranteed by design; what can be guaranteed is that every claim above is tied to a table this code produces.
