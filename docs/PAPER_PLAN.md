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

**Where the claims stand after two full Flickr8k runs** (one seed; `RESULTS.md`, section 0):

- C1, the hub: works at the level of "what is in the picture" (right kind of subject in 91 % of text → text
  outputs, 79 % for Hindi → English without parallel data; chance 22 %) and not at the level of wording (word
  error rate 78 %). The paper has to be written about meaning-level conversion and scored with content measures
  next to their chance levels. Counterfactual credit, the specific novelty, is mixed on validation data: the hub
  explains more with it and is harder to read; there is no test number yet.
- C2, the tonality gate: **no support on real data** in either run. Unless a longer representation phase with
  converged gates shows otherwise, it is a SynthWorld-only result and cannot carry a third of the paper.
- C3, attaching the island: Yorùbá speech → English text is right about the subject in 63 % of outputs from 985
  image pairs and no Yorùbá text (chance 18 %; BLEU 3.6 against about 1.1). Whether back-translation contributes
  on real data is unknown, and Yorùbá's validation scores fall while it runs. This is the next thing to measure.
- Missing for any submission: seeds, the baselines of section 3 on real data, and a recogniser listening to the
  written speech.

Negative result worth a paragraph: entropic Gromov–Wasserstein docking, the roadmap's central proposal, does not
dock an unpaired speech island (pure GW drives it to chance; semi-relaxed collapses; balanced fused GW ties with
no docking). It stays in the `island` ablation.

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
6. **Seeds and significance.** Three seeds minimum; paired bootstrap on BLEU for the main comparisons.
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
