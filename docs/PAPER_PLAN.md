# Paper plan

Working title: **GeoBridge-Flow: Speaking Through Pictures — A Sentence-Level Visual Interlingua with Tone-Gated,
Transport-Docked Flows for Unwritten Languages**

## 1. What to claim (and what not to)

Three contributions carry the paper. Each has one figure and one ablation table that isolate it.

| | Contribution | Claim, stated so it can be falsified | Evidence produced by this repository |
|---|---|---|---|
| C1 | **Barycentric visual interlingua** | A hub code made of utterance-selected barycentres of frozen patch tokens translates *what was said*; a scene-level visual pivot, a text anchor and a free learned anchor do not, under identical flows. | `ablate --groups anchor`; `01_slot_attention.png`; pointing-game accuracy (SynthWorld); `11_hub_agreement.png` |
| C2 | **Lexical-tonality gate** | A per-language gate on the speaker-normalised F0 contour (i) opens for tonal languages and closes for non-tonal ones without tone labels, and (ii) is necessary for tone-contrastive words to survive speech → hub → text. | `ablate --groups tone`; `05_tone.png` (gates, pitch-scaling response); tone-word accuracy with pitch removed; twin-confusion rate |
| C3 | **Gromov→Wasserstein docking** of an unpaired speech island — *currently a negative result* | Hypothesis: with few bridge pairs, annealed fused GW with anchor pinning beats no docking, and the gain grows as bridge pairs become scarce. **Not supported so far**: at a 20 % bridge budget balanced docking ties with no docking and semi-relaxed docking collapses (one seed, SynthWorld). Keep it only if the 1 % and 5 % budgets, with seeds, show a gain; otherwise report it as an analysis of why relational docking fails for arbitrary word forms. | `ablate --groups docking bridge`; `04_docking_coupling.png` (plans vs true partners; relational-isometry correlation) |

System-level claim: the first any-to-any (text ↔ speech ↔ image-embedding) model for an unwritten language that
uses no text of that language and no parallel sentences, including speech *into* that language.

Supporting components, **cited, not claimed**: hub ⊕ private spokes (MUNI, MUNITE, CrossFlow), transitive
hub-swap (AlignFlow and DDIB for why a plain ODE cycle is vacuous; BCM/BiFM for inversion consistency),
grapheme–byte compressor (LD4LG, MYTE, MAGNET, grapheme-pair encoding), per-language LoRA (LoRA-Whisper,
Zipper-LoRA). See `docs/LITERATURE_AUDIT.md`.

Do not claim: "first to couple GW with flows" (GENOT), "cycle-consistency loss" as a contribution, script-fair
tokenisation as a contribution, mathematical guarantees for docking (GW on minibatches has none), or that
contrastive alignment erases tone (the documented loss is at quantisation).

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

`anchor`, `docking`, `bridge`, `tone`, `text_unit`, `transitive`, `private`, `lora`, `protocol` — see
`geobridge/ablation.py:GROUPS`. Run every group on SynthWorld with 3 seeds and `anchor`, `docking`, `bridge`,
`tone` on Flickr8k-Tri.

### 2.4 Analyses (`python run.py infer`)

Slot attention across languages; hub manifold; transport plans against true partners; gate values and
pitch-scaling response; frame- and grapheme-level attribution; quality against ODE steps; fertility per script.

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
| Docking does not help once 5 % bridges exist | `docking` table flat | report honestly; the claim then narrows to the ≤ 1 % regime |
| Gate does not separate on real data (SSL features already encode tone) | `tone/gate_yo ≈ tone/gate_en`; no drop with pitch removed | this is itself a finding (consistent with Osakuade & King); use discrete units for the segmental stream, where tone is known to be lost |
| The YFACC mirror stores every clip as exactly 8.0 s | long utterances may be truncated | verify against the original release (Olaleye et al., 2022) before final numbers |
| Hindi captions are machine-translated | en ↔ hi scores measure agreement with an MT system | describe as silver; use only for the script-robustness analysis |

## 5. Venues

The honest positioning is a low-resource speech/multimodal paper with a controlled-benchmark component:
ACL / EMNLP (Speech and Multimodality; Multilinguality), Interspeech, or TACL. Workshops for early feedback:
AfricaNLP, MMLow, IWSLT (low-resource track), SIGTYP for the tonality-gate analysis. Acceptance cannot be
guaranteed by design; what can be guaranteed is that every claim above is tied to a table this code produces.
