# Literature audit (checked 2026-11-21)

Two things were checked against primary sources (arXiv metadata and paper text, ACL Anthology, conference sites):

1. Do the papers the roadmap builds on exist, and do they say what the roadmap says they say?
2. For every mechanism the roadmap proposes, what is the closest prior work, and what is left to claim?

arXiv ids resolve at `https://arxiv.org/abs/<id>`. "Not found" means not found in the searches run, not proof of
absence. Items marked *unverified* were seen only in a search snippet.

## 1. The papers the roadmap cites

| # | Roadmap entry | Status | What the source actually says |
|---|---|---|---|
| 1 | FlowBind (ICLR 2026) | **Verified** — arXiv 2512.15420, ICLR 2026 poster | Hub `z* = H(available modalities)`, learned, **not anchored to any modality**. One drift MLP per modality on the straight path between `z*` and the modality latent; deterministic ODE both ways. Every modality is **a single 768-d vector** of a frozen encoder (EmbeddingGemma, CLIP, CLAP) decoded by frozen generators. Collapse is avoided by updating the hub encoder only at `t = 0`. General audio, **no speech, no languages, no unpaired modality**. "6×/10×": the parameter ratio is 5.6× (568M vs 3.2B); the 10× compares against an OmniFlow GPU-hour figure that OmniFlow never published. |
| 2 | OmniFlow (Jun 2025) | Verified, **date wrong** — arXiv 2412.01169 (Dec 2024), CVPR 2025 | Joint velocity field over all modality latents with decoupled time steps; no shared latent; 3.4B parameters; needs triplets. |
| 3 | Modaliff (Oct 2026) | Verified, **description wrong** — arXiv 2610.25533 | A token **diffusion** model with **no shared latent**; explicitly criticises FlowBind's single-vector hub for discarding token structure. |
| 4 | MUNI (Jun 2026) | Verified — arXiv 2606.16408 | Shared **stochastic** latent with modality-specific flow decoders; criticises FlowBind as deterministic and dimension-matched. Follow-up MUNITE: arXiv 2610.09866. |
| 5 | Qwen3-Omni (Sep 2025) | Verified — arXiv 2509.17765 | Thinker–Talker MoE; speech understanding in 19 languages, generation in 10. |
| 6 | HyperCLOVA X 8B Omni (Jan 2026) | Verified — arXiv 2601.01792 | Interleaved discrete + continuous tokens. |
| 7 | OmniFusion (EMNLP Findings 2026) | Verified — arXiv 2512.00234 | Gated fusion of a frozen omni model into a translation LLM; no low-resource languages. |
| 8 | GSLM2 / SpidR 2.0 (Nov 2026) | Verified — arXiv 2611.01143 (SpidR: 2512.20308) | Six unit streams from layers 5–10; speech only. Affiliation is "Meta Superintelligence Labs". |
| 9 | SpeechMapper (ICASSP 2026) | Verified — arXiv 2601.20417 | Two-stage projector into a frozen LLM's embedding space. |
| 10 | "Audio–Image ASR pretraining" (Sep 2026) | Partially — arXiv 2606.24080 (Jun 2026) | Experiments use **Vaani only** (WAXAL is only cited) and **transcripts are still required** for fine-tuning. It is not transcription-free ASR. |
| 11 | MICL for ASR (Apr 2026) | Verified, date differs — arXiv 2601.05707 (Jan 2026), ACL 2026 Findings | In-context learning of unseen languages with speech LLMs. |
| 12 | MMLow 2026, "6.7–7.4× fertility penalty" | Workshop verified; **claim misquoted** | The paper (aclanthology.org/2026.mmlow-main.1) reports 6.7–7.4 **tokens per word** for Bengali vs ~1.3 for English (≈5×), for Bengali only, and finds fertility **weakly** related to scores (r = −0.23), calling it "a cost and segmentation concern rather than a driver of the gap". |
| 13 | JK-EGW (Aug 2026) | Verified — arXiv 2608.04234 | Post-hoc alignment of frozen encoders with entropic GW on a joint kernel; the bound is `O(n^-1/2)` **plus a kernel-truncation term**; image–text only; **paired samples required**. |
| 14 | "OT Multimodal Fusion" (Jun 2026) | **Not found** | No paper could be matched to this entry. Do not cite it without a title. |

Consequences for the paper: do not write "6.7–7.4× penalty", do not describe item 10 as transcription-free, do not
describe Modaliff as a flow model with a shared latent, and drop item 14.

## 2. Is each proposed mechanism new?

### Roadmap novelty 2 — entropic GW between unpaired sets, coupled to flows → **known**

The roadmap states GW alignment was "never coupled with continuous generative ODE flows". That is not correct.

- **GENOT** (Klein et al., NeurIPS 2024, 2310.09254) solves a minibatch entropic OT / GW / fused-GW problem and
  trains conditional flow matching on the coupling. Its authors already note that GW solutions are not unique.
- **Inter-modal bridges** (2505.12322): a few labelled pairs act as zero-cost anchors for an (unbalanced) fused-GW
  problem followed by flow matching between frozen text and image latents.
- **KPG-RL** (NeurIPS 2022, 2303.13102): keypoint-pinned OT with relation-to-keypoint costs.
- **GWL** (ICML 2019, 1901.06003): alternates GW coupling and embedding learning and **ramps the weight of the
  learned-embedding cost over iterations**, i.e. the Gromov→learned-cost anneal, for graph matching.
- Also relevant: Alvarez-Melis & Jaakkola (1809.00013), DeepJDOT (1803.10081), semi-relaxed GW (2110.02753),
  SOTAlign (2602.23353), NGMA (2611.11460), Coupled Flow Matching (2510.23015).

Not found: GW/FGW-coupled flow matching on **unpaired speech**; semi-relaxed GW against a **memory bank of hub
codes**; the anneal where the Wasserstein cost comes from **the flow's own backward map**. What is claimable is
the setting and this particular combination, not a new coupling.

### Roadmap novelty 5 — the 2-hop ODE cycle loss → **vacuous as written, and that is published**

Every map in `z_m1 → ẑ* → z̃_m2 → z̃* → z_m1` is an ODE flow and therefore invertible, so the loss is zero up to
discretisation error for *any* velocity field. AlignFlow (AAAI 2020, 1905.12892) states that per-domain invertible
flows into a shared latent are cycle-consistent by construction; DDIB (ICLR 2023, 2203.08382) states the cycle
holds "up to discretization errors". The repository keeps the term only to demonstrate this numerically
(`tests/test_smoke.py::test_ode_is_invertible_so_the_plain_cycle_is_vacuous`, figure `09_ode_steps.png`).

### Hub ⊕ private stochastic spokes → **claim taken**

MUNI and MUNITE already argue for a shared stochastic latent plus modality-specific generative variation, and
CrossFlow (2412.15213) flows from a variational content code. The exact construction used here (the spoke's source
is `[hub ; Gaussian]`, so the backward ODE returns both) was not found, but it must be presented as a design
choice with these citations.

### Visual anchoring / vision as interlingua → **the idea and the task exist; the generative system does not**

- Harwath et al., "Vision as an interlingua" (ICASSP 2018, 1804.03052): cross-lingual speech retrieval via images.
- **Oneata & Kamper, "Translating speech with just images" (Interspeech 2024, 2406.07133)**: Yorùbá speech →
  English text with no parallel data on YFACC, by training on captions of the paired image. **BLEU-4 14.2 with four
  references, against a 21.1 supervised topline.** They also try mapping speech to image features for a frozen
  captioner, and report short, hallucinated outputs. This is the baseline to beat or match on the same data.
- Pirlogeanu et al. (2608.26925): written English word → spoken Hindi word via images.
- YFACC (2210.04600), WAXAL (2602.02734), BIG-C (2305.17202): the corpora that make the setting realistic.

Not found: any-to-any generation through a visual hub; **speech output in the low-resource language**; **any
tone evaluation** in vision-pivoted translation; a sentence-level hub built from patch tokens.

Note on the roadmap's "frozen self-supervised vision (SigLIP / DINOv2)": SigLIP is trained with text supervision,
so only DINOv2 gives an anchor that has never seen text. Both are supported (`vision_encoder`); the strict
zero-text claim should be made with DINOv2 and the SigLIP number reported next to it.

### Tone → **the problem is documented; the gate is open**

- Tone is lost at **quantisation**, not in continuous SSL features: Osakuade & King (2410.19935, 2604.07467) for
  Mandarin and Yorùbá; Shen et al. (2403.16865); Gubian et al. (2506.10855) find tone, phone and speaker subspaces
  largely orthogonal. The roadmap's claim that contrastive alignment erases tone has no direct evidence.
- Tone-aware units: ToneUnit (2406.08989), SITA (2601.09050). F0 streams: Polyak et al. (2104.00355), pGSLM
  (2109.03264).
- **Liang & Levow (SIGTYP 2025)** already obtain a tonality signal by ablation: flattening pitch hurts ASR much
  more for tonal languages. A reviewer will cite this and ask for functional-load controls.
- Mokgosi et al. (2606.31642, *unverified details*): adapters gated by F0 statistics, per utterance.

Not found: a **learned per-language gate** on the speaker-normalised contour inside the semantic encoder whose
value is read out as tonality. This is the most open direction.

### Grapheme–byte compressor → **every part is published**

LD4LG (2212.09462: Perceiver resampler to a fixed 32×64 text latent for diffusion), ByT5, CANINE, Charformer, MrT5,
BLT (2412.09871), MYTE (2403.10691) and MAGNET (2407.08818) for script-fair byte segmentation, and grapheme-cluster
BPE for Indic scripts (2409.11501). Pooling bytes at grapheme-cluster boundaries into a fixed continuous latent was
not found as such, but it is a combination; it is used here as an enabling component and not claimed as a
contribution.

### Per-language LoRA on the velocity field → **citations only**

LoRA-Whisper (2406.06619), Zipper-LoRA (2603.17558), and a flow-matching TTS that adds a language with 1.72 % of the
parameters (Kwon et al., Interspeech 2025, *base model unverified*). No novelty to claim.

## 3. Where the room is

| Rank | Direction | Why it is open | Closest work to cite |
|---|---|---|---|
| 1 | Learned lexical-tonality gate, with tone evaluated end-to-end in vision-pivoted translation | Nobody gates pitch per language into the semantic path, and nobody measures tone in visually grounded translation | Liang & Levow 2025; Osakuade & King; Gubian et al. |
| 2 | Generative any-to-any conversion for an unwritten language through a **sentence-level** visual hub, including speech *into* that language | Prior work is speech→text only, scene-level, and reports short, hallucinated output | Oneata & Kamper 2024; FlowBind; MUNI |
| 3 | Docking an unpaired speech island onto a hub with annealed, semi-relaxed, anchor-pinned fused GW | The components exist; the setting and the self-induced cost do not | GENOT; inter-modal bridges; GWL; KPG-RL |

Closest complete system: **FlowBind** (with MUNI / MUNITE as stochastic successors). Closest on the task:
**Oneata & Kamper (2024)**.

## 4. Competitors worth adding to the related-work section

MUNITE (2610.09866), NExT-OMNI (2510.13721), CrossFlow (2412.15213), FlowTok (2503.10772), AR-Omni (2601.17761),
MODUS (2607.25948), Omnilingual SONAR (2603.16606), Translatotron 3 (2305.17547), NaijaS2ST (2604.16287),
It's a (Blind) Match (2503.24129), GeoWarP (2610.09411).
