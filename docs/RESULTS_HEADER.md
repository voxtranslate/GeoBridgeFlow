# Results measured so far

Everything below was produced by the code in this repository on a machine with **4 CPU cores and no GPU**.
The runs are therefore small: 3,000 optimisation steps, models of 5–17 M parameters, **one seed**, test sets of
240 (SynthWorld) and 460 (Flickr8k subset) caption groups. With 240 utterances a difference of fewer than about
8 points between two rows is within noise. The numbers show that the pipeline runs end to end and which
mechanisms have an effect in a controlled setting. They are **not** the numbers of a paper: those need the GPU
schedule in `scripts/paper_experiments.sh` (full data, three seeds, significance tests).

Sections 1–3 are generated from the JSON files under `runs/` by

```bash
python scripts/make_results.py --synth runs/sf_main --ablation runs/sf_ablation --real runs/f1200
```

## 0. What the runs show

**Works (SynthWorld, reference run = full model).**

- *Zero-pair speech translation of an unwritten, tonal language.* With 205 image–speech bridge pairs, no text of
  the island language and no parallel sentence anywhere in training, island speech → hub-language text reaches
  BLEU 22.4, and the first-mentioned object's shape word is right in 94 % of utterances.
- *Tone survives, and only where it is lexical.* Tone-contrastive (colour) words are translated correctly in
  81 % of island utterances and in 47.5 % when the pitch contour is removed from the input; scaling the contour
  from 0 to 1 moves that accuracy monotonically from 43 % to 83 % (96 utterances). The learned gate is 0.58 for the tonal
  language and 0.00 for the non-tonal one, whose accuracy (99 %) does not react to pitch at all.
- *Zero-pair text translation across scripts.* Latin-script hub language ↔ Devanagari-script language, with no
  pair of sentences ever shown together: 62 % and 68 % exact content, BLEU 43 and 51.
- *A plain ODE cycle is vacuous.* Its loss falls from 2.8 to 0.0002 as the solver goes from 1 to 32 steps, for the
  same weights (`09_ode_steps.png`): it measures the integrator, not alignment.

**What the ablations say** (section 2; one seed).

- *Sentence-level visual hub vs scene-level visual pivot:* the pivot loses on every translation and retrieval
  column (for example island content 12 % vs 28 %, text → third-language text 17 % vs 62 %).
- *Visual hub vs text anchor:* the text anchor is **better** when the target is hub-language text (island content
  41 % vs 28 %) and much worse for the third language (24 % vs 62 %) and for anything involving images. The claim
  that survives is "balanced across directions and not centred on one written language", not "better everywhere".
- *Tone stream:* without it, tone-word accuracy on the island falls from 81 % to 56 %, exactly its level with the
  pitch removed, island BLEU from 22 to 15 and island speech → image R@1 from 25 % to 13 %; the non-tonal hub
  language is unaffected (speech → text content 48 % vs 52 %).
- *Private tokens:* making the spokes deterministic costs 26 points of hub-language speech → text content
  accuracy (25 % vs 52 %) and 37 points of text → third-language content (25 % vs 62 %).
- *Transitive hub-swap:* removing it costs 17 points of hub-language speech → text content accuracy (35 % vs
  52 %; WER 53 vs 36) and leaves the island, which has no same-language pairs to swap, unchanged. Text → text and
  captioning are slightly better without it.
- *Docking:* with balanced marginals it is indistinguishable from no docking (28 % vs 33 % island content, BLEU
  22 vs 17), and so is pure Wasserstein docking (23 %). Pure Gromov–Wasserstein docking, which is what the
  roadmap specified, drives the island to chance (3 %), and so does the semi-relaxed plan, which collapses onto
  6 % of the hub codes. **At a 20 % bridge budget docking has not earned its place.** The 1 % and 5 % budgets,
  where it could matter, were not run.
- *Not run:* the bridge-budget sweep, the text-unit, LoRA and protocol groups, and four variants of the hub
  (independent slots, contrastive-only grounding, sharp attention from the start, free learned anchor).

**Does not work yet.**

- *The second thing a sentence says.* Both objects of a two-object caption now get a slot in about 80 % of
  cases, but spare slots also attend to objects the sentence does not mention, and exact content on two-object
  captions through the grounded hub stays in single digits. Exact-content numbers above are therefore carried
  almost entirely by one-object captions.
- *Speaker disentanglement.* A linear probe reads the speaker from the hub tokens of a speech read-out more
  easily than from the private tokens. The private tokens give sampling diversity, not a clean factorisation.
- *Real data at this scale.* On the 1,200-image Flickr8k subset the byte-level text codec is far from converged
  after 1,400 steps (67 % character error on English), so every text metric is near the floor (Yorùbá speech →
  English text BLEU 2.5). What does work there: retrieval through the hub from real speech and photos (Yorùbá
  speech → image R@10 31 % among 92 test images, chance 11 %), hub agreement between Yorùbá speech and English
  text above chance (AUC 0.66), and script-fair units (Hindi 3.2 units per word against 11.2 for bytes and for a
  BPE vocabulary learned on English). The tonality gates did not separate on real data (English 0.35, Yorùbá
  0.46) and the comparison with Oneata & Kamper (2024) cannot be made at this scale.

## 0.1 What changed because the code was run

Each row is a defect found by measurement during this session and the change that removed it (SynthWorld, single
runs; the "before" numbers come from intermediate checkpoints that are not kept).

| Symptom measured | Cause | Change | After |
|---|---|---|---|
| Grounding stuck at 14 % (48-way), all slots identical, text and speech hub codes identical | every slot settled on the global image token: one code per image | slots attend over patch tokens only; distinct slot initialisation | 51 % at the same step |
| Both tonality gates closed (0.01–0.03) within 400 steps | the price acted before grounding had learned whether pitch helps; a closed sigmoid cannot reopen | gates held open for the first half of phase A, then released and priced; own step size | separation appears |
| Gate stayed useful for the non-tonal language too | the gated stream also carried voicing and energy, which help every language | only the pitch contour is gated; bias-free, bounded FiLM | 0.00 (non-tonal) vs 0.58–0.94 (tonal) across runs |
| Hub with the image in hand reproduced 46 % of captions; 0 % of two-object captions had both objects covered | the contrastive score credits every slot for re-finding the same object | explanatory loss, sequential attention with inhibition, contrastive score under the same attention, null slot offered late, soft-to-sharp attention | first object right in 99–100 % (hub language), 94 % (island); both objects covered in 75–81 % |
| Island at chance although its grounded hub codes were accurate | semi-relaxed docking collapsed onto 6 % of the bank codes | balanced marginals | island content 28 % (no docking: 33 %) |
| Transport plan at 98 % of maximal entropy | the uniform plan is a stationary point of the GW iteration | warm start from the model-induced plan | row entropy 0.50 |
