# GeoBridge-Flow: method

Notation: `d` latent width, `K_h` hub slots, `K_p` private slots, `N = K_h + K_p`. A *view* is one utterance in one
modality `m ∈ {txt, aud}` and one language `ℓ`. Code pointers are given as `file:symbol`.

## 0. What changed with respect to the roadmap, and why

| Roadmap | Problem found | What the code does instead |
|---|---|---|
| Hub = frozen SigLIP/DINOv2 geometry (one code per image) | An image-level pivot cannot carry *what was said*: five captions of one image share one code, so "translation" degenerates into captioning. This is the short, hallucinated output reported by Oneata & Kamper (2024). | **Barycentric visual interlingua**: a hub code is `K_h` attention-barycentres of the frozen patch tokens, selected by the utterance. Still text-free and frozen-geometry, but sentence-level. |
| `L_EGW` "never coupled with flows" | It has been (GENOT; inter-modal bridges; GWL for the anneal). Also, pure GW is blind to *where* the island sits: it is invariant to isometries, so it cannot by itself make two languages land on the same code. | **Gromov→Wasserstein docking**: fused GW whose cost is annealed from relational (GW) to positional (model-induced W), semi-relaxed marginals, anchor pinning with the available bridge pairs, and the plan is also used to draw pseudo-pairs. Claimed as a combination in a new setting. |
| `L_Trans-Cycle` through ODE maps | Every map is invertible, so the loss is zero for any velocity field (AlignFlow; DDIB). It trains nothing. | **Transitive hub-swap**: each spoke is trained on hub codes *read by the other spoke's ODE* from the paired view. This removes the train/test mismatch of cross-modal conversion and is not satisfied by construction. The original cycle is kept as a diagnostic. |
| Two speech streams: `z_sem` → hub, `z_pros` (with tone) bypasses the hub, and `z_sem ⟂ z_pros` | Self-contradictory for tonal languages: if tone bypasses the hub it cannot affect translation; if it is fused into `z_sem` the orthogonality penalty fights it. | **Three factors**: segmental content (always to the hub), lexical tone = speaker-normalised F0 contour (to the hub through a learned per-language gate), paralinguistics = pitch register + timbre (bypass, decorrelated from the hub-bound latent). |
| Deterministic spokes | One hub code has many valid realisations (wordings, speakers). A bijection from the hub cannot represent that. | Spoke source is `[hub ; Gaussian private tokens]` (cf. MUNI, CrossFlow). |
| "Single-stage training" | Flows trained on moving codec latents chase a moving target. | One run, two phases on a step schedule (A: codecs + grounder, B: flows), one resumable checkpoint stream. |
| Text / speech pairs across languages may be used implicitly | The captions of an image are translations of each other, so a corpus with all languages visible is implicitly parallel through the image. | **Strict non-parallel protocol** (default): in training, a caption is visible in one language only. No loss pairs two languages. |

## 1. Frozen front-ends and latents

- **Vision** (`data/features.py:VisionFrontend`): a frozen tower gives a `g×g` grid of patch tokens plus a global
  token. They are mapped once by a frozen **anchor projection**: PCA to `d` dimensions with a single global scale,
  `v_p = s · Wᵀ(x_p − μ)`. It is an isometry up to scale on the top-`d` subspace, fitted in float64 from the
  covariance (no SVD at training time) and never updated.
- **Speech** (`SSLFrontend`): deep hidden states of a multilingual SSL model at 25 Hz are the *segmental* stream
  `S`. The *tone* stream `F` has four values per frame: z-scored log-F0, voicing, Δlog-F0, z-scored log-energy.
  F0 is tracked with YIN and **standardised per utterance over voiced frames**, which removes the speaker's
  register and keeps the contour. The removed register `(mean, std)` and a time-averaged early-layer vector form
  the paralinguistic input.
- **Text** (`models/text_codec.py`): NFC text → extended grapheme clusters → each cluster embedded from its UTF-8
  bytes (mean ⊕ max pooling) → transformer → Perceiver resampler → `z_txt ∈ R^{N×d}`. A byte-level
  autoregressive decoder reconstructs the string. Sequence length is the number of written symbols for every
  script.
- **Speech codec** (`models/audio_codec.py`): `h = Linear(S) + Linear(voicing, energy)`, then the pitch FiLM

  `h ← h ⊙ (1 + γ_ℓ · tanh(W_s τ)) + γ_ℓ · tanh(W_b τ)`, with `τ = Conv(z-log-F0, Δlog-F0)` and `γ_ℓ = σ(a_ℓ)`,

  then transformer → resampler → `z_aud ∈ R^{N×d}`. Three details matter and were found empirically:
  (i) only the **pitch contour** is gated; voicing and energy mark syllable structure in every language, and
  when they shared the gated path the gate stayed open everywhere; (ii) both FiLM terms are bounded and
  bias-free, so `γ_ℓ` is an upper bound on what pitch can contribute and a flat contour is exactly a closed gate;
  (iii) the decoder reconstructs **only the segmental stream** from `z_aud` and the bypass code
  `p = MLP(speaker vector, register)`, so no reconstruction term asks for pitch. A prosody head predicts the
  contour from `stopgrad(z_aud)`; a frame-synchronous renderer maps (segmental, contour, `p`) to a mel-spectrogram.
- All modality latents are token-standardised (parameter-free LayerNorm) and their decoders are trained with
  additive noise on the latent, so flows never need to hit a latent exactly.

## 2. N1 — Barycentric visual interlingua (`models/hub.py`)

For a view with latent `z` and paired image tokens `{v_p}`:

```
q_k   = Grounder(z, m, ℓ)_k                                   k = 1..K_h      (slot queries)
α_k   = softmax_{p ∪ null}( ⟨q_k, key(v_p)⟩/τ − (λ/τ) Σ_{j<k} α_jp )          (sequential, inhibited)
z*_k  = Σ_p α_kp v_p                                           ∈ conv({v_p} ∪ {0})
```

A hub code is therefore a point of `conv(V_image ∪ {0})^{K_h}` inside the frozen vision geometry. Five properties
were each needed in practice (the numbers are from the controlled benchmark), and each can be switched off:

1. **Patches only.** The global token is the mean of the image; when it was offered, every slot settled on it and
   the hub collapsed to one code per image (grounding stuck at 14 % where it otherwise reaches 55–70 %).
2. **Sequential attention with inhibition of return.** Slot `k` is penalised on the patches slots `< k` took, so
   the slots enumerate what a sentence mentions. Independent slots all pile onto the most salient mentioned
   object.
3. **A null option.** A slot with nothing left to say attends to a learned null key whose value is the origin of
   the anchored space. It is offered only in the second half of phase A and attention is then charged a small
   price (`w_null`); offered from the start, later slots retreat to null and, being saturated, never come back.
4. **Soft-to-sharp attention.** The temperature starts at `hub_temp_start × τ` and is annealed geometrically to
   `τ` during the first half of phase A. A sharp softmax gives the explanatory loss no gradient towards a patch
   the slot is not already on; with the anneal, both objects of a two-object caption get their own slot in
   75–81 % of cases (7–16 % without).
5. **A contrastive score that uses the same attention** (below).

Training:

```
L_ground = InfoNCE over the batch with  s(u, I) = (1/K_h) Σ_k Σ_p α_kp(u, I) ⟨q_k(u), key(v_p(I))⟩
           where α(u, I) is the sequential, inhibited attention of u on image I (both directions, same-image
           duplicates masked).  Each slot is credited with the match it finds among the patches earlier slots
           left free.  The usual "best patch per slot, independently" score gives every slot full credit for
           re-finding the same object, so nothing rewards a second slot for the second thing a sentence says.
L_expl   = ‖ P(z*, m, ℓ) − stopgrad(z) ‖²      a small predictor P must reconstruct the utterance latent from the
           hub code alone; latents are unit-variance, so the value is the share of variance the hub leaves
           unexplained (1 − R²)
L_agree  = ‖ z*(text) − z*(speech) ‖²   for text and speech of the SAME caption in the SAME language
```

`L_expl` asks the hub code to be sufficient for the sentence. With independent slots and the independent
contrastive score, the hub described a single object per sentence: in the controlled benchmark none of the
two-object captions had both objects covered and text → hub → text reached 46 % with the image in hand.
`L_expl` is the `t = 0` view of flow matching (what the hub does not determine is unexplained variance of
the latent, cf. FlowBind's Prop. 1), restricted here to hub codes that are barycentres of frozen patches. `P` is
discarded after phase A. An optional slot-diversity term (`w_slot_div`, off by default) is kept for the
ablation; it pushes spare slots onto patches the sentence does not mention.

The grounding gradient reaches the codecs, which is what lets the tone gate learn from meaning (§5). At test time
the grounder is not used: hub codes are read by the backward ODE.

Ablation anchors (`hub_mode`): `global` (scene-level pooling, identical for all captions of an image), `free`
(unconstrained learned code, FlowBind-style, trained jointly with the flows), `text` (first `K_h` tokens of a
hub-language text latent; for the island, a caption of the same image with a different caption index).

## 3. N2 — Hub ⊕ private flow spokes (`models/flow.py`)

One spoke per modality, state `x ∈ R^{N×d}`:

```
x_0 = [ z* ; ξ ],  ξ ~ N(0, I)^{K_p×d}        x_1 = z_m        x_t = (1−t) x_0 + t x_1
L_cfm^m = E_t ‖ v_{θ_m + Δθ_{ℓ,m}}(x_t, t) − (x_1 − x_0) ‖²
```

`v` is a transformer with AdaLN time conditioning; every linear map is `W + (α/r) B_ℓ A_ℓ` with one rank-`r` pair
per language (`layers.py:LoRALinear`, `B` zero-initialised). *Write* = integrate 0→1 from `[z* ; ξ]`. *Read* =
integrate 1→0 from `z_m`, giving the hub estimate (first `K_h` tokens) and the private code. The image spoke
targets the anchored patch grid pooled to `N` tokens.

Conversion `(m₁, ℓ₁) → (m₂, ℓ₂)`: read with spoke `m₁`, keep the hub part, draw fresh private tokens, write with
spoke `m₂`. No image is involved.

## 4. N4 — Transitive hub-swap (`losses.py:phase_b`)

For text `z_t` and speech `z_a` of the same caption in the same language:

```
ĥ_t = Read_txt(z_t),  ĥ_a = Read_aud(z_a)             (no gradient; few ODE steps)
L_swap = ½ [ CFM_aud( [ĥ_t ; ξ] → z_a ) + CFM_txt( [ĥ_a ; ξ] → z_t ) ]
```

Each spoke learns to write from hub codes produced by the *other* spoke's ODE, which is exactly the input it
receives at test time. An optional differentiable version (`twohop_weight`) reads both views, exchanges the hub
parts, keeps each view's private code, writes back and penalises the reconstruction error and `‖ĥ_t − ĥ_a‖²`.

## 5. N5 — Lexical-tonality gate

`γ_ℓ ∈ [0,1]` scales how much of the speaker-normalised contour enters the hub-bound latent. Its loss terms:

```
+ w_gate · γ_ℓ                         "pay for pitch": a constant price
+ (through L_ground and the flows)     the benefit, present only if pitch helps identify what is said
```

Because the codec reconstructs only the segmental stream, nothing else rewards pitch. The gate therefore stays
open for a language iff its contour is informative about meaning.

Schedule (dense, then sparsify): for the first `gate_release_frac` of phase A all gates are held at 1 and no price
is charged; then they are released at `σ(tone_gate_init)` and priced. The gates are scalars, so they use their own
constant step size `lr_gate`. Pricing from step 0 closed every gate within ~400 steps, before grounding had
learned whether pitch is useful, and a closed sigmoid gate has no gradient left to reopen.

Reported per language: the gate value, the **tone-reliance index** `E‖ĥ − ĥ_flat‖² / E‖ĥ_i − ĥ_j‖²` (how far the
hub code moves when the contour is removed, in units of the distance between two utterances), and the
translation score with the contour removed.
`L_orth` is the mean squared entry of the batch cross-correlation between the pooled latent and the bypass code.

## 6. N3 — Gromov→Wasserstein docking (`losses.py:dock`, `models/ot.py`)

Island speech `{z_u}` has no image. Let `ĥ_u = Read_aud(z_u)` (differentiable) and `{y_j}` a sample of the **hub
bank**, a FIFO memory of hub codes produced from image-paired views. With centred cosine distances
`C¹ = D(ĥ, ĥ)`, `C² = D(y, y)`, `M = D(ĥ, y)`, each divided by its mean:

```
Π = argmin_Π  (1−α) Σ_{ijkl} (C¹_ik − C²_jl)² Π_ij Π_kl + α ⟨M, Π⟩ + ⟨R, Π⟩ − ε H(Π)
    s.t.  Π 1 = a                       (semi-relaxed: the bank side is free)      or  Π ∈ U(a, b)   (balanced)
```

- `α` is annealed from 0 (trust only the island's internal geometry) to `α_end` (trust where it currently sits).
- `R` pins the bridge pairs present in the batch to their true hub codes (cost 0 on the true partner, `c_pin`
  elsewhere); through the GW term the pinned rows also constrain the unpaired rows.
- Solver: entropic mirror descent; the inner step is a row-softmax (semi-relaxed) or log-domain Sinkhorn
  (balanced). float32, outside autocast, products and logsumexp only. The iteration is **warm-started from the
  model-induced plan** `softmax(−(M + R)/ε)`: the uniform plan is a stationary point of the GW iteration (every
  row then prefers the same central columns), and from a uniform start the plan stayed at 98 % of maximal entropy.
- `L_dock` is the objective at the fixed plan (envelope theorem), differentiable through `ĥ_u`.
- `L_pseudo`: sample `j ~ Π(i, ·)` and train `CFM_aud([y_j ; ξ] → z_u,i)` weighted by `1 − H(Π_i)/log m`.

## 7. Objective and schedule

```
Phase A   L_A = L_text + L_audio(segmental, prosody†, duration, mel, w_orth·L_orth, w_gate·γ) + L_ground
                + w_expl·L_expl + w_agree·L_agree                                 († stop-gradient on the latent)
Phase B   L_B = Σ_m L_cfm^m + w_swap·L_swap + w_dock·L_dock + w_pseudo·L_pseudo
```

Phase B freezes the codecs and the grounder, keeps an EMA of the spokes, and evaluates with the EMA weights.
Bridge pairs are few, so every paired batch is guaranteed `bridge_per_batch` of them (drawn by a generator seeded
with the stream position, which keeps resumption exact).

## 8. Evaluation (`metrics.py`)

All conversions go through the hub and use no image at test time.

- Text out: BLEU and chrF++ against the parallel reference and against all captions of the image; WER/CER for
  same-language transcription; exact content accuracy on SynthWorld.
- Hub space: R@k of utterance → image (how well gallery patches explain the hub code), island speech → text.
- Speech out: feature-level round trip (hub → speech latent → decoded frame features and predicted contour →
  re-encoded → hub → text). It passes through the lossy codec, so ODE invertibility does not make it trivial; it
  is an intrinsic proxy and should be complemented by an external ASR for a camera-ready.
- Tone: gates, tone-reliance, score with the contour removed, twin-confusion rate on SynthWorld.
- Protocol: `strict_nonparallel` on by default; `bridge_frac` budgets are nested subsets stored in `splits.json`.

## 9. Known limits

- Meaning that has no visual correlate cannot live in a barycentric hub code; it can only be carried by the
  private tokens, which are not transferred across modalities. Abstract text will translate poorly.
- Image output is an embedding (the anchored patch grid), scored by retrieval. Pixels need a decoder trained on
  the same vision space (e.g. an unCLIP model); none is included.
- Waveforms come from Griffin-Lim on a predicted mel; a neural vocoder is needed for natural speech.
- GW coupling on minibatches is noisy and identifiable only up to the symmetries the pins and the shared backbone
  break; with zero bridge pairs nothing breaks them except the shared speech backbone.
