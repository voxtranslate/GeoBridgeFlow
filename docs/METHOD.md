# GeoBridge-Flow: method

Notation: `d` latent width, `K_h` hub slots, `K_p` private slots, `N = K_h + K_p`. A *view* is one utterance in one
modality `m ∈ {txt, aud}` and one language `ℓ`. Code pointers are given as `file:symbol`. Measured effects are in
`RESULTS.md`; this document describes what the code does and why.

```
 read  (a function: one code per sentence)            write (a distribution: many realisations per code)
 text   ─ codec ─▶ z_txt ─ reader ─┐                  ┌─ flow writer ─▶ z_txt ─ byte decoder ─▶ text
 speech ─ codec ─▶ z_aud ─ reader ─┼─▶ hub code z* ───┼─ flow writer ─▶ z_aud ─ renderer+vocoder ─▶ speech
 image  ─ frozen tower ─▶ z_img ─ backward ODE ─┘     └─ flow writer ─▶ z_img ─ preview decoder ─▶ image
```

## 0. What changed with respect to the roadmap, and why

| Roadmap | Problem found | What the code does instead |
|---|---|---|
| Hub = frozen SigLIP/DINOv2 geometry (one code per image) | An image-level pivot cannot carry *what was said*: five captions of one image share one code, so "translation" degenerates into captioning (the short, hallucinated outputs of Oneata & Kamper, 2024). | **Barycentric visual interlingua** (§2): a hub code is `K_h` attention-barycentres of the frozen patch tokens, selected by the utterance. Still text-free and inside the frozen geometry, but sentence-level. |
| Nothing says *which* part of the image a sentence is about | A contrastive loss is satisfied by one distinctive object; spare slots drift to things the sentence never mentions. | **Mutual sufficiency** (§2): the code must explain the sentence (sufficiency) and be predictable from the sentence alone (minimality). **Counterfactual patch credit** gives the sufficiency loss a signal towards patches a slot is not looking at yet. |
| Read the source with the backward ODE of its flow | Costs an ODE per read, leaks the speaker into the hub (a bijection has to put everything somewhere), and is the input distribution the writers were never trained on. | **Asymmetric read / write** (§3): reading is amortised by an image-free *reader* (one pass); writing stays a flow. The reader is the same network that enforces minimality, so it costs nothing extra. |
| `L_Trans-Cycle` through ODE maps | Every map is invertible, so the loss is zero for any velocity field (AlignFlow; DDIB). It trains nothing. | **Read-hub exposure** (§4): writers are trained on hub codes the reader produced (own view, the other modality's view, and with jitter). The original cycle is kept as a diagnostic that shows it vanishes with the step size. |
| `L_EGW`: entropic Gromov–Wasserstein docking of unpaired island speech | GW has been coupled with flows before (GENOT). Measured here: pure GW is blind to *where* the island sits and drives it to chance; the semi-relaxed plan collapses onto a few hub codes; the balanced fused plan is indistinguishable from no docking. | **Hub-space back-translation** (§5) with the hub as the pivot language, an optimal-transport *rectification* of the pseudo codes, and a *relay* to the text writers. GW docking is kept as an ablation. |
| Two speech streams: `z_sem` → hub, `z_pros` (with tone) bypasses the hub, and `z_sem ⟂ z_pros` | Self-contradictory for tonal languages: if tone bypasses the hub it cannot affect translation; if it is fused into `z_sem` the orthogonality penalty fights it. | **Three factors** (§6): segmental content (always to the hub), lexical tone = speaker-normalised F0 contour (through a learned per-language gate), paralinguistics = pitch register + timbre (bypass, decorrelated from the hub-bound latent). |
| Deterministic spokes | One hub code has many valid realisations (wordings, speakers). A bijection from the hub cannot represent that. | Writer source is `[hub ; Gaussian private tokens]` (cf. MUNI, CrossFlow). |
| "Single-stage training" | Flows trained on moving codec latents chase a moving target. | One run, two phases on a step schedule (A: representation, B: transport on a **latent cache**), one resumable checkpoint stream. |
| Text / speech pairs across languages may be used implicitly | The captions of an image are translations of each other, so a corpus with all languages visible is implicitly parallel through the image. | **Strict non-parallel protocol** (default): in training, a caption is visible in one language only. No loss pairs two languages. |

## 1. Frozen front-ends and latents

- **Vision** (`data/features.py:VisionFrontend`): a frozen tower gives a `g×g` grid of patch tokens plus a global
  token. They are mapped once by a frozen **anchor projection**: PCA to `d` dimensions with a single global scale,
  `v_p = s · Wᵀ(x_p − μ)`. It is an isometry up to scale on the top-`d` subspace, fitted in float64 from the
  covariance and never updated. The real presets read images through **SigLIP 2** (`google/siglip2-base-patch16-256`,
  whose training adds dense objectives to the image–text one) on an 8 × 8 grid; the hub's barycentres are
  barycentres of those 64 patches. On the same subset and schedule, two seeds, held-out English speech finds
  its image in the top ten in 38 % of cases against 22 % with SigLIP at 7 × 7 (`RESULTS.md`).
- **Recordings** (`data/features.py:prep_wave`): resampled, **high-passed at 70 Hz** (`highpass_hz`, real
  presets), trimmed of leading and trailing silence, truncated. The high-pass is not cosmetic. Half of the
  energy of the YFACC Yorùbá recordings is rumble below 80 Hz; with it the trimmer found no pauses (an utterance
  was 7.3 s long where 4.0 s is speech) and the pitch tracker found pitch in 9 % of the frames (35 % without
  it; English: 38 % and 39 %). Everything below is computed from the cleaned waveform.
- **Speech** (`SSLFrontend`): deep hidden states of a multilingual SSL model at 25 Hz are the *segmental* stream
  `S`. The *tone* stream `F` has four values per frame: z-scored log-F0, voicing, Δlog-F0, z-scored log-energy.
  F0 is tracked with YIN and **standardised per utterance over voiced frames**, which removes the speaker's
  register and keeps the contour. The removed register `(mean, std)` and a time-averaged early-layer vector form
  the paralinguistic input.
- **Text** (`models/text_codec.py`): NFC text → extended grapheme clusters → each cluster embedded from its UTF-8
  bytes (mean ⊕ max pooling) → transformer → Perceiver resampler → `z_txt ∈ R^{N×d}`. A byte-level
  autoregressive decoder reconstructs the string. Sequence length is the number of written symbols for every
  script; there is no vocabulary to be out of.
- **Speech codec** (`models/audio_codec.py`): `h = Linear(S) + Linear(voicing, energy)`, then the pitch FiLM

  `h ← h ⊙ (1 + γ_ℓ · tanh(W_s τ)) + γ_ℓ · tanh(W_b τ)`, with `τ = Conv(z-log-F0, Δlog-F0)` and `γ_ℓ = σ(a_ℓ)`,

  then transformer → resampler → `z_aud ∈ R^{N×d}`. (i) Only the **pitch contour** is gated; voicing and energy
  mark syllable structure in every language. (ii) Both FiLM terms are bounded and bias-free, so `γ_ℓ` is an upper
  bound on what pitch can contribute and a flat contour is exactly a closed gate. (iii) The decoder reconstructs
  **only the segmental stream** from `z_aud` and the bypass code `p = MLP(speaker vector, register)`, so no
  reconstruction term asks for pitch. A prosody head predicts the contour from `stopgrad(z_aud)`; a
  frame-synchronous renderer maps (segmental, contour, `p`) to a mel-spectrogram in the feature space of the
  vocoder (`mel_spec = speecht5`: the pretrained SpeechT5 HiFi-GAN; `internal`: Griffin-Lim).
- **What is said and how it is said: a speaking path beside the codec** (`audio_anchor = extra`,
  `audio_private_slots`; real presets). A caption's bytes are reconstructed by classification; speech features
  are reconstructed by regression, and regression has a resting place in which every latent token is the
  utterance mean and every frame is decoded to it. On real speech a resampler of free queries stays there: in
  three Flickr8k runs the codec's reconstruction was constant in time and everything the model said was a hum.
  The speech latent therefore has two kinds of token, trained by two objectives that do not touch each other.
  The first `N` (as many as every latent has) are the **content tokens**: the codec above, unchanged. They are
  free queries, each of which can settle on one thing the utterance is about wherever it is said, and they are
  all the hub ever reads: slot queries, the reader and the sufficiency target see nothing else. The remaining
  ones are **realisation tokens**, laid out in time. A small encoder of their own reads the segmental stream
  and the tone stream (ungated: the hub never sees these tokens); token `k` starts from the mean of its states
  over the `k`-th of `n` equal spans of the utterance's own length (plus its learned query, then
  cross-attention to all frames). A second, **speaking decoder** reads content and realisation tokens
  (stop-gradient on the content): every frame starts from the linear interpolation of the realisation tokens
  around its own relative position, then attends to all tokens, and predicts the segmental stream; the prosody
  has a head of its own on a stop-gradient, as in the codec. The spans are relative, so the layout does not
  depend on the speaking rate; a duration head gives the length when speech is written. (Reading the content
  encoder's states instead, with a stop-gradient, is not enough: trained for the gist, they keep too little
  of the frames.) The flow writes both kinds of token from the hub
  slots and private noise, like any latent. Laying out *every* token in time, or training ordered and free
  tokens with one decoder, gives the codec the same help and takes the content away from the hub
  (`RESULTS.md`): what is said and how it is said have to be two objectives. The separation is exact and
  tested: with the same listening weights, a model with the speaking path and one without give the same value
  of every loss term of the representation phase and the same gradient on every shared parameter, and the
  trainer clips the speaking path's gradient on its own (`clip_apart`), so the two paths do not even share a
  clipping budget. A run with a speaking path is the same listening model with other random numbers.
- **Listening tokens** (`listen_tokens`). What the hub reads of an utterance is, by default, its content tokens:
  free summaries of the whole utterance. The literature on speech–image retrieval reads speech at frame rate
  with attention and finds that pooling over frames is what costs the most. Listening tokens give the hub the
  same without changing what it already reads: the mean of the speech encoder's states over each of `n` equal
  spans of the utterance, projected to the latent width, appended to the encoded latent. The hub's slot
  queries and the reader attend to content and listening tokens together. Nothing is reconstructed from them
  and no writer writes them (a written latent has none, and the hub then reads its content tokens alone), so
  grounding and reading are the only things that shape them. **Off in every preset**: a first gain on real
  speech did not repeat with a second seed, and in the SigLIP 2 image space they cost 13 points of held-out
  speech → image R@10 in both seeds (`RESULTS.md`, fifth round). They remain an option and a variant of the
  `speech` ablation group.
- All modality latents are token-standardised (parameter-free LayerNorm) and their decoders are trained with
  additive noise on the latent, so flows never need to hit a latent exactly.
- **Extra image–text pairs for the hub language** (`extra_images`; `data/flickr8k.py`, `data/dataset.py`). The
  premise of the setting is that one language is rich: it can bring image–caption pairs the other languages do
  not have. They are captions as text, in the hub language only, appended to the training split; they add
  nothing to validation or test and no sentence to any other language, so the zero-pair protocol is unchanged.
  On the real benchmark they are the Flickr30k photos. Flickr30k contains the Flickr8k photos with the same
  captions, so all of them are removed by Flickr photo id, whatever their split: otherwise test images would
  be trained on and the captions that the strict protocol gives to Yorùbá and Hindi would come back as English.
  Every batch is the benchmark's own caption groups (an epoch is still an epoch of the benchmark, and speech
  and bridge pairs get the same number of updates as before) plus `extra_per_batch` extra groups from a stream
  that walks the extra pool without replacement; the stream is a function of (seed, epoch, batch), so resuming
  is exact. The extra images are also negatives of every grounding score and targets of the image writer.
- **Extra speech for the hub language** (`extra_speech`). Speech is the side that has the least to learn from:
  one corpus, 183 speakers. The test part of SpokenCOCO adds 5,000 images whose 25,031 captions exist as text
  and as a recording, from 1,766 speakers; the text is joined from the COCO annotations by the caption id in the
  clip name. They are full triplets of the hub language (grounding, text–speech agreement, codec, reader,
  writers) and enter like the extra text: training split only, own feature caches, normalised with the
  benchmark's statistics, a fixed share (`extra_speech_share`, a fifth) of the extra places of every batch.
  The share matters. On the controlled benchmark a fifth costs the island language nothing and a batch flooded
  with hub-language speech costs it half of its translation: the hub then makes room for the language that
  fills the batch (`docs/RESULTS.md`, fourth round).
- **Captions written by a pretrained captioner** (`captioner`, `gen_captions`; `data/captioner.py`). The
  image-pivoted literature trains on nothing else, and finds that several *sampled* captions per image work
  where one beam-searched caption does not. Here a captioner (GIT) writes five sampled English captions for
  every training image of the benchmark, once, cached. They are extra text views of images the benchmark
  already has: grounded like any caption, read by the text codec, and - their main use - sibling targets for
  the island's speech (section 4). Training images only; validation, test and references are untouched.
- **Every island utterance keeps its image** in the real presets (`bridge_frac = 1`). The scarce-pairs setting
  (20 %, with the rest as unpaired speech for back-translation) is the `bridge` ablation; published
  image-pivoted systems train on all pairs, and a comparison with them has to as well.
- **Extra speech that only listens** (`listen_only`, off). The speech views of extra spoken captions go through
  the codec and are then only read: the reader regresses them onto the hub code that the text of the same
  caption and the image define (one view, one vote, in the same batch as the benchmark's own speech). They are
  left out of grounding, sufficiency, credit, minimality and text–speech agreement, and pay for no tone gate,
  so nothing they say can move the hub. This undoes most of the damage of a flooded batch, which locates that
  damage in the terms that shape the hub; at a fifth of the places it is worse than grounding the same speech,
  so it is an option for large shares and an ablation variant, not the default.

## 2. Barycentric visual interlingua with mutual sufficiency (`models/hub.py`, `losses.py:phase_a`)

For a view with latent `z` and paired image tokens `{v_p}`:

```
q_k   = Grounder(z, m, ℓ)_k                                   k = 1..K_h      (slot queries)
α_k   = softmax_{p ∪ null}( ⟨q_k, key(v_p)⟩/τ − (λ/τ) Σ_{j<k} α_jp )          (sequential, inhibited)
z*_k  = Σ_p α_kp v_p                                           ∈ conv({v_p} ∪ {0})
```

A hub code is a point of `conv(V_image ∪ {0})^{K_h}` inside the frozen vision geometry: it contains no text, it
is sentence-level, and it can be drawn on the image. Patches only (the global token is the image mean; offered,
every slot settles on it), sequential attention with inhibition of return, a null option offered late, and a
soft-to-sharp temperature are each needed and each can be switched off for the ablation.

The grounder is trained so that the code is **what the sentence and the image have in common**:

```
L_ground  InfoNCE over the batch with  s(u, I) = (1/K_h) Σ_k Σ_p α_kp(u, I) ⟨q_k(u), key(v_p(I))⟩ ,
          α(u, I) the same sequential, inhibited attention of u on image I       (found in the right image)
L_expl    ‖ Explainer(z*, m, ℓ) − stopgrad(z) ‖²                                 (sufficiency: the code says
          everything the sentence says; unit-variance latents, so this is the unexplained variance 1 − R²)
L_read    ‖ Reader(stopgrad(z), m, ℓ) − stopgrad(z*) ‖²                          (an image-free reader)
L_min     ‖ z* − stopgrad(Reader(z)) ‖²                                          (minimality: the code says
          nothing the sentence does not; an unmentioned object cannot be predicted from the sentence)
L_agree   ‖ z*(speech) − z*(text) ‖²             same caption, same language
```

One attention pass (`hub.ground`) produces both the contrastive scores and the code.

**Counterfactual patch credit** (`losses.py:seek`). Sufficiency reaches the attention only through the softmax
Jacobian, `∂z*_k/∂logit_p = α_kp (v_p − z*_k)`: a patch the slot does not look at gets no credit, however much it
would explain. Once the contrastive loss (which is content with one object) has made the slots sharp, the second
thing a sentence says can stay unseen for ever. The counterfactual "slot `k` sits on patch `p`" can be ranked
for every `(k, p)` at once, to first order:

```
propose   gain_kp = − ⟨ ∂ err / ∂ z*_k ,  v_p − z*_k ⟩      first-order drop of the utterance's unexplained
          (v = 0 for the null option; patches taken by      variance err if slot k sat on patch p;
           earlier slots are not candidates)                 p*_k = argmax_p gain_kp
pull      L_seek = − Σ_k w_k · log α_{k, p*_k} ,   w_k = gain₊² / (gain₊² + g₀²)
```

A code that already explains its sentence is left alone (`w → 0`). The pull is a cross-entropy on the *unclamped*
log-attention, whose gradient with respect to the logits is `α − onehot`: it does not vanish for a patch with no
attention, which is the whole point.

On real photographs the mean weight is higher than on the controlled benchmark (0.78 against 0.4–0.5 in the first
Flickr8k run): with 49 diverse patches and half of the latent unexplained, the best of many first-order estimates
is usually large. Whether the credit helps there is not known. Making the rule scale-free (gain relative to the
utterance's unexplained variance) was tried on SynthWorld and halved the coverage it buys, so the rule stays
absolute.

Two stricter variants were tried and measured (`RESULTS.md`). *Veto* (`seek_verify`, kept as an ablation): each
proposed move is carried out, the explainer is run on it, and a move that verifiably explains less is dropped;
this sends fewer slots to objects the sentence does not mention but also finds fewer second objects. *Verified
gain* (requiring the explainer's error to drop before pulling) switches the mechanism off: the explainer has to
co-adapt before a newly covered object pays off, so demanding the gain up front is circular.

**Agreement.** Text and speech of the same caption in the same language pull each other's code (`L_agree`,
`agree_mode = mutual`). Making the text code a fixed target for speech (`teach`) looked natural and was worse:
text then sharpens early onto one object and the coverage of two-object captions drops (see `RESULTS.md`); it
is kept as an ablation.

Schedule ("structure first, sparsity later"): for the first `gate_release_frac` of phase A there is no null
option, attention is annealed from soft to sharp, and `L_min` is off; then the null option is offered and
minimality starts.

Ablation anchors (`hub_mode`): `global` (scene-level pooling, identical for all captions of an image), `free`
(unconstrained learned code, FlowBind-style, trained jointly with the flows), `text` (first `K_h` tokens of a
hub-language text latent; for the island, a caption of the same image with a different caption index).

## 3. Asymmetric read / write (`models/geobridge.py`, `models/flow.py`)

**Read** = the reader of §2: `z* = Reader(z_m, m, ℓ)`, one pass, no image, no ODE. It regresses onto codes that
do not depend on the speaker, so nothing forces speaker identity into the hub (measured: it leaks less speaker
information than the backward-ODE read-out, not none). For speech it is far better than the backward ODE; for
text the backward ODE is the better reader (`RESULTS.md`), so a per-modality read mode is the obvious next step. Images have no reader (an image
does not say one thing); an image is read by the backward ODE of its writer. `read_mode = ode` restores the
backward ODE for text and speech (ablation).

**Write** = one conditional-flow-matching writer per modality, state `x ∈ R^{N×d}`:

```
x_0 = [ z* ; ξ ],  ξ ~ N(0, I)^{K_p×d}        x_1 = z_m        x_t = (1−t) x_0 + t x_1
L_cfm^m = E_t ‖ v_{θ_m + Δθ_{ℓ,m}}(x_t, t) − (x_1 − x_0) ‖²
```

`v` is a transformer with AdaLN time conditioning; every linear map is `W + (α/r) B_ℓ A_ℓ` with one rank-`r` pair
per language (`layers.py:LoRALinear`, `B` zero-initialised). The private tokens absorb what the hub does not
determine (wording, speaker), so one code has many realisations. The image writer targets the anchored patch
grid pooled to `N` tokens; a small **preview decoder** (`models/pixel.py`) turns that latent into a thumbnail.
It is trained with a contrast-weighted L1 (`losses.py:preview_loss`): plain L1 is minimised by the median image,
which is the bare background wherever the content covers a small part of the picture, and a decoder trained that
way produced all-background previews. Its input is frozen, so it shares no gradient with the rest of the model.

Optionally (`draft_t0 > 0`) writing starts from the explainer's one-pass draft: the ODE begins at time `t0` on
the straight path between the source and the draft. Off by default: the draft is a conditional mean, and for
sentences with several valid realisations it is blurred.

Conversion `(m₁, ℓ₁) → (m₂, ℓ₂)`: reader of `m₁`, writer of `m₂`, fresh private tokens. No image is involved.

**Read-back selection** (`select_k`; `GeoBridge.write_best`). Writing is one-to-many: the private noise decides
the wording, the voice, the timing. Reading is a function. At test time several candidates are therefore
written from the same hub code and each is read back by the reader; the candidate whose code is closest to the
code it was written from is kept. It costs `k` short ODE solves and `k` reader passes, needs nothing but the
model, and is scored as an evaluation-only ablation (the same checkpoint, `k = 1`).

## 4. Transport phase on a latent cache (`cache.py`, `losses.py:phase_b`)

After phase A the codecs and the grounder are frozen, so every utterance latent and every grounded hub code is a
constant. They are computed once (`LatentCache`) and phase B runs on tensors only: no feature loading, no codec,
no grounder, no image. Rebuilding the cache after a resume is deterministic.

```
L_B = Σ_m L_cfm^m( [h̃ ; ξ] → z_m )  +  w_read · L_read  +  bt_weight · L_bt            (§5)
```

**Canonical code.** Where a caption has a transcript in the same language, its speech view takes the
text-grounded code in the cache (`canonical_text`): text is the cleaner of the two views, and the speech reader
then learns to produce what text would have produced. This is a transport-phase choice only; the grounder itself
is trained with mutual agreement.

**Write-latent exposure** (`w_dec`). The codecs are trained on exact encoder latents, but at test time a decoder
only ever sees latents that a writer produced from a hub code, and those are blurred wherever the hub leaves the
wording open. So in the transport phase the byte decoder (not the encoder, whose latents are cached) is also
trained, teacher-forced, on latents written by the text writer from the same hub sources the writer is trained
on:

```
ẑ = Write_txt(h̃, ℓ; ξ)   (no gradient)        L_dec = CE( Decoder(ẑ, ℓ), bytes of the sentence )
```

A share of the latents (`1 − p_dec_written`) stays exact, so the decoder keeps reading encoder latents too. This
is the mirror image of read-hub exposure and closes the loop that the two-phase schedule leaves open between
writer and codec. Measured on SynthWorld (`decoder` ablation): every text output from a text or hub-language
speech source improves; on real captions, where the hub explains far less of the latent, it is what the first
full run was missing (`RESULTS.md`). It is switched on in the real-data presets; the SynthWorld reference run and
the other ablation groups predate it.

**Read-hub exposure.** The hub part `h̃` of a writer's source is the grounded code for half of the views and a
*read* code for the rest: the reader's output for the same view, or for the other modality's view of the same
caption (same language). All of them receive Gaussian jitter of random scale (`hub_jitter`), so a writer has to
tolerate a code that was read with some error. This replaces the roadmap's cycle term: it removes the train /
test mismatch of cross-modal conversion and is not satisfied by construction. In the controlled benchmark the
reader's codes are close enough to the grounded ones that neither exposure nor jitter changes the scores
measurably (`RESULTS.md`); they are kept because they cost one reader pass that is computed anyway.

**Image-sibling transport** (`w_sib`). A view of a language that has no text of its own speech (the island) or
no speech (a written-only language) is tied to the hub language by one thing: its hub code should be the one
its image gives. Image-pivoted speech translation (Oneata & Kamper 2024) ties them more directly, by training
speech → text on captions of the same image. The same supervision inside the hub: from the code that the
reader reads out of such a view, the text writer is trained (flow matching) to write **another caption of the
same image**, human or generated, in a written language; the decoder is trained on what the writer then
writes, against that caption's bytes; and the gradient reaches the reader. Never the caption itself: under the
strict protocol the island's caption exists in no other language, so what is paired is two descriptions of one
picture, not a sentence and its translation. The reader's own target (the grounded code of the utterance)
stays, so a view is asked both to say what it says and to be something a description of its picture can be
written from. Views of the hub language, whose reader is trained on transcripts, do not get it.

**Write-latent exposure of the speaking decoder** (`w_dec_aud`; needs the speaking path). The text decoder is
trained on written latents against the caption's bytes. For speech the same recipe would teach a blur: a
written latent is one way of saying the sentence and the recorded frames are another. So the latent is written
from the utterance's **own private code**, read off its real latent by inverting the flow, and from its hub
code as the writer gets it (read, jittered). It is then a latent of this recording, with the writer's errors
in it, and the speaking decoder is trained to render the recording's frames from it. A few utterances per step
have their frames fetched from the feature cache for this.

**Write-latent exposure of the reader** (`w_reread`). The same gap exists on the reading side. Back-translation
shows the reader speech of the island language that the speech writer wrote; nothing showed it written speech of
a hub language, and on Flickr8k written English speech, read back, carried no content while written Yorùbá
speech did. At every transport step real hub codes `h` are written as speech in a spoken hub language (no
gradient through the write) and the reader is trained to read them back, `‖R(W_aud(h)) − h‖²`. Chains such as
island speech → hub-language speech → anything then stay on what the reader has seen.

## 5. Hub-space back-translation of unpaired island speech (`losses.py:back_translate`)

The island language has speech only, a few image-paired utterances (bridge pairs) and many unpaired ones. With the
hub as the pivot language:

```
write   real unpaired speech z_u,  pseudo code  ĥ = Rectify(Reader(z_u))     speech writer learns  ĥ → z_u
read    real hub code h,  synthetic island speech  s = Write_aud(h, ℓ_island)  reader learns        s → h
relay   the same s read back,  h' = Reader(s);  real text z_t of h            text writer learns    h' → z_t
```

The target side of every direction is real. *Relay* is what connects the island to the writers of the other
languages: no caption exists in both, so without it a text writer never sees a code carrying the island reader's
errors.

**Transport rectification** (optional; no measurable effect so far, see `RESULTS.md`). Pseudo codes fed back on
themselves can drift towards frequent content. Before use they
are moved part of the way (`bt_rectify`) along the entropic optimal-transport map onto a batch of real hub codes,
`ĥ ← ĥ + ρ (T(ĥ) − ĥ)`, `T` the barycentric projection of a balanced log-domain Sinkhorn plan: the batch of
pseudo codes has to cover the real distribution, which a nearest-neighbour snap or a semi-relaxed plan does not
enforce.

Fused Gromov–Wasserstein docking of the reader (`losses.py:dock`, `models/ot.py`) is kept as the ablation of the
original proposal.

## 6. Lexical-tonality gate

`γ_ℓ ∈ [0,1]` scales how much of the speaker-normalised contour enters the hub-bound latent. Its loss terms:

```
+ w_gate · γ_ℓ                         "pay for pitch": a constant price
+ (through L_ground, L_expl, L_agree)  the benefit, present only if pitch helps identify what is said
```

Because the codec reconstructs only the segmental stream, nothing else rewards pitch. The gate therefore stays
open for a language iff its contour is informative about meaning.

Who pays matters. With a price per language present in the batch (`gate_price = uniform`) a language with few
image pairs pays as much as one with many, while only image-paired utterances can show that pitch helps: on
Flickr8k the Yorùbá gate (985 pairs) closed completely and the English one (30,000 pairs) did not. With
`gate_price = evidence` every grounded utterance pays for the gate of its language, so price and benefit scale
with the same count. The gates are trained with Adam, which follows the sign of the net gradient, so this
changes which way a gate goes, not merely how fast.

Schedule (dense, then sparsify): for the first `gate_release_frac` of phase A all gates are held at 1 and no price
is charged; then they are released at `σ(tone_gate_init)` and priced. The gates are scalars, so they use their own
constant step size `lr_gate`. Pricing from step 0 closed every gate before grounding had learned whether pitch is
useful, and a closed sigmoid gate has no gradient left to reopen.

Reported per language: the gate value, the **tone-reliance index** `E‖ĥ − ĥ_flat‖² / E‖ĥ_i − ĥ_j‖²` (how far the
hub code moves when the contour is removed, in units of the distance between two utterances), and the
translation score with the contour removed. `L_orth` is the mean squared entry of the batch cross-correlation
between the pooled latent and the bypass code.

## 7. Objective, schedule and execution

```
Phase A   L_A = L_text + L_audio(segmental, prosody†, duration, mel, w_orth·L_orth, w_gate·γ)
                + L_ground + w_expl·L_expl + w_seek·L_seek + w_read·L_read + w_min·L_min + w_agree·L_agree + w_pix·L_pix
Phase B   L_B = Σ_m L_cfm^m + w_read·L_read + w_dec·L_dec + w_reread·L_reread + bt_weight·(L_bt_write + L_bt_read + bt_relay·L_bt_relay)
                                                                                  († stop-gradient on the latent)
```

- Phase B keeps an EMA of what it trains and evaluates with the EMA weights. Bridge pairs are few, so every
  paired batch is guaranteed `bridge_per_batch` of them (drawn by a generator seeded with the stream position,
  which keeps resumption exact).
- **Several GPUs** (`trainer.py:StepReplica`): one optimisation step consumes one batch per GPU. Batches are
  nested, ragged dictionaries (and, in phase B, slices of the latent cache), which `nn.DataParallel` cannot
  scatter; so the only thing scattered is an index, and each replica takes its own prepared batch, moves it to
  its device and computes the complete loss there under its own autocast context. Gradients are averaged. The
  frozen vision and speech encoders are wrapped the same way for feature extraction (`features.py:on_all_gpus`).
  On one device the same code runs the batches one after the other (gradient accumulation).
- **Small real corpora** (the `_REAL` block of `config.py`). Flickr8k has 6.8 k training images; in the first full
  run the validation grounding loss was best after 4,000 steps and rose for the remaining 12,000. The real-data
  presets use a shorter schedule, `phase_a_select` (the transport phase starts from the phase-A state with the
  best validation grounding + sufficiency + read losses after the release, not from the last one), noise on the
  cached latents the reader is trained on, and the evidence-proportional gate price. The transport phase is
  8,000 steps on the two-T4 preset: in the third run the validation scores were flat after 5,000, and the
  shorter phase lets two ablation variants finish in the same 12-hour session.
  Residual dropout in the grounder and explainer, speech time masking and stronger weight decay exist as options
  and are *not* in the presets: on SynthWorld dropout 0.1 there destroys the two-object coverage that the credit
  loss buys (13 % against 93 %), and so does dropout confined to the text and speech encoders (`enc_dropout`: 15 %
  against 97 %). The overfitting of the representation phase on Flickr8k is therefore answered only by selecting
  an early state.
- Resuming refuses to continue a run folder under different settings (anything but paths, logging, evaluation
  and the length of the transport phase), so a changed configuration can never silently inherit old checkpoints.
- Checkpoints hold the model, the EMA, both optimisers, the AMP scaler, RNG states, the history and the position
  inside the epoch (epoch index + batch offset of both data streams); the epoch order is a function of
  (seed, epoch), so training resumes from any step of any epoch.

## 8. Evaluation (`metrics.py`)

All conversions go through the hub and use no image at test time.

- Text out: BLEU and chrF++ against the parallel reference and against all captions of the image; WER/CER for
  same-language transcription; exact content accuracy on SynthWorld; and **content F1**: corpus-level word F1 in
  which every word is weighted by its inverse document frequency among the references, so that function words
  count for almost nothing and no stop-word list is needed in any language.
- **Chance level of every text score** (`chance_bleu`, `chance_chrf`, `chance_cf1`): the same outputs scored
  against the reference of another image. Writers are generative: given any hub code they write a fluent caption,
  and fluent captions share n-grams. On Flickr8k that is worth about 1 BLEU, 15 chrF++ and 6 content F1, so a BLEU
  of 3 is three times chance, and a score without its chance level cannot be read.
- **Selection score** (`score`, picks `best.pt`): content F1 minus its chance level, averaged over the text
  directions scored during training. The island BLEU alone, which was the score before, selected the first
  validation of the transport phase in the second Flickr8k run: the validation subset was 400 random caption
  groups with 67 island utterances among them. The subset now gives up to half of its places to island captions.
- Hub space: R@k of utterance → image (how well gallery patches explain the hub code), island speech → text.
- Image out: R@k of the *written* image latent against the gallery, and the preview thumbnail.
- Speech out: feature-level round trip (hub → speech latent → decoded frame features and predicted contour →
  re-encoded → hub → text). It passes through the lossy codec, so it is not trivially satisfied, but it is read
  by the model's own reader and so measures agreement of the model with itself.
- Speech out, heard from outside (`inferencer.py:speech_asr`; English hub language, SpeechT5 mel space): an
  off-the-shelf recogniser (Whisper) transcribes three rungs, each scored against the caption by word error rate
  and by content F1 with its chance level. *Vocoder*: the real utterance's own log-mel through the vocoder, the
  ceiling of the path. *Codec*: real speech → speech latent → decoded streams → log-mel, what the codec keeps.
  *Written*: caption text → hub → written speech latent → log-mel, the model's text-to-speech; and island
  speech → hub → written hub-language speech. The gap between two rungs says where intelligibility is lost.
- Text codec alone (encode, decode, no hub) is scored at every validation: it bounds everything that passes
  through the hub, and the decoder keeps training in the transport phase.
- Tone: gates, tone-reliance, score with the contour removed, twin-confusion rate on SynthWorld.
- **BLEU as the image-pivoted literature computes it** (`bleu_r1` … `bleu_r5`): sacrebleu, lower-cased, effective
  order, corpus level, `. , ? !` removed, against `n` = 1 … 5 of the image's captions with the caption of the
  source itself always among them (Oneata & Kamper 2024, from their evaluation script). Their published
  Yorùbá → English numbers are 6.65 (n = 1) to 15.82 (n = 5); the other references are sampled there with five
  repeats and taken in order here.
- Protocol: `strict_nonparallel` on by default; `bridge_frac` budgets are nested subsets stored in `splits.json`.

## 9. Known limits

- Meaning that has no visual correlate cannot live in a barycentric hub code; it can only be carried by the
  private tokens, which are not transferred across modalities. Abstract text will translate poorly. Measured on
  Flickr8k: the hub explains 47 % of the variance of a held-out caption's latent; text read into the hub and
  written again names the right kind of subject in 91 % of cases and has a word error rate of 78 %. The model
  converts meaning at the level of "what is in the picture"; it is not a recogniser or a translator of wording.
- On real data the tonality gate has not been tested: in the three runs so far the pitch tracker found pitch
  in 9 % of the Yorùbá frames (rumble in the recordings, removed since by the high-pass), so the gate of the
  tonal language closed over an empty input.
- Speech output was unintelligible in the three runs so far (a recogniser found no content in it), because the
  speech latent held no order of time. With the time-ordered latent a recogniser finds content in the codec's
  reconstructions in a short CPU run on real features; written speech has not been heard yet.
- Back-translation of unpaired island speech shows no effect on Flickr8k (one seed, with the old speech latent
  and the uncleaned recordings).
- The counterfactual credit is a first-order estimate through the explainer as it currently is; it also sends
  some slots to objects the sentence does not mention, which the minimality term then has to remove.
- Image output is a latent of the frozen vision space plus a low-resolution preview; photo-realistic pixels need
  a decoder trained on that space (e.g. an unCLIP model), which is not included.
- Back-translation starts from the bridge pairs: with none, nothing ties the island reader to the hub except the
  shared speech backbone.
- The multi-GPU path was written for two T4s and reviewed, but this repository's measurements were all made on a
  CPU-only machine; see `RESULTS.md`.
