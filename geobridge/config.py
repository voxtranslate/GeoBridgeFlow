"""Single source of truth for every hyper-parameter of GeoBridge-Flow.

Everything that changes behaviour lives in :class:`Config`.  A config is
  * built from a preset (``Config.preset("flickr8k_yo")``),
  * overridden from the CLI (``--set key=value``) or from an ablation variant,
  * saved next to every checkpoint, so a run is always reproducible from its folder.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ experiment
    exp_name: str = "geobridge"
    out_root: str = "runs"
    seed: int = 0
    device: str = "auto"                 # auto | cpu | cuda | cuda:0 | mps
    n_gpu: int = 0                       # GPUs used for training and feature extraction (0 = every visible one)
    step_batches: int = 0                # batches per optimisation step (0 = one per GPU; >1 on one device = accumulation)
    amp: bool = True                     # mixed precision (cuda only); OT always runs in fp32
    num_workers: int = 2
    torch_threads: int = 0               # 0 = leave PyTorch default

    # ------------------------------------------------------------------ data
    dataset: str = "flickr8k_yo"         # flickr8k_yo | synthworld
    data_root: str = "data"
    max_images: int = 0                  # >0: deterministic subset of images (quick runs)
    extra_images: int = 0                # extra image + hub-language TEXT pairs, training only (no speech, no other
                                         # language): -1 = all there are, N = the first N, 0 = none.  flickr8k_yo: the
                                         # Flickr30k photos that are not Flickr8k photos; synthworld: N more scenes
    extra_speech: int = 0                # extra images with hub-language text AND speech, training only: -1 = all,
                                         # N = the first N, 0 = none.  flickr8k_yo: the SpokenCOCO test part (5,000
                                         # images, 25,031 captions read by 1,766 speakers); synthworld: N more scenes
    lang_balance: bool = False           # every language has the same say in the losses of its modality (grounding,
                                         # sufficiency, credit, reading), however many of its utterances a batch
                                         # holds.  Measured on SynthWorld: worse on every count (the few island
                                         # views become a few very heavy, noisy ones); kept as an ablation only
    extra_per_batch: int = 0             # extra caption groups added to every batch (0 = as many as batch_size)
    extra_speech_share: float = 0.2      # ... share of them that are spoken ones, when both kinds exist.  A fifth is
                                         # about what the real pools hold (25k spoken of 143k extra captions).  Measured
                                         # on SynthWorld (island speech -> text; 74 % with extra text alone): a fifth
                                         # 77 %, a third 63 % (without the reader's exposure and augmentation), all 33 %
    listen_only: bool = False            # True: extra speech only LISTENS - its views are read into the hub code that
                                         # the text of the same caption and the image define, and take no part in the
                                         # terms that shape the hub (grounding, sufficiency, credit, minimality,
                                         # agreement).  Measured on SynthWorld: it undoes most of the damage of extra
                                         # speech that fills every extra place (island speech -> text 33 % -> 58 %),
                                         # but at a fifth of the places grounded speech does no damage and listening
                                         # is the worse of the two (68 % against 77 %).  For large shares; an ablation
    audio_anchor: bool = False           # speech latent that keeps the order of time (see AudioCodec.span_means)
    highpass_hz: float = 0.0             # high-pass every recording before trimming, pitch tracking and feature
                                         # extraction (0 = off; 70 in the real presets: below the lowest voice pitch)
    # Role of every language.  A language can appear in several roles.
    hub_langs: Tuple[str, ...] = ("en",)        # (image, text, audio) triplets
    island_langs: Tuple[str, ...] = ("yo",)     # speech only; text is never seen in training
    text_only_langs: Tuple[str, ...] = ("hi",)  # written, no speech (script-robustness branch)
    bridge_frac: float = 0.2             # fraction of island speech that keeps its image pairing
    island_unpaired: bool = True         # feed the remaining island speech as an unpaired stream
    use_island_text: bool = False        # True = "written" upper bound (island text used in training)
    strict_nonparallel: bool = True      # a caption is visible in ONE language only during training, so no
                                         # sentence of a non-hub language has its translation in the train set
    island_caps: Tuple[int, ...] = (0,)  # caption indices the island language may use in training
    text_only_caps: Tuple[int, ...] = (4,)   # caption indices given to the text-only language(s)
    split_fracs: Tuple[float, float, float] = (0.90, 0.05, 0.05)  # only for images w/o official split
    max_text_bytes: int = 160
    max_graphemes: int = 128
    max_audio_sec: float = 10.0
    trim_silence_db: float = 35.0

    # synthetic controlled benchmark (dataset == "synthworld")
    synth_n_images: int = 2400
    synth_caps_per_image: int = 4
    synth_n_colors: int = 8
    synth_n_shapes: int = 8
    synth_grid: int = 4
    synth_n_speakers: int = 6
    synth_two_obj_prob: float = 0.35
    synth_zipf: float = 0.7              # object types follow a Zipf law (0 = uniform).  Uniform, independent
                                         # attributes are the one case where no unpaired alignment can work
    synth_feat_noise: float = 0.15
    synth_vision_dim: int = 64

    # ------------------------------------------------------------------ frozen front-ends
    vision_encoder: str = "google/siglip-base-patch16-224"   # any HF vision tower | "synthetic"
    vision_grid: int = 7                 # patch tokens are pooled to grid x grid (+1 global token)
    speech_encoder: str = "utter-project/mHuBERT-147"        # any HF wav2vec2/HuBERT | "mfcc"
    speech_sem_layers: Tuple[int, ...] = (8, 9)   # hidden states averaged -> segmental/semantic stream
    speech_spk_layers: Tuple[int, ...] = (1, 2)   # early layers, time-averaged -> paralinguistic vector
    speech_pool: int = 2                 # 50 Hz -> 25 Hz
    sample_rate: int = 16000
    n_mels: int = 80
    mel_spec: str = "internal"           # internal (50 fps, Griffin-Lim) | speecht5 (62.5 fps, pretrained HiFi-GAN vocoder)
    mel_hop: int = 320                   # hop of the internal mel
    feat_batch_size: int = 16

    # ------------------------------------------------------------------ latent geometry
    latent_dim: int = 128                # d: token width shared by hub and all modality latents
    hub_slots: int = 8                   # K_h: tokens of the visual interlingua
    private_slots: int = 8               # K_p: modality-private tokens (Gaussian at t = 0)

    # ------------------------------------------------------------------ text codec (N4)
    text_unit: str = "grapheme"          # grapheme (ours) | byte | bpe
    bpe_vocab: int = 2000
    max_bytes_per_grapheme: int = 8
    text_width: int = 256
    text_enc_layers: int = 4
    text_dec_layers: int = 4
    text_heads: int = 4
    latent_noise_max: float = 0.5        # noise-robust latent training (decoder sees z + sigma*eps)
    dropout: float = 0.0                 # residual dropout in the codecs, grounder, explainer and reader (never the flows).
                                         # Measured on SynthWorld: 0.1 destroys the two-object coverage; keep 0 with w_seek > 0
    enc_dropout: float = 0.0             # residual dropout in the text and speech ENCODERS only (grounder and explainer
                                         # stay deterministic).  Measured on SynthWorld: 0.1 destroys the two-object
                                         # coverage as well (15 % against 97 %); kept as an ablation, not as a remedy

    # ------------------------------------------------------------------ audio codec (N5)
    audio_width: int = 256
    audio_enc_layers: int = 4
    audio_dec_layers: int = 3
    audio_heads: int = 4
    tone_mode: str = "gated"             # gated (ours) | on (ungated) | off (single stream)
    tone_gate_init: float = 2.0          # logit at release -> gamma = 0.88
    gate_price: str = "uniform"          # who pays for an open gate: every language in the batch (uniform) | every
                                         # grounded utterance, for the gate of its language (evidence)
    speech_mask: float = 0.0             # time masking of the speech encoder input: max share of an utterance per span
    gate_release_frac: float = 0.5       # share of phase A during which the tonality gates are held open and
                                         # the slots have no null option ("structure first, sparsity later")
    bypass_dim: int = 64                 # paralinguistic (speaker / register) bypass code
    use_bypass: bool = True
    mel_width: int = 192                 # mel renderer

    # ------------------------------------------------------------------ hub
    hub_mode: str = "bary"               # bary (ours) | global | free | text
    hub_width: int = 256
    hub_layers: int = 2
    hub_heads: int = 4
    hub_temp: float = 0.07               # attention temperature over patches
    hub_inhibit: float = 2.0             # inhibition of return, in units of the full cosine range (0 = independent slots)
    hub_temp_start: float = 6.0          # attention temperature multiplier at step 0 (annealed to 1 while dense)
    ground_temp: float = 0.07            # InfoNCE temperature
    agree_mode: str = "mutual"           # text and speech of a caption pull each other's code (mutual) | text is the target (teach)
    canonical_text: bool = True          # transport phase: speech with a same-language transcript takes the text-grounded code
    read_mode: str = "amortized"         # amortized (image-free reader, one pass) | ode (backward flow of the spoke)

    # ------------------------------------------------------------------ spokes (writers)
    flow_width: int = 384
    flow_layers: int = 6
    flow_heads: int = 6
    lora_rank: int = 8                   # 0 disables language-factorised LoRA
    lora_alpha: float = 16.0
    private_noise: bool = True           # False: K_p private tokens are zeros (deterministic spokes)
    ode_steps: int = 4                   # steps per hop at test time
    train_ode_steps: int = 2             # steps of the writes done inside the training losses
    ode_solver: str = "midpoint"         # euler | midpoint | heun
    sample_temp: float = 0.6             # std of the private noise at generation time
    eval_temp: float = 0.5               # same, for the scored (deterministic-seeded) evaluation
    asr_model: str = "openai/whisper-base.en"   # recogniser that listens to the written speech in `infer`
                                         # (English hub language and SpeechT5 mel space only; "" = skip)
    draft_t0: float = 0.0                # start writing from the explainer's draft at time t0 (0 = from the source)
    t_sampling: str = "uniform"          # uniform | logit_normal
    exposure: str = "reader"             # read hubs the writers are also trained on: reader | ode | none
    p_read: float = 0.5                  # share of paired views written from a read hub instead of the grounded one
    hub_jitter: float = 0.15             # std (upper bound) of the noise added to the hub part of a writer's source
    read_noise: float = 0.0              # std (upper bound) of the noise on the latent the reader is trained on (phase B)
    read_aug: int = 2                    # time-masked encodings of every spoken training utterance kept in the latent
                                         # cache; the reader reads a random one of them (or the clean one) and must
                                         # give the same hub code.  Transport phase only: nothing upstream moves
    read_aug_span: float = 0.15          # ... each hides two spans of up to this share of the utterance
    w_dec: float = 1.0                   # write-latent exposure: the byte decoder is trained on latents written by the
                                         # text writer (0 = decoder frozen after phase A)
    p_dec_written: float = 0.75          # ... share of those latents that are written (the rest are encoder latents)
    w_reread: float = 1.0                # write-latent exposure of the READER: real hub codes are written as speech in
                                         # the spoken hub language(s) and read back (back-translation does this for
                                         # the island language only).  0 = off
    image_spoke: bool = True
    thumb_size: int = 64                 # preview pixel decoder (image latent -> RGB)
    pix_width: int = 96
    pix_contrast: float = 4.0            # preview loss: extra weight of pixels far from the mean colour (0 = plain L1)

    # ------------------------------------------------------------------ unpaired island speech: hub-space back-translation
    bt_weight: float = 1.0               # 0 = use the bridge pairs only
    bt_relay: float = 1.0                # relay term: text writers learn from codes read off synthetic island speech
    bt_warmup_frac: float = 0.2          # share of phase B before back-translation starts
    bt_rectify: float = 0.5              # how far pseudo hub codes are moved along the OT map onto real ones (0 = off)
    bt_eps: float = 0.05
    bt_bank_mult: int = 4                # real hub codes drawn per unpaired utterance

    # ------------------------------------------------------------------ fused-GW docking of the reader (ablation; off)
    dock_mode: str = "none"              # none | g2w | gw | w
    dock_marginal: str = "balanced"      # balanced (log-Sinkhorn) | semi (semi-relaxed: collapses, see docs)
    dock_eps: float = 0.05
    dock_outer_iters: int = 10
    dock_sinkhorn_iters: int = 50
    dock_alpha_start: float = 0.0        # weight of the model-induced Wasserstein cost
    dock_alpha_end: float = 0.8
    dock_anneal_frac: float = 0.6        # fraction of phase B over which alpha is annealed
    dock_warmup_frac: float = 0.1        # fraction of phase B before docking is switched on
    naive_cycle_weight: float = 0.0      # roadmap's ODE cycle; kept to show it is vacuous

    # ------------------------------------------------------------------ loss weights
    w_text_rec: float = 1.0
    w_audio_rec: float = 1.0
    w_mel: float = 1.0
    w_orth: float = 0.1
    w_gate: float = 0.01                 # "pay for pitch" sparsity on the tonality gate
    w_ground: float = 1.0
    w_agree: float = 0.5
    w_expl: float = 1.0                  # sufficiency: the hub code must predict the utterance latent
    w_seek: float = 0.5                  # counterfactual patch credit: pull each slot to the patch that would explain more
    seek_gain: float = 0.02              # predicted drop of unexplained variance at which that pull is half on
    seek_verify: bool = False            # also carry each proposed move out and drop it if it verifiably explains less
    w_read: float = 1.0                  # the image-free reader must predict the hub code
    w_min: float = 0.5                   # minimality: the hub code must be predictable by the reader
    w_pix: float = 1.0                   # preview pixel decoder (0 removes it and its thumbnails)
    w_cfm: float = 1.0
    w_dock: float = 1.0
    w_free_hub_var: float = 1.0          # variance floor for the unconstrained ("free") hub baseline
    text_replay: int = 0                 # extra text-only captions per phase-A step (text is cheap; codec converges faster)

    # ------------------------------------------------------------------ optimisation
    batch_size: int = 64
    island_batch_size: int = 64
    phase_a_steps: int = 20000           # representation phase: codecs + grounder
    phase_a_select: bool = False         # start the transport phase from the phase-A state with the best validation
                                         # hub losses (grounding + sufficiency + read) instead of the last one
    phase_b_steps: int = 40000           # transport phase: writers, exposure, back-translation
    lr: float = 3e-4
    lr_flow: float = 3e-4
    lr_gate: float = 0.05                # the tonality gates are scalars: they need a much larger step
    bridge_per_batch: int = 8            # bridge pairs are few: guarantee this many in every paired batch
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 1000               # step checkpoints -> resume from any step
    keep_last: int = 3                   # number of step checkpoints kept (milestones are kept too)
    milestone_every: int = 10000
    eval_max_items: int = 400            # cap for the in-training generation metrics
    train_only: str = ""                 # e.g. "lora:yo" -> language-incremental docking
    init_from: str = ""                  # checkpoint to initialise weights from (phase reuse)

    # ------------------------------------------------------------------ derived helpers
    @property
    def n_tokens(self) -> int:
        return self.hub_slots + self.private_slots

    @property
    def mel_fps(self) -> float:
        return 62.5 if self.mel_spec == "speecht5" else self.sample_rate / self.mel_hop

    @property
    def total_steps(self) -> int:
        return self.phase_a_steps + self.phase_b_steps

    @property
    def all_langs(self) -> Tuple[str, ...]:
        out = []
        for l in (*self.hub_langs, *self.island_langs, *self.text_only_langs):
            if l not in out:
                out.append(l)
        return tuple(out)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_root, self.exp_name)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_root, "cache", self.dataset_tag)

    @property
    def extra_cache_dir(self) -> str:
        """Caches of the extra image-text pairs.  Separate from the benchmark's own cache, which they never touch."""
        n = "all" if self.extra_images < 0 else f"n{self.extra_images}"
        if self.dataset == "synthworld":
            return os.path.join(self.cache_dir, f"extra_{n}")
        return os.path.join(self.data_root, "cache", f"flickr30k_extra_{n}_b{self.max_text_bytes}")

    @property
    def extra_speech_cache_dir(self) -> str:
        """Caches of the extra spoken data (images, text, speech features)."""
        n = "all" if self.extra_speech < 0 else f"n{self.extra_speech}"
        if self.dataset == "synthworld":
            return os.path.join(self.cache_dir, f"extra_speech_{n}")
        return os.path.join(self.data_root, "cache", f"spokencoco_extra_{n}_b{self.max_text_bytes}_s{int(self.max_audio_sec)}")

    @property
    def dataset_tag(self) -> str:
        if self.dataset == "synthworld":
            key = (self.synth_n_images, self.synth_caps_per_image, self.synth_n_colors, self.synth_n_shapes,
                   self.synth_grid, self.synth_n_speakers, self.synth_two_obj_prob, self.synth_feat_noise,
                   self.synth_vision_dim, self.synth_zipf)
            return "synthworld_" + hashlib.md5(repr(key).encode()).hexdigest()[:8]
        return self.dataset + (f"_n{self.max_images}" if self.max_images else "")

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name: f for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in known:
                continue
            if isinstance(v, list):
                v = tuple(v)
            kw[k] = v
        return cls(**kw)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def clone(self, **overrides: Any) -> "Config":
        c = copy.deepcopy(self)
        c.update(overrides)
        return c

    def update(self, overrides: Dict[str, Any]) -> "Config":
        known = {f.name: f for f in fields(self)}
        for k, v in overrides.items():
            if k not in known:
                raise KeyError(f"unknown config field '{k}'")
            setattr(self, k, _coerce(v, getattr(self, k)))
        return self

    def signature(self, keys: Tuple[str, ...]) -> str:
        """Stable hash of a subset of fields (used to share phase-A checkpoints across ablations)."""
        d = self.to_dict()
        blob = json.dumps({k: d[k] for k in sorted(keys)}, sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    # ------------------------------------------------------------------ presets
    @classmethod
    def preset(cls, name: str) -> "Config":
        if name not in PRESETS:
            raise KeyError(f"unknown preset '{name}'. available: {sorted(PRESETS)}")
        return cls().update(PRESETS[name])


def _coerce(value: Any, ref: Any) -> Any:
    """Cast CLI strings to the type of the current field value."""
    if not isinstance(value, str) or isinstance(ref, str):
        if isinstance(ref, tuple) and isinstance(value, list):
            return tuple(value)
        return value
    if isinstance(ref, bool):
        return value.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(ref, int):
        return int(float(value))
    if isinstance(ref, float):
        return float(value)
    if isinstance(ref, tuple):
        items = [v for v in value.replace("(", "").replace(")", "").split(",") if v.strip() != ""]
        if len(ref) and not isinstance(ref[0], str):
            cast = type(ref[0])
            return tuple(cast(float(v)) if cast is int else cast(v) for v in items)
        return tuple(v.strip() for v in items)
    return value


# Fields that determine phase A (codecs + grounder).  Two configs with the same signature over these
# fields can share one phase-A checkpoint, which makes phase-B ablations several times cheaper.
PHASE_A_KEYS: Tuple[str, ...] = (
    "seed", "dataset", "max_images", "extra_images", "extra_speech", "extra_per_batch", "extra_speech_share", "listen_only", "lang_balance", "audio_anchor", "highpass_hz", "hub_langs", "island_langs", "text_only_langs", "bridge_frac",
    "use_island_text", "island_unpaired", "strict_nonparallel", "island_caps", "text_only_caps", "vision_encoder", "vision_grid", "speech_encoder", "speech_sem_layers",
    "speech_pool", "latent_dim", "hub_slots", "private_slots", "text_unit", "bpe_vocab", "text_width",
    "text_enc_layers", "text_dec_layers", "latent_noise_max", "audio_width", "audio_enc_layers",
    "audio_dec_layers", "tone_mode", "bypass_dim", "use_bypass", "hub_width", "hub_layers",
    "hub_temp", "hub_inhibit", "hub_temp_start", "ground_temp", "agree_mode", "w_text_rec", "w_audio_rec", "w_mel", "w_orth", "w_gate", "w_ground",
    "w_agree", "w_expl", "w_seek", "seek_gain", "seek_verify", "w_read", "w_min", "dropout", "enc_dropout", "gate_price", "speech_mask", "phase_a_select", "w_pix", "text_replay", "thumb_size", "pix_width", "pix_contrast", "batch_size", "island_batch_size", "phase_a_steps", "lr", "weight_decay",
    "lr_gate", "bridge_per_batch", "gate_release_frac", "tone_gate_init",
    "warmup_steps", "synth_n_images", "synth_caps_per_image", "synth_n_colors", "synth_n_shapes",
    "synth_grid", "synth_n_speakers", "synth_two_obj_prob", "synth_feat_noise", "synth_vision_dim", "synth_zipf",
    "max_text_bytes", "max_graphemes", "max_audio_sec", "mel_width", "n_mels", "mel_spec",
)


_SYNTH = dict(
    dataset="synthworld", vision_encoder="synthetic", speech_encoder="mfcc", vision_grid=4,
    hub_langs=("sa",), island_langs=("sb",), text_only_langs=("sc",), text_only_caps=(1,),
    max_text_bytes=96, max_graphemes=64, max_audio_sec=4.0, num_workers=0, amp=False,
)

# What the real benchmark changes relative to the defaults.  Each item answers something measured in a full
# Flickr8k run on 2 x T4, and each either was validated on SynthWorld or cannot disturb the representation phase:
#   phase_a_select  the validation grounding loss is best after about 4,000 steps and rises afterwards, whatever
#                   the schedule; this only selects a state, it does not change the training dynamics
#   read_noise      the reader's validation error rises throughout the transport phase (the cached latents never
#                   change); transport phase only
#   gate_price      with a uniform price the Yoruba gate closed to 0: it has few image pairs to defend it
# Schedule: 8,000 + 10,000 steps.  In the second run every validation curve of the transport phase had peaked
# within 9,000 of its steps (text read+write at 8,000-9,000, image -> text at 2,000) and declined afterwards.
# Deliberately NOT here: dropout, anywhere upstream of the hub.  Measured on SynthWorld it destroys the two-object
# coverage that the credit loss buys: 13 % against 93 % in the codecs, grounder and explainer; 15 % against 97 %
# confined to the encoders.
_REAL = dict(mel_spec="speecht5", phase_a_select=True, read_noise=0.2, gate_price="evidence", highpass_hz=70.0)
# More data for the hub language is the answer to the overfitting of the representation phase: with 6,847 training
# images every encoder, trained from scratch, memorises the captions within 15 epochs.  All Flickr30k photos that
# are not Flickr8k photos (about 23,700, five English captions each, text only) are added to the training split;
# as many of their caption groups as of the benchmark's own join every batch.
# Speech is the weak side (English speech finds its image three times less often than English text, and the
# speech codec rarely generalised), so the SpokenCOCO test part joins too: 5,000 more images whose 25,031 captions
# are text AND speech, from 1,766 speakers against Flickr8k's 183.  Spoken captions take a fifth of the extra
# places of a batch, not more: on SynthWorld extra speech next to extra text improves what speech is grounded on
# and leaves the rest about where it was, while extra speech alone, filling the batch, halves the island's scores.
_MORE = dict(extra_images=-1, extra_speech=-1)

PRESETS: Dict[str, Dict[str, Any]] = {
    # ---- real benchmark: Flickr8k images + English text/speech + Yoruba speech (+ Hindi text)
    "flickr8k_yo": dict(_REAL, **_MORE, text_replay=128, phase_a_steps=8000, phase_b_steps=10000, eval_every=1000),
    # same benchmark on two 16 GB GPUs (Kaggle "GPU T4 x2"); also runs on a single T4 (Colab), one batch per step.
    # Every step consumes one batch per GPU, so the schedule is counted in optimiser steps, not batches.
    "flickr8k_yo_t4": dict(
        _REAL, **_MORE, text_replay=128, batch_size=64, island_batch_size=64, feat_batch_size=32,
        phase_a_steps=8000, phase_b_steps=10000, eval_every=1000, save_every=1000, milestone_every=8000,
        num_workers=3, eval_max_items=600,
    ),
    # same benchmark, sized for a laptop / CPU sanity run
    "flickr8k_yo_cpu": dict(
        _REAL, text_replay=96, max_images=1200, latent_dim=96, hub_slots=6, private_slots=6, text_width=192, text_enc_layers=3,
        text_dec_layers=3, audio_width=192, audio_enc_layers=3, audio_dec_layers=2, hub_width=192,
        flow_width=256, flow_layers=4, flow_heads=4, lora_rank=4, batch_size=48, island_batch_size=48,
        phase_a_steps=2500, phase_b_steps=3500, eval_every=1000, save_every=500, milestone_every=2000,
        eval_max_items=200, num_workers=2, mel_width=128,
        bridge_frac=0.2, warmup_steps=200,
    ),
    # ---- controlled procedural benchmark (ground truth known for every factor)
    "synth": dict(
        _SYNTH, latent_dim=64, hub_slots=4, private_slots=4, text_width=128, text_enc_layers=2,
        text_dec_layers=2, audio_width=128, audio_enc_layers=2, audio_dec_layers=2, hub_width=128,
        flow_width=192, flow_layers=4, flow_heads=4, lora_rank=4, batch_size=64, island_batch_size=64,
        phase_a_steps=3000, phase_b_steps=4000, eval_every=1000, save_every=500, milestone_every=2000,
        eval_max_items=300, mel_width=96, warmup_steps=200,
        bridge_frac=0.2,
    ),
    # small world + small model: a full run takes minutes on a CPU; used for the reference ablation
    "synth_fast": dict(
        _SYNTH, synth_n_images=1200, latent_dim=48, hub_slots=4, private_slots=4, text_width=96,
        text_enc_layers=2, text_dec_layers=2, audio_width=96, audio_enc_layers=2, audio_dec_layers=2,
        hub_width=96, hub_layers=2, flow_width=128, flow_layers=3, flow_heads=4, lora_rank=4, batch_size=48,
        island_batch_size=48, phase_a_steps=1500, phase_b_steps=2500, eval_every=500, save_every=500,
        milestone_every=1000, eval_max_items=120, mel_width=64,
        warmup_steps=100, bridge_frac=0.2, bypass_dim=32, torch_threads=2,
    ),
    # seconds-long smoke test used by tests/ and CI
    "synth_tiny": dict(
        _SYNTH, synth_n_images=240, latent_dim=32, hub_slots=3, private_slots=3, text_width=64,
        text_enc_layers=1, text_dec_layers=1, text_heads=2, audio_width=64, audio_enc_layers=1,
        audio_dec_layers=1, audio_heads=2, hub_width=64, hub_layers=1, hub_heads=2, flow_width=64,
        flow_layers=2, flow_heads=2, lora_rank=2, batch_size=16, island_batch_size=16, phase_a_steps=30,
        phase_b_steps=30, eval_every=30, save_every=20, milestone_every=40, log_every=10,
        eval_max_items=24, mel_width=48, warmup_steps=5,
        bridge_frac=0.25, bypass_dim=16, bridge_per_batch=3,
    ),
}
