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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ experiment
    exp_name: str = "geobridge"
    out_root: str = "runs"
    seed: int = 0
    device: str = "auto"                 # auto | cpu | cuda | cuda:0 | mps
    amp: bool = True                     # mixed precision (cuda only); OT always runs in fp32
    num_workers: int = 2
    torch_threads: int = 0               # 0 = leave PyTorch default

    # ------------------------------------------------------------------ data
    dataset: str = "flickr8k_yo"         # flickr8k_yo | synthworld
    data_root: str = "data"
    max_images: int = 0                  # >0: deterministic subset of images (quick runs)
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
    mel_hop: int = 320                   # 50 fps mel for the renderer
    feat_batch_size: int = 16
    cache_mel: bool = True

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

    # ------------------------------------------------------------------ audio codec (N5)
    audio_width: int = 256
    audio_enc_layers: int = 4
    audio_dec_layers: int = 3
    audio_heads: int = 4
    tone_mode: str = "gated"             # gated (ours) | on (ungated) | off (single stream)
    tone_gate_init: float = 2.0          # logit at release -> gamma = 0.88
    gate_release_frac: float = 0.5       # share of phase A during which all gates are held open and unpriced
    bypass_dim: int = 64                 # paralinguistic (speaker / register) bypass code
    use_bypass: bool = True
    mel_width: int = 192                 # mel renderer

    # ------------------------------------------------------------------ hub (N1)
    hub_mode: str = "bary"               # bary (ours) | global | free | text
    hub_width: int = 256
    hub_layers: int = 2
    hub_heads: int = 4
    hub_temp: float = 0.07               # attention temperature over patches
    hub_inhibit: float = 2.0             # inhibition of return, in units of the full cosine range (0 = independent slots)
    ground_temp: float = 0.07            # InfoNCE temperature

    # ------------------------------------------------------------------ spokes (N2)
    flow_width: int = 384
    flow_layers: int = 6
    flow_heads: int = 6
    lora_rank: int = 8                   # 0 disables language-factorised LoRA
    lora_alpha: float = 16.0
    private_noise: bool = True           # False: K_p private tokens are zeros (deterministic spokes)
    ode_steps: int = 4                   # steps per hop at test time
    train_ode_steps: int = 2             # steps of the (cheaper) reads done inside the training losses
    ode_solver: str = "midpoint"         # euler | midpoint | heun
    sample_temp: float = 0.6             # std of the private noise at generation time
    eval_temp: float = 0.5               # same, for the scored (deterministic-seeded) evaluation
    t_sampling: str = "uniform"          # uniform | logit_normal
    image_spoke: bool = True

    # ------------------------------------------------------------------ docking (N3)
    dock_mode: str = "g2w"               # g2w (ours) | gw | w | none
    dock_marginal: str = "semi"          # semi (semi-relaxed, ours) | balanced (log-Sinkhorn)
    dock_eps: float = 0.05
    dock_outer_iters: int = 10
    dock_sinkhorn_iters: int = 50
    dock_alpha_start: float = 0.0        # weight of the model-induced Wasserstein cost
    dock_alpha_end: float = 0.8
    dock_anneal_frac: float = 0.6        # fraction of phase B over which alpha is annealed
    dock_bank_size: int = 2048
    dock_bank_batch: int = 192
    dock_pin: bool = True                # anchor pinning with the available bridge pairs
    dock_pin_cost: float = 4.0
    dock_pseudo_pair: bool = True        # coupling-guided CFM on the unpaired stream
    dock_warmup_frac: float = 0.1        # fraction of phase B before docking is switched on

    # ------------------------------------------------------------------ transitive consistency (N4')
    swap_frac: float = 0.5               # share of paired items trained with a hub from another spoke
    twohop_weight: float = 0.0           # differentiable 2-hop hub-swap reconstruction (costly)
    naive_cycle_weight: float = 0.0      # roadmap's ODE cycle; kept to show it is vacuous

    # ------------------------------------------------------------------ loss weights
    w_text_rec: float = 1.0
    w_audio_rec: float = 1.0
    w_mel: float = 1.0
    w_orth: float = 0.1
    w_gate: float = 0.01                 # "pay for pitch" sparsity on the tonality gate
    w_ground: float = 1.0
    w_agree: float = 0.5
    w_expl: float = 1.0                  # explanatory grounding: the hub code must predict the utterance latent
    w_slot_div: float = 0.0              # optional slot-diversity term (off: it pushes spare slots onto patches
                                         # the sentence does not mention)
    w_cfm: float = 1.0
    w_swap: float = 1.0
    w_dock: float = 1.0
    w_pseudo: float = 0.5
    w_free_hub_var: float = 1.0          # variance floor for the unconstrained ("free") hub baseline

    # ------------------------------------------------------------------ optimisation
    batch_size: int = 64
    island_batch_size: int = 64
    phase_a_steps: int = 20000           # representation phase: codecs + grounder
    phase_b_steps: int = 40000           # transport phase: spokes, swap, docking
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
    "seed", "dataset", "max_images", "hub_langs", "island_langs", "text_only_langs", "bridge_frac",
    "use_island_text", "island_unpaired", "strict_nonparallel", "island_caps", "text_only_caps", "vision_encoder", "vision_grid", "speech_encoder", "speech_sem_layers",
    "speech_pool", "latent_dim", "hub_slots", "private_slots", "text_unit", "bpe_vocab", "text_width",
    "text_enc_layers", "text_dec_layers", "latent_noise_max", "audio_width", "audio_enc_layers",
    "audio_dec_layers", "tone_mode", "bypass_dim", "use_bypass", "hub_width", "hub_layers",
    "hub_temp", "hub_inhibit", "ground_temp", "w_text_rec", "w_audio_rec", "w_mel", "w_orth", "w_gate", "w_ground",
    "w_agree", "w_slot_div", "w_expl", "batch_size", "island_batch_size", "phase_a_steps", "lr", "weight_decay",
    "lr_gate", "bridge_per_batch", "gate_release_frac", "tone_gate_init",
    "warmup_steps", "synth_n_images", "synth_caps_per_image", "synth_n_colors", "synth_n_shapes",
    "synth_grid", "synth_n_speakers", "synth_two_obj_prob", "synth_feat_noise", "synth_vision_dim", "synth_zipf",
    "max_text_bytes", "max_graphemes", "max_audio_sec", "mel_width", "n_mels",
)


_SYNTH = dict(
    dataset="synthworld", vision_encoder="synthetic", speech_encoder="mfcc", vision_grid=4,
    hub_langs=("sa",), island_langs=("sb",), text_only_langs=("sc",), text_only_caps=(1,),
    max_text_bytes=96, max_graphemes=64, max_audio_sec=4.0, num_workers=0, amp=False,
)

PRESETS: Dict[str, Dict[str, Any]] = {
    # ---- real benchmark: Flickr8k images + English text/speech + Yoruba speech (+ Hindi text)
    "flickr8k_yo": dict(),
    # same benchmark, sized for a laptop / CPU sanity run
    "flickr8k_yo_cpu": dict(
        max_images=1200, latent_dim=96, hub_slots=6, private_slots=6, text_width=192, text_enc_layers=3,
        text_dec_layers=3, audio_width=192, audio_enc_layers=3, audio_dec_layers=2, hub_width=192,
        flow_width=256, flow_layers=4, flow_heads=4, lora_rank=4, batch_size=48, island_batch_size=48,
        phase_a_steps=2500, phase_b_steps=3500, eval_every=1000, save_every=500, milestone_every=2000,
        dock_bank_size=1024, dock_bank_batch=128, eval_max_items=200, num_workers=2, mel_width=128,
        bridge_frac=0.2, warmup_steps=200,
    ),
    # ---- controlled procedural benchmark (ground truth known for every factor)
    "synth": dict(
        _SYNTH, latent_dim=64, hub_slots=4, private_slots=4, text_width=128, text_enc_layers=2,
        text_dec_layers=2, audio_width=128, audio_enc_layers=2, audio_dec_layers=2, hub_width=128,
        flow_width=192, flow_layers=4, flow_heads=4, lora_rank=4, batch_size=64, island_batch_size=64,
        phase_a_steps=3000, phase_b_steps=4000, eval_every=1000, save_every=500, milestone_every=2000,
        dock_bank_size=1024, dock_bank_batch=128, eval_max_items=300, mel_width=96, warmup_steps=200,
        bridge_frac=0.2,
    ),
    # small world + small model: a full run takes minutes on a CPU; used for the reference ablation
    "synth_fast": dict(
        _SYNTH, synth_n_images=1200, latent_dim=48, hub_slots=4, private_slots=4, text_width=96,
        text_enc_layers=2, text_dec_layers=2, audio_width=96, audio_enc_layers=2, audio_dec_layers=2,
        hub_width=96, hub_layers=2, flow_width=128, flow_layers=3, flow_heads=4, lora_rank=4, batch_size=48,
        island_batch_size=48, phase_a_steps=1500, phase_b_steps=2500, eval_every=500, save_every=500,
        milestone_every=1000, dock_bank_size=768, dock_bank_batch=96, eval_max_items=120, mel_width=64,
        warmup_steps=100, bridge_frac=0.2, bypass_dim=32, torch_threads=2,
    ),
    # seconds-long smoke test used by tests/ and CI
    "synth_tiny": dict(
        _SYNTH, synth_n_images=240, latent_dim=32, hub_slots=3, private_slots=3, text_width=64,
        text_enc_layers=1, text_dec_layers=1, text_heads=2, audio_width=64, audio_enc_layers=1,
        audio_dec_layers=1, audio_heads=2, hub_width=64, hub_layers=1, hub_heads=2, flow_width=64,
        flow_layers=2, flow_heads=2, lora_rank=2, batch_size=16, island_batch_size=16, phase_a_steps=30,
        phase_b_steps=30, eval_every=30, save_every=20, milestone_every=40, log_every=10,
        dock_bank_size=64, dock_bank_batch=32, eval_max_items=24, mel_width=48, warmup_steps=5,
        bridge_frac=0.25, bypass_dim=16, dock_outer_iters=4, dock_sinkhorn_iters=20, bridge_per_batch=3,
    ),
}
