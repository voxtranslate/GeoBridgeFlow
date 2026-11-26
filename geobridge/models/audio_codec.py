"""Tone-gated speech codec.

Three factors of a spoken utterance are kept apart on purpose:

  segmental content   deep SSL features                           -> always enters the latent
  lexical tone        speaker-normalised F0 contour (z log-F0, delta) -> enters the latent through a learned
                      per-language gate  gamma_l in [0, 1]  that is charged a sparsity price ("pay for pitch");
                      voicing and energy are not gated (they are segmental cues in every language)
  paralinguistics     pitch register + timbre (speaker vector)    -> never enters the latent; it travels in a
                      bypass code that only the decoder / renderer sees, and is decorrelated from the latent

The latent is trained to reconstruct the *segmental* stream only, so nothing in the reconstruction loss asks
for pitch.  The gate therefore stays open only if pitch helps the objectives that depend on meaning (visual
grounding, and through it the flows).  In a tonal language it does; in a non-tonal one the price wins and the
gate closes.  The gate values are thus a learned, per-language estimate of how lexical pitch is.

Schedule ("train dense, then sparsify"): for the first ``gate_release_frac`` of phase A all gates are held at 1
and no price is charged, so the model first learns *whether* pitch is useful; then the gates are released and
priced.  Releasing them closed, or pricing them from step 0, kills them before grounding has learned anything
(a closed sigmoid gate has no gradient left to reopen).

The F0 contour needed for synthesis is predicted by a prosody head that reads the latent with a
stop-gradient: for a tonal language the latent contains tone (the gate is open) and the head can recover the
lexical contour; for a non-tonal language it falls back to a neutral contour, which is the right behaviour.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, Block, Resampler, sincos, standardize_tokens


class MelRenderer(nn.Module):
    """(segmental features, tone contour, bypass code) at 25 Hz -> log-mel at the vocoder's frame rate.
    Frame-synchronous, so no duration model is needed; the waveform comes from the vocoder that owns the mel
    feature space (pretrained HiFi-GAN for ``mel_spec="speecht5"``, Griffin-Lim for the internal mel)."""

    def __init__(self, sem_dim: int, bypass_dim: int, width: int, n_mels: int):
        super().__init__()
        self.inp = nn.Linear(sem_dim + 3 + bypass_dim, width)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2) for _ in range(4)])
        self.out = nn.Linear(width, n_mels)

    def forward(self, sem: torch.Tensor, tone3: torch.Tensor, bypass: torch.Tensor, out_len: int) -> torch.Tensor:
        x = torch.cat([sem, tone3, bypass.unsqueeze(1).expand(-1, sem.shape[1], -1)], -1)
        h = F.interpolate(self.inp(x).transpose(1, 2), size=out_len, mode="linear", align_corners=False)
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return self.out(h.transpose(1, 2))


class AudioCodec(nn.Module):
    def __init__(self, cfg, sem_dim: int, spk_dim: int, n_lang: int, stats: Dict):
        super().__init__()
        self.cfg, W, d, N = cfg, cfg.audio_width, cfg.latent_dim, cfg.tokens_of("aud")
        self.n_tokens = N
        mode = {True: "all", False: "off"}.get(cfg.audio_anchor, cfg.audio_anchor)   # (a run saved when it was a switch)
        assert mode in ("off", "extra", "all"), cfg.audio_anchor
        self.mode = mode
        # tokens laid out in time: none, those beyond the n_tokens every latent has (they come last), or all
        self.n_ordered = {"off": 0, "extra": N - cfg.n_tokens, "all": N}[mode]
        self.n_content = N - self.n_ordered if mode == "extra" else N   # tokens of the codec the hub reads
        self.n_listen = cfg.listen_tokens                               # + listening tokens, appended after the N written
        assert mode != "extra" or self.n_ordered > 0, "audio_anchor = extra needs audio_private_slots > private_slots"
        self.sem_dim = sem_dim
        t = lambda k: torch.as_tensor(stats[k], dtype=torch.float32)
        for k in ("sem_mean", "sem_std", "spk_mean", "spk_std", "f0_mean", "f0_std"):
            self.register_buffer(k, t(k))
        self.register_buffer("mel_mean", torch.tensor(float(stats.get("mel_mean", 0.0))))
        self.register_buffer("mel_std", torch.tensor(float(stats.get("mel_std", 1.0))))
        # ---- encoder
        self.sem_in = nn.Linear(sem_dim, W)
        # The gated path sees ONLY the pitch contour (z-scored log-F0 and its delta).  Voicing and energy are
        # useful in every language (they mark syllable structure), so they enter ungated; otherwise the gate
        # would stay open everywhere and say nothing about lexical tone.  All maps of the gated path are
        # bias-free, so a flat contour (zeros) is exactly equivalent to a closed gate.
        self.aux_in = nn.Linear(2, W)
        self.tone_net = nn.Sequential(nn.Conv1d(2, W, 5, padding=2, bias=False), nn.GELU(),
                                      nn.Conv1d(W, W, 5, padding=2, bias=False))
        self.film_scale = nn.Linear(W, W, bias=False)
        self.film_shift = nn.Linear(W, W, bias=False)
        self.gate_logit = nn.Parameter(torch.full((n_lang,), float(cfg.tone_gate_init)))
        self.gate_forced_open = False  # set by the trainer during the first part of phase A (not a parameter)
        self.lang_emb = nn.Embedding(n_lang, W)
        p_enc = max(cfg.dropout, cfg.enc_dropout)
        self.enc = nn.ModuleList([Block(W, cfg.audio_heads, dropout=p_enc) for _ in range(cfg.audio_enc_layers)])
        self.resampler = Resampler(W, self.n_content, d, layers=2, heads=cfg.audio_heads, dropout=cfg.enc_dropout)
        if self.n_listen:
            self.listen_out = nn.Linear(W, d)
        # ---- paralinguistic bypass
        self.bypass = MLP(spk_dim + 2, 2 * cfg.bypass_dim, cfg.bypass_dim)
        self.byp_in = nn.Linear(cfg.bypass_dim, W)
        self.byp_gate = nn.Parameter(torch.zeros(1))
        # ---- decoder (latent -> frame-level segmental features) and heads
        self.lat_in = nn.Linear(d, W)
        self.lat_pos = nn.Parameter(torch.randn(self.n_content, W) * 0.02)
        self.q_proj = nn.Linear(2 * W, W)
        self.dec = nn.ModuleList([Block(W, cfg.audio_heads, ctx_dim=W, dropout=cfg.dropout) for _ in range(cfg.audio_dec_layers)])
        self.dec_norm = nn.LayerNorm(W)
        self.sem_out = nn.Linear(W, sem_dim)
        self.dur_head = MLP(d, W, 1)
        self.pros_dec = nn.ModuleList([Block(W, cfg.audio_heads, ctx_dim=W) for _ in range(1)])
        self.pros_out = nn.Linear(W, 3)  # z-scored log-F0, voicing logit, z-scored log-energy
        self.renderer = MelRenderer(sem_dim, cfg.bypass_dim, cfg.mel_width, cfg.n_mels)
        # ---- the speaking path (audio_anchor = extra): time-ordered tokens with an encoder and a decoder of their
        #      own.  It reads the recording's streams itself (the pitch contour too, ungated: the hub never sees
        #      these tokens) and the content tokens with a stop-gradient, so everything above trains exactly as it
        #      would without it: what the hub reads cannot be changed by what is added for speaking.
        if mode == "extra":
            self.speak_feat = nn.Linear(sem_dim + 4, W)
            self.speak_lang = nn.Embedding(n_lang, W)
            self.speak_enc = nn.ModuleList([Block(W, cfg.audio_heads, dropout=cfg.dropout) for _ in range(2)])
            self.time_resampler = Resampler(W, self.n_ordered, d, layers=2, heads=cfg.audio_heads)
            self.speak_in = nn.Linear(d, W)
            self.speak_pos = nn.Parameter(torch.randn(N, W) * 0.02)
            self.speak_q = nn.Linear(2 * W, W)
            self.speak_byp = nn.Linear(cfg.bypass_dim, W)
            self.speak_dec = nn.ModuleList([Block(W, cfg.audio_heads, ctx_dim=W, dropout=cfg.dropout) for _ in range(cfg.audio_dec_layers)])
            self.speak_norm = nn.LayerNorm(W)
            self.speak_out = nn.Linear(W, sem_dim)
            self.speak_pros_dec = Block(W, cfg.audio_heads, ctx_dim=W)
            self.speak_pros = nn.Linear(W, 3)

    # ------------------------------------------------------------------ gate
    def gate(self, lang: Optional[torch.Tensor] = None) -> torch.Tensor:
        mode = self.cfg.tone_mode
        if mode == "off":
            g = torch.zeros_like(self.gate_logit)
        elif mode == "on" or self.gate_forced_open:
            g = torch.ones_like(self.gate_logit)
        else:
            g = torch.sigmoid(self.gate_logit)
        return g if lang is None else g[lang]

    def norm_sem(self, sem: torch.Tensor) -> torch.Tensor:
        return (sem - self.sem_mean) / self.sem_std

    def bypass_code(self, spk: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_bypass:
            return torch.zeros(spk.shape[0], self.cfg.bypass_dim, device=spk.device, dtype=spk.dtype)
        x = torch.cat([(spk - self.spk_mean) / self.spk_std, (f0 - self.f0_mean) / self.f0_std], -1)
        return self.bypass(x)

    # ------------------------------------------------------------------ encoder
    def time_mask(self, mask: torch.Tensor, spans: int = 2) -> torch.Tensor:
        """SpecAugment-style time masking of the segmental stream: per utterance, ``spans`` random spans of up
        to ``speech_mask`` of its length are hidden from the encoder (the codec still has to reconstruct them).
        mask [B, T] valid frames -> keep [B, T] in {0, 1}."""
        B, T = mask.shape
        n = mask.sum(1, keepdim=True).float()
        pos = torch.arange(T, device=mask.device).unsqueeze(0)
        keep = torch.ones(B, T, device=mask.device)
        for _ in range(spans):
            width = torch.rand(B, 1, device=mask.device) * self.cfg.speech_mask * n
            start = torch.rand(B, 1, device=mask.device) * (n - width).clamp_min(0)
            keep = keep * ((pos < start) | (pos >= start + width)).float()
        return keep

    def encode(self, sem: torch.Tensor, tone: torch.Tensor, mask: torch.Tensor, lang: torch.Tensor,
               sem_is_normed: bool = False, gate_override: Optional[float] = None,
               keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``keep`` [B, T] in {0, 1}: frames of the segmental stream the encoder may see (the others are replaced by
        the feature mean); used to encode time-masked versions of an utterance."""
        x = sem if sem_is_normed else self.norm_sem(sem)
        if keep is not None:
            x = x * keep.unsqueeze(-1).to(x.dtype)
        elif self.training and self.cfg.speech_mask > 0:
            x = x * self.time_mask(mask).unsqueeze(-1).to(x.dtype)   # masked frames = the feature mean
        h = self.sem_in(x) + self.aux_in(tone[..., [1, 3]])
        g = self.gate(lang) if gate_override is None else torch.full((x.shape[0],), gate_override, device=x.device)
        if self.cfg.tone_mode != "off" or gate_override is not None:
            tau = self.tone_net(tone[..., [0, 2]].transpose(1, 2)).transpose(1, 2)
            g = g.view(-1, 1, 1).to(h.dtype)
            # both FiLM terms are bounded by tanh, so gamma is an upper bound on what pitch can contribute;
            # with an unbounded shift the network could cancel a shrinking gate by growing its weights
            h = h * (1 + g * torch.tanh(self.film_scale(tau))) + g * torch.tanh(self.film_shift(tau))
        h = h + sincos(torch.arange(h.shape[1], device=h.device), h.shape[-1]) + self.lang_emb(lang).unsqueeze(1)
        for blk in self.enc:
            h = blk(h, mask=mask)
        if self.mode == "all":
            z = standardize_tokens(self.resampler(h, mask, self.span_means(h, mask, self.n_tokens)))
        else:
            z = standardize_tokens(self.resampler(h, mask))                   # content tokens: free queries
        if self.mode == "extra":                                               # + how it is said, laid out in time
            e = self.speak_feat(torch.cat([x, tone.to(x.dtype)], -1))
            e = e + sincos(torch.arange(e.shape[1], device=e.device), e.shape[-1]) + self.speak_lang(lang).unsqueeze(1)
            for blk in self.speak_enc:
                e = blk(e, mask=mask)
            z = torch.cat([z, standardize_tokens(self.time_resampler(e, mask, self.span_means(e, mask, self.n_ordered)))], 1)
        if self.n_listen:   # + what the hub listens to: the encoder's own states, span by span (neither decoded nor written)
            z = torch.cat([z, standardize_tokens(self.listen_out(self.span_means(h, mask, self.n_listen)))], 1)
        return z

    # ------------------------------------------------------------------ tokens that keep the order of time
    # Speech, unlike a caption's bytes, is reconstructed by regression, and the regression has a comfortable
    # resting place: every latent token the utterance mean, every frame decoded to that mean.  On real speech
    # that is where a set of free queries stays (measured on Flickr8k: reconstructions constant in time, written
    # speech a hum).  A token laid out in time starts from the mean of its own span of the utterance, and each
    # frame of the decoder starts from the ordered tokens around its own position, so the order of time is in
    # the latent by construction and attention only has to add detail.
    #
    # Which tokens?  Laying out every token ("all") gives a codec that reconstructs and a hub that can no longer
    # find what the utterance is about (SynthWorld: island speech -> text 5 % against 77 %): free queries had
    # been putting each thing that is said into the same token wherever it is said.  Adding ordered tokens to
    # the free ones and training them together loses the same thing more slowly, because the ordered tokens then
    # do the reconstruction and nothing puts the content into the free ones.  So ("extra") the free content
    # tokens and their decoder are left exactly as they are - they are all the hub reads, grounds and explains -
    # and a SPEAKING PATH is added beside them: time-ordered tokens from a small encoder of their own (the content
    # encoder, trained for the gist, keeps too little of the frames: measured), and a second decoder that reads
    # content and ordered tokens with a stop-gradient on the content.  What is said is one objective, how it is
    # said is another, and the second cannot disturb the first.
    @staticmethod
    def span_means(h: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
        """Masked mean of the frames over n equal spans of each utterance's own length: [B, T, W] -> [B, n, W]."""
        T = h.shape[1]
        length = mask.sum(1, keepdim=True).clamp_min(1)                                    # [B, 1]
        span = (torch.arange(T, device=h.device).unsqueeze(0) * n // length).clamp_max(n - 1)   # [B, T]
        w = F.one_hot(span, n).to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)                 # [B, T, n]
        return torch.einsum("btn,btw->bnw", w, h) / w.sum(1).clamp_min(1).unsqueeze(-1)

    @staticmethod
    def span_upsample(ctx: torch.Tensor, n_frames: torch.Tensor, T: int) -> torch.Tensor:
        """The inverse layout: every frame gets the linear interpolation of the tokens around its own position.
        [B, n, W] -> [B, T, W]"""
        n, W = ctx.shape[1], ctx.shape[2]
        pos = torch.arange(T, device=ctx.device).unsqueeze(0).float()
        u = ((pos + 0.5) / n_frames.clamp_min(1).unsqueeze(1).float() * n - 0.5).clamp(0, n - 1)   # [B, T]
        lo = u.floor().long()
        hi = (lo + 1).clamp_max(n - 1)
        a = (u - lo.float()).unsqueeze(-1).to(ctx.dtype)
        take = lambda i: ctx.gather(1, i.unsqueeze(-1).expand(-1, -1, W))
        return (1 - a) * take(lo) + a * take(hi)

    # ------------------------------------------------------------------ decoder
    def _frame_queries(self, n_frames: torch.Tensor, T: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(T, device=device).unsqueeze(0)
        mask = pos < n_frames.unsqueeze(1)
        rel = pos.float() / n_frames.clamp_min(1).unsqueeze(1).float() * 1000.0
        q = self.q_proj(torch.cat([sincos(pos.expand(len(n_frames), -1), W), sincos(rel, W)], -1))
        return q, mask

    def decode(self, z: torch.Tensor, lang: torch.Tensor, n_frames: torch.Tensor, bypass: torch.Tensor,
               T: Optional[int] = None, content_only: bool = False) -> Dict[str, torch.Tensor]:
        """latent -> normalised segmental features [B, T, D_s] and prosody [B, T, 3].
        With a speaking path (audio_anchor = extra) this is the speaking decoder, which reads all tokens;
        ``content_only`` gives the content decoder instead, which reads the content tokens alone."""
        T = T or int(n_frames.max())
        z = z[:, : self.n_tokens]   # (listening tokens, if the latent carries them, are not for a decoder)
        if self.mode == "extra" and not content_only:
            return self._speak(z, n_frames, bypass, T)
        z = z[:, : self.n_content]
        W = self.lat_in.out_features
        byp = torch.tanh(self.byp_gate) * self.byp_in(bypass).unsqueeze(1)
        q, mask = self._frame_queries(n_frames, T, W, z.device)
        ctx = self.lat_in(z) + self.lat_pos
        h = q + byp
        up = self.span_upsample(ctx, n_frames, T) if self.mode == "all" else None
        if up is not None:
            h = h + up
        for blk in self.dec:
            h = blk(h, mask=mask, ctx=ctx)
        sem = self.sem_out(self.dec_norm(h))
        # prosody head: stop-gradient on the latent (it must not force pitch into the latent)
        ctx_p = self.lat_in(z.detach()).detach() + self.lat_pos.detach()
        hp = q.detach() + self.byp_in(bypass).unsqueeze(1)
        if up is not None:
            hp = hp + up.detach()
        for blk in self.pros_dec:
            hp = blk(hp, mask=mask, ctx=ctx_p)
        return {"sem": sem, "pros": self.pros_out(hp), "mask": mask}

    def _speak(self, z: torch.Tensor, n_frames: torch.Tensor, bypass: torch.Tensor, T: int) -> Dict[str, torch.Tensor]:
        """The speaking decoder: every frame starts from the time-ordered tokens around it and attends to all
        tokens.  The content tokens and the bypass code come in with a stop-gradient."""
        W = self.speak_in.out_features
        z = torch.cat([z[:, : self.n_content].detach(), z[:, self.n_content: self.n_tokens]], 1)
        pos = torch.arange(T, device=z.device).unsqueeze(0)
        mask = pos < n_frames.unsqueeze(1)
        rel = pos.float() / n_frames.clamp_min(1).unsqueeze(1).float() * 1000.0
        q = self.speak_q(torch.cat([sincos(pos.expand(len(n_frames), -1), W), sincos(rel, W)], -1))
        ctx = self.speak_in(z) + self.speak_pos
        h = q + self.speak_byp(bypass.detach()).unsqueeze(1) + self.span_upsample(ctx[:, self.n_content:], n_frames, T)
        for blk in self.speak_dec:
            h = blk(h, mask=mask, ctx=ctx)
        h = self.speak_norm(h)
        # prosody: a head of its own on a stop-gradient.  Trained on the decoder's own states, the three prosody
        # objectives slow the reconstruction of the segmental stream down (real speech, 300 steps: 5 % of the
        # frame variance beyond the mean with them, 14 % without)
        hp = self.speak_pros_dec(h.detach(), mask=mask, ctx=ctx.detach())
        return {"sem": self.speak_out(h), "pros": self.speak_pros(hp), "mask": mask}

    def predict_frames(self, z: torch.Tensor) -> torch.Tensor:
        n = torch.exp(self.dur_head(z[:, : self.n_tokens].mean(1)).squeeze(-1) + math.log(50.0))
        return n.round().clamp(4, int(self.cfg.max_audio_sec * 25)).long()

    @staticmethod
    def stream_losses(out: Dict[str, torch.Tensor], sem: torch.Tensor, tone: torch.Tensor,
                      mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segmental MSE, contour L1 (voiced frames), voicing BCE, energy L1 of decoded streams against the real
        ones (``sem`` normalised)."""
        m = mask.unsqueeze(-1).float()
        denom = m.sum().clamp_min(1)
        voiced = tone[..., 1:2]
        sem_ = (((out["sem"] - sem) ** 2) * m).sum() / denom / sem.shape[-1]
        f0_ = ((out["pros"][..., 0:1] - tone[..., 0:1]).abs() * voiced * m).sum() / (voiced * m).sum().clamp_min(1)
        voi_ = (F.binary_cross_entropy_with_logits(out["pros"][..., 1:2].float(), voiced, reduction="none") * m).sum() / denom
        en_ = ((out["pros"][..., 2:3] - tone[..., 3:4]).abs() * m).sum() / denom
        return sem_, f0_, voi_, en_

    def speaker(self) -> Optional[nn.Module]:
        """The speaking decoder as one module (a view on the same parameters, as ``TextCodec.decoder``): the
        transport phase trains it on written latents while everything else of the codec stays frozen."""
        if self.mode != "extra":
            return None
        view = nn.Module()
        for name in ("speak_in", "speak_q", "speak_byp", "speak_dec", "speak_norm", "speak_out", "speak_pros_dec", "speak_pros"):
            view.add_module(name, getattr(self, name))
        view.register_parameter("speak_pos", self.speak_pos)
        return view

    def speaking_parameters(self) -> List[nn.Parameter]:
        """Every parameter of the speaking path (its encoder, its tokens, its decoder).  No loss connects them to
        the rest of the model, so the trainer clips their gradient on its own: under one global norm the speaking
        path would take a share of the listening path's step whenever its own gradient is large."""
        if self.mode != "extra":
            return []
        return [p for n, p in self.named_parameters() if n.startswith(("speak_", "time_resampler."))]

    def speak_loss(self, z: torch.Tensor, aud: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loss of the speaking decoder on the latents ``z`` (written ones, in the transport phase) against the real
        streams of the same utterances.  -> (loss, segmental MSE)"""
        sem, mask = self.norm_sem(aud["sem"]), aud["mask"]
        out = self._speak(z, mask.sum(1), self.bypass_code(aud["spk"], aud["f0"]), sem.shape[1])
        s_sem, s_f0, s_voi, s_en = self.stream_losses(out, sem, aud["tone"], mask)
        return s_sem + 0.5 * (s_f0 + s_voi + s_en), s_sem

    # ------------------------------------------------------------------ training loss
    def loss(self, aud: Dict[str, torch.Tensor], n_grounded: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """``n_grounded``: the first n utterances of the batch are image-paired (the rest is unpaired speech);
        used by the evidence-proportional gate price."""
        cfg = self.cfg
        sem, tone, mask, lang = self.norm_sem(aud["sem"]), aud["tone"], aud["mask"], aud["lang"]
        z = self.encode(sem, tone, mask, lang, sem_is_normed=True)
        zin = z
        if self.training and cfg.latent_noise_max > 0:
            sigma = torch.rand(z.shape[0], 1, 1, device=z.device) * cfg.latent_noise_max
            sigma = sigma * (torch.rand_like(sigma) < 0.7)
            zin = z + sigma * torch.randn_like(z)
        n_frames = mask.sum(1)
        p = self.bypass_code(aud["spk"], aud["f0"])
        stream_losses = lambda out: self.stream_losses(out, sem, tone, mask)
        l_sem, l_f0, l_voi, l_en = stream_losses(self.decode(zin, lang, n_frames, p, T=sem.shape[1], content_only=True))
        l_dur = (self.dur_head(z[:, : self.n_tokens].detach().mean(1)).squeeze(-1) - (torch.log(n_frames.float()) - math.log(50.0))).abs().mean()
        res = {"z": z, "sem_mse": l_sem, "f0_l1": l_f0, "dur_l1": l_dur,
               "loss": l_sem + 0.5 * (l_f0 + l_voi + l_en) + 0.5 * l_dur}
        if self.mode == "extra":   # the speaking path: its own objective, no gradient into anything above
            s_sem, s_f0, s_voi, s_en = stream_losses(self._speak(zin, n_frames, p, sem.shape[1]))
            res["speak_mse"] = s_sem
            res["loss"] = res["loss"] + s_sem + 0.5 * (s_f0 + s_voi + s_en)
        # mel renderer (teacher-forced on real streams; independent of the latent)
        if "mel" in aud and cfg.w_mel > 0:
            tone3 = torch.stack([tone[..., 0], tone[..., 1], tone[..., 3]], -1)
            mel_hat = self.renderer(sem, tone3, p.detach(), aud["mel"].shape[1])
            mel = (aud["mel"] - self.mel_mean) / self.mel_std
            mm = (torch.arange(mel.shape[1], device=mel.device).unsqueeze(0) < aud["mel_len"].unsqueeze(1)).unsqueeze(-1).float()
            res["mel_l1"] = ((mel_hat - mel).abs() * mm).sum() / mm.sum().clamp_min(1) / mel.shape[-1]
            res["loss"] = res["loss"] + cfg.w_mel * res["mel_l1"]
        # decorrelate what goes to the hub from what bypasses it
        if cfg.use_bypass and cfg.w_orth > 0 and z.shape[0] > 3:
            res["orth"] = cross_correlation(z[:, : self.n_content].mean(1), p)
            res["loss"] = res["loss"] + cfg.w_orth * res["orth"]
        # pay-for-pitch.  "uniform": every language present in the batch pays for its gate.  "evidence": every
        # *grounded* utterance pays for the gate of its language, so price and benefit (which only grounded
        # utterances can provide) scale with the same count - otherwise a language with few image pairs is
        # priced out whatever its pitch is worth (measured on Flickr8k: the Yoruba gate closed to 0)
        if cfg.tone_mode == "gated" and not self.gate_forced_open:
            if cfg.gate_price == "evidence":
                n = lang.shape[0] if n_grounded is None else n_grounded
                res["gate_pen"] = self.gate(lang[:n]).mean() if n > 0 else z.new_zeros(())
            else:
                res["gate_pen"] = self.gate()[lang.unique()].mean()
            res["loss"] = res["loss"] + cfg.w_gate * res["gate_pen"]
        return res

    # ------------------------------------------------------------------ synthesis
    @torch.no_grad()
    def synthesize_mel(self, z: torch.Tensor, lang: torch.Tensor, spk: torch.Tensor, f0: torch.Tensor,
                       n_frames: Optional[torch.Tensor] = None, content_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """latent (+ a reference speaker for the bypass) -> log-mel [B, T_mel, n_mels], frame counts, streams.
        ``content_only``: rendered by the content decoder (what the model says without its speaking path)."""
        n_frames = n_frames if n_frames is not None else self.predict_frames(z)
        p = self.bypass_code(spk, f0)
        out = self.decode(z, lang, n_frames, p, content_only=content_only)
        pros = out["pros"]
        voiced = (torch.sigmoid(pros[..., 1]) > 0.5).float()
        tone3 = torch.stack([pros[..., 0] * voiced, voiced, pros[..., 2]], -1)
        ratio = self.cfg.mel_fps / 25.0
        mel = self.renderer(out["sem"], tone3, p, int(round(out["sem"].shape[1] * ratio))) * self.mel_std + self.mel_mean
        return mel, n_frames, {"sem": out["sem"], "tone3": tone3, "mel_len": torch.round(n_frames.float() * ratio).long()}


    @torch.no_grad()
    def render_real(self, aud: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Log-mel rendered from the REAL streams of an utterance (segmental features, contour, bypass code): what
        the renderer does on its own, with no latent in the way.  -> [B, T_mel, n_mels]"""
        tone = aud["tone"]
        tone3 = torch.stack([tone[..., 0], tone[..., 1], tone[..., 3]], -1)
        n_mel = int(round(aud["sem"].shape[1] * self.cfg.mel_fps / 25.0))
        mel = self.renderer(self.norm_sem(aud["sem"]), tone3, self.bypass_code(aud["spk"], aud["f0"]), n_mel)
        return mel * self.mel_std + self.mel_mean


def cross_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared entry of the batch cross-correlation matrix between two sets of features (the
    orthogonality penalty L_orth: zero iff no linear dependence between hub-bound and bypass codes)."""
    a = (a.float() - a.float().mean(0)) / (a.float().std(0) + 1e-4)
    b = (b.float() - b.float().mean(0)) / (b.float().std(0) + 1e-4)
    c = a.t() @ b / a.shape[0]
    return (c ** 2).mean()
