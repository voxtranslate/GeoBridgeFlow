"""Dependency-light audio DSP: mel / MFCC, YIN pitch tracking, tone features, Griffin-Lim.

The *tone stream* of GeoBridge-Flow is built here.  It is deliberately tiny and interpretable:

    tone[t] = ( z-scored log-F0 , voiced flag , delta log-F0 , z-scored log-energy )

log-F0 is standardised **per utterance over voiced frames**, which removes the speaker's pitch register
(non-lexical in every language) and keeps the relative contour (lexical in tonal languages).  The removed
register (mean, std of log-F0) travels in the paralinguistic bypass instead.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio.functional as AF
except Exception:  # pragma: no cover - a plain filterbank and resampler are used instead
    AF = None

_FB_CACHE: Dict[Tuple, torch.Tensor] = {}


def _mel_fb(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        if AF is not None:
            fb = AF.melscale_fbanks(n_fft // 2 + 1, 0.0, sr / 2, n_mels, sr, norm="slaney", mel_scale="slaney")
        else:  # pragma: no cover - plain triangular HTK filterbank
            f = torch.linspace(0, sr / 2, n_fft // 2 + 1)
            m = torch.linspace(0, 2595 * math.log10(1 + sr / 2 / 700), n_mels + 2)
            hz = 700 * (10 ** (m / 2595) - 1)
            fb = torch.zeros(n_fft // 2 + 1, n_mels)
            for i in range(n_mels):
                lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
                fb[:, i] = torch.clamp(torch.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), min=0)
        _FB_CACHE[key] = fb
    return _FB_CACHE[key]


def stft_mag(wav: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """wav [T] -> magnitude [frames, n_fft/2+1] (centered, Hann)."""
    win = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True,
                      pad_mode="reflect", return_complex=True)
    return spec.abs().transpose(0, 1)


def log_mel(wav: torch.Tensor, sr: int = 16000, n_fft: int = 1024, hop: int = 320, n_mels: int = 80) -> torch.Tensor:
    mag = stft_mag(wav, n_fft, hop)
    mel = mag @ _mel_fb(sr, n_fft, n_mels).to(mag.device)
    return torch.log(mel.clamp_min(1e-5))


def mfcc(wav: torch.Tensor, sr: int = 16000, n_mfcc: int = 13, hop: int = 160, n_fft: int = 400,
         n_mels: int = 40) -> torch.Tensor:
    """MFCC + deltas [frames, 2*n_mfcc].  Low-order cepstra are largely pitch-invariant, which makes them a
    faithful stand-in for the 'semantic but tone-poor' deep SSL layer in the synthetic benchmark."""
    lm = log_mel(wav, sr, n_fft, hop, n_mels)
    n = lm.shape[1]
    k = torch.arange(n_mfcc).unsqueeze(1).float()
    i = torch.arange(n).unsqueeze(0).float()
    dct = torch.cos(math.pi / n * (i + 0.5) * k) * math.sqrt(2.0 / n)
    c = lm @ dct.t()
    c = c - c.mean(0, keepdim=True)  # cepstral mean normalisation
    d = torch.zeros_like(c)
    if c.shape[0] > 2:
        d[1:-1] = (c[2:] - c[:-2]) / 2
    return torch.cat([c, d], dim=1)


# ---------------------------------------------------------------------------------------------- YIN
def yin_f0(wav: np.ndarray, sr: int = 16000, hop: int = 160, frame: int = 1024, fmin: float = 60.0,
           fmax: float = 500.0, thresh: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised YIN (de Cheveigne & Kawahara, 2002).  Returns (f0 [frames] in Hz, 0 = unvoiced; aperiodicity)."""
    wav = np.asarray(wav, dtype=np.float64)
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    pad = frame // 2
    x = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    fr = x[idx]
    tau_min, tau_max = max(2, int(sr / fmax)), min(frame // 2 - 1, int(sr / fmin))
    w = frame - tau_max  # integration window
    # difference function d(tau) = sum_j (x_j - x_{j+tau})^2 over j < w, via FFT cross-correlation
    nfft = 1 << int(np.ceil(np.log2(frame + w)))
    a = np.fft.rfft(fr[:, :w], nfft)
    b = np.fft.rfft(fr, nfft)
    corr = np.fft.irfft(np.conj(a) * b, nfft)[:, : tau_max + 1]
    cs = np.cumsum(np.concatenate([np.zeros((n_frames, 1)), fr ** 2], axis=1), axis=1)
    taus = np.arange(tau_max + 1)
    e0 = cs[:, w:w + 1] - cs[:, 0:1]
    et = cs[:, taus + w] - cs[:, taus]
    d = np.maximum(e0 + et - 2 * corr, 0.0)
    cmnd = np.ones_like(d)
    csum = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * taus[1:][None, :] / np.maximum(csum, 1e-12)
    seg = cmnd[:, tau_min: tau_max + 1]
    below = seg < thresh
    # first tau under the threshold, then walk to the local minimum; otherwise the global minimum
    first = np.where(below.any(1), below.argmax(1), seg.argmin(1))
    tau = first.copy()
    for _ in range(64):
        nxt = np.minimum(tau + 1, seg.shape[1] - 1)
        better = seg[np.arange(n_frames), nxt] < seg[np.arange(n_frames), tau]
        if not better.any():
            break
        tau = np.where(better, nxt, tau)
    ap = seg[np.arange(n_frames), tau]
    t = tau + tau_min
    tm, tp = np.clip(t - 1, 0, tau_max), np.clip(t + 1, 0, tau_max)
    y0, y1, y2 = cmnd[np.arange(n_frames), tm], cmnd[np.arange(n_frames), t], cmnd[np.arange(n_frames), tp]
    denom = y0 - 2 * y1 + y2
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(denom) > 1e-9, denom, 1.0), 0.0)
    f0 = sr / np.maximum(t + np.clip(shift, -1, 1), 1e-6)
    energy = np.sqrt((fr ** 2).mean(1))
    voiced = (ap < max(thresh, 0.25)) & (energy > 0.02 * (energy.max() + 1e-9)) & (f0 >= fmin) & (f0 <= fmax)
    return np.where(voiced, f0, 0.0), ap


def _pool_frames(x: np.ndarray, n: int) -> np.ndarray:
    """Average-pool / interpolate a [T, ...] sequence to exactly n frames."""
    if len(x) == n:
        return x
    src = np.linspace(0, len(x) - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    wgt = (src - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - wgt) + x[hi] * wgt


def tone_features(wav: np.ndarray, n_frames: int, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (tone [n_frames, 4], f0stats [2] = mean/std of log-F0 over voiced frames)."""
    f0, _ = yin_f0(wav, sr=sr)
    voiced = f0 > 0
    hop = 160
    pad = 512
    x = np.pad(np.asarray(wav, dtype=np.float64), (pad, pad))
    n = len(f0)
    idx = np.arange(1024)[None, :] + hop * np.arange(n)[:, None]
    idx = np.minimum(idx, len(x) - 1)
    loge = np.log(np.sqrt((x[idx] ** 2).mean(1)) + 1e-5)
    if voiced.sum() >= 3:
        lf = np.log(f0[voiced])
        mu, sd = float(lf.mean()), float(lf.std() + 1e-3)
    else:
        mu, sd = math.log(150.0), 0.2
    z = np.zeros(n)
    z[voiced] = (np.log(f0[voiced]) - mu) / sd
    # fill unvoiced gaps by linear interpolation so the delta is meaningful, then mask again
    if voiced.sum() >= 2:
        zi = np.interp(np.arange(n), np.flatnonzero(voiced), z[voiced])
    else:
        zi = z
    dz = np.gradient(zi) if n > 1 else np.zeros(n)
    ez = (loge - loge.mean()) / (loge.std() + 1e-3)
    feats = np.stack([z, voiced.astype(np.float64), dz * voiced, ez], axis=1)
    out = _pool_frames(feats, n_frames)
    out[:, 1] = (out[:, 1] > 0.5).astype(np.float64)
    out[:, 0] *= out[:, 1]
    out[:, 2] *= out[:, 1]
    return out.astype(np.float32), np.array([mu, sd], dtype=np.float32)


def flatten_tone(tone: torch.Tensor) -> torch.Tensor:
    """Counterfactual used by the tone-reliance probes: remove the pitch contour, keep voicing and energy."""
    out = tone.clone()
    out[..., 0] = 0.0
    out[..., 2] = 0.0
    return out


def trim_silence(wav: np.ndarray, top_db: float = 35.0, sr: int = 16000, pad_ms: float = 60.0) -> np.ndarray:
    """Trim leading / trailing silence.  The threshold adapts to the noise floor of the clip (10th percentile
    + 10 dB, never above -12 dB), so padded noisy recordings are trimmed and continuous speech is left alone."""
    if top_db <= 0 or len(wav) < sr // 10:
        return wav
    hop, win = sr // 100, sr // 40
    n = 1 + max(0, len(wav) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((wav[idx] ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / (np.percentile(rms, 99) + 1e-12) + 1e-12)
    thr = max(-top_db, min(float(np.percentile(db, 10)) + 10.0, -12.0))
    act = np.convolve((db > thr).astype(np.float64), np.ones(15) / 15, mode="same") > 0.5  # ignore clicks
    keep = np.flatnonzero(act)
    if len(keep) == 0:
        return wav
    pad = int(pad_ms / 1000 * sr)
    a, b = max(0, keep[0] * hop - pad), min(len(wav), keep[-1] * hop + win + pad)
    return wav[a:b]


def highpass(wav: np.ndarray, cutoff_hz: float, sr: int = 16000, ramp_hz: float = 20.0) -> np.ndarray:
    """Zero-phase high-pass: nothing below ``cutoff - ramp``, everything above ``cutoff``, a raised cosine between.
    Field recordings can carry most of their energy as rumble below the lowest voice pitch (YFACC Yoruba: half of
    it below 80 Hz).  That rumble hides the pauses from the silence trimmer and the periodicity from the pitch
    tracker, and it is what an utterance-normalising speech encoder then normalises by."""
    if cutoff_hz <= 0 or len(wav) < 32:
        return wav
    spec = np.fft.rfft(np.asarray(wav, dtype=np.float64))
    f = np.fft.rfftfreq(len(wav), 1.0 / sr)
    g = 0.5 - 0.5 * np.cos(np.pi * np.clip((f - (cutoff_hz - ramp_hz)) / ramp_hz, 0.0, 1.0))
    return np.fft.irfft(spec * g, len(wav)).astype(np.float32)


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if AF is not None:
        return AF.resample(t, sr_in, sr_out).numpy()
    n = int(round(len(wav) * sr_out / sr_in))  # pragma: no cover
    return F.interpolate(t[None, None], size=n, mode="linear", align_corners=False)[0, 0].numpy()


def griffin_lim(logmel: torch.Tensor, sr: int = 16000, n_fft: int = 1024, hop: int = 320, n_iter: int = 48) -> torch.Tensor:
    """log-mel [frames, n_mels] -> waveform [T] (pseudo-inverse mel + Griffin-Lim; no learned vocoder)."""
    fb = _mel_fb(sr, n_fft, logmel.shape[1]).to(logmel.device)
    mag = torch.clamp(torch.exp(logmel) @ torch.linalg.pinv(fb), min=1e-5).transpose(0, 1)  # [F, frames]
    win = torch.hann_window(n_fft, device=logmel.device)
    angles = torch.exp(2j * math.pi * torch.rand(mag.shape, device=logmel.device))
    wav: Optional[torch.Tensor] = None
    for _ in range(n_iter):
        wav = torch.istft(mag * angles, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True)
        spec = torch.stft(wav, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True,
                          pad_mode="reflect", return_complex=True)
        m = min(spec.shape[1], mag.shape[1])
        angles = torch.ones_like(mag, dtype=spec.dtype)
        angles[:, :m] = spec[:, :m] / spec[:, :m].abs().clamp_min(1e-8)
    assert wav is not None
    return wav / wav.abs().max().clamp_min(1e-5) * 0.95
