"""Frozen front-ends and the one-time feature cache.

Everything expensive (vision tower, speech SSL model, pitch tracking, mel) runs exactly once per dataset and
is stored as memory-mapped arrays.  Extraction is chunked by source file, so an interrupted run resumes from
the last finished chunk instead of starting again.

Cache layout (``<data_root>/cache/<dataset_tag>/``)
    manifest.json, splits.json
    vision_<tag>/tokens.npy          [n_images, P+1, D_v]  float16   (last token = global)
    audio_<tag>/sem.npy              [frames, D_s]         float16   deep-layer ("semantic") stream, 25 Hz
    audio_<tag>/tone.npy             [frames, 4]           float32   tone stream (see audio_dsp.tone_features)
    audio_<tag>/mel.npy              [2*frames, n_mels]    float16   50 fps log-mel for the renderer
    audio_<tag>/spk.npy, f0stats.npy [n_utts, D_spk], [n_utts, 2]    paralinguistic bypass inputs
    audio_<tag>/index.json           uid -> [row, frame offset, n_frames]
    audio_<tag>/stats.json           per-dimension mean / std of every stream (train split)
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import audio_dsp as dsp
from ..utils import load_json, save_json


def _tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


# ================================================================================== vision
class VisionFrontend:
    """Any Hugging Face vision tower -> (grid x grid pooled patch tokens, global token)."""

    def __init__(self, name: str, grid: int, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.grid, self.device = grid, device
        self.proc = AutoImageProcessor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        self.model = getattr(model, "vision_model", model).to(device).eval()  # SigLIP/CLIP: keep the vision tower
        self.n_prefix: Optional[int] = None

    @torch.no_grad()
    def __call__(self, images: List) -> np.ndarray:
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixel_values=px)
        h = out.last_hidden_state
        side = int(round((h.shape[1]) ** 0.5))
        if side * side == h.shape[1]:  # SigLIP: no CLS; pooled head output is the global token
            patches = h
            glob = out.pooler_output if getattr(out, "pooler_output", None) is not None else h.mean(1)
        else:  # ViT / DINOv2 / CLIP: CLS (+ registers) first
            n_patch = (int((h.shape[1]) ** 0.5)) ** 2
            patches, glob = h[:, h.shape[1] - n_patch:], h[:, 0]
            side = int(round(n_patch ** 0.5))
        g = patches.transpose(1, 2).reshape(patches.shape[0], -1, side, side)
        g = F.adaptive_avg_pool2d(g.float(), self.grid).flatten(2).transpose(1, 2)
        return torch.cat([g, glob.float().unsqueeze(1)], 1).cpu().numpy()


def cache_vision(cfg, builder, manifest: Dict, device: torch.device, logger) -> str:
    out = os.path.join(cfg.cache_dir, f"vision_{_tag(cfg.vision_encoder)}_g{cfg.vision_grid}")
    meta_p = os.path.join(out, "meta.json")
    if os.path.exists(meta_p):
        return out
    os.makedirs(out, exist_ok=True)
    n = len(manifest["images"])
    if cfg.vision_encoder == "synthetic":
        toks = np.stack([builder.vision_tokens(i) for i in range(n)]).astype(np.float16)
        np.save(os.path.join(out, "tokens.npy"), toks)
    else:
        fe = VisionFrontend(cfg.vision_encoder, cfg.vision_grid, device)
        arr = None
        for grp in tqdm(builder.image_groups(manifest["images"]), desc="vision features", dynamic_ncols=True):
            part_p = os.path.join(out, f"part_{_tag(grp['key'])}.npz")
            if os.path.exists(part_p):
                continue
            idx, feats, buf, buf_idx = [], [], [], []
            for i, img in grp["loader"]():
                buf.append(img)
                buf_idx.append(i)
                if len(buf) == cfg.feat_batch_size:
                    feats.append(fe(buf)); idx += buf_idx; buf, buf_idx = [], []
            if buf:
                feats.append(fe(buf)); idx += buf_idx
            np.savez(part_p, idx=np.array(idx), feats=np.concatenate(feats).astype(np.float16))
        for fp in sorted(p for p in os.listdir(out) if p.startswith("part_")):
            z = np.load(os.path.join(out, fp))
            if arr is None:
                arr = np.zeros((n, *z["feats"].shape[1:]), dtype=np.float16)
            arr[z["idx"]] = z["feats"]
        np.save(os.path.join(out, "tokens.npy"), arr)
        for fp in [p for p in os.listdir(out) if p.startswith("part_")]:
            os.remove(os.path.join(out, fp))
        toks = arr
    save_json({"n": n, "tokens": int(toks.shape[1]), "dim": int(toks.shape[2]), "encoder": cfg.vision_encoder}, meta_p)
    logger.info(f"[cache] vision tokens {tuple(toks.shape)} -> {out}")
    return out


def fit_anchor(tokens: np.ndarray, train_idx: np.ndarray, d: int, path: str) -> Dict[str, torch.Tensor]:
    """Frozen anchor projection: PCA to d dims with ONE global scale, i.e. an isometry (up to scale) of the
    vision geometry restricted to its top-d subspace.  Fitted once in float64 via eigh of the covariance,
    never touched by training."""
    if os.path.exists(path):
        return torch.load(path)
    rng = np.random.default_rng(0)
    sel = train_idx if len(train_idx) <= 4000 else rng.choice(train_idx, 4000, replace=False)
    x = np.asarray(tokens[np.sort(sel)], dtype=np.float64).reshape(-1, tokens.shape[-1])
    mean = x.mean(0)
    cov = np.cov((x - mean).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    k = min(d, v.shape[1])
    basis = np.zeros((v.shape[1], d))
    basis[:, :k] = v[:, :k]
    scale = 1.0 / np.sqrt(max(w[:k].mean(), 1e-8))
    out = {"mean": torch.tensor(mean, dtype=torch.float32), "basis": torch.tensor(basis, dtype=torch.float32),
           "scale": torch.tensor(scale, dtype=torch.float32),
           "explained": torch.tensor(float(w[:k].sum() / max(w.sum(), 1e-12)))}
    torch.save(out, path)
    return out


# ================================================================================== speech
class MFCCFrontend:
    """No-download front-end (synthetic benchmark, unit tests)."""

    dim, spk_dim = 26, 40

    def __init__(self, cfg) -> None:
        self.pool = 4  # 100 fps -> 25 fps

    def __call__(self, wavs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        out = []
        for w in wavs:
            t = torch.from_numpy(w).float()
            m = dsp.mfcc(t)
            n = max(1, m.shape[0] // self.pool)
            sem = m[: n * self.pool].reshape(n, self.pool, -1).mean(1) if m.shape[0] >= self.pool else m.mean(0, keepdim=True)
            spk = dsp.log_mel(t, n_fft=400, hop=160, n_mels=40).mean(0)
            out.append((sem.numpy(), spk.numpy()))
        return out


class SSLFrontend:
    """HuBERT / wav2vec2-family encoder.  ``sem`` = mean of deep hidden states (pooled to 25 Hz),
    ``spk`` = time-average of early hidden states."""

    def __init__(self, cfg, device: torch.device) -> None:
        from transformers import AutoConfig, AutoModel

        self.cfg, self.device, self.pool = cfg, device, cfg.speech_pool
        self.model = AutoModel.from_pretrained(cfg.speech_encoder).to(device).eval()
        conf = AutoConfig.from_pretrained(cfg.speech_encoder)
        self.dim = self.spk_dim = conf.hidden_size
        self.normalize = getattr(conf, "feat_extract_norm", "group") == "layer"
        try:
            from transformers import AutoFeatureExtractor

            self.normalize = bool(AutoFeatureExtractor.from_pretrained(cfg.speech_encoder).do_normalize)
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self, wavs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        lens = [len(w) for w in wavs]
        x = torch.zeros(len(wavs), max(max(lens), 400))
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w).float()
            if self.normalize:
                t = (t - t.mean()) / (t.std() + 1e-7)
            x[i, : len(w)] = t
        hs = self.model(x.to(self.device), output_hidden_states=True).hidden_states
        sem = torch.stack([hs[l] for l in self.cfg.speech_sem_layers]).mean(0).float()
        early = torch.stack([hs[l] for l in self.cfg.speech_spk_layers]).mean(0).float()
        fl = self.model._get_feat_extract_output_lengths(torch.tensor(lens)).tolist()
        out = []
        for i, n in enumerate(fl):
            n = max(1, min(int(n), sem.shape[1]))
            s = sem[i, :n]
            m = max(1, n // self.pool)
            s = s[: m * self.pool].reshape(m, self.pool, -1).mean(1) if n >= self.pool else s.mean(0, keepdim=True)
            out.append((s.cpu().numpy(), early[i, :n].mean(0).cpu().numpy()))
        return out


def speech_tag(cfg) -> str:
    if cfg.speech_encoder == "mfcc":
        return "mfcc"
    return f"{_tag(cfg.speech_encoder)}_L{'-'.join(map(str, cfg.speech_sem_layers))}_p{cfg.speech_pool}"


def cache_audio(cfg, builder, manifest: Dict, splits: Dict, device: torch.device, logger) -> str:
    out = os.path.join(cfg.cache_dir, f"audio_{speech_tag(cfg)}")
    if os.path.exists(os.path.join(out, "index.json")):
        return out
    os.makedirs(os.path.join(out, "chunks"), exist_ok=True)
    utts = [u for u in manifest["utts"] if u["has_audio"]]
    groups = builder.audio_groups(utts)
    fe = None
    max_len = int(cfg.max_audio_sec * cfg.sample_rate)
    pbar = tqdm(total=sum(len(g["uids"]) for g in groups), desc="speech features", dynamic_ncols=True)
    for grp in groups:
        cp = os.path.join(out, "chunks", _tag(grp["key"]) + ".npz")
        if os.path.exists(cp):
            pbar.update(len(grp["uids"]))
            continue
        if fe is None:
            fe = MFCCFrontend(cfg) if cfg.speech_encoder == "mfcc" else SSLFrontend(cfg, device)
        uids, sems, tones, mels, spks, f0s, lens = [], [], [], [], [], [], []
        buf: List[Tuple[str, np.ndarray]] = []

        def flush() -> None:
            if not buf:
                return
            order = np.argsort([len(w) for _, w in buf])
            feats = fe([buf[i][1] for i in order])
            for i, (sem, spk) in zip(order, feats):
                uid, wav = buf[i]
                n = sem.shape[0]
                tone, f0s_ = dsp.tone_features(wav, n, cfg.sample_rate)
                mel = dsp.log_mel(torch.from_numpy(wav).float(), cfg.sample_rate, 1024, cfg.mel_hop, cfg.n_mels).numpy()
                mel = dsp._pool_frames(mel, 2 * n) if cfg.cache_mel else np.zeros((0, cfg.n_mels), np.float32)
                uids.append(uid); sems.append(sem.astype(np.float16)); tones.append(tone)
                mels.append(mel.astype(np.float16)); spks.append(spk.astype(np.float32)); f0s.append(f0s_); lens.append(n)
            buf.clear()

        for uid, wav, sr in grp["loader"]():
            wav = dsp.resample(wav, sr, cfg.sample_rate)
            wav = dsp.trim_silence(wav, cfg.trim_silence_db, cfg.sample_rate)[:max_len]
            if len(wav) < cfg.sample_rate // 10:
                wav = np.pad(wav, (0, cfg.sample_rate // 10 - len(wav)))
            buf.append((uid, np.ascontiguousarray(wav, dtype=np.float32)))
            if len(buf) >= cfg.feat_batch_size:
                flush()
            pbar.update(1)
        flush()
        np.savez(cp + ".tmp.npz", uids=np.array(uids), lens=np.array(lens), sem=np.concatenate(sems),
                 tone=np.concatenate(tones), mel=np.concatenate(mels), spk=np.stack(spks), f0=np.stack(f0s))
        os.replace(cp + ".tmp.npz", cp)
    pbar.close()

    # ---- consolidate chunks into memory-mapped arrays
    chunk_files = sorted(os.listdir(os.path.join(out, "chunks")))
    zs = [np.load(os.path.join(out, "chunks", c)) for c in chunk_files]
    total = int(sum(z["lens"].sum() for z in zs))
    n_utt = int(sum(len(z["lens"]) for z in zs))
    d_s, d_spk = zs[0]["sem"].shape[1], zs[0]["spk"].shape[1]
    sem = np.lib.format.open_memmap(os.path.join(out, "sem.npy"), "w+", np.float16, (total, d_s))
    tone = np.lib.format.open_memmap(os.path.join(out, "tone.npy"), "w+", np.float32, (total, 4))
    mel = np.lib.format.open_memmap(os.path.join(out, "mel.npy"), "w+", np.float16,
                                    (2 * total if cfg.cache_mel else 0, cfg.n_mels))
    spk, f0 = np.zeros((n_utt, d_spk), np.float32), np.zeros((n_utt, 2), np.float32)
    index: Dict[str, List[int]] = {}
    off = row = 0
    for z in zs:
        m = int(z["lens"].sum())
        sem[off: off + m], tone[off: off + m] = z["sem"], z["tone"]
        if cfg.cache_mel:
            mel[2 * off: 2 * (off + m)] = z["mel"]
        o = off
        for uid, n, s, f in zip(z["uids"].tolist(), z["lens"].tolist(), z["spk"], z["f0"]):
            index[uid] = [row, o, int(n)]
            spk[row], f0[row] = s, f
            o += int(n); row += 1
        off += m
    sem.flush(); tone.flush(); mel.flush()
    np.save(os.path.join(out, "spk.npy"), spk)
    np.save(os.path.join(out, "f0stats.npy"), f0)

    # ---- normalisation statistics from the training split only
    img_split = splits["images"]
    lang_of = {u["uid"]: (u["lang"], manifest["images"][u["img"]]["id"]) for u in utts}
    tr_rows = [v for k, v in index.items() if img_split[lang_of[k][1]] == "train"]
    rng = np.random.default_rng(0)
    pick = [tr_rows[i] for i in rng.permutation(len(tr_rows))[:3000]] or list(index.values())[:3000]
    xs = np.concatenate([np.asarray(sem[o: o + n], np.float32) for _, o, n in pick])
    stats = {"sem_mean": xs.mean(0), "sem_std": xs.std(0) + 1e-4, "dim": d_s, "spk_dim": d_spk,
             "spk_mean": spk.mean(0), "spk_std": spk.std(0) + 1e-4,
             "f0_mean": f0.mean(0), "f0_std": f0.std(0) + 1e-4, "frame_rate": 25.0}
    if cfg.cache_mel:
        ms = np.concatenate([np.asarray(mel[2 * o: 2 * (o + n)], np.float32) for _, o, n in pick])
        stats.update(mel_mean=float(ms.mean()), mel_std=float(ms.std() + 1e-4))
    else:
        stats.update(mel_mean=0.0, mel_std=1.0)
    save_json(stats, os.path.join(out, "stats.json"))
    save_json(index, os.path.join(out, "index.json"))  # written last: marks the cache as complete
    for c in chunk_files:
        os.remove(os.path.join(out, "chunks", c))
    logger.info(f"[cache] speech: {n_utt} utterances, {total} frames ({total / 25 / 3600:.2f} h) -> {out}")
    return out


class AudioStore:
    """Read-only random access to the cached speech streams."""

    def __init__(self, path: str, use_mel: bool = True) -> None:
        self.index: Dict[str, List[int]] = load_json(os.path.join(path, "index.json"))
        self.stats = {k: (np.array(v, dtype=np.float32) if isinstance(v, list) else v)
                      for k, v in load_json(os.path.join(path, "stats.json")).items()}
        self.sem = np.load(os.path.join(path, "sem.npy"), mmap_mode="r")
        self.tone = np.load(os.path.join(path, "tone.npy"), mmap_mode="r")
        mel_p = os.path.join(path, "mel.npy")
        self.mel = np.load(mel_p, mmap_mode="r") if use_mel and os.path.exists(mel_p) else None
        if self.mel is not None and self.mel.shape[0] == 0:
            self.mel = None
        self.spk = np.load(os.path.join(path, "spk.npy"))
        self.f0 = np.load(os.path.join(path, "f0stats.npy"))

    def get(self, uid: str, max_frames: int) -> Dict[str, np.ndarray]:
        row, off, n = self.index[uid]
        n = min(n, max_frames)
        out = {"sem": np.array(self.sem[off: off + n], dtype=np.float32),
               "tone": np.array(self.tone[off: off + n], dtype=np.float32),
               "spk": self.spk[row], "f0": self.f0[row]}
        if self.mel is not None:
            out["mel"] = np.array(self.mel[2 * off: 2 * (off + n)], dtype=np.float32)
        return out
