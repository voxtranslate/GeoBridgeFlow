"""Frozen front-ends and the one-time feature cache.

Everything expensive (vision tower, speech SSL model, pitch tracking, mel) runs exactly once per dataset and
is stored as memory-mapped arrays.  Extraction is chunked by source file, so an interrupted run resumes from
the last finished chunk instead of starting again.

Cache layout (``<data_root>/cache/<dataset_tag>/``)
    manifest.json, splits.json
    vision_<tag>/tokens.npy          [n_images, P+1, D_v]  float16   (last token = global)
    audio_<tag>/sem.npy              [frames, D_s]         float16   deep-layer ("semantic") stream, 25 Hz
    audio_<tag>/tone.npy             [frames, 4]           float32   tone stream (see audio_dsp.tone_features)
    mel_<spec>_<tag>/mel.npy         [mel frames, n_mels]  float16   renderer targets in the vocoder's feature space
    audio_<tag>/spk.npy, f0stats.npy [n_utts, D_spk], [n_utts, 2]    paralinguistic bypass inputs
    audio_<tag>/index.json           uid -> [row, frame offset, n_frames]
    audio_<tag>/stats.json           per-dimension mean / std of every stream (train split)
"""
from __future__ import annotations

import multiprocessing
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .. import audio_dsp as dsp
from ..utils import gpu_ids, load_json, quiet_transformers, save_json


def _tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def on_all_gpus(module: nn.Module, cfg, device: torch.device) -> nn.Module:
    """Frozen encoder on ``device``; with several GPUs every batch is split across all of them."""
    module = module.to(device).eval().requires_grad_(False)
    ids = gpu_ids(cfg, device)
    return nn.DataParallel(module, device_ids=ids) if len(ids) > 1 else module


# ================================================================================== vision
class _VisionTokens(nn.Module):
    """pixels [B, 3, H, W] -> [B, grid * grid + 1, D]: pooled patch tokens, then the global token."""

    def __init__(self, tower: nn.Module, grid: int) -> None:
        super().__init__()
        self.tower, self.grid = tower, grid

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        out = self.tower(pixel_values=px)
        h = out.last_hidden_state
        side = int(round(h.shape[1] ** 0.5))
        if side * side == h.shape[1]:  # SigLIP: no CLS; pooled head output is the global token
            patches = h
            glob = out.pooler_output if getattr(out, "pooler_output", None) is not None else h.mean(1)
        else:  # ViT / DINOv2 / CLIP: CLS (+ registers) first
            side = int(h.shape[1] ** 0.5)
            patches, glob = h[:, h.shape[1] - side * side:], h[:, 0]
        g = patches.transpose(1, 2).reshape(patches.shape[0], -1, side, side)
        g = F.adaptive_avg_pool2d(g.float(), self.grid).flatten(2).transpose(1, 2)
        return torch.cat([g, glob.float().unsqueeze(1)], 1)


class VisionFrontend:
    """Any Hugging Face vision tower -> (grid x grid pooled patch tokens, global token)."""

    def __init__(self, cfg, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device, self.amp = device, bool(cfg.amp and device.type == "cuda")
        with quiet_transformers():
            self.proc = AutoImageProcessor.from_pretrained(cfg.vision_encoder)
            model = AutoModel.from_pretrained(cfg.vision_encoder)
        tower = getattr(model, "vision_model", model)  # SigLIP / CLIP: keep the vision tower only
        self.net = on_all_gpus(_VisionTokens(tower, cfg.vision_grid), cfg, device)

    @torch.no_grad()
    def __call__(self, images: List) -> np.ndarray:
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            return self.net(px).float().cpu().numpy()


def cache_vision(cfg, builder, manifest: Dict, device: torch.device, logger, extra_from: Optional[int] = None,
                 root: Optional[str] = None) -> str:
    """Frozen vision tokens of every image of ``manifest``.  ``extra_from`` / ``root``: the manifest is that of
    extra data, cached under ``root``; its images follow the first ``extra_from`` images in the builder's numbering."""
    extra = extra_from is not None
    out = os.path.join(root or cfg.cache_dir, f"vision_{_tag(cfg.vision_encoder)}_g{cfg.vision_grid}")
    meta_p = os.path.join(out, "meta.json")
    if os.path.exists(meta_p):
        return out
    os.makedirs(out, exist_ok=True)
    n = len(manifest["images"])
    if cfg.vision_encoder == "synthetic":
        toks = np.stack([builder.vision_tokens((extra_from or 0) + i) for i in range(n)]).astype(np.float16)
        np.save(os.path.join(out, "tokens.npy"), toks)
    else:
        fe = VisionFrontend(cfg, device)
        arr = None
        groups = builder.extra_image_groups(manifest["images"]) if extra else builder.image_groups(manifest["images"])
        for grp in tqdm(groups, desc="vision features" + (" (extra images)" if extra else ""), dynamic_ncols=True):
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


def cache_thumbs(cfg, builder, manifest: Dict, vdir: str, logger, extra_from: Optional[int] = None) -> str:
    """Small RGB thumbnails of every image (uint8 [n, S, S, 3]): targets of the preview pixel decoder."""
    path = os.path.join(vdir, f"thumbs_{cfg.thumb_size}.npy")
    if os.path.exists(path):
        return path
    n, S = len(manifest["images"]), cfg.thumb_size
    out = np.zeros((n, S, S, 3), dtype=np.uint8)
    if hasattr(builder, "render"):
        for i in range(n):
            out[i] = np.asarray(builder.render((extra_from or 0) + i, S))
    else:
        groups = builder.extra_image_groups(manifest["images"]) if extra_from is not None else builder.image_groups(manifest["images"])
        for grp in tqdm(groups, desc="thumbnails" + (" (extra images)" if extra_from is not None else ""), dynamic_ncols=True):
            for i, img in grp["loader"]():
                out[i] = np.asarray(img.resize((S, S)))
    np.save(path + ".tmp.npy", out)
    os.replace(path + ".tmp.npy", path)
    logger.info(f"[cache] thumbnails {out.shape} -> {path}")
    return path


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


class _SpeechStreams(nn.Module):
    """waveforms [B, T] -> (deep-layer mean [B, frames, D], early-layer mean [B, frames, D])."""

    def __init__(self, model: nn.Module, sem_layers: Tuple[int, ...], spk_layers: Tuple[int, ...], use_mask: bool) -> None:
        super().__init__()
        self.model, self.sem_layers, self.spk_layers, self.use_mask = model, tuple(sem_layers), tuple(spk_layers), use_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.model(x, attention_mask=mask if self.use_mask else None, output_hidden_states=True).hidden_states
        sem = torch.stack([hs[l] for l in self.sem_layers]).mean(0).float()
        early = torch.stack([hs[l] for l in self.spk_layers]).mean(0).float()
        return sem, early


class SSLFrontend:
    """HuBERT / wav2vec2-family encoder.  ``sem`` = mean of deep hidden states (pooled to 25 Hz),
    ``spk`` = time-average of early hidden states."""

    def __init__(self, cfg, device: torch.device) -> None:
        from transformers import AutoConfig, AutoModel

        self.cfg, self.device, self.pool = cfg, device, cfg.speech_pool
        self.amp = bool(cfg.amp and device.type == "cuda")
        model = AutoModel.from_pretrained(cfg.speech_encoder)
        conf = AutoConfig.from_pretrained(cfg.speech_encoder)
        self.dim = self.spk_dim = conf.hidden_size
        layer_norm = getattr(conf, "feat_extract_norm", "group") == "layer"
        self.normalize = layer_norm
        try:
            from transformers import AutoFeatureExtractor

            self.normalize = bool(AutoFeatureExtractor.from_pretrained(cfg.speech_encoder).do_normalize)
        except Exception:
            pass
        self.frames = model._get_feat_extract_output_lengths
        # group-norm models were trained on zero-padded batches without a mask; layer-norm models need one
        self.net = on_all_gpus(_SpeechStreams(model, cfg.speech_sem_layers, cfg.speech_spk_layers, layer_norm), cfg, device)

    @torch.no_grad()
    def __call__(self, wavs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        lens = [len(w) for w in wavs]
        x = torch.zeros(len(wavs), max(max(lens), 400))
        mask = torch.zeros(x.shape, dtype=torch.long)
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w).float()
            if self.normalize:
                t = (t - t.mean()) / (t.std() + 1e-7)
            x[i, : len(w)] = t
            mask[i, : len(w)] = 1
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            sem, early = self.net(x.to(self.device), mask.to(self.device))
        sem, early = sem.float().cpu(), early.float().cpu()   # one transfer per batch, then cheap slicing
        out = []
        for i, n in enumerate(self.frames(torch.tensor(lens)).tolist()):
            n = max(1, min(int(n), sem.shape[1]))
            s = sem[i, :n]
            m = max(1, n // self.pool)
            s = s[: m * self.pool].reshape(m, self.pool, -1).mean(1) if n >= self.pool else s.mean(0, keepdim=True)
            out.append((s.numpy(), early[i, :n].mean(0).numpy()))
        return out


def prep_wave(cfg, wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample, remove rumble, trim silence, truncate: the one waveform every stream of an utterance is computed
    from."""
    wav = dsp.highpass(dsp.resample(wav, sr, cfg.sample_rate), cfg.highpass_hz, cfg.sample_rate)
    wav = dsp.trim_silence(wav, cfg.trim_silence_db, cfg.sample_rate)[: int(cfg.max_audio_sec * cfg.sample_rate)]
    if len(wav) < cfg.sample_rate // 10:
        wav = np.pad(wav, (0, cfg.sample_rate // 10 - len(wav)))
    return np.ascontiguousarray(wav, dtype=np.float32)


_ST5 = {}


def compute_mel(cfg, wav: np.ndarray, n_frames: int) -> np.ndarray:
    """Log-mel of an utterance, resampled in time to exactly round(n_frames * mel_fps / 25) frames."""
    if cfg.mel_spec == "speecht5":  # the feature space of the pretrained SpeechT5 HiFi-GAN vocoder
        if "fe" not in _ST5:
            from transformers import SpeechT5FeatureExtractor

            _ST5["fe"] = SpeechT5FeatureExtractor()
        mel = _ST5["fe"](audio_target=wav, sampling_rate=cfg.sample_rate, return_tensors="np")["input_values"][0]
    else:
        mel = dsp.log_mel(torch.from_numpy(wav).float(), cfg.sample_rate, 1024, cfg.mel_hop, cfg.n_mels).numpy()
    return dsp._pool_frames(mel, max(1, int(round(n_frames * cfg.mel_fps / 25.0)))).astype(np.float16)


_MEL_JOB: Dict = {}   # what the worker processes of cache_mel inherit when they are forked


def _mel_of_group(i: int) -> List[Tuple[str, np.ndarray]]:
    cfg, frames = _MEL_JOB["cfg"], _MEL_JOB["frames"]
    return [(uid, compute_mel(cfg, prep_wave(cfg, wav, sr), frames[uid])) for uid, wav, sr in _MEL_JOB["groups"][i]["loader"]()]


def cache_mel(cfg, builder, manifest: Dict, splits: Optional[Dict], adir: str, logger, root: Optional[str] = None) -> str:
    """Mel targets of the renderer, in the feature space of the configured vocoder.  A separate, cheap pass
    (no speech encoder), so switching vocoder never re-extracts the expensive streams.  ``root``: the manifest is
    that of extra spoken data (training only, cached under ``root``)."""
    out = os.path.join(root or cfg.cache_dir, f"mel_{cfg.mel_spec}_{os.path.basename(adir)[6:]}")
    if os.path.exists(os.path.join(out, "index.json")):
        return out
    os.makedirs(out, exist_ok=True)
    frames = {k: v[2] for k, v in load_json(os.path.join(adir, "index.json")).items()}
    utts = [u for u in manifest["utts"] if u["has_audio"] and u["uid"] in frames]
    mels, index, off = [], {}, 0
    groups = builder.extra_audio_groups(utts) if root else builder.audio_groups(utts)
    # Pure CPU work, one source file per task: spread over worker processes where the platform can fork (the tasks
    # are closures over open corpora).  Results come back in order, so the cache is the same as from one process.
    n_proc = min(len(groups), max(1, cfg.num_workers))
    pool = None
    if n_proc > 1 and "fork" in multiprocessing.get_all_start_methods():
        _MEL_JOB.update(cfg=cfg, groups=groups, frames=frames)
        pool = multiprocessing.get_context("fork").Pool(n_proc, initializer=torch.set_num_threads, initargs=(1,))
        parts = pool.imap(_mel_of_group, range(len(groups)))
    else:
        _MEL_JOB.update(cfg=cfg, groups=groups, frames=frames)
        parts = map(_mel_of_group, range(len(groups)))
    for part in tqdm(parts, total=len(groups), desc=f"mel ({cfg.mel_spec})", dynamic_ncols=True):
        for uid, m in part:
            index[uid] = [off, int(m.shape[0])]
            mels.append(m)
            off += m.shape[0]
    if pool is not None:
        pool.close()
        pool.join()
    _MEL_JOB.clear()
    arr = np.concatenate(mels)
    np.save(os.path.join(out, "mel.npy"), arr)
    img_split = splits["images"] if splits else {}
    train = [index[u["uid"]] for u in utts if img_split.get(manifest["images"][u["img"]]["id"], "train") == "train"][:3000] or list(index.values())[:3000]
    sample = np.concatenate([np.asarray(arr[o: o + n], np.float32) for o, n in train])
    save_json({"mel_mean": float(sample.mean()), "mel_std": float(sample.std() + 1e-4), "fps": cfg.mel_fps}, os.path.join(out, "stats.json"))
    save_json(index, os.path.join(out, "index.json"))
    logger.info(f"[cache] mel ({cfg.mel_spec}, {cfg.mel_fps} fps): {len(index)} utterances -> {out}")
    return out


def speech_tag(cfg) -> str:
    hp = f"_hp{cfg.highpass_hz:g}" if cfg.highpass_hz > 0 else ""   # another waveform, another cache
    if cfg.speech_encoder == "mfcc":
        return "mfcc" + hp
    return f"{_tag(cfg.speech_encoder)}_L{'-'.join(map(str, cfg.speech_sem_layers))}_p{cfg.speech_pool}{hp}"


def cache_audio(cfg, builder, manifest: Dict, splits: Optional[Dict], device: torch.device, logger,
                root: Optional[str] = None) -> str:
    """Speech streams of every spoken utterance of ``manifest``.  ``root``: the manifest is that of extra spoken
    data (training only, cached under ``root``; its own statistics are stored but the model normalises with the
    benchmark's)."""
    out = os.path.join(root or cfg.cache_dir, f"audio_{speech_tag(cfg)}")
    if os.path.exists(os.path.join(out, "index.json")):
        return out
    os.makedirs(os.path.join(out, "chunks"), exist_ok=True)
    utts = [u for u in manifest["utts"] if u["has_audio"]]
    groups = builder.extra_audio_groups(utts) if root else builder.audio_groups(utts)
    fe = None
    pbar = tqdm(total=sum(len(g["uids"]) for g in groups), desc="speech features" + (" (extra speech)" if root else ""),
                dynamic_ncols=True)
    for grp in groups:
        cp = os.path.join(out, "chunks", _tag(grp["key"]) + ".npz")
        if os.path.exists(cp):
            pbar.update(len(grp["uids"]))
            continue
        if fe is None:
            fe = MFCCFrontend(cfg) if cfg.speech_encoder == "mfcc" else SSLFrontend(cfg, device)
        uids, sems, tones, spks, f0s, lens = [], [], [], [], [], []
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
                uids.append(uid); sems.append(sem.astype(np.float16)); tones.append(tone)
                spks.append(spk.astype(np.float32)); f0s.append(f0s_); lens.append(n)
            buf.clear()

        for uid, wav, sr in grp["loader"]():
            buf.append((uid, prep_wave(cfg, wav, sr)))
            if len(buf) >= cfg.feat_batch_size:
                flush()
            pbar.update(1)
        flush()
        np.savez(cp + ".tmp.npz", uids=np.array(uids), lens=np.array(lens), sem=np.concatenate(sems),
                 tone=np.concatenate(tones), spk=np.stack(spks), f0=np.stack(f0s))
        os.replace(cp + ".tmp.npz", cp)
    pbar.close()

    # ---- consolidate chunks into memory-mapped arrays (the chunks of THIS list of groups: a directory listing
    #      would also pick up what another process working on the same cache has left or not yet written)
    chunk_files = sorted(_tag(g["key"]) + ".npz" for g in groups)
    zs = [np.load(os.path.join(out, "chunks", c)) for c in chunk_files]
    total = int(sum(z["lens"].sum() for z in zs))
    n_utt = int(sum(len(z["lens"]) for z in zs))
    d_s, d_spk = zs[0]["sem"].shape[1], zs[0]["spk"].shape[1]
    sem = np.lib.format.open_memmap(os.path.join(out, "sem.npy"), "w+", np.float16, (total, d_s))
    tone = np.lib.format.open_memmap(os.path.join(out, "tone.npy"), "w+", np.float32, (total, 4))
    spk, f0 = np.zeros((n_utt, d_spk), np.float32), np.zeros((n_utt, 2), np.float32)
    index: Dict[str, List[int]] = {}
    off = row = 0
    for z in zs:
        m = int(z["lens"].sum())
        sem[off: off + m], tone[off: off + m] = z["sem"], z["tone"]
        o = off
        for uid, n, s, f in zip(z["uids"].tolist(), z["lens"].tolist(), z["spk"], z["f0"]):
            index[uid] = [row, o, int(n)]
            spk[row], f0[row] = s, f
            o += int(n); row += 1
        off += m
    sem.flush(); tone.flush()
    np.save(os.path.join(out, "spk.npy"), spk)
    np.save(os.path.join(out, "f0stats.npy"), f0)

    # ---- normalisation statistics from the training split only
    img_split = splits["images"] if splits else {}
    lang_of = {u["uid"]: (u["lang"], manifest["images"][u["img"]]["id"]) for u in utts}
    tr_rows = [v for k, v in index.items() if img_split.get(lang_of[k][1], "train") == "train"]
    rng = np.random.default_rng(0)
    pick = [tr_rows[i] for i in rng.permutation(len(tr_rows))[:3000]] or list(index.values())[:3000]
    xs = np.concatenate([np.asarray(sem[o: o + n], np.float32) for _, o, n in pick])
    stats = {"sem_mean": xs.mean(0), "sem_std": xs.std(0) + 1e-4, "dim": d_s, "spk_dim": d_spk,
             "spk_mean": spk.mean(0), "spk_std": spk.std(0) + 1e-4,
             "f0_mean": f0.mean(0), "f0_std": f0.std(0) + 1e-4, "frame_rate": 25.0}
    save_json(stats, os.path.join(out, "stats.json"))
    save_json(index, os.path.join(out, "index.json"))  # written last: marks the cache as complete
    for c in chunk_files:
        os.remove(os.path.join(out, "chunks", c))
    logger.info(f"[cache] speech: {n_utt} utterances, {total} frames ({total / 25 / 3600:.2f} h) -> {out}")
    return out


class AudioStore:
    """Read-only random access to the cached speech streams (and, if given, the mel targets)."""

    def __init__(self, path: str, mel_path: Optional[str] = None) -> None:
        self.index: Dict[str, List[int]] = load_json(os.path.join(path, "index.json"))
        self.stats = {k: (np.array(v, dtype=np.float32) if isinstance(v, list) else v)
                      for k, v in load_json(os.path.join(path, "stats.json")).items()}
        self.stats.update(mel_mean=0.0, mel_std=1.0)
        self.sem = np.load(os.path.join(path, "sem.npy"), mmap_mode="r")
        self.tone = np.load(os.path.join(path, "tone.npy"), mmap_mode="r")
        self.mel, self.mel_index = None, {}
        if mel_path:
            self.mel = np.load(os.path.join(mel_path, "mel.npy"), mmap_mode="r")
            self.mel_index = load_json(os.path.join(mel_path, "index.json"))
            ms = load_json(os.path.join(mel_path, "stats.json"))
            self.stats.update(mel_mean=ms["mel_mean"], mel_std=ms["mel_std"])
        self.spk = np.load(os.path.join(path, "spk.npy"))
        self.f0 = np.load(os.path.join(path, "f0stats.npy"))

    def get(self, uid: str, max_frames: int, mel_fps: float = 50.0) -> Dict[str, np.ndarray]:
        row, off, n = self.index[uid]
        n_use = min(n, max_frames)
        out = {"sem": np.array(self.sem[off: off + n_use], dtype=np.float32),
               "tone": np.array(self.tone[off: off + n_use], dtype=np.float32),
               "spk": self.spk[row], "f0": self.f0[row]}
        if self.mel is not None:
            mo, mn = self.mel_index[uid]
            out["mel"] = np.array(self.mel[mo: mo + min(mn, int(round(n_use * mel_fps / 25.0)))], dtype=np.float32)
        return out


class AudioStores:
    """The benchmark's own speech cache and those of extra spoken data, read as one.  Normalisation statistics are
    the benchmark's: the model is built from them and extra data must not move them."""

    def __init__(self, stores: List[AudioStore]) -> None:
        self.stores, self.stats = list(stores), stores[0].stats
        self.index: Dict[str, int] = {uid: k for k, st in enumerate(self.stores) for uid in st.index}

    def get(self, uid: str, max_frames: int, mel_fps: float = 50.0) -> Dict[str, np.ndarray]:
        return self.stores[self.index[uid]].get(uid, max_frames, mel_fps)
