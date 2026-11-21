"""Inferencer: any-to-any conversion with a trained model, plus everything needed to look inside it.

``Inferencer(cfg).run_all()`` writes to ``<run_dir>/inference/``:

  outputs/translations.md|json     every text-producing direction, with references
  outputs/speech/*.wav + *.png     synthesised speech (Griffin-Lim) with mel and pitch contour
  outputs/private_diversity.json   one hub code, several private codes; hub interpolation
  01_slot_attention.png            where each hub slot looks, for text / speech / island speech of one caption
  02_flow_trajectories.png         ODE paths hub <-> modality (PCA), straightness, speed profile
  03_hub_manifold.png              all languages and modalities in hub coordinates (PCA + t-SNE)
  04_docking_coupling.png          transport plans island <-> hub at alpha = 0, .5, 1, with the true matches
  05_tone.png                      learned tonality gates, tone-reliance, pitch-scaling response
  06_speech_attribution.png        which frames (segmental vs tone stream) drive which hub slot
  07_text_attribution.png          which graphemes drive which hub slot
  08_retrieval.png                 island-speech -> image similarity matrix and top-k galleries
  09_ode_steps.png                 quality vs number of ODE steps; the plain ODE cycle is solver error
  10_fertility.png                 units per word / per grapheme for each script and text unit
  11_hub_agreement.png             same-caption vs different-caption hub distance across modalities
  report.json                      every number behind the figures
"""
from __future__ import annotations

import math
import os
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import audio_dsp as dsp
from .config import Config
from .data import GeoBridgeData
from .losses import GeoBridgeLoss
from .metrics import Evaluator, _to, recall_at_k
from .models import GeoBridge, build_spec, ot
from .text_units import TextUnitizer, graphemes, normalize_text
from .utils import get_logger, load_json, resolve_device, save_json, seed_everything

LANG_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def load_model(cfg: Config, data: GeoBridgeData, device: torch.device, checkpoint: Optional[str] = None,
               use_ema: bool = True) -> Tuple[GeoBridge, Dict]:
    path = checkpoint or os.path.join(cfg.run_dir, "checkpoints", "best.pt")
    if not os.path.exists(path):
        path = os.path.join(cfg.run_dir, "checkpoints", "last.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GeoBridge(cfg, build_spec(data)).to(device)
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}, strict=False)
    if use_ema and ck.get("ema") and ck["step"] > cfg.phase_a_steps:
        named = {f"{i}.{n}": p for i, m in enumerate(model.phase_b_modules()) for n, p in m.named_parameters()}
        with torch.no_grad():
            for k, p in named.items():
                if k in ck["ema"] and ck["ema"][k].shape == p.shape:
                    p.copy_(ck["ema"][k].to(device))
    model.eval()
    return model, ck


class Inferencer:
    def __init__(self, cfg: Config, checkpoint: Optional[str] = None, data: Optional[GeoBridgeData] = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.log = get_logger("gb.infer")
        self.data = data or GeoBridgeData(cfg, self.log, self.device)
        self.model, self.ck = load_model(cfg, self.data, self.device, checkpoint)
        self.evaluator = Evaluator(cfg, self.data, self.device)
        self.out = os.path.join(cfg.run_dir, "inference")
        os.makedirs(os.path.join(self.out, "outputs", "speech"), exist_ok=True)
        self.report: Dict = {"checkpoint_step": int(self.ck["step"])}
        self.hub_lang = cfg.hub_langs[0]
        self.island = [l for l in cfg.island_langs]
        seed_everything(0)

    # ================================================================== public conversion API
    @torch.no_grad()
    def text_to_hub(self, texts: List[str], lang: str) -> torch.Tensor:
        sub = self.data.unitizer.collate(texts)
        sub["lang"] = torch.full((len(texts),), self.model.lang2id[lang], dtype=torch.long)
        sub = _to(sub, self.device)
        return self.model.to_hub("txt", self.model.encode("txt", sub), sub["lang"])

    @torch.no_grad()
    def translate_text(self, texts: List[str], src: str, tgt: str, temp: Optional[float] = None) -> List[str]:
        return self.model.hub_to_text(self.text_to_hub(texts, src), tgt, temp=temp)

    @torch.no_grad()
    def utterances_to_hub(self, utt_ids: List[int]) -> Tuple[torch.Tensor, Dict]:
        sub = _to(self.data._collate_audio(utt_ids, list(range(len(utt_ids)))), self.device)
        return self.model.to_hub("aud", self.model.encode("aud", sub), sub["lang"]), sub

    @torch.no_grad()
    def speech_to_text(self, utt_ids: List[int], tgt: str) -> List[str]:
        return self.model.hub_to_text(self.utterances_to_hub(utt_ids)[0], tgt)

    @torch.no_grad()
    def hub_to_wave(self, hub: torch.Tensor, lang: str, ref_utt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """hub [1, K, d] -> (waveform, log-mel, z-scored F0 contour) in the voice of ``ref_utt``."""
        ref = _to(self.data._collate_audio([ref_utt], [0]), self.device)
        mel, n_frames, _, streams = self.model.hub_to_speech(hub, lang, ref["spk"], ref["f0"])
        m = mel[0, : 2 * int(n_frames[0])].float().cpu()
        wav = dsp.griffin_lim(m, self.cfg.sample_rate, 1024, self.cfg.mel_hop).numpy()
        return wav, m.numpy(), streams["tone3"][0, : int(n_frames[0]), 0].float().cpu().numpy()

    def evaluate(self, split: str = "test", max_items: int = 0) -> Dict[str, float]:
        path = os.path.join(self.cfg.run_dir, f"eval_{split}.json")
        m = self.evaluator.evaluate(self.model, split, max_items, quick=False, save_path=path)
        self.log.info(f"[eval] {split}: " + " | ".join(f"{k} {v:.2f}" for k, v in m.items()
                                                        if k.endswith("/bleu") or k.endswith("/r1") or k == "score"))
        return m

    # ================================================================== everything
    def run_all(self, split: str = "test") -> Dict:
        from .utils import setup_matplotlib

        setup_matplotlib()
        self.split = split
        self.groups = self.data.eval_groups(split, 400)
        self.store = self.evaluator.collect(self.model, split, 400)
        steps = [self.save_outputs, self.fig_slot_attention, self.fig_flow_trajectories, self.fig_hub_manifold,
                 self.fig_docking_coupling, self.fig_tone, self.fig_speech_attribution, self.fig_text_attribution,
                 self.fig_retrieval, self.fig_ode_steps, self.fig_fertility, self.fig_hub_agreement]
        for fn in steps:
            try:
                fn()
                self.log.info(f"[infer] {fn.__name__} done")
            except Exception as e:  # one failed figure must not lose the others
                self.log.info(f"[infer] {fn.__name__} FAILED: {type(e).__name__}: {e}")
                self.report.setdefault("failed", []).append(f"{fn.__name__}: {e}")
        save_json(self.report, os.path.join(self.out, "report.json"))
        self.log.info(f"[infer] everything written to {self.out}")
        return self.report

    # ------------------------------------------------------------------ small helpers
    def _view_utt(self, g: Dict, mod: str, lang: str) -> Optional[int]:
        for m, l, ui in g["all_views"]:
            if m == mod and l == lang:
                return ui
        return None

    def _image(self, img_idx: int, size: int = 224):
        b = self.data.builder
        if hasattr(b, "render"):
            return b.render(img_idx, size)
        return b.load_image(self.data.images[img_idx]["id"]).resize((size, size))

    def _attn_grid(self, attn: np.ndarray) -> np.ndarray:
        """attention over the patch tokens -> [grid, grid] map."""
        return attn.reshape(self.model.hub.grid)

    def _examples(self, n: int, need: List[Tuple[str, str]]) -> List[Dict]:
        gs = [g for g in self.groups if all(self._view_utt(g, m, l) is not None for m, l in need)]
        return gs[:n]

    @torch.no_grad()
    def _ground(self, g: Dict, mod: str, lang: str):
        """Grounder attention (needs the image) and the ODE hub estimate (does not) for one view."""
        ui = self._view_utt(g, mod, lang)
        batch = _to(self.data.collate_groups([{"img": g["img"], "cap": g["cap"], "views": [(mod, lang, ui)]}]), self.device)
        sub = batch[mod]
        V = self.model.hub.project(batch["img"])
        z = self.model.encode(mod, sub)
        o = self.model.paired_hub(mod, z, sub["lang"], V)
        hub_ode = self.model.to_hub(mod, z, sub["lang"])
        implied = torch.softmax(torch.einsum("nkd,npd->nkp", F.normalize(hub_ode.float(), dim=-1),
                                             F.normalize(V[:, :-1].float(), dim=-1)) / 0.05, dim=-1)
        return o["attn"][0].float().cpu().numpy(), implied[0].cpu().numpy(), hub_ode, V

    # ================================================================== saved outputs
    @torch.no_grad()
    def save_outputs(self) -> None:
        cfg, data, model, H = self.cfg, self.data, self.model, self.hub_lang
        rows, md = [], ["# GeoBridge-Flow: sample conversions (all through the hub; no image at test time)\n"]
        views = self.store["views"]
        directions = [("aud", L, H) for L in self.island if ("aud", L) in views] + [("aud", H, H), ("img", "-", H)]
        for T in cfg.text_only_langs:
            if ("txt", T) in views:
                directions += [("txt", H, T), ("txt", T, H)]
        for mod, src, tgt in directions:
            if mod == "img":
                hub, gi = self.store["img_hub"][:12], list(range(12))
            elif (mod, src) in views:
                hub, gi = views[(mod, src)]["hub"][:12], views[(mod, src)]["g"][:12].tolist()
            else:
                continue
            hyps = self.evaluator._gen_text(model, hub, tgt)
            md.append(f"\n## {mod}:{src} -> text:{tgt}\n\n| source | reference | output |\n|---|---|---|")
            for h, g_i in zip(hyps, gi):
                g = self.store["groups"][g_i]
                src_txt = self.evaluator._ref(g, src) if src != "-" else f"image {data.images[g['img']]['id']}"
                ref = self.evaluator._ref(g, tgt) or ""
                rows.append({"direction": f"{mod}:{src}>{tgt}", "source": src_txt, "reference": ref, "output": h})
                md.append(f"| {src_txt} | {ref} | {h} |")
        save_json(rows, os.path.join(self.out, "outputs", "translations.json"))
        with open(os.path.join(self.out, "outputs", "translations.md"), "w", encoding="utf8") as f:
            f.write("\n".join(md) + "\n")

        # ---- one hub code, many private codes; and a walk between two hub codes
        div = []
        if ("txt", H) in views:
            for i in range(min(4, len(views[("txt", H)]["hub"]))):
                hub = views[("txt", H)]["hub"][i:i + 1].to(self.device)
                g = self.store["groups"][views[("txt", H)]["g"][i].item()]
                outs = {f"temp={t}": [model.hub_to_text(hub, H, temp=t)[0] for _ in range(3)] for t in (0.0, 0.5, 1.0)}
                div.append({"source": self.evaluator._ref(g, H), "samples": outs})
            a, b = views[("txt", H)]["hub"][0:1].to(self.device), views[("txt", H)]["hub"][1:2].to(self.device)
            walk = [model.hub_to_text((1 - w) * a + w * b, H, temp=0.0)[0] for w in np.linspace(0, 1, 6)]
            div.append({"hub_interpolation": walk})
        save_json(div, os.path.join(self.out, "outputs", "private_diversity.json"))

        # ---- speech synthesis: text -> speech (hub and island language), island speech -> hub-language speech
        import matplotlib.pyplot as plt

        synth = []
        ex = self._examples(3, [("txt", H), ("aud", H)])
        for n, g in enumerate(ex):
            hub = self.text_to_hub([self.evaluator._ref(g, H)], H)
            jobs = [(H, self._view_utt(g, "aud", H), f"tts_{H}_{n}")]
            for L in self.island:
                ref_u = next((self._view_utt(x, "aud", L) for x in self.groups if self._view_utt(x, "aud", L) is not None), None)
                if ref_u is not None:
                    jobs.append((L, ref_u, f"tts_{L}_{n}"))
                    if self._view_utt(g, "aud", L) is not None:
                        hub_l, _ = self.utterances_to_hub([self._view_utt(g, "aud", L)])
                        jobs.append((H, self._view_utt(g, "aud", H), f"s2st_{L}2{H}_{n}", hub_l))
            for job in jobs:
                lang, ref_u, name = job[0], job[1], job[2]
                wav, mel, f0 = self.hub_to_wave(job[3] if len(job) > 3 else hub, lang, ref_u)
                _write_wav(os.path.join(self.out, "outputs", "speech", name + ".wav"), wav, cfg.sample_rate)
                fig, ax = plt.subplots(2, 1, figsize=(8, 4.2), sharex=False)
                ax[0].imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
                ax[0].set_title(f"{name}: generated log-mel  |  text: {self.evaluator._ref(g, H)}", fontsize=8)
                ax[1].plot(f0, color="#d62728")
                ax[1].set_title("predicted speaker-normalised log-F0 (0 = unvoiced)", fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(self.out, "outputs", "speech", name + ".png"), dpi=110)
                plt.close(fig)
                synth.append({"file": name + ".wav", "lang": lang, "text": self.evaluator._ref(g, H), "seconds": len(wav) / cfg.sample_rate})
        self.report["synthesised_speech"] = synth

    # ================================================================== 01 slot attention
    def fig_slot_attention(self) -> None:
        import matplotlib.pyplot as plt

        H, K = self.hub_lang, self.cfg.hub_slots
        views = [("txt", H), ("aud", H)] + [("aud", L) for L in self.island]
        ex = self._examples(3, [v for v in views]) or self._examples(3, views[:2])
        if not ex:
            return
        rows = []
        for g in ex:
            for mod, lang in views:
                if self._view_utt(g, mod, lang) is not None:
                    rows.append((g, mod, lang))
        fig, axes = plt.subplots(len(rows), K + 2, figsize=(2.1 * (K + 2), 2.2 * len(rows)), squeeze=False)
        for r, (g, mod, lang) in enumerate(rows):
            attn, implied, _, _ = self._ground(g, mod, lang)
            img = self._image(g["img"])
            axes[r, 0].imshow(img)
            txt = self.evaluator._ref(g, lang) or ""
            axes[r, 0].set_title(f"{mod}:{lang}\n{txt[:34]}", fontsize=6.5)
            for k in range(K):
                axes[r, k + 1].imshow(img, alpha=0.45)
                axes[r, k + 1].imshow(self._attn_grid(attn[k]), cmap="inferno", alpha=0.75, extent=(0, img.size[0], img.size[1], 0),
                                      interpolation="nearest")
                axes[r, k + 1].set_title(f"slot {k} (grounder)", fontsize=6.5)
            axes[r, K + 1].imshow(img, alpha=0.45)
            axes[r, K + 1].imshow(self._attn_grid(implied.mean(0)), cmap="viridis", alpha=0.75,
                                  extent=(0, img.size[0], img.size[1], 0), interpolation="nearest")
            axes[r, K + 1].set_title("ODE hub -> patches\n(no image used)", fontsize=6.5)
            for a in axes[r]:
                a.axis("off")
        fig.suptitle("Barycentric visual interlingua: each hub slot is a barycentre of frozen patch tokens", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "01_slot_attention.png"), dpi=120)
        plt.close(fig)
        # pointing game (SynthWorld has ground-truth object cells)
        if self.cfg.dataset == "synthworld":
            res = {}
            for mod, lang in views:
                hit = prec = tot = 0
                for g in self.groups[:80]:
                    ui = self._view_utt(g, mod, lang)
                    if ui is None:
                        continue
                    attn = self._ground(g, mod, lang)[0]
                    cells = set(self.data.utts[ui]["cells"])
                    tops = [int(a.argmax()) for a in attn]
                    hit += int(any(t in cells for t in tops))         # some slot points at a described object
                    prec += sum(t in cells for t in tops) / len(tops)  # share of slots on described objects
                    tot += 1
                if tot:
                    res[f"{mod}:{lang}"] = {"any_slot_hits": 100.0 * hit / tot, "slot_precision": 100.0 * prec / tot,
                                            "chance_precision": 100.0 * 1.35 / (self.cfg.synth_grid ** 2)}
            self.report["pointing_game"] = res

    # ================================================================== 02 flow trajectories
    @torch.no_grad()
    def fig_flow_trajectories(self) -> None:
        import matplotlib.pyplot as plt

        model, H = self.model, self.hub_lang
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
        stats = {}
        panels = []
        if ("txt", H) in self.store["views"]:
            panels.append(("txt", H, +1, "hub -> text (write)"))
        for L in (self.island or [H]):
            if ("aud", L) in self.store["views"]:
                panels.append(("aud", L, -1, f"speech:{L} -> hub (read)"))
        for ax, (mod, lang, direction, title) in zip(axes[:2], panels[:2]):
            rec = self.store["views"][(mod, lang)]
            n = min(24, len(rec["z"]))
            lid = torch.full((n,), model.lang2id[lang], dtype=torch.long, device=self.device)
            spoke = model.spokes[mod]
            if direction > 0:
                x = spoke.source(rec["hub"][:n].to(self.device), self.cfg.eval_temp)
            else:
                x = rec["z"][:n].to(self.device)
            _, path = spoke.solve(x, lid, direction, steps=32, return_path=True)
            P = torch.stack(path, 1).flatten(2).float().cpu()                 # [n, 33, D]
            flat = P.reshape(-1, P.shape[-1])
            mu = flat.mean(0)
            _, _, Vt = torch.pca_lowrank(flat - mu, q=2, center=False)
            xy = ((P - mu) @ Vt[:, :2]).numpy()
            for i in range(n):
                ax.plot(xy[i, :, 0], xy[i, :, 1], lw=0.9, color="#888")
                ax.scatter(xy[i, 0, 0], xy[i, 0, 1], s=14, color="#1f77b4", zorder=3)
                ax.scatter(xy[i, -1, 0], xy[i, -1, 1], s=14, color="#d62728", zorder=3)
            seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1).sum(1)
            chord = (P[:, -1] - P[:, 0]).norm(dim=-1)
            stats[title] = float((chord / seg.clamp_min(1e-8)).mean())
            ax.set_title(f"{title}\nblue = start, red = end | straightness {stats[title]:.3f}", fontsize=9)
            # speed profile
            ts = torch.linspace(0, 1, 17)
            speeds = []
            for t in ts:
                tt = torch.full((n,), float(t), device=self.device)
                xt = P[:, int(round(float(t if direction > 0 else 1 - t) * 32))].reshape(n, -1, self.cfg.latent_dim).to(self.device)
                speeds.append(float(spoke.vel(xt, tt, lid).float().flatten(1).norm(dim=-1).mean()))
            axes[2].plot(ts.numpy(), speeds, label=title)
        axes[2].set_title("velocity norm along the path", fontsize=9)
        axes[2].set_xlabel("t")
        axes[2].legend(fontsize=7)
        # endpoint error of few-step solvers against a 64-step reference
        if panels:
            mod, lang, direction, title = panels[0]
            rec = self.store["views"][(mod, lang)]
            n = min(64, len(rec["z"]))
            lid = torch.full((n,), model.lang2id[lang], dtype=torch.long, device=self.device)
            x = model.spokes[mod].source(rec["hub"][:n].to(self.device), self.cfg.eval_temp)
            ref = model.spokes[mod].solve(x, lid, +1, steps=64, solver="midpoint")
            for solver in ("euler", "midpoint", "heun"):
                errs = [float(((model.spokes[mod].solve(x, lid, +1, steps=s, solver=solver) - ref) ** 2).mean()) for s in (1, 2, 4, 8, 16)]
                axes[3].semilogy([1, 2, 4, 8, 16], errs, marker="o", label=solver)
            axes[3].set_title("end-point error vs 64-step reference", fontsize=9)
            axes[3].set_xlabel("ODE steps")
            axes[3].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "02_flow_trajectories.png"), dpi=120)
        plt.close(fig)
        self.report["path_straightness"] = stats

    # ================================================================== 03 hub manifold
    def fig_hub_manifold(self) -> None:
        import matplotlib.pyplot as plt

        views = self.store["views"]
        keys = [k for k in views if not (k[0] == "txt" and k[1] in self.island and not self.cfg.use_island_text)]
        X, lab, grp = [], [], []
        for i, k in enumerate(keys):
            n = min(250, len(views[k]["hub"]))
            X.append(views[k]["hub"][:n].flatten(1))
            lab += [i] * n
            grp += views[k]["g"][:n].tolist()
        if self.store["img_hub"] is not None:
            n = min(250, len(self.store["img_hub"]))
            X.append(self.store["img_hub"][:n].flatten(1))
            lab += [len(keys)] * n
            grp += list(range(n))
            keys = keys + [("img", "-")]
        X = torch.cat(X).float()
        lab, grp = np.array(lab), np.array(grp)
        Xc = X - X.mean(0)
        _, _, Vt = torch.pca_lowrank(Xc, q=2, center=False)
        pca = (Xc @ Vt[:, :2]).numpy()
        try:
            from sklearn.manifold import TSNE

            emb = TSNE(n_components=2, init="pca", perplexity=min(30, max(5, len(X) // 20)), random_state=0).fit_transform(
                F.normalize(Xc, dim=-1).numpy())
        except Exception:
            emb = pca
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, Y, title in ((axes[0], pca, "PCA of hub codes"), (axes[1], emb, "t-SNE of hub codes")):
            for i, k in enumerate(keys):
                m = lab == i
                ax.scatter(Y[m, 0], Y[m, 1], s=9, alpha=0.6, color=LANG_COLORS[i % 6], label=f"{k[0]}:{k[1]}")
            ax.set_title(title + " (all read by backward ODE)", fontsize=10)
            ax.legend(fontsize=8, markerscale=2)
        # same caption across views: lines in t-SNE for a few groups
        for gidx in np.unique(grp)[:40]:
            pts = emb[grp == gidx]
            if len(pts) > 1:
                axes[1].plot(pts[:, 0], pts[:, 1], lw=0.4, color="k", alpha=0.35)
        # modality gap: distance between view centroids relative to within-view spread
        cent = torch.stack([X[lab == i].mean(0) for i in range(len(keys))])
        spread = torch.stack([X[lab == i].std(0).norm() for i in range(len(keys))])
        gap = torch.cdist(cent, cent) / spread.mean()
        im = axes[2].imshow(gap.numpy(), cmap="Blues")
        axes[2].set_xticks(range(len(keys)), [f"{k[0]}:{k[1]}" for k in keys], rotation=45, fontsize=8)
        axes[2].set_yticks(range(len(keys)), [f"{k[0]}:{k[1]}" for k in keys], fontsize=8)
        for i in range(len(keys)):
            for j in range(len(keys)):
                axes[2].text(j, i, f"{gap[i, j]:.2f}", ha="center", va="center", fontsize=7)
        axes[2].set_title("centroid gap between views / mean within-view spread\n(0 = views share one distribution in the hub)", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "03_hub_manifold.png"), dpi=120)
        plt.close(fig)
        self.report["hub_centroid_gap"] = {f"{keys[i][0]}:{keys[i][1]}|{keys[j][0]}:{keys[j][1]}": float(gap[i, j])
                                           for i in range(len(keys)) for j in range(i + 1, len(keys))}

    # ================================================================== 04 docking coupling
    @torch.no_grad()
    def fig_docking_coupling(self) -> None:
        import matplotlib.pyplot as plt

        views, cfg, H = self.store["views"], self.cfg, self.hub_lang
        if not self.island or ("aud", self.island[0]) not in views or ("txt", H) not in views:
            return
        a, t = views[("aud", self.island[0])], views[("txt", H)]
        pos = {g: n for n, g in enumerate(t["g"].tolist())}
        keep = [n for n, g in enumerate(a["g"].tolist()) if g in pos][:48]
        if len(keep) < 4:
            return
        src = a["hub"][keep]
        order = [pos[a["g"][n].item()] for n in keep]
        extra = [i for i in range(len(t["hub"])) if i not in set(order)][:48]
        tgt = torch.cat([t["hub"][order], t["hub"][extra]])   # first len(keep) columns are the true partners
        mu = tgt.mean(0, keepdim=True)
        fs, ft = (src - mu).flatten(1), (tgt - mu).flatten(1)
        C1, C2 = ot.scale_free(ot.pairwise_cos_dist(fs)), ot.scale_free(ot.pairwise_cos_dist(ft))
        Mx = ot.scale_free(ot.pairwise_cos_dist(fs, ft))
        fig, axes = plt.subplots(1, 4, figsize=(21, 5))
        res = {}
        for ax, alpha in zip(axes[:3], (0.0, 0.5, 1.0)):
            Pi = ot.solve_fgw(C1, C2, Mx, alpha, cfg.dock_eps, max(cfg.dock_outer_iters, 20), cfg.dock_sinkhorn_iters, cfg.dock_marginal)
            acc = float((Pi.argmax(1) == torch.arange(len(keep))).float().mean()) * 100
            mass = float(Pi[torch.arange(len(keep)), torch.arange(len(keep))].sum() / Pi.sum()) * 100
            res[f"alpha={alpha}"] = {"match_acc": acc, "mass_on_true_pairs": mass, **ot.coupling_stats(Pi)}
            ax.imshow(Pi.numpy(), cmap="magma", aspect="auto")
            ax.plot(range(len(keep)), range(len(keep)), "c.", ms=2)
            name = {0.0: "pure Gromov-Wasserstein", 0.5: "fused", 1.0: "pure Wasserstein"}[alpha]
            ax.set_title(f"alpha = {alpha} ({name})\nargmax = true partner: {acc:.0f}%  mass on true pairs: {mass:.0f}%", fontsize=9)
            ax.set_xlabel(f"hub codes of {H} text (first {len(keep)} = true partners, cyan)")
            ax.set_ylabel(f"island speech ({self.island[0]}), read by backward ODE")
        axes[3].scatter(C1.flatten().numpy(), C2[: len(keep), : len(keep)].flatten().numpy(), s=3, alpha=0.3)
        rho = float(np.corrcoef(C1.flatten().numpy(), C2[: len(keep), : len(keep)].flatten().numpy())[0, 1])
        axes[3].set_title(f"relational geometry: island vs hub pairwise distances\n(Gromov premise) Pearson r = {rho:.3f}", fontsize=9)
        axes[3].set_xlabel("distance between two island utterances (hub coords)")
        axes[3].set_ylabel("distance between their true partners")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "04_docking_coupling.png"), dpi=120)
        plt.close(fig)
        res["relational_isometry_r"] = rho
        self.report["docking_coupling"] = res

    # ================================================================== 05 tone
    @torch.no_grad()
    def fig_tone(self) -> None:
        import matplotlib.pyplot as plt

        model, cfg, H = self.model, self.cfg, self.hub_lang
        spoken = [l for l in model.langs if ("aud", l) in self.store["views"]]
        gates = model.audio.gate().detach().cpu().numpy()
        fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
        cols = [LANG_COLORS[i % 6] for i in range(len(spoken))]
        axes[0].bar(spoken, [gates[model.lang2id[l]] for l in spoken], color=cols)
        for i, l in enumerate(spoken):
            axes[0].text(i, gates[model.lang2id[l]] + 0.02, "tonal" if self.data.lang_info[l].get("tonal") else "non-tonal",
                         ha="center", fontsize=8)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title("learned lexical-tonality gate (label = ground truth)", fontsize=9)
        scales = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
        res: Dict = {"gate": {l: float(gates[model.lang2id[l]]) for l in spoken}, "pitch_scaling": {}}
        for i, l in enumerate(spoken):
            utts = self.store["views"][("aud", l)]["utt"][:96].tolist()
            sub = _to(self.data._collate_audio(utts, list(range(len(utts)))), self.device)
            base = model.to_hub("aud", model.encode("aud", sub), sub["lang"])
            ref = ((base - base.roll(1, 0)) ** 2).mean()
            disp, accs = [], []
            for s in scales:
                tone = sub["tone"].clone()
                tone[..., 0] *= s
                tone[..., 2] *= s
                hub = model.to_hub("aud", model.encode("aud", dict(sub, tone=tone)), sub["lang"])
                disp.append(float(((hub - base) ** 2).mean() / ref.clamp_min(1e-8)))
                if cfg.dataset == "synthworld":
                    hy = self.evaluator._gen_text(model, hub.cpu(), H)
                    gs = [self.store["groups"][g] for g in self.store["views"][("aud", l)]["g"][:96].tolist()]
                    accs.append(self.evaluator._content_acc(hy, gs, H)["color_acc"])
            axes[1].plot(scales, disp, marker="o", color=cols[i], label=l)
            if accs:
                axes[2].plot(scales, accs, marker="o", color=cols[i], label=l)
            res["pitch_scaling"][l] = {"scales": scales, "hub_displacement": disp, "color_acc": accs}
        axes[1].set_title("hub displacement when the pitch contour is scaled\n(relative to the distance between two utterances)", fontsize=9)
        axes[1].set_xlabel("contour scale (1 = natural, 0 = flattened)")
        axes[1].legend(fontsize=8)
        axes[2].set_title(f"colour accuracy of speech -> {H} text vs contour scale" if cfg.dataset == "synthworld"
                          else "(content accuracy needs ground truth: SynthWorld only)", fontsize=9)
        axes[2].set_xlabel("contour scale")
        axes[2].legend(fontsize=8)
        # natural vs re-predicted contour for one island utterance
        L = (self.island or spoken)[0]
        if ("aud", L) in self.store["views"]:
            ui = int(self.store["views"][("aud", L)]["utt"][0])
            sub = _to(self.data._collate_audio([ui], [0]), self.device)
            z = model.encode("aud", sub)
            out = model.audio.decode(z, sub["lang"], sub["mask"].sum(1), model.audio.bypass_code(sub["spk"], sub["f0"]))
            nat, voiced = sub["tone"][0, :, 0].cpu().numpy(), sub["tone"][0, :, 1].cpu().numpy() > 0.5
            pred = out["pros"][0, :, 0].float().cpu().numpy()
            axes[3].plot(np.where(voiced, nat, np.nan), color="k", lw=1.6, label="natural contour")
            axes[3].plot(np.where(voiced, pred, np.nan), color="#d62728", lw=1.2, label="re-predicted from the latent")
            axes[3].set_title(f"{L}: pitch contour recovered from the hub-bound latent\n{(self.data.utts[ui].get('text') or '')[:60]}", fontsize=8)
            axes[3].legend(fontsize=8)
            v = voiced & np.isfinite(pred)
            if v.sum() > 3:
                res["contour_corr_" + L] = float(np.corrcoef(nat[v], pred[v])[0, 1])
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "05_tone.png"), dpi=120)
        plt.close(fig)
        self.report["tone"] = res

    # ================================================================== 06 / 07 attribution
    def fig_speech_attribution(self) -> None:
        import matplotlib.pyplot as plt

        model, K = self.model, self.cfg.hub_slots
        langs = [l for l in ([self.hub_lang] + self.island) if ("aud", l) in self.store["views"]]
        fig, axes = plt.subplots(3, len(langs), figsize=(8 * len(langs), 8), squeeze=False)
        res = {}
        for c, l in enumerate(langs):
            ui = int(self.store["views"][("aud", l)]["utt"][0])
            sub = _to(self.data._collate_audio([ui], [0]), self.device)
            sem = sub["sem"].clone().requires_grad_(True)
            tone = sub["tone"].clone().requires_grad_(True)
            with torch.enable_grad():
                z = model.audio.encode(sem, tone, sub["mask"], sub["lang"])
                hub, _ = model.spokes["aud"].read(z, sub["lang"], grad=True)
                sal_s, sal_t = [], []
                for k in range(K):
                    gs, gt = torch.autograd.grad((hub[0, k] ** 2).sum(), [sem, tone], retain_graph=True)
                    sal_s.append((gs * sem).abs().sum(-1)[0])
                    sal_t.append((gt * tone)[0, :, [0, 2]].abs().sum(-1))
            S, T = torch.stack(sal_s).detach().cpu().numpy(), torch.stack(sal_t).detach().cpu().numpy()
            share = float(T.sum() / (S.sum() + T.sum() + 1e-12))
            res[l] = {"tone_stream_share_of_attribution": share}
            axes[0, c].imshow(S / (S.max() + 1e-9), aspect="auto", cmap="Blues")
            axes[0, c].set_title(f"{l}: segmental-stream attribution per hub slot\n{(self.data.utts[ui].get('text') or '')[:70]}", fontsize=8)
            axes[1, c].imshow(T / (S.max() + 1e-9), aspect="auto", cmap="Reds", vmin=0, vmax=1)
            axes[1, c].set_title(f"{l}: pitch-contour attribution per hub slot (same colour scale; "
                                 f"share of total = {100 * share:.1f}%)", fontsize=8)
            f0 = sub["tone"][0, :, 0].cpu().numpy()
            v = sub["tone"][0, :, 1].cpu().numpy() > 0.5
            axes[2, c].plot(np.where(v, f0, np.nan), color="k")
            axes[2, c].set_xlim(0, len(f0) - 1)
            axes[2, c].set_title("speaker-normalised log-F0", fontsize=8)
            axes[2, c].set_xlabel("frame (25 Hz)")
            for r in (0, 1):
                axes[r, c].set_ylabel("hub slot")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "06_speech_attribution.png"), dpi=120)
        plt.close(fig)
        self.report["speech_attribution"] = res

    def fig_text_attribution(self) -> None:
        import matplotlib.pyplot as plt

        model, K = self.model, self.cfg.hub_slots
        langs = [l for l in [self.hub_lang] + list(self.cfg.text_only_langs) if ("txt", l) in self.store["views"]]
        fig, axes = plt.subplots(len(langs), 1, figsize=(15, 2.6 * len(langs)), squeeze=False)
        for r, l in enumerate(langs):
            ui = int(self.store["views"][("txt", l)]["utt"][0])
            text = self.data.utts[ui]["text"]
            sub = self.data.unitizer.collate([text])
            sub["lang"] = torch.tensor([model.lang2id[l]])
            sub = _to(sub, self.device)
            with torch.enable_grad():
                h0 = model.text.embed_units(sub["units"]).detach().requires_grad_(True)
                z = model.text.encode_from_embeddings(h0, sub["unit_mask"])
                hub, _ = model.spokes["txt"].read(z, sub["lang"], grad=True)
                sal = torch.stack([(torch.autograd.grad((hub[0, k] ** 2).sum(), h0, retain_graph=True)[0] * h0).abs().sum(-1)[0]
                                   for k in range(K)])
            S = sal.detach().cpu().numpy()
            units = [u.decode("utf8", errors="replace") for u in self.data.unitizer.segment(text)][: S.shape[1]]
            axes[r, 0].imshow(S / (S.max() + 1e-9), aspect="auto", cmap="Purples")
            axes[r, 0].set_xticks(range(len(units)), [u if u.strip() else "_" for u in units], fontsize=7)
            axes[r, 0].set_ylabel("hub slot")
            axes[r, 0].set_title(f"{l}: grapheme-level attribution of each hub slot  ({len(units)} units for "
                                 f"{len(text.encode('utf8'))} bytes)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "07_text_attribution.png"), dpi=120)
        plt.close(fig)

    # ================================================================== 08 retrieval
    @torch.no_grad()
    def fig_retrieval(self) -> None:
        import matplotlib.pyplot as plt

        views = self.store["views"]
        key = next((("aud", L) for L in self.island if ("aud", L) in views), ("txt", self.hub_lang))
        rec = views[key]
        n = min(40, len(rec["hub"]))
        img_ids = self.store["img_ids"]
        gal = self.store["gallery"].to(self.device)
        sc = self.model.hub.hub_to_image_scores(rec["hub"][:n].to(self.device), gal).cpu()
        tgt = torch.tensor([img_ids.index(self.store["img_of_group"][g]) for g in rec["g"][:n].tolist()])
        order = torch.argsort(tgt)
        fig = plt.figure(figsize=(18, 7))
        ax = fig.add_subplot(1, 2, 1)
        cols = torch.unique(tgt[order], sorted=False)
        ax.imshow(sc[order][:, tgt[order]].numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"{key[0]}:{key[1]} -> image scores (hub code explained by gallery patches)\n"
                     f"diagonal = correct image | R@1 over {len(img_ids)} images: {recall_at_k(sc, tgt)['r1']:.1f}%", fontsize=9)
        ax.set_xlabel("image of query j")
        ax.set_ylabel("query i")
        for q in range(3):
            top = sc[q].topk(min(5, sc.shape[1])).indices.tolist()
            for j, gi in enumerate(top):
                a = fig.add_subplot(3, 10, q * 10 + 6 + j)
                a.imshow(self._image(img_ids[gi], 112))
                a.axis("off")
                if gi == int(tgt[q]):
                    a.set_title("correct", fontsize=7, color="green")
                if j == 0:
                    g = self.store["groups"][rec["g"][q].item()]
                    a.text(0, -18, (self.evaluator._ref(g, self.hub_lang) or "")[:60], fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "08_retrieval.png"), dpi=120)
        plt.close(fig)

    # ================================================================== 09 ODE steps / the plain cycle
    @torch.no_grad()
    def fig_ode_steps(self) -> None:
        import matplotlib.pyplot as plt

        model, cfg, H = self.model, self.cfg, self.hub_lang
        views = self.store["views"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        res: Dict = {"score_vs_steps": {}, "plain_cycle_vs_steps": {}}
        src = next((("aud", L) for L in self.island if ("aud", L) in views), ("aud", H) if ("aud", H) in views else None)
        if src is not None:
            utts = views[src]["utt"][:160].tolist()
            gs = [self.store["groups"][g] for g in views[src]["g"][:160].tolist()]
            sub = _to(self.data._collate_audio(utts, list(range(len(utts)))), self.device)
            z = model.encode("aud", sub)
            xs, ys = [1, 2, 4, 8, 16], []
            for s in xs:
                hub = model.to_hub("aud", z, sub["lang"], steps=s)
                lid = torch.full((len(utts),), model.lang2id[H], dtype=torch.long, device=self.device)
                zt = model.from_hub("txt", hub, lid, temp=cfg.eval_temp, steps=s)
                m, _ = self.evaluator._score_text(model.text.generate(zt, lid), gs, H, same_lang=(src[1] == H))
                ys.append(m.get("content_acc", m.get("bleu", 0.0)))
            axes[0].plot(xs, ys, marker="o")
            axes[0].set_xscale("log", base=2)
            axes[0].set_title(f"{src[0]}:{src[1]} -> text:{H}  ({'content accuracy' if 'content_acc' in m else 'BLEU'}) "
                              f"vs ODE steps per hop", fontsize=9)
            axes[0].set_xlabel("ODE steps (midpoint)")
            res["score_vs_steps"] = dict(zip(map(str, xs), ys))
        if ("txt", H) in views and ("aud", H) in views:
            loss = GeoBridgeLoss(cfg, model.lang2id)
            n = min(64, len(views[("txt", H)]["z"]))
            z1 = views[("txt", H)]["z"][:n].to(self.device)
            lid = torch.full((n,), model.lang2id[H], dtype=torch.long, device=self.device)
            xs = [1, 2, 4, 8, 16, 32]
            for solver in ("euler", "midpoint"):
                old = cfg.ode_solver
                cfg.ode_solver = solver
                ys = [float(loss.naive_cycle(model, "txt", z1, lid, "aud", lid, steps=s)) for s in xs]
                cfg.ode_solver = old
                axes[1].loglog(xs, ys, marker="o", label=solver)
                res["plain_cycle_vs_steps"][solver] = dict(zip(map(str, xs), ys))
            axes[1].set_title("plain ODE cycle  text -> hub -> speech -> hub -> text\n"
                              "vanishes with the step size for ANY field: it measures the solver, not alignment", fontsize=9)
            axes[1].set_xlabel("ODE steps")
            axes[1].set_ylabel("cycle MSE")
            axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "09_ode_steps.png"), dpi=120)
        plt.close(fig)
        self.report["ode"] = res

    # ================================================================== 10 fertility
    def fig_fertility(self) -> None:
        import matplotlib.pyplot as plt

        cfg, data = self.cfg, self.data
        langs = [l for l in data.langs if any(u["lang"] == l and u.get("text") for u in data.utts[:5000])]
        texts = {l: [u["text"] for u in data.utts if u["lang"] == l and u.get("text")][:1500] for l in langs}
        modes = {"grapheme (ours)": TextUnitizer("grapheme", 10 ** 6, cfg.max_bytes_per_grapheme, 10 ** 6),
                 "byte": TextUnitizer("byte", 10 ** 6, 1, 10 ** 6)}
        bpe_path = os.path.join(cfg.cache_dir, f"bpe_{cfg.bpe_vocab}_{'-'.join(cfg.hub_langs)}.json")
        try:
            if not os.path.exists(bpe_path):
                tr = [u["text"] for u in data.utts if u["lang"] in cfg.hub_langs and data.img_split[u["img"]] == "train" and u.get("text")]
                TextUnitizer.train_bpe(tr, bpe_path, cfg.bpe_vocab)
            modes[f"BPE-{cfg.bpe_vocab} (hub language only)"] = TextUnitizer("bpe", 10 ** 6, 1, 10 ** 6, bpe_path)
        except Exception:
            pass
        res = {name: {l: u.fertility(texts[l]) for l in langs} for name, u in modes.items()}
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
        w = 0.8 / len(modes)
        for ax, key, title in ((axes[0], "units_per_word", "encoder units per word (fertility)"),
                               (axes[1], "units_per_grapheme", "encoder units per written symbol"),
                               (axes[2], "bytes_per_grapheme", "UTF-8 bytes per written symbol (script cost)")):
            for i, name in enumerate(modes):
                vals = [res[name][l][key] for l in langs]
                ax.bar(np.arange(len(langs)) + i * w, vals, w, label=name)
                for x, v in zip(np.arange(len(langs)) + i * w, vals):
                    ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(np.arange(len(langs)) + 0.4 - w / 2, [f"{l}\n{data.lang_info[l]['script'][:14]}" for l in langs], fontsize=8)
            ax.set_title(title, fontsize=9)
            if key != "bytes_per_grapheme":
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "10_fertility.png"), dpi=120)
        plt.close(fig)
        self.report["fertility"] = res

    # ================================================================== 11 hub agreement
    def fig_hub_agreement(self) -> None:
        import matplotlib.pyplot as plt

        views, H = self.store["views"], self.hub_lang
        pairs = [(("txt", H), ("aud", H))] + [(("aud", L), ("txt", H)) for L in self.island] + [(("aud", L), ("aud", H)) for L in self.island]
        pairs = [p for p in pairs if p[0] in views and p[1] in views]
        if not pairs:
            return
        fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 4.4), squeeze=False)
        res = {}
        for ax, (a, b) in zip(axes[0], pairs):
            pos = {g: n for n, g in enumerate(views[b]["g"].tolist())}
            keep = [n for n, g in enumerate(views[a]["g"].tolist()) if g in pos]
            if len(keep) < 3:
                continue
            ha = F.normalize(views[a]["hub"][keep].flatten(1), dim=-1)
            hb = F.normalize(views[b]["hub"][[pos[views[a]["g"][n].item()] for n in keep]].flatten(1), dim=-1)
            same = (ha * hb).sum(-1).numpy()
            diff = (ha * hb.roll(1, 0)).sum(-1).numpy()
            ax.hist(same, bins=30, alpha=0.7, color="#2ca02c", label="same caption")
            ax.hist(diff, bins=30, alpha=0.7, color="#888", label="different caption")
            auc = float((same[:, None] > diff[None, :]).mean())
            res[f"{a[0]}:{a[1]}|{b[0]}:{b[1]}"] = {"same_mean_cos": float(same.mean()), "diff_mean_cos": float(diff.mean()), "auc": auc}
            ax.set_title(f"hub cosine  {a[0]}:{a[1]}  vs  {b[0]}:{b[1]}\nAUC(same > different) = {auc:.3f}"
                         + ("   [no paired data between these two]" if a[1] != b[1] else ""), fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "11_hub_agreement.png"), dpi=120)
        plt.close(fig)
        self.report["hub_agreement"] = res


def _write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    x = np.clip(np.asarray(wav, dtype=np.float32), -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((x * 32767).astype("<i2").tobytes())
