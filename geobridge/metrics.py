"""Evaluation of every conversion direction, always through the hub and never with the paired image.

Text out      BLEU / chrF++ (sacrebleu) against the parallel reference and against all captions of the image;
              WER / CER for same-language transcription; exact content accuracy on SynthWorld.
Hub space     recall@k of utterance -> image (hub code explained by gallery patches), cross-lingual
              speech -> text retrieval.
Image out     the image latent written from a hub code, retrieved among the true image latents (recall@k).
Speech out    feature-level round trip: hub -> speech latent -> decoded frame features and predicted pitch ->
              re-encoded -> read -> hub -> text, scored against the reference ("self-transcribed", an intrinsic
              proxy for ASR-BLEU that needs no external recogniser).
Tone          learned gate per language, tone-reliance index (how far the hub code moves when the pitch
              contour is removed), and the translation score with the contour removed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import audio_dsp as dsp
from .text_units import normalize_text
from .utils import save_json


# ------------------------------------------------------------------------------------------ string metrics
def edit_distance(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(hyps: Sequence[str], refs: Sequence[str], unit: str = "word") -> float:
    num = den = 0
    for h, r in zip(hyps, refs):
        h, r = normalize_text(h).lower(), normalize_text(r).lower()
        hs, rs = (h.split(), r.split()) if unit == "word" else (list(h), list(r))
        num += edit_distance(hs, rs)
        den += max(1, len(rs))
    return 100.0 * num / max(1, den)


def bleu_chrf(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> Dict[str, float]:
    """refs: one list of references per hypothesis (ragged allowed)."""
    import sacrebleu

    hyps = [normalize_text(h) or "." for h in hyps]
    n_ref = max(len(r) for r in refs)
    streams = [[normalize_text(r[min(k, len(r) - 1)]) for r in refs] for k in range(n_ref)]
    bleu = sacrebleu.corpus_bleu(hyps, streams, tokenize="13a")
    chrf = sacrebleu.corpus_chrf(hyps, streams, word_order=2)
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def recall_at_k(scores: torch.Tensor, target: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """scores [n, G]; target [n] index of the correct gallery item (ties with identical galleries allowed)."""
    rank = (scores > scores.gather(1, target.view(-1, 1))).sum(1)
    return {f"r{k}": 100.0 * float((rank < k).float().mean()) for k in ks}


# ------------------------------------------------------------------------------------------ evaluator
class Evaluator:
    def __init__(self, cfg, data, device: torch.device):
        self.cfg, self.data, self.device = cfg, data, device
        self.synth = cfg.dataset == "synthworld"

    # ---- one pass over the split: read every view into hub coordinates
    @torch.no_grad()
    def collect(self, model, split: str, max_items: int = 0, flat_tone: bool = True, steps: Optional[int] = None,
                read_mode: Optional[str] = None) -> Dict:
        data, dev = self.data, self.device
        groups = data.eval_groups(split, max_items)
        store: Dict = {"groups": groups, "views": {}, "img_of_group": [], "V": {}, "Z": {}, "img_hub": [], "ground": {}}
        check_ground = self.synth and self.cfg.hub_mode == "bary"
        g0 = 0
        for part, batch in data.eval_batches(groups, 64):
            batch = _to(batch, dev)
            V = model.hub.project(batch["img"])
            lat = model.image_latent(V)
            for j, g in enumerate(part):
                store["img_of_group"].append(g["img"])
                if g["img"] not in store["V"]:
                    store["V"][g["img"]], store["Z"][g["img"]] = V[j].float().cpu(), lat[j].float().cpu()
            if "img" in model.spokes:
                store["img_hub"].append(model.read("img", lat, steps=steps).float().cpu())
            for mod in ("txt", "aud"):
                sub = batch[mod]
                if sub is None:
                    continue
                z = model.encode(mod, sub)
                hub = model.read(mod, z, sub["lang"], mode=read_mode, steps=steps)
                if check_ground:  # where the grounder looks, with the image in hand (ground truth: SynthWorld)
                    attn = model.hub.attend(model.hub.queries(z, mod, sub["lang"]), V[sub["item"]])[1].float().cpu()
                    self._slot_stats(store["ground"], attn, mod, [model.langs[i] for i in sub["lang"].tolist()],
                                     sub["utt"].tolist(), [part[j]["img"] for j in sub["item"].tolist()])
                hub_flat = None
                if mod == "aud" and flat_tone:
                    sub_f = dict(sub, tone=dsp.flatten_tone(sub["tone"]))
                    hub_flat = model.read("aud", model.encode("aud", sub_f), sub["lang"], mode=read_mode, steps=steps)
                for li in sub["lang"].unique().tolist():
                    sel = (sub["lang"] == li).nonzero().squeeze(1)
                    rec = store["views"].setdefault((mod, model.langs[li]), {"g": [], "hub": [], "z": [], "utt": [],
                                                                            "hub_flat": [], "spk": [], "f0": []})
                    rec["g"].append(sub["item"][sel].cpu() + g0)
                    rec["hub"].append(hub[sel].float().cpu())
                    rec["z"].append(z[sel].float().cpu())
                    rec["utt"].append(sub["utt"][sel].cpu())
                    if hub_flat is not None:
                        rec["hub_flat"].append(hub_flat[sel].float().cpu())
                    if mod == "aud":
                        rec["spk"].append(sub["spk"][sel].cpu())
                        rec["f0"].append(sub["f0"][sel].cpu())
            g0 += len(part)
        for rec in store["views"].values():
            for k in list(rec):
                rec[k] = torch.cat(rec[k]) if len(rec[k]) else None
        store["img_hub"] = torch.cat(store["img_hub"]) if store["img_hub"] else None
        store["img_ids"] = sorted(store["V"])
        store["gallery"] = torch.stack([store["V"][i] for i in store["img_ids"]])
        store["gallery_lat"] = torch.stack([store["Z"][i] for i in store["img_ids"]])
        return store

    # ---- helpers
    def _slot_stats(self, acc: Dict, attn: torch.Tensor, mod: str, langs: List[str], utts: List[int], imgs: List[int]) -> None:
        """Count, per view type, what the slots of each caption sit on: a mentioned object, an object the
        sentence does not mention, background, or nothing (null); and whether a two-object caption has both
        of its objects covered.  attn [n, K, P]."""
        for a, lang, u, img in zip(attn, langs, utts, imgs):
            c = acc.setdefault((mod, lang), {"slots": 0, "mentioned": 0, "unmentioned": 0, "background": 0, "null": 0,
                                             "two": 0, "both": 0})
            cells = set(self.data.utts[u]["cells"])
            objects = {cell for cell, _, _ in self.data.images[img]["scene"]}
            for k in range(a.shape[0]):
                top = int(a[k].argmax())
                kind = "null" if float(a[k].sum()) < 0.5 else "mentioned" if top in cells else \
                    "unmentioned" if top in objects else "background"
                c[kind] += 1
            c["slots"] += a.shape[0]
            if len(cells) == 2:
                c["two"] += 1
                c["both"] += int(all(float(a[:, cell].max()) > 0.3 for cell in cells))

    def _ref(self, g: Dict, lang: str) -> Optional[str]:
        ui = g["utts"].get(lang)
        return self.data.utts[ui]["text"] if ui is not None and self.data.utts[ui].get("text") else None

    @torch.no_grad()
    def _gen_text(self, model, hub: torch.Tensor, lang: str, bs: int = 64, seed: int = 0) -> List[str]:
        out: List[str] = []
        gen = torch.Generator(device="cpu").manual_seed(seed)
        d, kp = self.cfg.latent_dim, self.cfg.private_slots
        for i in range(0, len(hub), bs):
            h = hub[i:i + bs].to(self.device)
            priv = torch.randn(h.shape[0], kp, d, generator=gen).to(self.device) * self.cfg.eval_temp
            if not self.cfg.private_noise:
                priv = torch.zeros_like(priv)
            lang_id = torch.full((h.shape[0],), model.lang2id[lang], dtype=torch.long, device=self.device)
            z = model.write("txt", h, lang_id, private=priv)
            out += model.text.generate(z, lang_id)
        return out

    def _score_text(self, hyps: List[str], groups: List[Dict], tgt: str, same_lang: bool) -> Tuple[Dict[str, float], List[Dict]]:
        keep = [i for i, g in enumerate(groups) if self._ref(g, tgt) is not None]
        if not keep:
            return {}, []
        hy = [hyps[i] for i in keep]
        refs1 = [[self._ref(groups[i], tgt)] for i in keep]
        m = bleu_chrf(hy, refs1)
        out = {"bleu": m["bleu"], "chrf": m["chrf"]}
        multi = [self.data.references(groups[i]["img"], tgt) or refs1[n] for n, i in enumerate(keep)]
        out["bleu_multi"] = bleu_chrf(hy, multi)["bleu"]
        if same_lang:
            out["wer"] = error_rate(hy, [r[0] for r in refs1], "word")
            out["cer"] = error_rate(hy, [r[0] for r in refs1], "char")
        if self.synth:
            out.update(self._content_acc(hy, [groups[i] for i in keep], tgt))
        samples = [{"ref": refs1[n][0], "hyp": hy[n], "img": groups[i]["img"], "cap": groups[i]["cap"]}
                   for n, i in enumerate(keep[:60])]
        return out, samples

    def _content_acc(self, hyps: List[str], groups: List[Dict], lang: str) -> Dict[str, float]:
        """SynthWorld only: parse the output and compare with the true content of the source utterance."""
        w = self.data.builder
        exact = col = shp = twin = n_col_err = 0
        by_len = {1: [0, 0], 2: [0, 0]}  # objects in the sentence -> [exact, count]
        for h, g in zip(hyps, groups):
            ui = next(iter(g["utts"].values()))
            truth = [tuple(c) for c in self.data.utts[ui]["content"]]
            pred = w.parse(h, lang)
            exact += int(pred == truth)
            if len(truth) in by_len:
                by_len[len(truth)][0] += int(pred == truth)
                by_len[len(truth)][1] += 1
            if pred:
                col += int(pred[0][0] == truth[0][0])
                shp += int(pred[0][1] == truth[0][1])
                if pred[0][0] != truth[0][0]:
                    n_col_err += 1
                    twin += int(pred[0][0] // 2 == truth[0][0] // 2)  # confused with its tonal twin
        n = max(1, len(hyps))
        out = {"content_acc": 100.0 * exact / n, "color_acc": 100.0 * col / n, "shape_acc": 100.0 * shp / n,
               "twin_confusion": 100.0 * twin / max(1, n_col_err)}
        out.update({f"content{k}_acc": 100.0 * e / c for k, (e, c) in by_len.items() if c})
        return out

    # ---- main entry
    @torch.no_grad()
    def evaluate(self, model, split: str = "valid", max_items: int = 0, quick: bool = False,
                 save_path: Optional[str] = None, steps: Optional[int] = None, read_mode: Optional[str] = None) -> Dict[str, float]:
        cfg, data = self.cfg, self.data
        was_training = model.training
        model.eval()
        store = self.collect(model, split, max_items, flat_tone=True, steps=steps, read_mode=read_mode)
        groups, views = store["groups"], store["views"]
        M: Dict[str, float] = {}
        samples: Dict[str, List[Dict]] = {}
        hub_langs = [l for l in cfg.hub_langs if ("txt", l) in views]
        island = [l for l in cfg.island_langs if ("aud", l) in views]
        written = list(hub_langs) + [l for l in cfg.text_only_langs if ("txt", l) in views and l not in hub_langs]
        if cfg.use_island_text:
            written += [l for l in island if ("txt", l) in views]

        # ---------------- text generation
        tasks: List[Tuple[str, str, str]] = []
        for H in hub_langs:
            tasks += [("aud", L, H) for L in island]
            if ("aud", H) in views:
                tasks.append(("aud", H, H))
            if store["img_hub"] is not None:
                tasks.append(("img", "-", H))
            if not quick:
                tasks.append(("txt", H, H))  # read a sentence and write it again: what the hub code keeps
                for T in written:
                    if T != H:
                        tasks += [("txt", H, T), ("txt", T, H)]
                        tasks += [("aud", L, T) for L in island if not cfg.use_island_text or T != L]
        seen = set()
        for mod, src, tgt in tasks:
            if (mod, src, tgt) in seen:
                continue
            seen.add((mod, src, tgt))
            if mod == "img":
                hub, gi = store["img_hub"], torch.arange(len(groups))
            else:
                hub, gi = views[(mod, src)]["hub"], views[(mod, src)]["g"]
            hyps = self._gen_text(model, hub, tgt)
            name = f"{mod}_{src}>txt_{tgt}".replace("_->", ">")
            m, s = self._score_text(hyps, [groups[i] for i in gi.tolist()], tgt, same_lang=(src == tgt))
            M.update({f"{name}/{k}": v for k, v in m.items()})
            samples[name] = s
            if mod == "aud" and views[(mod, src)].get("hub_flat") is not None and (not quick or src in island):
                hy_f = self._gen_text(model, views[(mod, src)]["hub_flat"], tgt)
                mf, _ = self._score_text(hy_f, [groups[i] for i in gi.tolist()], tgt, same_lang=(src == tgt))
                M.update({f"{name}/flat_{k}": v for k, v in mf.items() if k in ("bleu", "content_acc", "color_acc")})

        # ---------------- hub-space retrieval
        img_index = {i: n for n, i in enumerate(store["img_ids"])}
        gal = store["gallery"].to(self.device)
        for (mod, lang), rec in views.items():
            if mod == "txt" and lang in cfg.island_langs and not cfg.use_island_text:
                continue  # the island orthography is held out: the model was never shown it
            if quick and not (lang in island or (mod == "txt" and lang in hub_langs)):
                continue
            target = torch.tensor([img_index[store["img_of_group"][g]] for g in rec["g"].tolist()])
            sc = torch.cat([model.hub.hub_to_image_scores(rec["hub"][i:i + 32].to(self.device), gal).cpu()
                            for i in range(0, len(rec["hub"]), 32)])
            M.update({f"{mod}_{lang}>img/{k}": v for k, v in recall_at_k(sc, target).items()})
        if "img" in model.spokes:  # the image-writing direction: generated image latent vs the true ones
            lat = F.normalize(store["gallery_lat"].flatten(1), dim=-1)
            gen = torch.Generator().manual_seed(0)
            for (mod, lang), rec in views.items():
                if (mod == "txt" and lang in cfg.island_langs and not cfg.use_island_text) or \
                        (quick and not (lang in island or (mod == "txt" and lang in hub_langs))):
                    continue
                target = torch.tensor([img_index[store["img_of_group"][g]] for g in rec["g"].tolist()])
                outs = []
                for i in range(0, len(rec["hub"]), 64):
                    h = rec["hub"][i:i + 64].to(self.device)
                    priv = torch.randn(h.shape[0], cfg.private_slots, cfg.latent_dim, generator=gen).to(self.device) * cfg.eval_temp
                    outs.append(model.write("img", h, private=priv if cfg.private_noise else torch.zeros_like(priv)).float().cpu())
                sc = F.normalize(torch.cat(outs).flatten(1), dim=-1) @ lat.t()
                M.update({f"{mod}_{lang}>img_gen/{k}": v for k, v in recall_at_k(sc, target).items()})
        for L in island:
            for H in hub_langs:
                a, t = views[("aud", L)], views[("txt", H)]
                pos = {g: n for n, g in enumerate(t["g"].tolist())}
                keep = [n for n, g in enumerate(a["g"].tolist()) if g in pos]
                if len(keep) > 1:
                    sim = F.normalize(a["hub"][keep].flatten(1), dim=-1) @ F.normalize(t["hub"].flatten(1), dim=-1).t()
                    tgt_idx = torch.tensor([pos[a["g"][n].item()] for n in keep])
                    M.update({f"aud_{L}>txt_{H}/hub_{k}": v for k, v in recall_at_k(sim, tgt_idx, (1, 5)).items()})

        # ---------------- speech generation (feature-level round trip, scored as text in the hub language)
        if not quick and hub_langs:
            H = hub_langs[0]
            for S in [l for l in list(hub_langs) + island if ("aud", l) in views]:
                for mod, src in ([("txt", H)] + ([("aud", L) for L in island if L != S] if S == H else [("aud", H)])):
                    if (mod, src) not in views:
                        continue
                    name = f"{mod}_{src}>aud_{S}"
                    hy, gi = self._speech_roundtrip(model, views[(mod, src)], views[("aud", S)], S, H)
                    m, s = self._score_text(hy, [groups[i] for i in gi], H, same_lang=False)
                    M.update({f"{name}/rt_{k}": v for k, v in m.items() if k in ("bleu", "chrf", "content_acc", "color_acc")})
                    samples[name + " (round trip)"] = s

        # ---------------- codecs
        if not quick:
            for lang in written:
                rec = views[("txt", lang)]
                z = rec["z"][:200].to(self.device)
                lid = torch.full((z.shape[0],), model.lang2id[lang], dtype=torch.long, device=self.device)
                hy = model.text.generate(z, lid)
                refs = [data.utts[u]["text"] for u in rec["utt"][:200].tolist()]
                M[f"codec_txt_{lang}/cer"] = error_rate(hy, refs, "char")

        # ---------------- grounding (controlled benchmark): what the hub slots sit on
        for (mod, lang), c in store["ground"].items():
            if mod == "txt" and lang in cfg.island_langs and not cfg.use_island_text:
                continue  # the island's orthography is never trained on
            key = f"ground/{mod}_{lang}"
            M[f"{key}/slots_mentioned"] = 100.0 * c["mentioned"] / max(1, c["slots"])
            M[f"{key}/slots_unmentioned"] = 100.0 * c["unmentioned"] / max(1, c["slots"])
            M[f"{key}/slots_null"] = 100.0 * c["null"] / max(1, c["slots"])
            if c["two"]:
                M[f"{key}/both_covered"] = 100.0 * c["both"] / c["two"]

        # ---------------- tone
        gates = model.audio.gate().detach().cpu()
        for name, i in model.lang2id.items():
            if ("aud", name) in views:
                M[f"tone/gate_{name}"] = float(gates[i])
                rec = views[("aud", name)]
                if rec.get("hub_flat") is not None and len(rec["hub"]) > 2:
                    d_flat = ((rec["hub"] - rec["hub_flat"]) ** 2).mean()
                    d_ref = ((rec["hub"] - rec["hub"].roll(1, 0)) ** 2).mean()
                    M[f"tone/reliance_{name}"] = float(d_flat / d_ref.clamp_min(1e-8))

        # ---------------- headline score (used for best-checkpoint selection)
        score_keys = [k for k in M if any(k.startswith(f"aud_{L}>txt_") for L in island) and k.endswith("/bleu")]
        if not score_keys:
            score_keys = [k for k in M if k.endswith("/bleu")]
        M["score"] = float(np.mean([M[k] for k in score_keys])) if score_keys else 0.0
        if save_path:
            save_json({"metrics": M, "samples": samples}, save_path)
        model.train(was_training)
        return M

    @torch.no_grad()
    def _speech_roundtrip(self, model, src: Dict, tgt_ref: Dict, S: str, H: str, max_n: int = 256):
        """hub(src) -> speech latent in language S -> frame features + predicted pitch -> codec encoder ->
        speech spoke -> hub -> text in H.  Returns hypotheses and the group index of each source item."""
        cfg, dev = self.cfg, self.device
        n = min(len(src["hub"]), max_n)
        hyps: List[str] = []
        gen = torch.Generator().manual_seed(0)
        for i in range(0, n, 64):
            hub = src["hub"][i:i + 64].to(dev)
            b = hub.shape[0]
            lid = torch.full((b,), model.lang2id[S], dtype=torch.long, device=dev)
            pick = torch.randint(len(tgt_ref["spk"]), (b,), generator=gen)  # a real voice of language S
            spk, f0 = tgt_ref["spk"][pick].to(dev), tgt_ref["f0"][pick].to(dev)
            priv = torch.randn(b, cfg.private_slots, cfg.latent_dim, generator=gen).to(dev) * cfg.eval_temp
            z = model.write("aud", hub, lid, private=priv if cfg.private_noise else torch.zeros_like(priv))
            n_frames = model.audio.predict_frames(z)
            p = model.audio.bypass_code(spk, f0)
            out = model.audio.decode(z, lid, n_frames, p)
            voiced = (torch.sigmoid(out["pros"][..., 1]) > 0.5).float()
            zf0 = out["pros"][..., 0] * voiced
            dz = torch.zeros_like(zf0)
            dz[:, 1:-1] = (zf0[:, 2:] - zf0[:, :-2]) / 2
            tone = torch.stack([zf0, voiced, dz * voiced, out["pros"][..., 2]], -1) * out["mask"].unsqueeze(-1)
            z2 = model.audio.encode(out["sem"], tone, out["mask"], lid, sem_is_normed=True)
            hub2 = model.read("aud", z2, lid)
            hyps += self._gen_text(model, hub2.cpu(), H)
        return hyps, src["g"][:n].tolist()


def _to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to(v, device) for k, v in batch.items()}
    return batch
