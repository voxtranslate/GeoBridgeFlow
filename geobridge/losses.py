"""All training objectives of GeoBridge-Flow in one class.

Phase A - representation (codecs, grounder, explainer, reader, preview decoder)
    L_A = w_text_rec * L_text + w_audio_rec * L_audio(segmental, prosody, duration, mel, orth, gate price)
        + w_ground * L_ground        the hub code is found in the right image          (contrastive)
        + w_expl   * L_expl          the hub code explains the sentence                (sufficiency)
        + w_read   * L_read          an image-free reader predicts the hub code
        + w_min    * L_min           ... and the hub code must be predictable by it    (minimality)
        + w_agree  * L_agree  + w_pix * L_pix

Phase B - transport (writers = flows, plus the reader), computed on *cached* latents: the frozen codecs and the
grounder are not run again.
    L_B = w_cfm * (L_cfm^txt + L_cfm^aud + L_cfm^img)     hub (+) private flow matching; a share of the views
                                                          is written from a *read* hub (exposure)
        + w_read * L_read                                  keeps the reader on the grounded codes
        + bt_weight * (L_bt_write + L_bt_read)             hub-space back-translation of unpaired island speech
        [+ w_dock * L_dock + naive_cycle_weight * L_cycle] ablation / diagnostic terms, off by default

No term ever pairs two different languages: cross-lingual transfer can only pass through the hub.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import ot


def _pad_time(x: torch.Tensor, L: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, L - x.shape[1])) if x.dim() == 3 else F.pad(x, (0, L - x.shape[1]))


def cat_audio(a: Optional[Dict], b: Optional[Dict]) -> Optional[Dict]:
    """Concatenate two collated speech sub-batches (padding the time axis)."""
    if a is None or b is None:
        return a if b is None else b
    T = max(a["sem"].shape[1], b["sem"].shape[1])
    Tm = max(a["mel"].shape[1], b["mel"].shape[1]) if "mel" in a and "mel" in b else 0
    out = {}
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if k in ("sem", "tone", "mask", "mel"):  # time axes; "mel_len" travels along untouched
            L = Tm if k == "mel" else T
            x, y = _pad_time(x, L), _pad_time(y, L)
        out[k] = torch.cat([x, y], 0)
    return out


def cat_text(a: Optional[Dict], b: Optional[Dict], pad_id: int) -> Optional[Dict]:
    """Concatenate two collated text sub-batches (padding units and bytes with their pad ids)."""
    if a is None or b is None:
        return a if b is None else b
    L, T = max(a["units"].shape[1], b["units"].shape[1]), max(a["bytes"].shape[1], b["bytes"].shape[1])

    def pad(d: Dict) -> Dict:
        return {"units": F.pad(d["units"], (0, 0, 0, L - d["units"].shape[1]), value=pad_id),
                "unit_mask": F.pad(d["unit_mask"], (0, L - d["unit_mask"].shape[1])),
                "bytes": F.pad(d["bytes"], (0, T - d["bytes"].shape[1]), value=256), "lang": d["lang"]}

    pa, pb = pad(a), pad(b)
    return {k: torch.cat([pa[k], pb[k]], 0) for k in pa}


def same_lang_pairs(txt: Optional[Dict], aud: Optional[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indices (i_txt, i_aud) of text / speech views of the same caption in the same language."""
    if txt is None or aud is None:
        e = torch.zeros(0, dtype=torch.long)
        return e, e
    key = {(int(i), int(l)): n for n, (i, l) in enumerate(zip(txt["item"].tolist(), txt["lang"].tolist()))}
    it, ia = [], []
    for n, (i, l) in enumerate(zip(aud["item"].tolist(), aud["lang"].tolist())):
        if (i, l) in key:
            it.append(key[(i, l)])
            ia.append(n)
    dev = txt["item"].device
    return torch.tensor(it, dtype=torch.long, device=dev), torch.tensor(ia, dtype=torch.long, device=dev)


class GeoBridgeLoss:
    def __init__(self, cfg, lang2id: Dict[str, int]):
        self.cfg = cfg
        self.hub_ids = [lang2id[l] for l in cfg.hub_langs if l in lang2id]

    # ============================================================================ phase A
    def phase_a(self, model, batch: Dict, island: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg, hub, logs = self.cfg, model.hub, {}
        V = hub.project(batch["img"])
        keys = hub.keys(V)
        total = V.new_zeros(())
        txt, aud = batch["txt"], batch["aud"]
        z: Dict[str, torch.Tensor] = {}

        # ---- codecs (one forward per modality: paired views + replay text / unpaired island speech)
        txt_all = cat_text(txt, batch.get("replay"), model.text.unit_pad)
        if txt_all is not None:
            r = model.text.loss(txt_all)
            total = total + cfg.w_text_rec * r["loss"]
            if txt is not None:
                z["txt"] = r["z"][: txt["units"].shape[0]]
            logs.update(txt_ce=r["loss"].item(), txt_acc=r["byte_acc"].item())
        aud_all = cat_audio(aud, island["aud"] if island is not None else None)
        if aud_all is not None:
            r = model.audio.loss(aud_all)
            total = total + cfg.w_audio_rec * r["loss"]
            if aud is not None:
                z["aud"] = r["z"][: aud["sem"].shape[0]]
            logs.update({k: r[k].item() for k in ("sem_mse", "f0_l1", "dur_l1", "mel_l1", "orth", "gate_pen") if k in r})

        # ---- grounding: one attention pass gives the contrastive scores and the hub code
        scores, items, codes = [], [], {}
        released = hub.null_enabled  # minimality starts when the dense stage ends
        for mod, sub in (("txt", txt), ("aud", aud)):
            if mod not in z:
                continue
            q = hub.queries(z[mod], mod, sub["lang"])
            score, attn = hub.ground(q, keys, sub["item"])
            code = torch.einsum("nkp,npd->nkd", attn.to(V.dtype), V[sub["item"]][:, :-1])
            codes[mod] = code
            scores.append(score)
            items.append(sub["item"])
            mid = hub.MODS[mod]
            if cfg.w_expl > 0:  # sufficiency: the code alone reconstructs the utterance latent
                unexp = F.mse_loss(hub.explainer(code, mid, sub["lang"]).float(), z[mod].detach().float())
                total = total + cfg.w_expl * unexp
                logs[f"unexplained_{mod}"] = unexp.item()
            pred = hub.reader(z[mod].detach(), mod, sub["lang"]).float()
            l_read = F.mse_loss(pred, code.detach().float())
            total = total + cfg.w_read * l_read
            logs[f"read_{mod}"] = l_read.item()
            if cfg.w_min > 0 and released:  # minimality: nothing in the code that the sentence does not say
                total = total + cfg.w_min * F.mse_loss(code.float(), pred.detach())
            logs[f"slot_use_{mod}"] = attn.sum(-1).mean().item()
        if scores:
            logits = torch.cat(scores) / cfg.ground_temp
            item = torch.cat(items)
            same = batch["img_idx"][item].unsqueeze(1) == batch["img_idx"].unsqueeze(0)  # [n, B]
            pos = F.one_hot(item, batch["B"]).bool()
            masked = logits.masked_fill(same & ~pos, -1e4)
            l_u2i = F.cross_entropy(masked, item)
            logp = logits.t().log_softmax(dim=1)                                          # [B, n]
            l_i2u = -torch.logsumexp(logp.masked_fill(~same.t(), -1e4), dim=1)[same.t().any(1)].mean()
            ground = 0.5 * (l_u2i + l_i2u)
            total = total + cfg.w_ground * ground
            logs.update(ground=ground.item(), ground_acc=(masked.argmax(1) == item).float().mean().item())
            it, ia = same_lang_pairs(txt, aud)
            if len(it) and len(codes) == 2 and cfg.w_agree > 0:
                agree = F.mse_loss(codes["txt"][it], codes["aud"][ia])
                total = total + cfg.w_agree * agree
                logs["agree"] = agree.item()
            logs["unexplained"] = sum(v for k, v in logs.items() if k.startswith("unexplained_")) / len(codes)
            logs["read"] = sum(v for k, v in logs.items() if k.startswith("read_")) / len(codes)

        # ---- preview pixel decoder (frozen image latent -> thumbnail)
        if "thumb" in batch and hasattr(model, "pixel"):
            target = batch["thumb"].permute(0, 3, 1, 2).float() / 255.0
            pix = F.l1_loss(model.pixel(model.image_latent(V)).float(), target)
            total = total + cfg.w_pix * pix
            logs["pix_l1"] = pix.item()
        gates = model.audio.gate().detach()
        for name, i in model.lang2id.items():
            logs[f"gate_{name}"] = gates[i].item()
        logs["loss"] = total.item()
        return total, logs

    # ============================================================================ phase B
    def text_anchor(self, z: Dict[str, torch.Tensor], mod: str, sub: Dict, pairs, pivot: Optional[Dict],
                    n_items: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Text-anchor baseline: which text latent serves as the hub of each view, and whether one exists."""
        n, dev = z[mod].shape[0], z[mod].device
        hub_lang = torch.tensor([int(l) in self.hub_ids for l in sub["lang"].tolist()], device=dev)
        anchor = z[mod].clone() if mod == "txt" else torch.zeros_like(z[mod])
        ok = hub_lang.clone() if mod == "txt" else torch.zeros(n, dtype=torch.bool, device=dev)
        if mod == "aud" and "txt" in z and len(pairs[0]):  # hub-language speech: its own transcript
            anchor[pairs[1]] = z["txt"][pairs[0]]
            ok[pairs[1]] = True
        if pivot is not None and "pivot" in z:  # everything else: a caption of the same image (image pivot)
            row = torch.full((n_items,), -1, dtype=torch.long, device=dev)
            row[pivot["item"]] = torch.arange(len(pivot["item"]), device=dev)
            r = row[sub["item"]]
            use = (~ok) & (r >= 0)
            anchor[use] = z["pivot"][r[use]]
            ok = ok | use
        return anchor, ok

    def phase_b(self, model, batch: Dict, island: Optional[Dict], progress: float,
                bank: Optional[Callable[[int], torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``batch`` comes from the latent cache: per modality the latents ``z``, the grounded hub codes
        ``hub``, ``lang``, ``item`` and ``ok``; plus the image latents ``img_lat`` [B, N, d]."""
        cfg, logs = self.cfg, {}
        sp, reader = model.spokes, model.hub.reader
        views = {m: batch[m] for m in ("txt", "aud") if batch.get(m) is not None}
        dev = batch["img_lat"].device
        total = batch["img_lat"].new_zeros(())

        # ---- grounded hub codes (cached; the unconstrained 'free' anchor is computed live and trained)
        hubs: Dict[str, torch.Tensor] = {}
        for mod, sub in views.items():
            if cfg.hub_mode == "free":
                hubs[mod] = model.hub.free_code(sub["z"], mod, sub["lang"], batch["scene"][sub["item"]])
            else:
                hubs[mod] = sub["hub"]
        if cfg.hub_mode == "free" and hubs:
            var = F.relu(1.0 - torch.cat(list(hubs.values())).float().flatten(1).std(0)).mean()
            total = total + cfg.w_free_hub_var * var
            logs["free_var"] = var.item()

        # ---- reader stays on the grounded codes; its (detached) outputs are the read hubs for exposure
        reads: Dict[str, torch.Tensor] = {}
        for mod, sub in views.items():
            r = reader(sub["z"], mod, sub["lang"])
            l_read = F.mse_loss(r[sub["ok"]].float(), hubs[mod][sub["ok"]].detach().float())
            total = total + cfg.w_read * l_read
            logs[f"read_{mod}"] = l_read.item()
            reads[mod] = r.detach()

        # ---- exposure: a share of the views is written from a hub that was *read*, as it will be at test time
        pairs = same_lang_pairs(views.get("txt"), views.get("aud"))
        src = {m: hubs[m] for m in views}
        if cfg.exposure != "none" and cfg.p_read > 0:
            for mod, sub in views.items():
                other = "aud" if mod == "txt" else "txt"
                u = torch.rand(sub["z"].shape[0], device=dev)
                mine, theirs = (pairs[0], pairs[1]) if mod == "txt" else (pairs[1], pairs[0])
                new = src[mod].clone()
                if cfg.exposure == "reader":
                    self_read = u < cfg.p_read / 2
                    new[self_read] = reads[mod][self_read].to(new.dtype)
                    if len(mine):  # the same caption read from the other modality
                        cross = (u[mine] >= cfg.p_read / 2) & (u[mine] < cfg.p_read)
                        new[mine[cross]] = reads[other][theirs[cross]].to(new.dtype)
                elif len(mine):  # "ode": hub read by the other spoke's backward flow (costly; ablation)
                    cross = u[mine] < cfg.p_read
                    if cross.any():
                        o = views[other]
                        h, _ = sp[other].read(o["z"][theirs[cross]], o["lang"][theirs[cross]], steps=cfg.train_ode_steps)
                        new[mine[cross]] = h.to(new.dtype)
                src[mod] = new

        # ---- hub (+) private conditional flow matching
        for mod, sub in views.items():
            l = sp[mod].cfm_loss(sp[mod].source(src[mod]), sub["z"], sub["lang"], weight=sub["ok"].float())
            total = total + cfg.w_cfm * l
            logs[f"cfm_{mod}"] = l.item()
        if "img" in sp and views:
            h_all = torch.cat([src[m][views[m]["ok"]] for m in views])
            it_all = torch.cat([views[m]["item"][views[m]["ok"]] for m in views])
            if len(h_all) > batch["B"]:
                pick = torch.randperm(len(h_all), device=dev)[: batch["B"]]
                h_all, it_all = h_all[pick], it_all[pick]
            l = sp["img"].cfm_loss(sp["img"].source(h_all), batch["img_lat"][it_all],
                                   torch.zeros(len(h_all), dtype=torch.long, device=dev))
            total = total + cfg.w_cfm * l
            logs["cfm_img"] = l.item()

        # ---- unpaired island speech
        if island is not None and bank is not None:
            if cfg.bt_weight > 0 and progress >= cfg.bt_warmup_frac:
                l_bt, bl = self.back_translate(model, island, bank)
                total = total + cfg.bt_weight * l_bt
                logs.update(bl)
            if cfg.dock_mode != "none" and progress >= cfg.dock_warmup_frac:
                l_dock, dl = self.dock(model, island, bank, progress)
                total = total + cfg.w_dock * l_dock
                logs.update(dl)
        if cfg.naive_cycle_weight > 0 and len(pairs[0]) > 1:
            t, a = views["txt"], views["aud"]
            lc = self.naive_cycle(model, "txt", t["z"][pairs[0]], t["lang"][pairs[0]], "aud", a["lang"][pairs[1]],
                                  steps=cfg.train_ode_steps, grad=True)
            total = total + cfg.naive_cycle_weight * lc
            logs["naive_cycle"] = lc.item()
        logs["loss"] = total.item()
        return total, logs

    # ============================================================================ hub-space back-translation
    def back_translate(self, model, island: Dict, bank: Callable[[int], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Both directions of back-translation, with the hub as the pivot language.

        write   real unpaired speech z_u, pseudo code h = Reader(z_u)      -> the writer learns  h -> z_u
        read    real hub code h (from any language), synthetic speech Write(h) -> the reader learns  Write(h) -> h

        The target side of each direction is real.  Pseudo codes drift towards frequent content when they are
        fed back on themselves, so before use they are moved part of the way along the entropic optimal
        transport map onto a batch of real hub codes (balanced marginals: the batch of pseudo codes has to
        cover the real distribution; ``bt_rectify`` is the displacement, 0 = off, 1 = full transport)."""
        cfg = self.cfg
        reader, spoke = model.hub.reader, model.spokes["aud"]
        z_u, lang_u = island["z"], island["lang"]
        n = z_u.shape[0]
        real = bank(cfg.bt_bank_mult * n).to(z_u.dtype)
        logs: Dict[str, float] = {}
        with torch.no_grad():
            h = reader(z_u, "aud", lang_u)
            if cfg.bt_rectify > 0:
                with torch.autocast(device_type=z_u.device.type, enabled=False):
                    cost = ot.scale_free(torch.cdist(h.float().flatten(1), real.float().flatten(1)) ** 2)
                    a = torch.full((n,), 1.0 / n, device=z_u.device)
                    b = torch.full((real.shape[0],), 1.0 / real.shape[0], device=z_u.device)
                    Pi = ot.log_sinkhorn(cost, a, b, cfg.bt_eps, cfg.dock_sinkhorn_iters)
                    rows = Pi / Pi.sum(1, keepdim=True).clamp_min(1e-30)
                    moved = (rows @ real.float().flatten(1)).view_as(h)
                    logs["bt_shift"] = ((moved - h.float()) ** 2).mean().item()
                    logs["bt_plan_entropy"] = ot.coupling_stats(Pi)["row_entropy"]
                h = (h.float() + cfg.bt_rectify * (moved - h.float())).to(z_u.dtype)
            synth = spoke.write(real[:n], lang_u, temp=1.0, steps=cfg.train_ode_steps)
        l_write = spoke.cfm_loss(spoke.source(h), z_u, lang_u)
        l_read = F.mse_loss(reader(synth, "aud", lang_u).float(), real[:n].float())
        logs.update(bt_write=l_write.item(), bt_read=l_read.item())
        return l_write + l_read, logs

    # ============================================================================ ablation: fused-GW docking
    def alpha(self, progress: float) -> float:
        cfg = self.cfg
        if cfg.dock_mode in ("gw", "w"):
            return 0.0 if cfg.dock_mode == "gw" else 1.0
        s = min(1.0, max(0.0, (progress - cfg.dock_warmup_frac) / max(1e-6, cfg.dock_anneal_frac)))
        return cfg.dock_alpha_start + (cfg.dock_alpha_end - cfg.dock_alpha_start) * s * s * (3 - 2 * s)

    def dock(self, model, island: Dict, bank: Callable[[int], torch.Tensor], progress: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Entropic fused Gromov-Wasserstein between the reader's codes for unpaired island speech and real hub
        codes, as a loss on the reader (envelope theorem).  Kept as the ablation of the original proposal:
        pure GW is blind to where the island sits, and neither variant has beaten the bridge pairs alone."""
        cfg = self.cfg
        h = model.hub.reader(island["z"], "aud", island["lang"])
        Y = bank(cfg.bt_bank_mult * h.shape[0])
        with torch.autocast(device_type=h.device.type, enabled=False):
            mu = Y.float().mean(0, keepdim=True)
            fs, ft = (h.float() - mu).flatten(1), (Y.float() - mu).flatten(1)
            C1, C2 = ot.scale_free(ot.pairwise_cos_dist(fs)), ot.scale_free(ot.pairwise_cos_dist(ft))
            M = ot.scale_free(ot.pairwise_cos_dist(fs, ft))
            alpha = self.alpha(progress)
            a = torch.full((fs.shape[0],), 1.0 / fs.shape[0], device=fs.device)
            init = a.unsqueeze(1) * torch.softmax(-M.detach() / cfg.dock_eps, dim=1)  # uniform is a fixed point
            Pi = ot.solve_fgw(C1.detach(), C2, M.detach(), alpha, cfg.dock_eps, cfg.dock_outer_iters,
                              cfg.dock_sinkhorn_iters, cfg.dock_marginal, init=init)
            value, gw, w = ot.fgw_value(C1, C2, Pi, M, alpha)
        stats = ot.coupling_stats(Pi)
        return value, {"dock": value.item(), "dock_alpha": alpha, "pi_row_entropy": stats["row_entropy"],
                       "pi_used_cols": stats["used_cols"]}

    # ============================================================================ diagnostic
    def naive_cycle(self, model, m1: str, z1, lang1, m2: str, lang2, steps: Optional[int] = None,
                    grad: bool = False) -> torch.Tensor:
        """The cycle of the original roadmap:  z_1 -> hub -> z_2 -> hub -> z_1  through ODE maps only.
        Every map is invertible, so the loss is zero up to solver error for *any* velocity field
        (cf. AlignFlow, DDIB).  Kept to demonstrate exactly that."""
        s1, s2 = model.spokes[m1], model.spokes[m2]
        x0 = s1.solve(z1, lang1, -1, steps=steps, grad=grad)
        z2 = s2.solve(x0, lang2, +1, steps=steps, grad=grad)
        x0b = s2.solve(z2, lang2, -1, steps=steps, grad=grad)
        z1b = s1.solve(x0b, lang1, +1, steps=steps, grad=grad)
        return F.mse_loss(x0b, x0) + F.mse_loss(z1b, z1)
