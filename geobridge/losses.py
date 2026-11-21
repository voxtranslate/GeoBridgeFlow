"""All training objectives of GeoBridge-Flow in one class.

Phase A - representation (codecs + grounder)
    L_A = w_text_rec * L_text  +  w_audio_rec * L_audio(sem, prosody, duration, mel, orth, gate price)
        + w_ground * L_ground  +  w_expl * L_expl  +  w_agree * L_agree  [+ w_slot_div * L_div]

Phase B - transport (flow spokes; codecs and grounder frozen)
    L_B = w_cfm * (L_cfm^txt + L_cfm^aud + L_cfm^img)          hub (+) private conditional flow matching
        + w_swap * L_swap                                       transitive hub-swap (cross-spoke exposure)
        + w_dock * L_dock  +  w_pseudo * L_pseudo               Gromov->Wasserstein docking of the island
        [+ twohop_weight * L_2hop  + naive_cycle_weight * L_cycle]   optional / diagnostic

No term ever pairs two different languages: cross-lingual transfer can only pass through the hub.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import ot


def cat_audio(a: Optional[Dict], b: Optional[Dict]) -> Optional[Dict]:
    """Concatenate two collated speech sub-batches (padding the time axis)."""
    if a is None or b is None:
        return a if b is None else b
    T = max(a["sem"].shape[1], b["sem"].shape[1])
    out = {}
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if k in ("sem", "tone", "mask", "mel"):
            L = 2 * T if k == "mel" else T
            x = F.pad(x, (0, 0, 0, L - x.shape[1])) if x.dim() == 3 else F.pad(x, (0, L - x.shape[1]))
            y = F.pad(y, (0, 0, 0, L - y.shape[1])) if y.dim() == 3 else F.pad(y, (0, L - y.shape[1]))
        if k == "item":
            y = torch.full_like(y, -1)
        out[k] = torch.cat([x, y], 0)
    return out


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
        self.island_ids = [lang2id[l] for l in cfg.island_langs if l in lang2id]
        self.hub_ids = [lang2id[l] for l in cfg.hub_langs if l in lang2id]
        self.last_coupling: Optional[Dict[str, torch.Tensor]] = None  # kept for the inferencer's figures

    # ============================================================================ phase A
    def phase_a(self, model, batch: Dict, island: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg, logs = self.cfg, {}
        total = batch["img"].new_zeros(())
        V = model.hub.project(batch["img"])
        keys = model.hub.keys(V)
        txt, aud = batch["txt"], batch["aud"]
        z: Dict[str, torch.Tensor] = {}
        if txt is not None:
            r = model.text.loss(txt)
            total = total + cfg.w_text_rec * r["loss"]
            z["txt"] = r["z"]
            logs.update(txt_ce=r["loss"].item(), txt_acc=r["byte_acc"].item())
        aud_all = cat_audio(aud, island["aud"] if island is not None else None)
        if aud_all is not None:
            r = model.audio.loss(aud_all)
            total = total + cfg.w_audio_rec * r["loss"]
            if aud is not None:
                z["aud"] = r["z"][: aud["sem"].shape[0]]
            logs.update(aud_sem=r["sem_mse"].item(), aud_f0=r["f0_l1"].item(), aud_dur=r["dur_l1"].item())
            for k in ("mel_l1", "orth", "gate_pen"):
                if k in r:
                    logs[k] = r[k].item()

        # ---- grounding: which patches does each utterance talk about?
        sims, items, outs, expl = [], [], {}, []
        for mod, sub in (("txt", txt), ("aud", aud)):
            if mod not in z:
                continue
            o = model.paired_hub(mod, z[mod], sub["lang"], V[sub["item"]])
            o["bary"], _ = model.hub.attend(o["q"], V[sub["item"]])
            outs[mod] = o
            sims.append(model.hub.similarity(o["q"], keys))
            items.append(sub["item"])
            if cfg.w_expl > 0:  # the hub code alone must explain the utterance
                pred = model.hub.explainer(o["bary"], model.hub.MODS[mod], sub["lang"])
                expl.append(((pred.float() - z[mod].detach().float()) ** 2).mean(dim=(1, 2)))
        if sims:
            logits = torch.cat(sims).float() / cfg.ground_temp
            item = torch.cat(items)
            B = batch["B"]
            same = batch["img_idx"][item].unsqueeze(1) == batch["img_idx"].unsqueeze(0)  # [n, B]
            pos = F.one_hot(item, B).bool()
            l_u2i = F.cross_entropy(logits.masked_fill(same & ~pos, -1e4), item)
            logp = logits.t().log_softmax(dim=1)                                          # [B, n]
            has = same.t().any(1)
            l_i2u = -torch.logsumexp(logp.masked_fill(~same.t(), -1e4), dim=1)[has].mean()
            ground = 0.5 * (l_u2i + l_i2u)
            attn = torch.cat([o["attn"] for o in outs.values()])
            a = F.normalize(attn, dim=-1)
            sim = a @ a.transpose(1, 2)
            K = sim.shape[-1]
            div = (sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)).mean() / max(1, K * (K - 1))
            total = total + cfg.w_ground * ground + cfg.w_slot_div * div
            if cfg.w_null > 0 and model.hub.null_enabled:  # pay for attention: unused slots retire to null
                use = attn.sum(-1).mean()
                total = total + cfg.w_null * use
                logs["slot_use"] = use.item()
            if expl:
                unexplained = torch.cat(expl).mean()  # latents are unit-variance: this is 1 - R^2
                total = total + cfg.w_expl * unexplained
                logs["unexplained"] = unexplained.item()
            logs.update(ground=ground.item(), slot_div=div.item(),
                        ground_acc=(logits.masked_fill(same & ~pos, -1e4).argmax(1) == item).float().mean().item())
            it, ia = same_lang_pairs(txt, aud)
            if len(it) and "txt" in outs and "aud" in outs:
                agree = F.mse_loss(outs["txt"]["bary"][it], outs["aud"]["bary"][ia])
                total = total + cfg.w_agree * agree
                logs["agree"] = agree.item()
            if cfg.hub_mode == "free":
                hub = torch.cat([o["hub"] for o in outs.values()])
                var = F.relu(1.0 - hub.float().flatten(1).std(0)).mean()
                total = total + cfg.w_free_hub_var * var
                logs["free_var"] = var.item()
        gates = model.audio.gate().detach()
        for name, i in model.lang2id.items():
            logs[f"gate_{name}"] = gates[i].item()
        logs["loss"] = total.item()
        return total, logs

    # ============================================================================ phase B
    def _text_anchor(self, model, batch, z, mod, sub, pairs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Text-anchor baseline: which text latent serves as the hub of each view, and whether one exists."""
        n = z[mod].shape[0]
        dev = z[mod].device
        hub_lang = torch.tensor([int(l) in self.hub_ids for l in sub["lang"].tolist()], device=dev)
        anchor = z[mod].clone() if mod == "txt" else torch.zeros_like(z[mod])
        valid = hub_lang.clone() if mod == "txt" else torch.zeros(n, dtype=torch.bool, device=dev)
        if mod == "aud" and "txt" in z and len(pairs[0]):  # hub-language speech: its own transcript
            anchor[pairs[1]] = z["txt"][pairs[0]]
            valid[pairs[1]] = True
        piv = batch.get("pivot")
        if piv is not None and "pivot" in z:  # everything else: a caption of the same image (image pivot)
            row = torch.full((batch["B"],), -1, dtype=torch.long, device=dev)
            row[piv["item"]] = torch.arange(len(piv["item"]), device=dev)
            r = row[sub["item"]]
            use = (~valid) & (r >= 0)
            anchor[use] = z["pivot"][r[use]]
            valid = valid | use
        return anchor, valid

    def phase_b(self, model, batch: Dict, island: Optional[Dict], progress: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg, logs = self.cfg, {}
        sp = model.spokes
        free = cfg.hub_mode == "free"
        txt, aud = batch["txt"], batch["aud"]
        with torch.no_grad():
            V = model.hub.project(batch["img"])
            z = {m: model.encode(m, s) for m, s in (("txt", txt), ("aud", aud)) if s is not None}
            if cfg.hub_mode == "text" and batch.get("pivot") is not None:
                z["pivot"] = model.encode("txt", batch["pivot"])
            z_img = model.image_latent(V)
        pairs = same_lang_pairs(txt, aud)
        hubs: Dict[str, torch.Tensor] = {}
        valid: Dict[str, torch.Tensor] = {}
        with torch.set_grad_enabled(free):
            for mod, sub in (("txt", txt), ("aud", aud)):
                if mod not in z:
                    continue
                piv_z, ok = (None, None)
                if cfg.hub_mode == "text":
                    piv_z, ok = self._text_anchor(model, batch, z, mod, sub, pairs)
                hubs[mod] = model.paired_hub(mod, z[mod], sub["lang"], V[sub["item"]], piv_z)["hub"]
                valid[mod] = ok if ok is not None else torch.ones(z[mod].shape[0], dtype=torch.bool, device=V.device)

        total = V.new_zeros(())
        # ---- hub (+) private conditional flow matching on every image-paired view
        for mod, sub in (("txt", txt), ("aud", aud)):
            if mod not in z:
                continue
            w = valid[mod].float()
            l, _ = sp[mod].cfm_loss(sp[mod].source(hubs[mod]), z[mod], sub["lang"], weight=w)
            total = total + cfg.w_cfm * l
            logs[f"cfm_{mod}"] = l.item()
        if "img" in sp and hubs:
            h_all = torch.cat([hubs[m][valid[m]] for m in hubs])
            it_all = torch.cat([(txt if m == "txt" else aud)["item"][valid[m]] for m in hubs])
            if len(h_all) > batch["B"]:
                pick = torch.randperm(len(h_all), device=V.device)[: batch["B"]]
                h_all, it_all = h_all[pick], it_all[pick]
            lang0 = torch.zeros(len(h_all), dtype=torch.long, device=V.device)
            l, _ = sp["img"].cfm_loss(sp["img"].source(h_all), z_img[it_all], lang0)
            total = total + cfg.w_cfm * l
            logs["cfm_img"] = l.item()
        if free:
            hub = torch.cat(list(hubs.values()))
            var = F.relu(1.0 - hub.float().flatten(1).std(0)).mean()
            total = total + cfg.w_free_hub_var * var
            logs["free_var"] = var.item()

        # ---- transitive hub-swap: train each spoke on hub codes *read by the other spoke's ODE*
        it, ia = pairs
        if cfg.w_swap > 0 and cfg.swap_frac > 0 and len(it) > 1:
            n = max(1, int(round(cfg.swap_frac * len(it))))
            sel = torch.randperm(len(it), device=V.device)[:n]
            st, sa = it[sel], ia[sel]
            h_from_t, _ = sp["txt"].read(z["txt"][st], txt["lang"][st], steps=cfg.train_ode_steps)
            h_from_a, _ = sp["aud"].read(z["aud"][sa], aud["lang"][sa], steps=cfg.train_ode_steps)
            l1, _ = sp["aud"].cfm_loss(sp["aud"].source(h_from_t), z["aud"][sa], aud["lang"][sa])
            l2, _ = sp["txt"].cfm_loss(sp["txt"].source(h_from_a), z["txt"][st], txt["lang"][st])
            swap = 0.5 * (l1 + l2)
            total = total + cfg.w_swap * swap
            logs["swap"] = swap.item()
            with torch.no_grad():
                logs["hub_gap"] = F.mse_loss(h_from_t, h_from_a).item()
        if cfg.twohop_weight > 0 and len(it) > 1:
            l2h = self.twohop(model, z["txt"][it], txt["lang"][it], z["aud"][ia], aud["lang"][ia])
            total = total + cfg.twohop_weight * l2h
            logs["twohop"] = l2h.item()
        if cfg.naive_cycle_weight > 0 and len(it) > 1:
            lc = self.naive_cycle(model, "txt", z["txt"][it], txt["lang"][it], "aud", aud["lang"][ia],
                                  steps=cfg.train_ode_steps, grad=True)
            total = total + cfg.naive_cycle_weight * lc
            logs["naive_cycle"] = lc.item()

        # ---- docking the unpaired island onto the hub
        if hubs:
            model.bank.push(torch.cat([hubs[m][valid[m]] for m in hubs]))
        if island is not None and cfg.dock_mode != "none" and progress >= cfg.dock_warmup_frac \
                and int(model.bank.count) >= min(cfg.dock_bank_batch, cfg.dock_bank_size) // 2:
            pin = None
            if cfg.dock_pin and "aud" in z and self.island_ids:
                is_isl = torch.isin(aud["lang"], torch.tensor(self.island_ids, device=V.device))
                if is_isl.any():
                    pin = (z["aud"][is_isl], aud["lang"][is_isl], hubs["aud"][is_isl].detach())
            d_loss, p_loss, dl = self.dock(model, island["aud"], progress, pin)
            total = total + cfg.w_dock * d_loss + cfg.w_pseudo * p_loss
            logs.update(dl)
        logs["loss"] = total.item()
        return total, logs

    # ============================================================================ docking (N3)
    def alpha(self, progress: float) -> float:
        cfg = self.cfg
        if cfg.dock_mode == "gw":
            return 0.0
        if cfg.dock_mode == "w":
            return 1.0
        s = (progress - cfg.dock_warmup_frac) / max(1e-6, cfg.dock_anneal_frac)
        s = min(1.0, max(0.0, s))
        s = s * s * (3 - 2 * s)  # smoothstep
        return cfg.dock_alpha_start + (cfg.dock_alpha_end - cfg.dock_alpha_start) * s

    def dock(self, model, isl: Dict, progress: float, pin=None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Gromov->Wasserstein docking.

        Island speech is read by its own spoke's backward ODE into hub coordinates.  Its batch is coupled to a
        sample of the hub bank by an entropic fused-GW plan whose cost starts as pure relational geometry
        (alpha = 0: trust only the shape of the island) and is annealed towards the model-induced Wasserstein
        cost (alpha -> 1: trust where the island currently sits).  The plan is used twice:
        (i) as a differentiable docking loss on the ODE end-points (envelope theorem), and
        (ii) to draw pseudo-pairs for coupling-guided flow matching, weighted by the plan's own confidence.
        """
        cfg = self.cfg
        spoke = model.spokes["aud"]
        with torch.no_grad():
            z_u = model.encode("aud", isl)
        lang_u = isl["lang"]
        h_u, _ = spoke.read(z_u, lang_u, grad=cfg.w_dock > 0, steps=cfg.train_ode_steps)
        Y = model.bank.sample(cfg.dock_bank_batch).to(h_u.dtype)
        mu = model.bank.mean()
        n_u, n_b = h_u.shape[0], Y.shape[0]
        src, tgt, R = h_u, Y, None
        if pin is not None:
            z_p, lang_p, hub_p = pin
            with torch.no_grad():
                h_p, _ = spoke.read(z_p, lang_p, steps=cfg.train_ode_steps)
            src, tgt = torch.cat([h_u, h_p]), torch.cat([Y, hub_p.to(Y.dtype)])
            n_p = h_p.shape[0]
            R = torch.zeros(n_u + n_p, n_b + n_p, device=h_u.device)
            R[n_u:, :] = cfg.dock_pin_cost
            R[n_u + torch.arange(n_p), n_b + torch.arange(n_p)] = 0.0
        with torch.autocast(device_type=h_u.device.type, enabled=False):
            fs = (src.float() - mu.float()).flatten(1)
            ft = (tgt.float() - mu.float()).flatten(1)
            C1 = ot.scale_free(ot.pairwise_cos_dist(fs))
            C2 = ot.scale_free(ot.pairwise_cos_dist(ft))
            M = ot.scale_free(ot.pairwise_cos_dist(fs, ft))
            alpha = self.alpha(progress)
            # warm start from the model-induced plan: from a uniform start the GW iteration sits on the
            # uniform plan, which is a stationary point (every row then prefers the same "central" columns)
            a = torch.full((fs.shape[0],), 1.0 / fs.shape[0], device=fs.device)
            init = a.unsqueeze(1) * torch.softmax(-(M.detach() + (R if R is not None else 0.0)) / cfg.dock_eps, dim=1)
            Pi = ot.solve_fgw(C1.detach(), C2, M.detach(), alpha, cfg.dock_eps, cfg.dock_outer_iters,
                              cfg.dock_sinkhorn_iters, cfg.dock_marginal, R, init=init)
            value, gw, w = ot.fgw_value(C1, C2, Pi, M, alpha)
        logs = {"dock": value.item(), "dock_gw": gw.item(), "dock_w": w.item(), "dock_alpha": alpha}
        logs.update({f"pi_{k}": v for k, v in ot.coupling_stats(Pi[:n_u]).items()})
        self.last_coupling = {"Pi": Pi.detach().cpu(), "n_island": n_u, "n_bank": n_b}
        pseudo = h_u.new_zeros(())
        if cfg.dock_pseudo_pair and cfg.w_pseudo > 0:
            rows = Pi[:n_u] / Pi[:n_u].sum(1, keepdim=True).clamp_min(1e-30)
            j = torch.multinomial(rows, 1).squeeze(1)
            conf = ot.row_confidence(Pi[:n_u])
            pseudo, _ = spoke.cfm_loss(spoke.source(tgt[j].detach()), z_u, lang_u, weight=conf)
            logs.update(pseudo=pseudo.item(), pi_conf=conf.mean().item())
        return value, pseudo, logs

    # ============================================================================ optional / diagnostic terms
    def twohop(self, model, z_t, lang_t, z_a, lang_a) -> torch.Tensor:
        """Differentiable 2-hop hub-swap reconstruction: read both views, exchange the hub parts, keep each
        view's own private code, write back, and ask for the original latents.  Unlike a plain ODE cycle this
        is not satisfied by construction: it holds only if the two spokes agree on the hub content."""
        sp = model.spokes
        n = self.cfg.train_ode_steps
        ht, pt = sp["txt"].read(z_t, lang_t, grad=True, steps=n)
        ha, pa = sp["aud"].read(z_a, lang_a, grad=True, steps=n)
        za = sp["aud"].write(ht, lang_a, private=pa, grad=True, steps=n)
        zt = sp["txt"].write(ha, lang_t, private=pt, grad=True, steps=n)
        return F.mse_loss(za, z_a) + F.mse_loss(zt, z_t) + F.mse_loss(ht, ha)

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
