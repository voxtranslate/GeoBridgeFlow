"""All training objectives of GeoBridge-Flow in one class.

Phase A - representation (codecs, grounder, explainer, reader, preview decoder)
    L_A = w_text_rec * L_text + w_audio_rec * L_audio(segmental, prosody, duration, mel, orth, gate price)
        + w_ground * L_ground        the hub code is found in the right image          (contrastive)
        + w_expl   * L_expl          the hub code explains the sentence                (sufficiency)
        + w_seek   * L_seek          ... and every slot is pulled to the patch that would explain more
                                     (counterfactual patch credit)
        + w_read   * L_read          an image-free reader predicts the hub code
        + w_min    * L_min           ... and the hub code must be predictable by it    (minimality)
        + w_agree  * L_agree  + w_pix * L_pix

Phase B - transport (writers = flows, plus the reader), computed on *cached* latents: the frozen codecs and the
grounder are not run again.
    L_B = w_cfm * (L_cfm^txt + L_cfm^aud + L_cfm^img)     hub (+) private flow matching; a share of the views
                                                          is written from a *read* hub (exposure)
        + w_read * L_read                                  keeps the reader on the grounded codes
        + w_dec * L_dec                                    the byte decoder learns to read written latents
        + bt_weight * (L_bt_write + L_bt_read + bt_relay * L_bt_relay)   hub-space back-translation of unpaired
                                                          island speech, relayed to the text writers
        [+ w_dock * L_dock + naive_cycle_weight * L_cycle] ablation / diagnostic terms, off by default

No term ever pairs two different languages: cross-lingual transfer can only pass through the hub.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def preview_loss(pred: torch.Tensor, target: torch.Tensor, contrast: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contrast-weighted L1 between RGB images [B, 3, S, S] in [0, 1]; also returns the plain L1.

    A pixel counts in proportion to how far it is from the batch's mean colour.  Plain L1 is minimised by the
    median image, which is the bare background wherever the content covers a small part of the picture; a
    decoder trained that way never starts to use its input (measured: an all-background preview)."""
    err = (pred - target).abs()
    away = (target - target.mean(dim=(0, 2, 3), keepdim=True)).abs().mean(1, keepdim=True)
    w = 1.0 + contrast * away / away.mean().clamp_min(1e-6)
    return (w * err).sum() / (w.sum() * err.shape[1]), err.mean()


def lang_weights(lang: torch.Tensor, balance: bool) -> torch.Tensor:
    """Per-view weights of the views of one modality.  ``balance``: every language present gets the same total
    weight (the weights still sum to the number of views), so the loss is the mean over languages of the
    per-language means.  Otherwise all ones."""
    if not balance or lang.numel() == 0:
        return torch.ones(lang.shape[0], device=lang.device)
    ids, inv, counts = torch.unique(lang, return_inverse=True, return_counts=True)
    return (lang.shape[0] / ids.numel()) / counts[inv].float()


def wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean of the per-view values ``x`` under the per-view weights ``w`` (the plain mean when they are all ones)."""
    return (x * w).sum() / w.sum().clamp_min(1e-6)


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
            n_paired = 0 if aud is None else aud["sem"].shape[0]
            # (listening speech is the last of the paired views and is not grounded: it pays for no gate)
            r = model.audio.loss(aud_all, n_grounded=n_paired - (int(aud["listen"].sum()) if aud is not None and "listen" in aud else 0))
            total = total + cfg.w_audio_rec * r["loss"]
            if aud is not None:
                z["aud"] = r["z"][:n_paired]
            logs.update({k: r[k].item() for k in ("sem_mse", "speak_mse", "f0_l1", "dur_l1", "mel_l1", "orth") if k in r})
            if "gate_pen" in r:
                logs["tone_price"] = r["gate_pen"].item()   # what the batch paid for open gates (not a gate value)

        # ---- listening speech (extra spoken captions, cfg.listen_only): it has been through the codec like all
        #      speech; from here on it is only READ - into the code that the text of the same caption and the image
        #      define - and is left out of every term that shapes the hub.  A hub that also has to make room for a
        #      flood of speech in the high-resource language does so at the expense of the island language.
        heard = None
        if aud is not None and "listen" in aud and "aud" in z and "txt" in z:
            hl = aud["listen"]
            it_h, ia_h = same_lang_pairs(txt, {"item": aud["item"][hl], "lang": aud["lang"][hl]})
            if len(ia_h):
                heard = {"z": z["aud"][hl][ia_h].detach(), "lang": aud["lang"][hl][ia_h], "txt": it_h}
            aud = {"item": aud["item"][~hl], "lang": aud["lang"][~hl]}   # the speech views that are grounded
            z["aud"] = z["aud"][~hl]
            if not len(aud["item"]):
                del z["aud"]

        # ---- grounding: one attention pass gives the contrastive scores and the hub code
        scores, items, codes, attns, logps, vw = [], [], {}, {}, {}, {}
        released = hub.null_enabled  # minimality starts when the dense stage ends
        for mod, sub in (("txt", txt), ("aud", aud)):
            if mod not in z:
                continue
            q = hub.queries(z[mod], mod, sub["lang"])
            score, attns[mod], logps[mod] = hub.ground(q, keys, sub["item"])
            codes[mod] = torch.einsum("nkp,npd->nkd", attns[mod].to(V.dtype), V[sub["item"]][:, :-1])
            scores.append(score)
            items.append(sub["item"])
            vw[mod] = lang_weights(sub["lang"], cfg.lang_balance)   # per-view weights (ones unless lang_balance)
        if scores:
            logits = torch.cat(scores) / cfg.ground_temp
            item = torch.cat(items)
            same = batch["img_idx"][item].unsqueeze(1) == batch["img_idx"].unsqueeze(0)  # [n, B]
            pos = F.one_hot(item, batch["B"]).bool()
            masked = logits.masked_fill(same & ~pos, -1e4)
            l_u2i = wmean(F.cross_entropy(masked, item, reduction="none"), torch.cat([vw[m] for m in codes]))
            logp = logits.t().log_softmax(dim=1)                                          # [B, n]
            l_i2u = -torch.logsumexp(logp.masked_fill(~same.t(), -1e4), dim=1)[same.t().any(1)].mean()
            ground = 0.5 * (l_u2i + l_i2u)
            total = total + cfg.w_ground * ground
            logs.update(ground=ground.item(), ground_acc=(masked.argmax(1) == item).float().mean().item())

            # the canonical code of a caption: where a transcript exists, the text-grounded code teaches speech
            # (same language only); text is the cleaner view, and speech then inherits everything it covers
            teacher = {m: c.detach() for m, c in codes.items()}
            it, ia = same_lang_pairs(txt, aud)
            if len(it) and len(codes) == 2:
                if cfg.agree_mode == "teach":
                    teacher["aud"] = teacher["aud"].clone()
                    teacher["aud"][ia] = teacher["txt"][it]
                if cfg.w_agree > 0:
                    agree = F.mse_loss(codes["aud"][ia], teacher["txt"][it] if cfg.agree_mode == "teach" else codes["txt"][it])
                    total = total + cfg.w_agree * agree
                    logs["agree"] = agree.item()
            for mod, sub in (("txt", txt), ("aud", aud)):
                if mod not in codes:
                    continue
                mid = hub.MODS[mod]
                if cfg.w_expl > 0:  # sufficiency: the code alone reconstructs the utterance latent
                    target, lang = z[mod][:, : cfg.hub_tokens_of(mod)].detach().float(), sub["lang"]   # what the hub reads

                    def unexplained(code: torch.Tensor, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
                        """Per-utterance unexplained variance of code [m, K, d] (for the utterances ``rows``)."""
                        t, l = (target, lang) if rows is None else (target[rows], lang[rows])
                        return ((hub.explainer(code, mid, l).float() - t) ** 2).mean(dim=(1, 2))

                    err = unexplained(codes[mod])
                    unexp = err.mean()
                    total = total + cfg.w_expl * wmean(err, vw[mod])
                    logs[f"unexplained_{mod}"] = unexp.item()
                    if cfg.w_seek > 0 and codes[mod].requires_grad:  # a training signal only (needs a gradient)
                        seek, moved = self.seek(unexplained, err, codes[mod], logps[mod], V[sub["item"]][:, :-1], released,
                                                vw[mod])
                        total = total + cfg.w_seek * seek
                        logs[f"seek_{mod}"] = moved
                n, z_in, lang_in, goal, w = len(sub["lang"]), z[mod].detach(), sub["lang"], teacher[mod], vw[mod]
                if mod == "aud" and heard is not None:   # listening speech joins the reader's batch, one view one vote
                    z_in, lang_in = torch.cat([z_in, heard["z"]]), torch.cat([lang_in, heard["lang"]])
                    goal, w = torch.cat([goal, teacher["txt"][heard["txt"]]]), torch.cat([w, torch.ones_like(heard["lang"], dtype=w.dtype)])
                pred = hub.reader(z_in, mod, lang_in).float()
                per_view = ((pred - goal.float()) ** 2).mean(dim=(1, 2))
                total = total + cfg.w_read * wmean(per_view, w)
                logs[f"read_{mod}"] = per_view[:n].mean().item()
                if len(per_view) > n:
                    logs["heard"] = per_view[n:].mean().item()   # reader error on the listening speech
                pred = pred[:n]
                if cfg.w_min > 0 and released:  # minimality: nothing in the code that the sentence does not say
                    total = total + cfg.w_min * wmean(((codes[mod].float() - pred.detach()) ** 2).mean(dim=(1, 2)), vw[mod])
                logs[f"slot_use_{mod}"] = attns[mod].sum(-1).mean().item()
            if cfg.w_expl > 0:
                logs["unexplained"] = sum(v for k, v in logs.items() if k.startswith("unexplained_")) / len(codes)
            logs["read"] = sum(v for k, v in logs.items() if k.startswith("read_")) / len(codes)

        # ---- preview pixel decoder (frozen image latent -> thumbnail)
        if "thumb" in batch and hasattr(model, "pixel"):
            target = batch["thumb"].permute(0, 3, 1, 2).float() / 255.0
            pix, l1 = preview_loss(model.pixel(model.image_latent(V)).float(), target, cfg.pix_contrast)
            total = total + cfg.w_pix * pix
            logs["pix_l1"] = l1.item()
        gates = model.audio.gate().detach()
        for name, i in model.lang2id.items():
            if name not in cfg.text_only_langs:   # a language without speech has a gate parameter nothing ever trains
                logs[f"gate_{name}"] = gates[i].item()
        logs["loss"] = total.item()
        return total, logs

    def seek(self, unexplained, err: torch.Tensor, code: torch.Tensor, logp: torch.Tensor, patches: torch.Tensor,
             allow_null: bool, view_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Counterfactual patch credit ("look where it would explain more").

        A slot's hub token is a barycentre, so sufficiency reaches the attention only through the softmax
        Jacobian  d code_k / d logit_p = a_kp (v_p - code_k): a patch the slot does not look at yet (a_kp ~ 0)
        gets no credit, however much it would explain.  Once the contrastive loss (which is content with one
        object) has made the slots sharp, the second thing a sentence says can stay unseen for ever.

        propose  the counterfactual "slot k sits on patch p" is cheap to rank for all (k, p) at once:
                     gain_kp = - < d err / d code_k ,  v_p - code_k >        (v = 0 for the null option)
                 is the first-order drop of the utterance's unexplained variance.  Patches already taken by
                 earlier slots are not candidates.  Every slot keeps its best candidate.
        pull     cross-entropy on the slot's log-attention towards its candidate, weighted by
                 gain^2 / (gain^2 + seek_gain^2): a code that already explains its sentence is left alone.
                 The log-attention is not clamped, so the gradient (a - onehot) reaches unattended patches.
                 The threshold is absolute, in units of latent variance.  Judging the gain relative to the
                 utterance's unexplained variance was tried and halves the coverage it buys (measured): a
                 nearly explained sentence then gets pulled on noise.
        veto     (``seek_verify``, off by default) each proposed move is carried out and the explainer is run
                 on it; a move that verifiably explains *less* is dropped.  Measured: fewer slots on objects
                 the sentence does not mention, but also fewer second objects found, because the explainer has
                 to co-adapt before a newly covered object pays off.  Requiring a verified *gain* up front
                 switches the mechanism off altogether.

        unexplained(code, rows) -> per-utterance error;  err = unexplained(code) [n];  code [n, K, d];
        logp [n, K, P + 1] log-attention with the null option last;  patches [n, P, d].
        Returns the loss and the mean weight (share of slots being moved)."""
        cfg = self.cfg
        n, K, _ = code.shape
        rows = torch.arange(n, device=code.device)
        scale = float(cfg.max_tokens * cfg.latent_dim)  # differentiate the sum of squares: O(1) values, safe in fp16
        g = torch.autograd.grad(err.sum() * scale, code, retain_graph=True)[0].float() / scale   # [n, K, d]
        with torch.no_grad():
            attn = logp[..., :-1].exp()
            here = (g * code.float()).sum(-1, keepdim=True)                                   # [n, K, 1]
            gain = here - torch.einsum("nkd,npd->nkp", g, patches.float())                    # [n, K, P]
            gain = gain.masked_fill(attn.cumsum(1) - attn > 0.5, -1e4)                        # taken by earlier slots
            gain = torch.cat([gain, here if allow_null else torch.full_like(here, -1e4)], -1)  # the origin
            best, idx = gain.max(-1)                                                          # [n, K]
            w = best.clamp_min(0) ** 2
            w = w / (w + cfg.seek_gain ** 2)
            if cfg.seek_verify:
                value = torch.cat([patches, torch.zeros_like(patches[:, :1])], 1)[rows.unsqueeze(1), idx]   # [n, K, d]
                trial = code.detach().unsqueeze(1).repeat(1, K, 1, 1)                         # [n, move, slot, d]
                k = torch.arange(K, device=code.device)
                trial[:, k, k] = value.to(trial.dtype)                                        # move k replaces slot k
                after = unexplained(trial.flatten(0, 1), rows.repeat_interleave(K)).view(n, K)
                w = w * (after <= err.detach().unsqueeze(1) + cfg.seek_gain)
        pull = w if view_weight is None else w * view_weight.unsqueeze(1)   # (the reported share of slots moved is unweighted)
        return -(pull * logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1).float()).mean(), float(w.mean())

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
                bank=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``batch`` comes from the latent cache: per modality the latents ``z``, the grounded hub codes
        ``hub``, ``lang``, ``item`` and ``ok``; plus the image latents ``img_lat`` [B, N, d].  ``bank`` is the
        latent cache itself (random real hub codes / views for the unpaired-speech terms)."""
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

        # ---- the reader keeps training, now on codes that no longer move; its (detached) outputs are the read
        #      hubs for exposure
        reads: Dict[str, torch.Tensor] = {}
        for mod, sub in views.items():
            ok, target = sub["ok"], hubs[mod].detach()
            z_in = sub.get("z_read", sub["z"])   # speech: possibly a time-masked encoding of the same utterance
            if cfg.read_noise > 0 and reader.training:   # the cached latents never change: without noise the
                sigma = torch.rand(z_in.shape[0], 1, 1, device=dev) * cfg.read_noise   # reader memorises them
                z_in = z_in + sigma.to(z_in.dtype) * torch.randn_like(z_in)
            r = reader(z_in, mod, sub["lang"])
            per_view = ((r[ok].float() - target[ok].float()) ** 2).mean(dim=(1, 2))
            total = total + cfg.w_read * (per_view * lang_weights(sub["lang"][ok], cfg.lang_balance)).mean()
            logs[f"read_{mod}"] = per_view.mean().item()
            reads[mod] = r.detach()
            if cfg.draft_t0 > 0 and cfg.hub_mode == "bary":  # the explainer is refined only if writing starts from its draft
                l_expl = F.mse_loss(model.hub.explainer(target[ok], model.hub.MODS[mod], sub["lang"][ok]).float(),
                                    sub["z"][ok][:, : cfg.hub_tokens_of(mod)].float())
                total = total + cfg.w_expl * l_expl
                logs[f"unexplained_{mod}"] = l_expl.item()

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

        # ---- hub (+) private conditional flow matching.  The hub part of the source is jittered: a writer has
        #      to tolerate a code that was read with some error (the island reader's most of all).
        if cfg.hub_jitter > 0:
            for mod in src:
                sigma = torch.rand(src[mod].shape[0], 1, 1, device=dev) * cfg.hub_jitter
                src[mod] = src[mod] + sigma.to(src[mod].dtype) * torch.randn_like(src[mod])
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

        # ---- write-latent exposure of the byte decoder.  The codec was trained on encoder latents; at test
        #      time it only ever sees latents that a writer produced from a hub code, which are blurred wherever
        #      the hub leaves the wording open.  Here the decoder learns to turn those into text (the encoder is
        #      frozen, so the cached latents stay valid).
        if cfg.w_dec > 0 and "txt" in views and "bytes" in views["txt"]:
            t = views["txt"]
            temp = cfg.eval_temp + (1.0 - cfg.eval_temp) * float(torch.rand(()))
            with torch.no_grad():
                z_w = sp["txt"].write(src["txt"], t["lang"], temp=temp, steps=cfg.train_ode_steps)
                written = (torch.rand(z_w.shape[0], device=dev) < cfg.p_dec_written) & t["ok"]
                z_d = torch.where(written.view(-1, 1, 1), z_w.to(t["z"].dtype), t["z"])
            l_dec, acc = model.text.decode_loss(z_d, t["bytes"], t["lang"])
            total = total + cfg.w_dec * l_dec
            logs.update(dec_ce=l_dec.item(), dec_acc=acc.item())

        # ---- write-latent exposure of the reader.  Back-translation teaches the reader to read island speech that
        #      the speech writer wrote; nothing taught it to read written speech of a hub language, and on Flickr8k
        #      it could not (written English speech, read back, carried no content while written Yoruba did).
        #      Real hub codes are written as speech in a spoken hub language and read back.
        if cfg.w_reread > 0 and "aud" in sp and views:
            h_real = torch.cat([hubs[m][views[m]["ok"]] for m in views]).detach()
            h_real = h_real[torch.randperm(len(h_real), device=dev)[: cfg.batch_size]]
            spoken = torch.tensor([model.lang2id[l] for l in cfg.hub_langs], device=dev)
            lang_s = spoken[torch.randint(len(spoken), (len(h_real),), device=dev)]
            with torch.no_grad():
                synth = sp["aud"].write(h_real, lang_s, temp=1.0, steps=cfg.train_ode_steps)
            l_rr = F.mse_loss(reader(synth, "aud", lang_s).float(), h_real.float())
            total = total + cfg.w_reread * l_rr
            logs["reread"] = l_rr.item()

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
    def back_translate(self, model, island: Dict, bank) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Back-translation of unpaired island speech, with the hub as the pivot language.

        write   real unpaired speech z_u, pseudo code h = Reader(z_u)          -> speech writer learns  h -> z_u
        read    real hub code h, synthetic island speech s = Write(h)          -> reader learns         s -> h
        relay   the same s, read back: h' = Reader(s); real text z_t of h      -> text writer learns    h' -> z_t

        The target side of every direction is real.  ``relay`` is what connects the island to the writers of
        the other languages: no caption exists in both, so without it a text writer only ever sees hub codes
        read from its own language, never ones carrying the island reader's errors.

        Pseudo codes drift towards frequent content when they are fed back on themselves, so before use they
        are moved part of the way along the entropic optimal transport map onto a batch of real hub codes
        (balanced marginals: the batch of pseudo codes has to cover the real distribution; ``bt_rectify`` is
        the displacement, 0 = off, 1 = full transport).  ``bank`` is the latent cache."""
        cfg = self.cfg
        reader, spoke = model.hub.reader, model.spokes["aud"]
        z_u, lang_u = island["z"], island["lang"]
        n, dev = z_u.shape[0], z_u.device
        logs: Dict[str, float] = {}
        text = bank.sample_views("txt", n, own_only=True) if cfg.bt_relay > 0 else None
        real = (text["hub"] if text is not None else bank.sample_hubs(n)).to(dev, z_u.dtype)
        with torch.no_grad():
            h = reader(z_u, "aud", lang_u)
            if cfg.bt_rectify > 0:
                ref = bank.sample_hubs(cfg.bt_bank_mult * n).to(dev)
                with torch.autocast(device_type=dev.type, enabled=False):
                    cost = ot.scale_free(torch.cdist(h.float().flatten(1), ref.float().flatten(1)) ** 2)
                    a = torch.full((n,), 1.0 / n, device=dev)
                    b = torch.full((ref.shape[0],), 1.0 / ref.shape[0], device=dev)
                    Pi = ot.log_sinkhorn(cost, a, b, cfg.bt_eps, cfg.dock_sinkhorn_iters)
                    rows = Pi / Pi.sum(1, keepdim=True).clamp_min(1e-30)
                    moved = (rows @ ref.float().flatten(1)).view_as(h)
                    logs["bt_shift"] = ((moved - h.float()) ** 2).mean().item()
                    logs["bt_plan_entropy"] = ot.coupling_stats(Pi)["row_entropy"]
                h = (h.float() + cfg.bt_rectify * (moved - h.float())).to(z_u.dtype)
            synth = spoke.write(real, lang_u, temp=1.0, steps=cfg.train_ode_steps)
        l_write = spoke.cfm_loss(spoke.source(h), z_u, lang_u)
        back = reader(synth, "aud", lang_u)
        l_read = F.mse_loss(back.float(), real.float())
        total = l_write + l_read
        logs.update(bt_write=l_write.item(), bt_read=l_read.item())
        if text is not None:
            writer = model.spokes["txt"]
            l_relay = writer.cfm_loss(writer.source(back.detach()), text["z"].to(dev, z_u.dtype), text["lang"].to(dev))
            total = total + cfg.bt_relay * l_relay
            logs["bt_relay"] = l_relay.item()
        return total, logs

    # ============================================================================ ablation: fused-GW docking
    def alpha(self, progress: float) -> float:
        cfg = self.cfg
        if cfg.dock_mode in ("gw", "w"):
            return 0.0 if cfg.dock_mode == "gw" else 1.0
        s = min(1.0, max(0.0, (progress - cfg.dock_warmup_frac) / max(1e-6, cfg.dock_anneal_frac)))
        return cfg.dock_alpha_start + (cfg.dock_alpha_end - cfg.dock_alpha_start) * s * s * (3 - 2 * s)

    def dock(self, model, island: Dict, bank, progress: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Entropic fused Gromov-Wasserstein between the reader's codes for unpaired island speech and real hub
        codes, as a loss on the reader (envelope theorem).  Kept as the ablation of the original proposal:
        pure GW is blind to where the island sits, and neither variant has beaten the bridge pairs alone."""
        cfg = self.cfg
        h = model.hub.reader(island["z"], "aud", island["lang"])
        Y = bank.sample_hubs(cfg.bt_bank_mult * h.shape[0]).to(h.device)
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
        same = s1.n_private == s2.n_private   # latents of different length share the hub slots only
        src = x0 if same else torch.cat([x0[:, : s1.K], x0.new_zeros(x0.shape[0], s2.n_private, x0.shape[2])], 1)
        z2 = s2.solve(src, lang2, +1, steps=steps, grad=grad)
        x0b = s2.solve(z2, lang2, -1, steps=steps, grad=grad)
        back = x0b if same else torch.cat([x0b[:, : s1.K], x0[:, s1.K:]], 1)
        z1b = s1.solve(back, lang1, +1, steps=steps, grad=grad)
        return F.mse_loss(x0b, src) + F.mse_loss(z1b, z1)
