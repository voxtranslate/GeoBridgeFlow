"""Fast checks that need no downloads (SynthWorld, CPU).  Run:  pytest -q tests"""
import math
import os

import numpy as np
import pytest
import torch

from geobridge import audio_dsp as dsp
from geobridge.config import Config
from geobridge.models import ot
from geobridge.text_units import TextUnitizer, bytes_to_text, graphemes


def test_graphemes_are_script_fair():
    assert graphemes("क्षि") == ["क्षि"]                # conjunct + vowel sign = one written symbol
    assert len(graphemes("ọmọkùnrin")) == 9             # combining tone / dot marks stay attached
    u = TextUnitizer("grapheme")
    hi, en = u.fertility(["तिदे नादि बो"]), u.fertility(["tide nadi bo"])
    assert hi["units_per_grapheme"] == en["units_per_grapheme"] == 1.0
    assert TextUnitizer("byte").fertility(["तिदे नादि बो"])["units_per_grapheme"] > 2.0


def test_text_roundtrip_bytes():
    u = TextUnitizer("grapheme")
    for s in ["A man climbs a rock .", "Ẹnikan ti o wọ ibori alawọ buluu", "सड़क दौड़र कवच में एक आदमी"]:
        units, target = u.encode(s)
        assert bytes_to_text(target[1:].tolist()) == s
        assert bytes(b for row in units.tolist() for b in row if b < 256).decode("utf8") == s


def test_yin_tracks_pitch():
    sr = 16000
    t = np.arange(sr) / sr
    for f in (110.0, 220.0, 330.0):
        wav = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6)).astype(np.float32)
        f0, _ = dsp.yin_f0(wav, sr)
        assert abs(np.median(f0[f0 > 0]) - f) / f < 0.02


def test_fgw_recovers_a_hidden_permutation():
    torch.manual_seed(0)
    x = torch.randn(24, 8)
    perm = torch.randperm(24)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    y = (x @ q)[perm] + 3.0                                 # rotated, shifted, shuffled copy of x
    C1, C2 = ot.scale_free(torch.cdist(x, x)), ot.scale_free(torch.cdist(y, y))
    for marginal in ("balanced", "semi"):
        Pi = ot.solve_fgw(C1, C2, None, 0.0, eps=0.02, outer=40, inner=100, marginal=marginal)
        inv = torch.argsort(perm)                           # x_i sits at position inv[i] of y
        assert (Pi.argmax(1) == inv).float().mean() > 0.9, marginal
        assert abs(float(Pi.sum()) - 1.0) < 1e-4
    val, gw, _ = ot.fgw_value(C1, C2, Pi, None, 0.0)
    assert float(gw) < 0.05


def test_log_sinkhorn_is_stable_where_the_plain_kernel_underflows():
    torch.manual_seed(0)
    cost = 5.0 + torch.rand(16, 20) * 50.0
    a, b = torch.full((16,), 1 / 16), torch.full((20,), 1 / 20)
    eps = 0.01
    assert float(torch.exp(-cost / eps).sum()) == 0.0        # the textbook Gibbs kernel is exactly zero here
    Pi = ot.log_sinkhorn(cost, a, b, eps=eps, iters=300)
    assert torch.isfinite(Pi).all() and abs(float(Pi.sum()) - 1.0) < 1e-4
    assert torch.allclose(Pi.sum(0), b, atol=1e-4)           # last projection is exact
    assert float((Pi.sum(1) - a).abs().max()) < 0.02         # near-assignment regime converges slowly, but safely
    Pi = ot.log_sinkhorn(torch.rand(16, 20), a, b, eps=0.1, iters=200)
    assert torch.allclose(Pi.sum(1), a, atol=1e-4) and torch.allclose(Pi.sum(0), b, atol=1e-4)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    from geobridge.trainer import Trainer

    root = tmp_path_factory.mktemp("gb")
    cfg = Config.preset("synth_tiny").update({"data_root": str(root / "data"), "out_root": str(root / "runs"),
                                              "exp_name": "t"})
    tr = Trainer(cfg)
    tr.fit()
    return cfg, tr


def test_content_f1_ignores_function_words_and_has_a_chance_level():
    from geobridge.metrics import content_f1, other_image, words

    assert words("एक कुत्ता घास में खेलता है.") == ["एक", "कुत्ता", "घास", "में", "खेलता", "है"]   # vowel signs stay in the word
    assert words("Ọmọkùnrin kan , ń sáré !") == ["ọmọkùnrin", "kan", "ń", "sáré"]
    refs = ["a dog runs in the snow .", "a man climbs a rock .", "a girl in a red dress .", "a boat on the water .",
            "a child plays in the sand .", "a bird sits in a tree ."]
    assert content_f1(refs, refs) == pytest.approx(100.0)
    generic = ["a man in the ."] * len(refs)                   # a fluent caption that ignores its source
    right_content = ["dog snow", "man rock", "girl dress red", "boat water", "child sand", "bird tree"]
    assert content_f1(right_content, refs) > 60.0
    imgs = [0, 0, 0, 1, 1, 2]                                  # captions of the same image are never their own chance pair
    wrong = other_image(imgs)
    assert all(imgs[i] != imgs[j] for i, j in enumerate(wrong))
    shuffled = [refs[j] for j in wrong]
    # what a generic caption scores, it scores against any reference: the distance from chance is what counts
    lift = lambda hyps: content_f1(hyps, refs) - content_f1(hyps, shuffled, pool=refs)
    assert content_f1(generic, refs) > 5.0 and abs(lift(generic)) < 5.0 < 50.0 < lift(right_content)


def test_training_runs_and_checkpoints(tiny_run):
    cfg, tr = tiny_run
    assert tr.step == cfg.total_steps
    ck = os.path.join(cfg.run_dir, "checkpoints")
    for name in ("last.pt", "best.pt", "phase_a.pt"):
        assert os.path.exists(os.path.join(ck, name))
    assert os.path.exists(os.path.join(cfg.run_dir, "curves", "losses_train_vs_valid.png"))
    assert os.path.exists(os.path.join(cfg.cache_dir, "splits.json"))
    assert all(math.isfinite(h["loss"]) for h in tr.history["train"])


def test_resume_from_any_step_continues_the_same_stream(tiny_run):
    from geobridge.trainer import Trainer

    cfg, tr = tiny_run
    mid = os.path.join(cfg.run_dir, "checkpoints", "step_00000040.pt")
    assert os.path.exists(mid)
    cfg2 = cfg.clone(exp_name="t_resume")
    tr2 = Trainer(cfg2, data=tr.data, resume=False)
    tr2.resume_from(mid)
    assert tr2.step == 40 and tr2.phase == "B"
    tr2.sampler.set_state(tr2.epoch, tr2.offset)
    a = next(iter(tr2.sampler))
    tr.sampler.set_state(tr2.epoch, tr2.offset)
    assert a == next(iter(tr.sampler))                      # identical next batch
    tr2.fit()
    assert tr2.step == cfg.total_steps


def test_no_cross_lingual_pairs_in_training(tiny_run):
    cfg, tr = tiny_run
    for g in tr.data.groups["train"]:
        langs = {l for _, l, _ in g["train_views"]}
        assert len(langs) <= 1                              # strict protocol: one language per caption
        assert not any(m == "txt" and l in cfg.island_langs for m, l, _ in g["train_views"])
    isl = {tr.data.utts[i]["uid"] for i in tr.data.island_ids}
    bridge = {tr.data.utts[ui]["uid"] for g in tr.data.groups["train"] for m, l, ui in g["train_views"] if l in cfg.island_langs}
    assert not (isl & bridge)                               # an utterance is either a bridge pair or unpaired


def test_ode_is_invertible_so_the_plain_cycle_is_vacuous(tiny_run):
    from geobridge.losses import GeoBridgeLoss

    cfg, tr = tiny_run
    m = tr.model.eval()
    z = torch.randn(8, cfg.n_tokens, cfg.latent_dim)
    lang = torch.zeros(8, dtype=torch.long)
    loss = GeoBridgeLoss(cfg, m.lang2id)
    with torch.no_grad():
        coarse = float(loss.naive_cycle(m, "txt", z, lang, "aud", lang, steps=2))
        fine = float(loss.naive_cycle(m, "txt", z, lang, "aud", lang, steps=64))
    assert fine < 1e-4 and fine <= coarse


def test_inference_and_evaluation(tiny_run):
    from geobridge.inferencer import Inferencer

    cfg, tr = tiny_run
    inf = Inferencer(cfg, data=tr.data)
    out = inf.translate_text(["sipa matu"], cfg.hub_langs[0], cfg.hub_langs[0])
    assert isinstance(out[0], str)
    metrics = inf.evaluate("test", 16)
    assert "score" in metrics and all(math.isfinite(v) for v in metrics.values())
    H = cfg.hub_langs[0]
    for k in ("cf1", "chance_cf1", "bleu", "chance_bleu", "chrf", "chance_chrf"):      # every text score has its chance level
        assert f"txt_{H}>txt_{H}/{k}" in metrics, k
    lifts = [metrics[k] - metrics[k[:-3] + "chance_cf1"] for k in metrics
             if k.endswith("/cf1") and ">txt_" in k and "/flat_" not in k and "/rt_" not in k]
    assert abs(metrics["score"] - sum(lifts) / len(lifts)) < 1e-9                      # selection: content above chance
    import json
    meta = json.load(open(os.path.join(cfg.run_dir, "eval_test.json")))["meta"]
    assert meta["step"] == inf.ck["step"] and meta["split"] == "test" and meta["items"] == 16
    rep = inf.run_all("test")
    assert not rep.get("failed"), rep.get("failed")
    assert os.path.exists(os.path.join(cfg.run_dir, "inference", "01_slot_attention.png"))


def test_flat_pitch_equals_closed_gate_and_hub_codes_stay_in_the_image_hull(tiny_run):
    cfg, tr = tiny_run
    m = tr.model.eval()
    batch = next(iter(tr.data.paired_loader("valid", shuffle=False, batch_size=8)))
    aud = batch["aud"]
    with torch.no_grad():
        flat = m.audio.encode(aud["sem"], dsp.flatten_tone(aud["tone"]), aud["mask"], aud["lang"], gate_override=1.0)
        closed = m.audio.encode(aud["sem"], aud["tone"], aud["mask"], aud["lang"], gate_override=0.0)
        assert torch.allclose(flat, closed, atol=1e-5)       # the gated path is bias-free by construction
        V = m.hub.project(batch["img"])
        q = m.hub.queries(m.encode("aud", aud), "aud", aud["lang"])
        hub, attn = m.hub.attend(q, V[aud["item"]])
        score, own, logp = m.hub.ground(q, m.hub.keys(V), aud["item"])
    assert torch.allclose(own, attn, atol=1e-5)               # one-pass grounding = the own-image attention
    assert torch.allclose(logp.exp().sum(-1), torch.ones_like(logp[..., 0]), atol=1e-4)   # patches + null = 1
    assert score.shape == (len(aud["item"]), batch["B"])
    assert attn.shape[-1] == V.shape[1] - 1                   # patches only: the global token is not offered
    assert (attn >= 0).all() and (attn.sum(-1) <= 1 + 1e-4).all()   # the rest of each row is the null option
    pts = torch.cat([V[aud["item"]][:, :-1], torch.zeros_like(V[aud["item"]][:, :1])], 1)
    lo, hi = pts.min(1).values.unsqueeze(1), pts.max(1).values.unsqueeze(1)
    assert ((hub >= lo - 1e-4) & (hub <= hi + 1e-4)).all()    # barycentres lie in the hull of the patches + origin


def test_reader_needs_no_image_and_cache_matches_live_computation(tiny_run):
    cfg, tr = tiny_run
    m = tr.model.eval()
    cache = tr.cache["valid"]
    assert cache.n_groups == len(tr.data.groups["valid"])
    b = cache.batch([0, 1, 2])
    raw = tr.data.collate_groups([{"img": g["img"], "cap": g["cap"], "views": g["valid_views"]} for g in tr.data.groups["valid"][:3]])
    with torch.no_grad():
        z_live = m.encode("txt", raw["txt"])
        hub_live = m.paired_hub("txt", z_live, raw["txt"]["lang"], m.hub.project(raw["img"])[raw["txt"]["item"]])
        read = m.read("txt", b["txt"]["z"], b["txt"]["lang"])            # latent in, hub code out: no image
    assert torch.allclose(b["txt"]["z"], z_live, atol=1e-4) and torch.allclose(b["txt"]["hub"], hub_live, atol=1e-4)
    assert read.shape == b["txt"]["hub"].shape == (z_live.shape[0], cfg.hub_slots, cfg.latent_dim)
    assert (b["txt"]["item"] == raw["txt"]["item"]).all()


def test_back_translation_and_draft_writing(tiny_run):
    from geobridge.losses import GeoBridgeLoss

    cfg, tr = tiny_run
    m = tr.model.eval()
    c = tr.cache["train"]
    isl = c.island_batch(list(range(8)))
    loss, logs = GeoBridgeLoss(cfg, m.lang2id).back_translate(m, isl, c)
    assert torch.isfinite(loss) and {"bt_write", "bt_read", "bt_relay", "bt_shift"} <= set(logs)
    views = c.sample_views("txt", 5)
    assert views["z"].shape[0] == views["hub"].shape[0] == views["lang"].shape[0] == 5
    hub = c.sample_hubs(4)
    lang = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        draft = m.hub.explainer(hub, 0, lang)
        assert torch.allclose(m.write("txt", hub, lang, t0=1.0), draft)   # t0 = 1: the draft itself, no ODE step
        assert m.write("txt", hub, lang, t0=0.5).shape == draft.shape
        z_img, rgb = m.hub_to_image(hub)
    assert rgb.shape == (4, 3, cfg.thumb_size, cfg.thumb_size) and float(rgb.min()) >= 0 and float(rgb.max()) <= 1


def test_counterfactual_credit_points_at_the_patch_that_would_explain_more():
    from geobridge.losses import GeoBridgeLoss

    torch.manual_seed(0)
    cfg = Config.preset("synth_tiny")
    loss = GeoBridgeLoss(cfg, {"sa": 0, "sb": 1, "sc": 2})
    patches = torch.randn(1, 9, 6)
    logits = torch.full((1, 1, 10), -12.0)                   # 9 patches + the null option (last)
    logits[0, 0, 0] = 12.0                                   # the slot looks, sharply, at patch 0 ...
    logits.requires_grad_(True)
    logp = torch.log_softmax(logits, -1)
    code = logp[..., :-1].exp() @ patches

    def unexplained(c, rows=None):                           # ... but the "sentence" is explained by patch 3
        return ((c - patches[:, 3:4]) ** 2).mean(dim=(1, 2))

    err = unexplained(code)
    plain = torch.autograd.grad(err.sum(), logits, retain_graph=True)[0][0, 0]
    seek, weight = loss.seek(unexplained, err, code, logp, patches, allow_null=False)
    credit = torch.autograd.grad(seek, logits)[0][0, 0]
    assert plain.abs().max() < 1e-6                          # softmax Jacobian: no signal towards the unseen patch
    assert weight > 0.5 and credit.argmin().item() == 3 and credit[3] < -0.1   # counterfactual credit: go to patch 3
    # the code is already right: no candidate promises anything, nothing is pulled
    logits2 = torch.full((1, 1, 10), -12.0)
    logits2[0, 0, 3] = 12.0
    logits2.requires_grad_(True)
    logp2 = torch.log_softmax(logits2, -1)
    code2 = logp2[..., :-1].exp() @ patches
    seek2, weight2 = loss.seek(unexplained, unexplained(code2), code2, logp2, patches, allow_null=True)
    assert weight2 < 1e-6 and abs(float(seek2)) < 1e-6
    # a proposal that verifiably explains less is vetoed (here: the first-order ranking is deliberately misled)
    def misleading(c, rows=None):                            # locally "more of patch 5 helps", but going there hurts
        return unexplained(c, rows) + 4.0 * torch.relu((c - patches[:, 0:1]) * (patches[:, 5:6] - patches[:, 0:1])).mean(dim=(1, 2)) ** 2 \
            - 0.5 * ((c - patches[:, 0:1]) * (patches[:, 5:6] - patches[:, 0:1])).mean(dim=(1, 2))
    err3 = misleading(code)
    _, w_free = loss.seek(misleading, err3, code, logp, patches, allow_null=False)
    cfg.seek_verify = True
    _, w_veto = loss.seek(misleading, err3, code, logp, patches, allow_null=False)
    assert w_veto <= w_free


def test_one_step_can_consume_several_batches(tiny_run):
    from geobridge.trainer import StepReplica, Trainer, chunked

    cfg, tr = tiny_run
    assert [len(c) for c in chunked(range(5), 2)] == [2, 2, 1]
    cfg2 = cfg.clone(exp_name="t_two", step_batches=2, phase_a_steps=4, phase_b_steps=4, eval_every=4, save_every=4)
    tr2 = Trainer(cfg2, data=tr.data, resume=False)
    assert isinstance(tr2.replica, StepReplica) and tr2.step_batches == 2
    tr2.fit()
    assert tr2.step == 8
    assert tr2.epoch * len(tr2.sampler) + tr2.offset == 16   # 8 optimiser steps consumed 16 batches
    # the replica computes exactly the loss of the batch it is pointed at
    batch = tr.data.collate_groups([{"img": g["img"], "cap": g["cap"], "views": g["train_views"]}
                                    for g in tr.data.groups["train"][:8]])
    rep = tr.replica
    tr.model.eval()
    rep.job = {"batches": [None, batch], "islands": [None, None], "logs": [None, None], "phase": "A", "progress": 0.0, "bank": None}
    with torch.no_grad():
        torch.manual_seed(0)
        via = rep(torch.tensor([1]))
        torch.manual_seed(0)
        direct, logs = tr.loss.phase_a(tr.model, batch, None)
    assert torch.allclose(via, direct.reshape(1)) and rep.job["logs"][1]["loss"] == logs["loss"]


def test_real_data_options(tiny_run):
    """The options the real-data presets switch on: decoder exposure, phase-A selection, regularisation,
    vetoed credit, evidence-proportional gate price - and the guard against resuming a run
    under different settings."""
    from geobridge.trainer import Trainer

    cfg, tr = tiny_run
    opts = dict(dropout=0.1, speech_mask=0.08, weight_decay=0.05, phase_a_select=True, read_noise=0.2, w_dec=1.0,
                seek_verify=True, gate_price="evidence")
    cfg2 = cfg.clone(exp_name="t_real", phase_a_steps=20, phase_b_steps=12, eval_every=5, save_every=10, step_batches=2, **opts)
    tr2 = Trainer(cfg2, data=tr.data, resume=False)
    enc_before = tr2.model.text.unit_emb.weight.detach().clone()
    tr2.fit()
    assert tr2.step == 32
    # phase-A selection: a validated state after the release (step 10 of 20) was kept
    assert os.path.exists(os.path.join(cfg2.run_dir, "checkpoints", "phase_a_best.pt")) and tr2.best_a["step"] >= 10
    # decoder exposure: the cache carries the byte targets, the loss is logged, and only the decoder half of the
    # text codec is trained in the transport phase
    b = tr2.cache["train"].batch([0, 1, 2, 3])
    raw = tr.data.collate_groups([{"img": g["img"], "cap": g["cap"], "views": g["train_views"]} for g in tr.data.groups["train"][:4]])
    assert (b["txt"]["bytes"] == raw["txt"]["bytes"]).all()
    assert any("dec_ce" in h for h in tr2.history["train"] if h["phase"] == "B")
    names = set(tr2.b_named)
    assert any(n.endswith("byte_emb.weight") for n in names) and not any("unit_emb" in n or "resampler" in n for n in names)
    ck = torch.load(os.path.join(cfg2.run_dir, "checkpoints", "phase_a.pt"), map_location="cpu", weights_only=False)["model"]
    assert torch.equal(ck["text.unit_emb.weight"], tr2.model.text.unit_emb.weight.cpu())      # encoder frozen in phase B
    assert not torch.equal(ck["text.byte_emb.weight"], tr2.model.text.byte_emb.weight.cpu())  # decoder trained
    assert not torch.equal(enc_before, tr2.model.text.unit_emb.weight)                        # (it did train in phase A)
    # evidence-proportional gate price: grounded utterances pay, unpaired ones do not
    batch = next(iter(tr.data.paired_loader("valid", shuffle=False, batch_size=6)))
    m = tr2.model.eval()
    m.audio.gate_forced_open = False
    with torch.no_grad():
        res = m.audio.loss(batch["aud"], n_grounded=3)
        assert torch.allclose(res["gate_pen"], m.audio.gate(batch["aud"]["lang"][:3]).mean())
        assert float(m.audio.loss(batch["aud"], n_grounded=0)["gate_pen"]) == 0.0
    # resuming the same folder with other settings is refused; with the same settings it continues
    with pytest.raises(RuntimeError, match="different settings"):
        Trainer(cfg2.clone(w_dec=0.0), data=tr.data)
    assert Trainer(cfg2.clone(eval_every=7, num_workers=0), data=tr.data).step == 32


def test_time_budget_stops_on_a_saved_step_and_the_run_continues(tiny_run):
    """A hosted session that is cut off may keep nothing: with a time budget the trainer stops by itself, on a
    step it has saved, and the same command continues from that step."""
    from geobridge.trainer import Trainer
    from geobridge.utils import OutOfTime, set_time_budget, time_left

    cfg, tr = tiny_run
    cfg2 = cfg.clone(exp_name="t_budget", phase_a_steps=6, phase_b_steps=4, eval_every=5, save_every=50)
    assert time_left() == float("inf")
    set_time_budget(1e-9)                                   # already used up
    try:
        with pytest.raises(OutOfTime):
            Trainer(cfg2, data=tr.data).fit()
    finally:
        set_time_budget(0)
    last = torch.load(os.path.join(cfg2.run_dir, "checkpoints", "last.pt"), map_location="cpu", weights_only=False)
    assert last["step"] == 1                                # not a multiple of save_every: saved because of the stop
    tr2 = Trainer(cfg2, data=tr.data)
    assert tr2.step == 1
    tr2.fit()
    assert tr2.step == 10


def test_ablation_never_retrains_the_main_run(tiny_run):
    """The main run is the full model of its own ablation and provides the representation of every variant that
    leaves phase A alone; a run with other settings is not mistaken for it."""
    from geobridge.ablation import AblationRunner
    from geobridge.config import PHASE_A_KEYS

    cfg, tr = tiny_run
    runner = AblationRunner(cfg.clone(), seeds=[cfg.seed])
    full, full_dir = runner._cell("private", "full", {}, cfg.seed)
    assert full.run_dir == cfg.run_dir and full_dir.startswith(runner.root)
    variant, _ = runner._cell("private", "no private", {"private_noise": False}, cfg.seed)
    assert variant.signature(PHASE_A_KEYS) == cfg.signature(PHASE_A_KEYS)
    assert runner._phase_a(variant) == os.path.join(cfg.run_dir, "checkpoints", "phase_a.pt")
    summary = runner.run(["private"])
    assert set(summary) == {"private"}
    assert next(iter(summary["private"]["stats"])).startswith("hub + private Gaussian tokens")   # the full model leads
    # a group never compares a run with itself: exactly one variant is the full model, whatever the base settings
    from geobridge.ablation import GROUPS
    for base in (cfg.clone(w_dec=1.0), cfg.clone(w_dec=0.0)):
        r = AblationRunner(base, seeds=[cfg.seed])
        for g, spec in GROUPS.items():
            # ("bridge" compares budgets against its own 20 % reference, which is not this base's budget)
            assert sum(r._is_full(o) for _, o in spec["variants"]) == (1 if g != "bridge" else 0), g
    # cheapest first: what needs no new representation runs before what does, whatever the order asked for
    cells = [("hub", n, o, cfg.seed) for n, o in GROUPS["hub"]["variants"][:2]] + \
            [("island", n, o, cfg.seed) for n, o in GROUPS["island"]["variants"][:2]]
    order = [runner._cost(*c) for c in runner._cheapest_first(cells)]
    assert order == sorted(order) and order[-1] == 2 and runner._cheapest_first(cells)[-1][0] == "hub"
    assert os.path.exists(os.path.join(full_dir, "results.json")) and not os.path.isdir(os.path.join(full_dir, "checkpoints"))
    assert not os.path.isdir(os.path.join(runner.root, "_phase_a"))          # nothing was trained for phase A
    # another seed, or a main run trained with other settings, gets a run of its own
    assert runner._cell("private", "full", {}, cfg.seed + 1)[0].run_dir.startswith(runner.root)
    other = AblationRunner(cfg.clone(lr_flow=cfg.lr_flow * 2), seeds=[cfg.seed])
    assert other._cell("private", "full", {}, cfg.seed)[0].run_dir.startswith(other.root)
    changed = other._cell("private", "no private", {"private_noise": False}, cfg.seed)[0]
    assert other._phase_a(changed) == os.path.join(cfg.run_dir, "checkpoints", "phase_a.pt")   # phase A is unchanged
