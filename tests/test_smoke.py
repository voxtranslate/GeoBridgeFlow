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
        score, own = m.hub.ground(q, m.hub.keys(V), aud["item"])
    assert torch.allclose(own, attn, atol=1e-5)               # one-pass grounding = the own-image attention
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
    loss, logs = GeoBridgeLoss(cfg, m.lang2id).back_translate(m, isl, c.sample_hubs)
    assert torch.isfinite(loss) and {"bt_write", "bt_read", "bt_shift"} <= set(logs)
    hub = c.sample_hubs(4)
    lang = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        draft = m.hub.explainer(hub, 0, lang)
        assert torch.allclose(m.write("txt", hub, lang, t0=1.0), draft)   # t0 = 1: the draft itself, no ODE step
        assert m.write("txt", hub, lang, t0=0.5).shape == draft.shape
        z_img, rgb = m.hub_to_image(hub)
    assert rgb.shape == (4, 3, cfg.thumb_size, cfg.thumb_size) and float(rgb.min()) >= 0 and float(rgb.max()) <= 1
