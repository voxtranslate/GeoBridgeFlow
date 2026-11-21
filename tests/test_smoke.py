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
        V = m.hub.project(batch["img"])[aud["item"]]
        out = m.paired_hub("aud", m.encode("aud", aud), aud["lang"], V)
    attn, hub = out["attn"], out["hub"]
    assert attn.shape[-1] == V.shape[1] - 1                   # patches only: the global token is not offered
    assert (attn >= 0).all() and (attn.sum(-1) <= 1 + 1e-4).all()   # the rest of each row is the null option
    zero = torch.zeros_like(V[:, :1])
    pts = torch.cat([V[:, :-1], zero], 1)
    lo, hi = pts.min(1).values.unsqueeze(1), pts.max(1).values.unsqueeze(1)
    assert ((hub >= lo - 1e-4) & (hub <= hi + 1e-4)).all()    # barycentres lie in the hull of the patches + origin
