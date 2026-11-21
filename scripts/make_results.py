#!/usr/bin/env python
"""Assemble docs/RESULTS.md from the JSON files written by `eval`, `infer` and `ablate`.

Every number in the document is read from a file under runs/; nothing is typed by hand.

    python scripts/make_results.py --synth runs/sf_main --ablation runs/sf_ablation --real runs/f1200
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional


def load(path: str) -> Optional[Dict]:
    return json.load(open(path, encoding="utf8")) if os.path.exists(path) else None


def fmt(v) -> str:
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def metric_table(m: Dict[str, float], rows: List[tuple]) -> List[str]:
    out = ["| Direction | Metric | Value |", "|---|---|---|"]
    for key, direction, name in rows:
        if key in m:
            out.append(f"| {direction} | {name} | {fmt(float(m[key]))} |")
    return out


def run_section(title: str, run: str, H: str, L: str, T: str, note: str) -> List[str]:
    ev = load(os.path.join(run, "eval_test.json"))
    cfg = load(os.path.join(run, "config.json"))
    data = load(os.path.join(run, "data.json"))
    params = load(os.path.join(run, "params.json"))
    rep = load(os.path.join(run, "inference", "report.json"))
    if ev is None or cfg is None:
        return [f"## {title}", "", f"_No evaluation found in `{run}`._", ""]
    m = ev["metrics"]
    md = [f"## {title}", "", note, ""]
    md += [f"- Run directory: `{run}`; {cfg['phase_a_steps']} representation steps + {cfg['phase_b_steps']} flow steps, "
           f"batch {cfg['batch_size']}, seed {cfg['seed']}.",
           f"- Data: {data['images']['train']} / {data['images']['valid']} / {data['images']['test']} images "
           f"(train / valid / test), {data['bridge_pairs_train']} bridge pairs and "
           f"{data['island_unpaired_train']} unpaired island utterances in training "
           f"(bridge budget {int(100 * cfg['bridge_frac'])} %).",
           f"- Model: {params['total'] / 1e6:.2f} M parameters; one language costs "
           f"{params['per_language_flow_params'] / 1e3:.1f} k flow parameters "
           f"({params['per_language_pct_of_spokes']:.2f} % of the shared flow backbone).", ""]
    rows = [
        (f"aud_{L}>txt_{H}/bleu", f"island speech ({L}) → {H} text, zero-pair", "BLEU (parallel reference)"),
        (f"aud_{L}>txt_{H}/bleu_multi", "", "BLEU (all captions of the image)"),
        (f"aud_{L}>txt_{H}/chrf", "", "chrF++"),
        (f"aud_{L}>txt_{H}/content_acc", "", "exact content accuracy (%)"),
        (f"aud_{L}>txt_{H}/color_acc", "", "tone-contrastive word accuracy (%)"),
        (f"aud_{L}>txt_{H}/flat_color_acc", "", "… with the pitch contour removed (%)"),
        (f"aud_{L}>txt_{H}/twin_confusion", "", "share of colour errors that are the tonal twin (%)"),
        (f"aud_{L}>txt_{H}/flat_bleu", "", "BLEU with the pitch contour removed"),
        (f"aud_{L}>img/r1", f"island speech ({L}) → image", "R@1 (%)"),
        (f"aud_{L}>img/r5", "", "R@5 (%)"),
        (f"aud_{L}>txt_{H}/hub_r1", f"island speech ({L}) → {H} text (hub retrieval)", "R@1 (%)"),
        (f"aud_{H}>txt_{H}/wer", f"{H} speech → {H} text", "WER (%)"),
        (f"aud_{H}>txt_{H}/content_acc", "", "exact content accuracy (%)"),
        (f"aud_{H}>txt_{H}/bleu", "", "BLEU"),
        (f"txt_{H}>img/r1", f"{H} text → image", "R@1 (%)"),
        (f"aud_{H}>img/r1", f"{H} speech → image", "R@1 (%)"),
        (f"img>txt_{H}/bleu_multi", f"image → {H} text", "BLEU (all captions of the image)"),
        (f"txt_{H}>txt_{T}/bleu", f"{H} text → {T} text, zero-pair", "BLEU"),
        (f"txt_{H}>txt_{T}/content_acc", "", "exact content accuracy (%)"),
        (f"txt_{T}>txt_{H}/bleu", f"{T} text → {H} text, zero-pair", "BLEU"),
        (f"txt_{T}>txt_{H}/content_acc", "", "exact content accuracy (%)"),
        (f"aud_{L}>txt_{T}/bleu", f"island speech ({L}) → {T} text, zero-pair", "BLEU"),
        (f"txt_{H}>aud_{H}/rt_bleu", f"{H} text → {H} speech (round trip)", "BLEU"),
        (f"txt_{H}>aud_{H}/rt_content_acc", "", "exact content accuracy (%)"),
        (f"txt_{H}>aud_{L}/rt_bleu", f"{H} text → island speech ({L}) (round trip)", "BLEU"),
        (f"txt_{H}>aud_{L}/rt_content_acc", "", "exact content accuracy (%)"),
        (f"aud_{L}>aud_{H}/rt_bleu", f"island speech ({L}) → {H} speech (round trip)", "BLEU"),
        (f"codec_txt_{H}/cer", f"text codec, {H}", "CER (%)"),
        (f"codec_txt_{T}/cer", f"text codec, {T}", "CER (%)"),
        (f"tone/gate_{H}", f"tonality gate, {H}", "γ"),
        (f"tone/gate_{L}", f"tonality gate, {L}", "γ"),
        (f"tone/reliance_{H}", f"tone-reliance index, {H}", ""),
        (f"tone/reliance_{L}", f"tone-reliance index, {L}", ""),
    ]
    md += metric_table(m, rows) + [""]
    if rep:
        md += ["Inside the model (`inference/report.json`):", ""]
        if "pointing_game" in rep:
            for k, v in rep["pointing_game"].items():
                md.append(f"- Pointing game, {k}: {v['slot_precision']:.1f} % of hub slots sit on a described object "
                          f"(chance {v['chance_precision']:.1f} %); some slot hits one in {v['any_slot_hits']:.1f} % of captions.")
        if "docking_coupling" in rep:
            d = rep["docking_coupling"]
            for a in ("alpha=0.0", "alpha=0.5", "alpha=1.0"):
                if a in d:
                    md.append(f"- Transport plan at {a}: argmax is the true partner for {d[a]['match_acc']:.1f} % of island "
                              f"utterances; {d[a]['mass_on_true_pairs']:.1f} % of the mass is on true pairs.")
            md.append(f"- Relational premise: Pearson r = {d['relational_isometry_r']:.3f} between island and hub pairwise distances.")
        if "hub_agreement" in rep:
            for k, v in rep["hub_agreement"].items():
                md.append(f"- Hub agreement {k}: AUC(same caption > different caption) = {v['auc']:.3f}.")
        if "path_straightness" in rep:
            md.append("- Path straightness (chord / arc length): " +
                      "; ".join(f"{k} {v:.3f}" for k, v in rep["path_straightness"].items()) + ".")
        if "ode" in rep and rep["ode"].get("plain_cycle_vs_steps"):
            c = rep["ode"]["plain_cycle_vs_steps"].get("midpoint", {})
            if c:
                md.append("- Plain ODE cycle loss against solver steps (midpoint): " +
                          ", ".join(f"{k} steps {v:.2e}" for k, v in c.items()) + ".")
        if "ode" in rep and rep["ode"].get("score_vs_steps"):
            md.append("- Island translation score against ODE steps per hop: " +
                      ", ".join(f"{k}: {v:.1f}" for k, v in rep["ode"]["score_vs_steps"].items()) + ".")
        if "tone" in rep and "pitch_scaling" in rep["tone"]:
            for lang, v in rep["tone"]["pitch_scaling"].items():
                md.append(f"- Pitch-contour scaling, {lang}: hub displacement " +
                          ", ".join(f"×{s:g} → {x:.3f}" for s, x in zip(v["scales"], v["hub_displacement"])) + ".")
        if "speech_attribution" in rep:
            md.append("- Share of the gradient attribution that falls on the pitch contour: " +
                      ", ".join(f"{k} {100 * v['tone_stream_share_of_attribution']:.1f} %" for k, v in rep["speech_attribution"].items()) + ".")
        if "fertility" in rep:
            for name, per in rep["fertility"].items():
                md.append(f"- Units per word, {name}: " + ", ".join(f"{l} {v['units_per_word']:.2f}" for l, v in per.items()) + ".")
        if rep.get("failed"):
            md.append(f"- Figures that failed: {rep['failed']}")
        md.append("")
    samples = ev.get("samples", {})
    key = f"aud_{L}>txt_{H}"
    if key in samples and samples[key]:
        md += [f"Sample outputs, island speech ({L}) → {H} text:", "", "| reference | output |", "|---|---|"]
        md += [f"| {s['ref']} | {s['hyp']} |" for s in samples[key][:8]] + [""]
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="")
    ap.add_argument("--ablation", default="")
    ap.add_argument("--real", default="")
    ap.add_argument("--header", default="docs/RESULTS_HEADER.md")
    ap.add_argument("--out", default="docs/RESULTS.md")
    a = ap.parse_args()
    md: List[str] = []
    if os.path.exists(a.header):
        md += open(a.header, encoding="utf8").read().rstrip().split("\n") + [""]
    if a.synth:
        md += run_section("1. SynthWorld reference run", a.synth, "sa", "sb", "sc",
                          "Controlled benchmark; `sa` = hub language, `sb` = tonal, speech-only island, "
                          "`sc` = Devanagari-script text.  Content accuracy is exact (the output is parsed and "
                          "compared with the true content of the source utterance).")
    if a.ablation and os.path.exists(os.path.join(a.ablation, "SUMMARY.md")):
        body = open(os.path.join(a.ablation, "SUMMARY.md"), encoding="utf8").read().split("\n")
        md += ["## 2. Ablations on SynthWorld", ""] + [l.replace("](plot_", f"](../{a.ablation}/plot_") for l in body[1:]] + [""]
    if a.real:
        md += run_section("3. Flickr8k-Tri subset (real data)", a.real, "en", "yo", "hi",
                          "English text and speech, Yorùbá speech (orthography held out), Hindi text. "
                          "This is a pipeline check on a small subset with a small model, not a result to cite.")
    with open(a.out, "w", encoding="utf8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
