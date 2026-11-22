#!/usr/bin/env python
"""Assemble docs/RESULTS.md from the JSON files written by `eval`, `infer` and `ablate`.

Every number in the document is read from a file under runs/; nothing is typed by hand.

    python scripts/make_results.py --synth runs/sf_ablation/_full/full_s0 --ablation runs/sf_ablation \
        --real_check runs/real_smoke
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
           f"- Retrieval gallery: the {data['images']['test']} test images (chance R@1 "
           f"{100 / data['images']['test']:.1f} %, R@10 {min(100.0, 1000 / data['images']['test']):.1f} %).",
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
        (f"aud_{L}>img/r10", "", "R@10 (%)"),
        (f"aud_{L}>img_gen/r1", f"island speech ({L}) → written image latent", "R@1 (%)"),
        (f"txt_{H}>img_gen/r1", f"{H} text → written image latent", "R@1 (%)"),
        (f"aud_{L}>txt_{H}/hub_r1", f"island speech ({L}) → {H} text (hub retrieval)", "R@1 (%)"),
        (f"aud_{H}>txt_{H}/wer", f"{H} speech → {H} text", "WER (%)"),
        (f"aud_{H}>txt_{H}/content_acc", "", "exact content accuracy (%)"),
        (f"aud_{H}>txt_{H}/bleu", "", "BLEU"),
        (f"txt_{H}>img/r1", f"{H} text → image", "R@1 (%)"),
        (f"txt_{H}>img/r10", "", "R@10 (%)"),
        (f"aud_{H}>img/r1", f"{H} speech → image", "R@1 (%)"),
        (f"aud_{H}>img/r10", "", "R@10 (%)"),
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
        if "hub_alignment" in rep:
            d = rep["hub_alignment"]
            md.append(f"- Island speech read with no image vs the hub codes of its (never seen) translations: Pearson r = "
                      f"{d['relational_isometry_r']:.3f} between the two sets of pairwise distances; an optimal-transport "
                      f"plan on position alone (alpha = 1) finds the true partner for "
                      f"{d['alpha=1.0']['match_acc']:.1f} % of utterances, on relational structure alone (alpha = 0) "
                      f"for {d['alpha=0.0']['match_acc']:.1f} %.")
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
        if "hub_vs_private_probe" in rep:
            for k, v in rep["hub_vs_private_probe"].items():
                if isinstance(v, dict):
                    md.append(f"- Linear probe for {k}: reader hub {v['reader hub']:.1f} %, backward-ODE hub "
                              f"{v['ODE hub']:.1f} %, backward-ODE private tokens {v['ODE private']:.1f} % "
                              f"(majority class {v['majority_class']:.1f} %).")
        if "read_write_modes" in rep:
            for k, v in rep["read_write_modes"].items():
                md.append(f"- {k} → hub-language text: " + ", ".join(f"{n} {x:.1f}" for n, x in v["read_mode"].items()) +
                          "; starting the writer from the explainer's draft at t0 = " +
                          ", ".join(f"{t}: {x:.1f}" for t, x in v["draft_t0"].items()) + ".")
        if "lora_relative_norm" in rep:
            for sp, per in rep["lora_relative_norm"].items():
                md.append(f"- Relative size of the language LoRA in the {sp} spoke (‖BA‖/‖W‖, mean over layers): " +
                          ", ".join(f"{l} {v:.3f}" for l, v in per.items()) + ".")
        if "codec_reconstruction" in rep:
            md.append("- Speech codec, frame cosine between natural and reconstructed segmental stream (one utterance): " +
                      ", ".join(f"{l} {v['segmental_cosine']:.3f}" for l, v in rep["codec_reconstruction"].items()) + ".")
        if rep.get("failed"):
            md.append(f"- Figures that failed: {rep['failed']}")
        md.append("")
    samples = ev.get("samples", {})
    key = f"aud_{L}>txt_{H}"
    if key in samples and samples[key]:
        md += [f"Sample outputs, island speech ({L}) → {H} text:", "", "| reference | output |", "|---|---|"]
        md += [f"| {s['ref']} | {s['hyp']} |" for s in samples[key][:8]] + [""]
    return md


FIGURES = [
    ("01_slot_attention.png", "Where each hub slot looks, for text, speech and island speech of the same caption"),
    ("05_tone.png", "Tonality gates, hub displacement and tone-word accuracy against pitch-contour scaling"),
    ("04_hub_alignment.png", "Island speech read with no image against the hub codes of its translations"),
    ("16_read_write_modes.png", "Reading with the one-pass reader vs the backward ODE; writing from the explainer's draft"),
    ("15_generated_images.png", "Image writing from text and speech: hub code, image latent, preview"),
    ("09_ode_steps.png", "Quality against ODE steps; the plain ODE cycle vanishes with the step size"),
    ("03_hub_manifold.png", "All views in hub coordinates"),
    ("13_hub_vs_private.png", "Linear probes on the hub and private halves of a speech read-out"),
    ("10_fertility.png", "Units per word and per written symbol for each script"),
]


def export_assets(a) -> List[str]:
    """Copy the figures, tables, sample outputs and metric files behind this document into docs/results/,
    so they are versioned with the code (runs/ itself is not)."""
    import glob
    import shutil

    root = os.path.join(os.path.dirname(a.out) or ".", "results")
    md = ["## 4. Files behind these numbers", "",
          "`runs/` is not versioned; the artefacts behind this document are copied to `docs/results/`.", ""]
    for name, run in (("synth", a.synth), ("real", a.real)):
        if not run or not os.path.isdir(run):
            continue
        dst = os.path.join(root, name)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(os.path.join(dst, "figures"), exist_ok=True)
        os.makedirs(os.path.join(dst, "speech"), exist_ok=True)
        for p in glob.glob(os.path.join(run, "inference", "*.png")) + glob.glob(os.path.join(run, "curves", "*.png")):
            shutil.copy(p, os.path.join(dst, "figures"))
        for p in glob.glob(os.path.join(run, "inference", "outputs", "speech", "*")):
            shutil.copy(p, os.path.join(dst, "speech"))
        for p in ("eval_test.json", "config.json", "params.json", "data.json", "history.json",
                  os.path.join("inference", "report.json"), os.path.join("inference", "outputs", "translations.md"),
                  os.path.join("inference", "outputs", "private_diversity.json")):
            if os.path.exists(os.path.join(run, p)):
                shutil.copy(os.path.join(run, p), dst)
        label = "SynthWorld reference run" if name == "synth" else "Flickr8k-Tri subset"
        md += [f"### {label} (`docs/results/{name}/`)", "",
               "Metrics `eval_test.json`, internals `report.json`, sample conversions `translations.md`, "
               "synthesised speech `speech/*.wav`, training curves and all figures in `figures/`.", ""]
        for fig, cap in FIGURES:
            if os.path.exists(os.path.join(dst, "figures", fig)):
                md += [f"**{cap}**", "", f"![{fig}](results/{name}/figures/{fig})", ""]
    if a.ablation and os.path.isdir(a.ablation):
        dst = os.path.join(root, "ablation")
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(os.path.join(dst, "cells"), exist_ok=True)
        for p in glob.glob(os.path.join(a.ablation, "table_*")) + glob.glob(os.path.join(a.ablation, "plot_*")) + \
                glob.glob(os.path.join(a.ablation, "SUMMARY.md")):
            shutil.copy(p, dst)
        for p in glob.glob(os.path.join(a.ablation, "*", "*", "results.json")):
            parts = p.split(os.sep)
            shutil.copy(p, os.path.join(dst, "cells", f"{parts[-3]}__{parts[-2]}.json"))
        md += ["### Ablation (`docs/results/ablation/`)", "",
               "Tables as Markdown, CSV and LaTeX, bar charts, and the full test metrics of every cell in `cells/`.", ""]
    return md


def real_check_section(run: str) -> List[str]:
    """Real data, pipeline check only: says what ran, reports no metric."""
    import glob

    cfg, data = load(os.path.join(run, "config.json")), load(os.path.join(run, "data.json"))
    rep = load(os.path.join(run, "inference", "report.json")) or {}
    if cfg is None or data is None:
        return ["## 3. Real data (Flickr8k-Tri)", "", f"_No run found in `{run}`._", ""]
    figs = len(glob.glob(os.path.join(run, "inference", "*.png")))
    wavs = len(glob.glob(os.path.join(run, "inference", "outputs", "speech", "*.wav")))
    done = [name for name, path in (("prepare", "data.json"), ("train", os.path.join("checkpoints", "last.pt")),
                                    ("eval", "eval_test.json"), ("infer", os.path.join("inference", "report.json")))
            if os.path.exists(os.path.join(run, path))]
    return ["## 3. Real data (Flickr8k-Tri): pipeline check only", "",
            f"The redesigned model was run on real data only to check that every stage works: {data['images']['train']} / "
            f"{data['images']['valid']} / {data['images']['test']} Flickr8k images (train / valid / test), English text and "
            f"speech ({data['lang_en']['with_audio']} utterances), Yorùbá speech ({data['lang_yo']['with_audio']} "
            f"utterances, orthography held out), Hindi text; frozen `{cfg['vision_encoder']}` and "
            f"`{cfg['speech_encoder']}` features, `{cfg['mel_spec']}` mel targets and vocoder; "
            f"{cfg['phase_a_steps']} + {cfg['phase_b_steps']} optimisation steps with {cfg['step_batches']} batches of "
            f"{cfg['batch_size']} per step.", "",
            f"- Stages completed: {', '.join(done)}.",
            f"- Inference wrote {figs} figures and {wavs} synthesised utterances; figures that failed: "
            f"{rep.get('failed') or 'none'}.",
            "- **No metric is reported**: after this many steps the model has learned nothing, and the numbers in "
            "its `eval_test.json` mean nothing. Real-data results have to come from the GPU schedule "
            "(`GeoBridgeFlow_single_cell.ipynb`, preset `flickr8k_yo_t4`, or `scripts/paper_experiments.sh`).", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="")
    ap.add_argument("--ablation", default="")
    ap.add_argument("--real", default="", help="a real-data run whose metrics are worth a table")
    ap.add_argument("--real_check", default="", help="a real-data run that only shows the pipeline works (no metrics)")
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
        md += ["## 2. Ablations on SynthWorld", ""] + [l.replace("](plot_", "](results/ablation/plot_") for l in body[1:]] + [""]
    if a.real:
        md += run_section("3. Flickr8k-Tri subset (real data)", a.real, "en", "yo", "hi",
                          "English text and speech, Yorùbá speech (orthography held out), Hindi text. "
                          "This is a pipeline check on a small subset with a small model, not a result to cite.")
    if a.real_check:
        md += real_check_section(a.real_check)
    md += export_assets(a)
    with open(a.out, "w", encoding="utf8") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
