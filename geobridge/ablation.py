"""Ablation runner: every architectural choice against its alternatives, multi-seed, resumable.

Each *group* isolates one design decision; its first variant is always the full model ("ours").  A variant is
a set of config overrides.  For every (variant, seed) the runner trains (or resumes), evaluates on the test
split and stores ``results.json``; finished cells are skipped on re-run.  Variants that share the same
representation phase reuse one phase-A checkpoint (matched by a hash of the fields that determine phase A),
so a transport-only ablation costs only its phase B.

Outputs in ``<out_root>/<exp>_ablation/``
    _full/full_s<seed>/...            the full model (shared by every group)
    _phase_a/<hash>/...               representation phases shared between variants
    <group>/<variant>_s<seed>/...     complete run directories (checkpoints, curves, eval json)
    table_<group>.md|csv|tex          mean +- std over seeds of the headline metrics
    plot_<group>.png                  bar chart per metric with seed error bars
    SUMMARY.md                        all tables + which variant wins each column
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PHASE_A_KEYS, Config
from .utils import get_logger, load_json, save_json


GROUPS: Dict[str, Dict] = {
    "hub": {
        "question": "What should the hub be, and what makes it say exactly what the sentence says?",
        "variants": [
            ("barycentric hub, mutual sufficiency (ours)", {}),
            ("no counterfactual patch credit (softmax gradient only)", {"w_seek": 0.0}),
            ("counterfactual credit with a veto on moves that verifiably explain less", {"seek_verify": True}),
            ("no minimality (reader does not shape the hub)", {"w_min": 0.0}),
            ("no sufficiency (no explainer loss, hence no credit either)", {"w_expl": 0.0}),
            ("text code as a fixed target for speech, instead of mutual agreement", {"agree_mode": "teach"}),
            ("independent slots (no inhibition of return)", {"hub_inhibit": 0.0}),
            ("sharp attention from the start (no anneal)", {"hub_temp_start": 1.0}),
            ("global visual pivot (scene-level)", {"hub_mode": "global"}),
            ("text anchor (image-pivoted)", {"hub_mode": "text"}),
            ("free learned anchor (FlowBind-style)", {"hub_mode": "free"}),
        ]},
    "read": {
        "question": "How should a source be read into the hub at test time?",
        "variants": [
            ("image-free reader, one pass (ours)", {}),
            ("backward ODE of the flow, trained with ODE hub-swap", {"read_mode": "ode", "exposure": "ode"}),
            ("reader, writers never trained on read hubs", {"exposure": "none", "hub_jitter": 0.0}),
            ("reader, writers trained without hub jitter", {"hub_jitter": 0.0}),
            ("reader, speech keeps its own code (no canonical text code)", {"canonical_text": False}),
        ]},
    "decoder": {
        "question": "Should the byte decoder be trained on the latents the text writer actually writes?",
        "variants": [
            ("decoder frozen after the representation phase (reference)", {}),
            ("decoder trained on written latents (write-latent exposure)", {"w_dec": 1.0}),
        ]},
    "island": {
        "question": "What should be done with the unpaired speech of the island language?",
        "variants": [
            ("hub-space back-translation, transport-rectified (ours)", {}),
            ("back-translation without the relay to the text writers", {"bt_relay": 0.0}),
            ("back-translation without the transport step", {"bt_rectify": 0.0}),
            ("bridge pairs only (unpaired speech unused in phase B)", {"bt_weight": 0.0}),
            ("fused Gromov-Wasserstein docking instead", {"bt_weight": 0.0, "dock_mode": "g2w"}),
            ("pure Gromov-Wasserstein docking instead (roadmap)", {"bt_weight": 0.0, "dock_mode": "gw"}),
        ]},
    "bridge": {
        "question": "How many image-speech bridge pairs does the island need, with and without back-translation?",
        "variants": [("20% bridge, back-translation (reference)", {"bridge_frac": 0.2})] + [
            (f"{int(100 * f)}% bridge, {'back-translation' if w else 'bridge pairs only'}", {"bridge_frac": f, "bt_weight": w})
            for f in (0.01, 0.05, 0.2) for w in (1.0, 0.0) if not (f == 0.2 and w == 1.0)] + [
            ("100% bridge (every island utterance paired)", {"bridge_frac": 1.0})]},
    "tone": {
        "question": "Does lexical tone need its own gated path into the hub?",
        "variants": [
            ("gated tone stream (ours)", {}),
            ("tone stream always on (no gate)", {"tone_mode": "on"}),
            ("no tone stream (single stream)", {"tone_mode": "off"}),
            ("gate without the sparsity price", {"w_gate": 0.0}),
            ("no paralinguistic bypass", {"use_bypass": False, "w_orth": 0.0}),
        ]},
    "text_unit": {
        "question": "Which text unit feeds the compressor?",
        "variants": [
            ("grapheme clusters over bytes (ours)", {}),
            ("raw bytes", {"text_unit": "byte"}),
            ("BPE learned on the hub language", {"text_unit": "bpe"}),
        ]},
    "private": {
        "question": "Do the writers need private (non-hub) degrees of freedom?",
        "variants": [
            ("hub + private Gaussian tokens (ours)", {}),
            ("private tokens fixed to zero (deterministic writers)", {"private_noise": False}),
        ]},
    "lora": {
        "question": "How much language-specific capacity do the velocity fields need?",
        "variants": [
            ("language LoRA, default rank (ours)", {}),
            ("no language LoRA (shared field + language embedding)", {"lora_rank": 0}),
            ("rank x4", {"lora_rank": "x4"}),
        ]},
    "protocol": {
        "question": "How much does an implicitly parallel corpus inflate the zero-pair numbers?",
        "variants": [
            ("strict non-parallel protocol (ours)", {}),
            ("captions visible in every language (implicitly parallel)", {"strict_nonparallel": False}),
            ("island orthography available (written upper bound)", {"use_island_text": True}),
        ]},
}

# headline columns: (pattern, short name, higher-is-better)
COLUMNS: List[Tuple[str, str, bool]] = [
    ("aud_{L}>txt_{H}/bleu", "island ST BLEU", True),
    ("aud_{L}>txt_{H}/chrf", "island ST chrF++", True),
    ("aud_{L}>txt_{H}/content_acc", "island ST content acc", True),
    ("aud_{L}>txt_{H}/content2_acc", "  ... two-object sentences", True),
    ("aud_{L}>txt_{H}/color_acc", "island ST tone-word acc", True),
    ("aud_{L}>txt_{H}/flat_color_acc", "  ... with pitch removed", True),
    ("aud_{L}>img/r1", "island speech->image R@1", True),
    ("aud_{L}>img_gen/r1", "island speech->written image R@1", True),
    ("aud_{H}>txt_{H}/content_acc", "hub ASR content acc", True),
    ("aud_{H}>txt_{H}/wer", "hub ASR WER", False),
    ("img>txt_{H}/bleu_multi", "captioning BLEU (multi-ref)", True),
    ("txt_{H}>txt_{T}/content_acc", "MT hub->written content acc", True),
    ("txt_{H}>txt_{T}/bleu", "MT hub->written BLEU", True),
    ("txt_{T}>txt_{H}/bleu", "MT written->hub BLEU", True),
    ("txt_{H}>img_gen/r1", "text->written image R@1", True),
    ("txt_{H}>aud_{L}/rt_bleu", "TTS into island (round-trip BLEU)", True),
    ("txt_{H}>aud_{L}/rt_content_acc", "TTS into island (round-trip content)", True),
    ("txt_{H}>txt_{H}/content2_acc", "text read+write, two-object sentences", True),
    ("ground/txt_{H}/both_covered", "both objects covered (text)", True),
    ("ground/aud_{L}/both_covered", "both objects covered (island speech)", True),
    ("ground/txt_{H}/slots_unmentioned", "slots on unmentioned objects (text)", False),
    ("tone/gate_{L}", "gate (island)", True),
    ("tone/gate_{H}", "gate (hub lang)", False),
]


class AblationRunner:
    def __init__(self, base: Config, seeds: Sequence[int] = (0,), test_items: int = 0):
        self.base, self.seeds, self.test_items = base, list(seeds), test_items
        self.root = os.path.join(base.out_root, f"{base.exp_name}_ablation")
        os.makedirs(self.root, exist_ok=True)
        self.log = get_logger("gb.ablation", os.path.join(self.root, "ablation.log"))

    def describe(self) -> None:
        for g, spec in GROUPS.items():
            print(f"\n[{g}] {spec['question']}")
            for name, over in spec["variants"]:
                print(f"    - {name:55s} {over}")

    # ------------------------------------------------------------------ one cell
    def _cfg(self, group: str, name: str, over: Dict, seed: int) -> Config:
        over = dict(over)
        if over.get("lora_rank") == "x4":
            over["lora_rank"] = self.base.lora_rank * 4
        if all(getattr(self.base, k) == v for k, v in over.items()):
            # the full model is the same run in every group: train it once, under _full/
            return self.base.clone(seed=seed, out_root=os.path.join(self.root, "_full"), exp_name=f"full_s{seed}")
        slug = "".join(c if c.isalnum() else "_" for c in name).strip("_")[:48]
        return self.base.clone(seed=seed, out_root=os.path.join(self.root, group), exp_name=f"{slug}_s{seed}", **over)

    def _phase_a(self, cfg: Config) -> str:
        """Train (once) and return the phase-A checkpoint shared by all configs with this representation."""
        sig = cfg.signature(PHASE_A_KEYS)
        a_cfg = cfg.clone(out_root=os.path.join(self.root, "_phase_a"), exp_name=sig, phase_b_steps=0, init_from="")
        path = os.path.join(a_cfg.run_dir, "checkpoints", "phase_a.pt")
        if not os.path.exists(path):
            from .trainer import Trainer

            self.log.info(f"[phase A] training shared representation {sig}")
            Trainer(a_cfg).fit()
        return path

    def run_cell(self, group: str, name: str, over: Dict, seed: int) -> Dict[str, float]:
        from .inferencer import load_model
        from .metrics import Evaluator
        from .trainer import Trainer

        cfg = self._cfg(group, name, over, seed)
        res_path = os.path.join(cfg.run_dir, "results.json")
        if os.path.exists(res_path):
            return load_json(res_path)
        if cfg.phase_a_steps > 0 and not os.path.exists(os.path.join(cfg.run_dir, "checkpoints", "last.pt")):
            cfg.init_from = self._phase_a(cfg)
        self.log.info(f"[cell] {group} / {name} / seed {seed}")
        tr = Trainer(cfg)
        tr.fit()
        model, _ = load_model(cfg, tr.data, tr.device, os.path.join(cfg.run_dir, "checkpoints", "last.pt"))
        res = Evaluator(cfg, tr.data, tr.device).evaluate(model, "test", self.test_items, quick=False,
                                                         save_path=os.path.join(cfg.run_dir, "eval_test.json"))
        res["params_total"] = tr.param_report["total"]
        res["per_language_pct"] = tr.param_report["per_language_pct_of_spokes"]
        save_json(res, res_path)
        return res

    # ------------------------------------------------------------------ whole study
    def run(self, groups: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None) -> Dict:
        """Run every variant of the given groups (all groups by default).  ``only`` restricts the run to
        variants whose name contains one of the given substrings; the full model is always included."""
        groups = list(groups) if groups else list(GROUPS)
        for g in groups:
            for i, (name, over) in enumerate(GROUPS[g]["variants"]):
                if only and i > 0 and not any(o.lower() in name.lower() for o in only):
                    continue
                for s in self.seeds:
                    self.run_cell(g, name, over, s)
        return self.collect(groups)

    def collect(self, groups: Optional[Sequence[str]] = None) -> Dict:
        """Rebuild tables from finished cells only (no training)."""
        summary: Dict = {}
        for g in (list(groups) if groups else list(GROUPS)):
            rows: Dict[str, List[Dict]] = {}
            for name, over in GROUPS[g]["variants"]:
                done = []
                for s in self.seeds:
                    p = os.path.join(self._cfg(g, name, over, s).run_dir, "results.json")
                    if os.path.exists(p):
                        done.append(load_json(p))
                if done:
                    rows[name] = done
            if len(rows) >= 2:  # a group with only the full model has nothing to compare
                summary[g] = self._report(g, GROUPS[g]["question"], rows)
        self._write_summary(summary)
        return summary

    # ------------------------------------------------------------------ reporting
    def _columns(self, rows: Dict[str, List[Dict]]) -> List[Tuple[str, str, bool]]:
        b = self.base
        H = b.hub_langs[0]
        L = b.island_langs[0] if b.island_langs else "-"
        T = next((t for t in b.text_only_langs if t != H), "-")
        keys = set(k for rs in rows.values() for r in rs for k in r)
        out = []
        for pat, short, hib in COLUMNS:
            k = pat.format(H=H, L=L, T=T)
            if k in keys:
                out.append((k, short, hib))
        return out

    def _report(self, group: str, question: str, rows: Dict[str, List[Dict]]) -> Dict:
        cols = self._columns(rows)
        stats = {name: {k: (float(np.mean([r[k] for r in rs if k in r])) if any(k in r for r in rs) else float("nan"),
                            float(np.std([r[k] for r in rs if k in r])) if sum(k in r for r in rs) > 1 else 0.0)
                        for k, _, _ in cols} for name, rs in rows.items()}
        best = {}
        for k, _, hib in cols:
            vals = {n: s[k][0] for n, s in stats.items() if not np.isnan(s[k][0])}
            if vals and max(vals.values()) - min(vals.values()) > 1e-9:  # no winner when every variant ties
                best[k] = (max if hib else min)(vals, key=vals.get)
        n_seeds = max(len(r) for r in rows.values())
        fmt = lambda n, k: (f"**{stats[n][k][0]:.2f}**" if best.get(k) == n else f"{stats[n][k][0]:.2f}") + \
                           (f" ± {stats[n][k][1]:.2f}" if n_seeds > 1 else "")
        md = [f"### {group}: {question}", "", f"_{n_seeds} seed(s); test split; mean" + (" ± std" if n_seeds > 1 else "") +
              "; bold = best in column (arrows give the direction)._", "",
              "| variant | " + " | ".join(f"{s} {'↑' if h else '↓'}" for _, s, h in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for n in rows:
            md.append(f"| {n} | " + " | ".join(fmt(n, k) for k, _, _ in cols) + " |")
        with open(os.path.join(self.root, f"table_{group}.md"), "w", encoding="utf8") as f:
            f.write("\n".join(md) + "\n")
        with open(os.path.join(self.root, f"table_{group}.csv"), "w", encoding="utf8") as f:
            f.write("variant," + ",".join(f"{s} mean,{s} std" for _, s, _ in cols) + "\n")
            for n in rows:
                f.write('"' + n + '",' + ",".join(f"{stats[n][k][0]:.4f},{stats[n][k][1]:.4f}" for k, _, _ in cols) + "\n")
        tex = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
               "Variant & " + " & ".join(s.replace("%", "\\%").replace("_", "\\_").replace(">", "$\\to$") for _, s, _ in cols) + " \\\\",
               "\\midrule"]
        for n in rows:
            cells = []
            for k, _, _ in cols:
                v = f"{stats[n][k][0]:.2f}" + (f"$\\pm${stats[n][k][1]:.2f}" if n_seeds > 1 else "")
                cells.append(f"\\textbf{{{v}}}" if best.get(k) == n else v)
            tex.append(n.replace("%", "\\%").replace("_", "\\_").replace(">", "$\\to$") + " & " + " & ".join(cells) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(self.root, f"table_{group}.tex"), "w", encoding="utf8") as f:
            f.write("\n".join(tex) + "\n")
        self._plot(group, question, rows, cols, stats)
        save_json({"question": question, "stats": {n: {k: list(v) for k, v in s.items()} for n, s in stats.items()},
                   "best": best, "seeds": n_seeds}, os.path.join(self.root, f"table_{group}.json"))
        return {"md": "\n".join(md), "best": best, "cols": cols, "stats": stats}

    def _plot(self, group, question, rows, cols, stats) -> None:
        from .utils import setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        cols = cols[:8]
        if not cols:
            return
        names = list(rows)
        fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 0.42 * len(names) + 2.6), squeeze=False)
        for ax, (k, short, hib) in zip(axes[0], cols):
            means = [stats[n][k][0] for n in names]
            stds = [stats[n][k][1] for n in names]
            colors = ["#d62728" if i == 0 else "#7f9ec9" for i in range(len(names))]
            ax.barh(range(len(names)), means, xerr=stds, color=colors, capsize=2)
            ax.set_yticks(range(len(names)), names if ax is axes[0, 0] else [""] * len(names), fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{short} {'↑' if hib else '↓'}", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle(f"{group}: {question}   (red = full model)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(self.root, f"plot_{group}.png"), dpi=130)
        plt.close(fig)

    def _write_summary(self, summary: Dict) -> None:
        md = [f"# Ablation study: {self.base.exp_name}", "",
              f"Dataset `{self.base.dataset}`, seeds {self.seeds}.  Every number is measured on the test split, "
              "through the hub, with no image and no text of the island language at test time.", ""]
        for g, rep in summary.items():
            md += [rep["md"], ""]
            ours = next(iter(rep["stats"]))
            wins = [s for k, s, _ in rep["cols"] if rep["best"].get(k) == ours]
            loses = [f"{s} (best: {rep['best'][k]})" for k, s, _ in rep["cols"] if k in rep["best"] and rep["best"][k] != ours]
            md += [f"Full model best on: {', '.join(wins) if wins else 'none'}.",
                   f"Full model not best on: {', '.join(loses) if loses else 'none'}.", "", f"![{g}](plot_{g}.png)", ""]
        with open(os.path.join(self.root, "SUMMARY.md"), "w", encoding="utf8") as f:
            f.write("\n".join(md) + "\n")
        self.log.info(f"[ablation] tables and plots in {self.root}")
