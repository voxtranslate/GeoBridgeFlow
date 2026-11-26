"""Ablation runner: every architectural choice against its alternatives, multi-seed, resumable.

Each *group* isolates one design decision.  A variant is a set of config overrides; the variant whose settings
are the base settings is the full model and is the first row of the group's table.  For every (variant, seed)
the runner trains (or resumes), evaluates the state selected by the run's own validation score on the test split
and stores ``results.json``; finished cells are skipped on re-run.  Cells run cheapest first, because a session
has a time limit.  Nothing is trained twice:

  * the full model of the base seed is the main run ``<out_root>/<exp>`` itself when that run has the base
    settings (it is trained there if it does not exist yet);
  * variants that share the same representation phase reuse one phase-A checkpoint, matched by a hash of the
    fields that determine phase A - the main run's when it matches - so a transport-only variant costs only
    its phase B.

Outputs in ``<out_root>/<exp>_ablation/``
    _full/full_s<seed>/...            results of the full model (shared by every group); for other seeds than
                                      the main run's, also the run itself
    _phase_a/<hash>/...               representation phases that the main run does not provide
    <group>/<variant>_s<seed>/...     complete run directories (checkpoints, curves, eval json)
    table_<group>.md|csv|tex          mean +- std over seeds of the headline metrics
    plot_<group>.png                  bar chart per metric with seed error bars
    SUMMARY.md                        all tables + which variant wins each column
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PHASE_A_KEYS, Config
from .utils import OutOfTime, get_logger, load_json, save_json


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
            ("language-balanced losses (every language the same weight in its modality)", {"lang_balance": True}),
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
            ("reader never shown written hub-language speech", {"w_reread": 0.0}),
            ("reader trained on clean speech latents only (no time-masked encodings)", {"read_aug": 0}),
        ]},
    "sibling": {
        "question": "Does writing another caption of the same image from a weak view's code help that view's language?",
        "variants": [
            ("no image-sibling transport", {"w_sib": 0.0}),
            ("image-sibling transport: writer, decoder and reader", {"w_sib": 1.0, "sib_views": "weak", "sib_reader_grad": True}),
            ("image-sibling transport: writer and decoder only", {"w_sib": 1.0, "sib_views": "weak", "sib_reader_grad": False}),
            ("image-sibling transport for every view", {"w_sib": 1.0, "sib_views": "all", "sib_reader_grad": True}),
        ]},
    "decode": {
        "question": "What does the way of reading and writing at test time change? (no training: the same model, scored again)",
        "variants": [
            ("read-back selection among 4 candidates", {"select_k": 4}),
            ("one candidate, no selection", {"select_k": 1}),
            ("read-back selection among 8 candidates", {"select_k": 8}),
            ("read by inverting the flow instead of the reader", {"read_mode": "ode"}),
            ("8 integration steps per hop instead of 4", {"ode_steps": 8}),
        ]},
    "decoder": {
        "question": "Should the byte decoder be trained on the latents the text writer actually writes?",
        "variants": [
            ("decoder trained on written latents (write-latent exposure)", {"w_dec": 1.0}),
            ("decoder frozen after the representation phase", {"w_dec": 0.0}),
            ("speaking decoder never shown written speech latents", {"w_dec_aud": 0.0}),
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
    "speech": {
        "question": "What does the speech latent need in order to be said again?",
        "variants": [
            ("content tokens + a speaking path with time-ordered tokens", {"audio_anchor": "extra", "audio_private_slots": "speak"}),
            ("the hub reads the content tokens only (no listening tokens)", {"listen_tokens": 0}),
            ("free queries only (no order of time)", {"audio_anchor": "off", "audio_private_slots": 0}),
            ("every token laid out in time, all read by the hub", {"audio_anchor": "all", "audio_private_slots": 0}),
        ]},
    "data": {
        "question": "What does extra data in the hub language (text, speech) buy the other languages?",
        "variants": [
            ("extra hub-language text and speech", {"extra_images": -1, "extra_speech": -1}),
            ("... the extra speech only listens (read into the hub, shaping none of it)",
             {"extra_images": -1, "extra_speech": -1, "listen_only": True}),
            ("extra hub-language text only", {"extra_images": -1, "extra_speech": 0}),
            ("extra hub-language speech only", {"extra_images": 0, "extra_speech": -1}),
            ("the benchmark's own data only", {"extra_images": 0, "extra_speech": 0}),
        ]},
    "regularise": {
        "question": "Can the encoders be regularised without losing what the credit loss buys?",
        "variants": [
            ("no encoder dropout (ours)", {"enc_dropout": 0.0, "speech_mask": 0.0}),
            ("encoder dropout 0.1 (grounder and explainer untouched)", {"enc_dropout": 0.1, "speech_mask": 0.0}),
            ("encoder dropout 0.1 + speech time masking", {"enc_dropout": 0.1, "speech_mask": 0.1}),
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
    ("aud_{L}>txt_{H}/cf1", "island ST content F1", True),
    ("aud_{L}>txt_{H}/chance_cf1", "  ... chance level", False),
    ("aud_{L}>txt_{H}/bleu", "island ST BLEU", True),
    ("aud_{L}>txt_{H}/chrf", "island ST chrF++", True),
    ("aud_{L}>txt_{H}/content_acc", "island ST content acc", True),
    ("aud_{L}>txt_{H}/content2_acc", "  ... two-object sentences", True),
    ("aud_{L}>txt_{H}/color_acc", "island ST tone-word acc", True),
    ("aud_{L}>txt_{H}/flat_color_acc", "  ... with pitch removed", True),
    ("aud_{L}>img/r1", "island speech->image R@1", True),
    ("aud_{L}>img_gen/r1", "island speech->written image R@1", True),
    ("aud_{H}>txt_{H}/content_acc", "hub ASR content acc", True),
    ("aud_{H}>txt_{H}/cf1", "hub speech->text content F1", True),
    ("aud_{H}>txt_{H}/wer", "hub ASR WER", False),
    ("txt_{H}>txt_{H}/cf1", "text read+write content F1", True),
    ("img>txt_{H}/bleu_multi", "captioning BLEU (multi-ref)", True),
    ("txt_{H}>txt_{T}/content_acc", "MT hub->written content acc", True),
    ("txt_{H}>txt_{T}/bleu", "MT hub->written BLEU", True),
    ("txt_{T}>txt_{H}/bleu", "MT written->hub BLEU", True),
    ("txt_{T}>txt_{H}/cf1", "MT written->hub content F1", True),
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


def _eval_keys() -> Tuple[str, ...]:
    from .metrics import EVAL_KEYS

    return EVAL_KEYS


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
    def _main_is_full(self, seed: int) -> bool:
        """The main run (``base.run_dir``) *is* the full model of the study when it has this seed and was - or
        will be - trained with exactly the base settings.  It is then never trained a second time."""
        from .trainer import RESUME_FREE, settings_diff

        if seed != self.base.seed:
            return False
        record = os.path.join(self.base.run_dir, "config.json")
        if not os.path.exists(record):
            return True   # not started yet: the full model is trained there and becomes the main run
        return not settings_diff(load_json(record), self.base.to_dict(), free=RESUME_FREE - {"phase_b_steps", "train_only"})

    def _resolve(self, over: Dict) -> Dict:
        """Overrides with relative values ("x4") turned into settings."""
        over = dict(over)
        if over.get("lora_rank") == "x4":
            over["lora_rank"] = self.base.lora_rank * 4
        if over.get("audio_private_slots") == "speak":   # the base's own speaking path, or one of twice the latent's length
            b = self.base
            over["audio_private_slots"] = b.audio_private_slots if b.audio_private_slots > b.private_slots else 3 * b.private_slots
        return over

    def _is_full(self, over: Dict) -> bool:
        """A variant whose settings are the base settings is the full model, wherever its group lists it."""
        return all(getattr(self.base, k) == v for k, v in self._resolve(over).items())

    def _eval_only(self, over: Dict) -> bool:
        """A variant that changes nothing but how the trained model is read and written at test time: it is the
        full model, scored again with other settings.  Nothing is trained for it."""
        from .metrics import EVAL_KEYS

        over = self._resolve(over)
        return bool(over) and not self._is_full(over) and set(over) <= set(EVAL_KEYS)

    def _cost(self, group: str, name: str, over: Dict, seed: int) -> int:
        """0: nothing to train (finished, the full model, or the full model scored another way); 1: one transport
        phase (its representation exists); 2: a representation phase as well."""
        cfg, out_dir = self._cell(group, name, over, seed)
        if os.path.exists(os.path.join(out_dir, "results.json")) or self._is_full(over) or self._eval_only(over):
            return 0
        if cfg.phase_a_steps == 0 or os.path.exists(os.path.join(cfg.run_dir, "checkpoints", "last.pt")):
            return 1
        sig = cfg.signature(PHASE_A_KEYS)
        if os.path.exists(os.path.join(self.root, "_phase_a", sig, "checkpoints", "phase_a.pt")):
            return 1
        return 1 if seed == self.base.seed and sig == self.base.signature(PHASE_A_KEYS) else 2

    def _cheapest_first(self, cells: List[Tuple]) -> List[Tuple]:
        """A session has a time limit, so the order decides how many answers it gives.  Cells that need no new
        representation come first, in the order asked for; the rest follow grouped by the representation they share
        (largest group first), so that each extra representation phase is paid for once and used at once."""
        cost = [self._cost(*c) for c in cells]
        sig = [self._cell(*c)[0].signature(PHASE_A_KEYS) if k == 2 else "" for c, k in zip(cells, cost)]
        shared, first = Counter(sig), {}
        for i, sg in enumerate(sig):
            first.setdefault(sg, i)
        late = [self._eval_only(c[2]) for c in cells]   # (scored from the full model's checkpoint: after it)
        order = sorted(range(len(cells)), key=lambda i: (cost[i], late[i], -shared[sig[i]] if cost[i] == 2 else 0, first[sig[i]], i))
        return [cells[i] for i in order]

    def _cell(self, group: str, name: str, over: Dict, seed: int) -> Tuple[Config, str]:
        """(configuration to train, directory that holds the cell's results)."""
        over = self._resolve(over)
        if self._is_full(over):
            # the full model is the same run in every group: one run per seed, under _full/ ...
            own = self.base.clone(seed=seed, out_root=os.path.join(self.root, "_full"), exp_name=f"full_s{seed}")
            started = os.path.exists(os.path.join(own.run_dir, "checkpoints", "last.pt"))
            if not started and self._main_is_full(seed):
                return self.base.clone(), own.run_dir   # ... or the main run itself; only its results go there
            return own, own.run_dir
        slug = "".join(c if c.isalnum() else "_" for c in name).strip("_")[:48]
        cfg = self.base.clone(seed=seed, out_root=os.path.join(self.root, group), exp_name=f"{slug}_s{seed}", **over)
        if set(over) <= set(_eval_keys()):   # the full model scored another way: its run, these settings, own results
            full, _ = self._cell(group, name, {}, seed)
            return full.clone(**over), cfg.run_dir
        return cfg, cfg.run_dir

    def _phase_a(self, cfg: Config) -> str:
        """The phase-A checkpoint shared by every configuration with this representation: an earlier one of the
        study, else the main run's when nothing that shapes phase A differs, else it is trained (once)."""
        sig = cfg.signature(PHASE_A_KEYS)
        a_cfg = cfg.clone(out_root=os.path.join(self.root, "_phase_a"), exp_name=sig, phase_b_steps=0, init_from="")
        path = os.path.join(a_cfg.run_dir, "checkpoints", "phase_a.pt")
        if os.path.exists(path):
            return path
        main, record = (os.path.join(self.base.run_dir, f) for f in ("checkpoints/phase_a.pt", "config.json"))
        if os.path.exists(main) and os.path.exists(record) and Config.load(record).signature(PHASE_A_KEYS) == sig:
            self.log.info(f"[phase A] representation {sig} taken from the main run: {main}")
            return main
        from .trainer import Trainer

        self.log.info(f"[phase A] training shared representation {sig}")
        Trainer(a_cfg).fit()
        return path

    def run_cell(self, group: str, name: str, over: Dict, seed: int) -> Dict[str, float]:
        from .inferencer import load_model
        from .metrics import Evaluator, eval_settings
        from .trainer import Trainer

        cfg, out_dir = self._cell(group, name, over, seed)
        res_path = os.path.join(out_dir, "results.json")
        if os.path.exists(res_path):
            return load_json(res_path)
        if self._eval_only(over):
            return self._score_again(group, name, cfg, out_dir, seed)
        is_main = cfg.run_dir == self.base.run_dir
        if cfg.phase_a_steps > 0 and not is_main and not os.path.exists(os.path.join(cfg.run_dir, "checkpoints", "last.pt")):
            cfg.init_from = self._phase_a(cfg)
        self.log.info(f"[cell] {group} / {name} / seed {seed}" + (f"  (the main run, {cfg.run_dir})" if is_main else ""))
        tr = Trainer(cfg)
        tr.fit()
        # every cell is scored the same way: the state its own validation score selected, on the test split
        best = tr.best["step"] if tr.best["step"] >= 0 else tr.step
        os.makedirs(out_dir, exist_ok=True)
        want = {"step": best, "split": "test", "items": self.test_items, "quick": False, **eval_settings(cfg)}
        stored = os.path.join(cfg.run_dir, "eval_test.json")   # written by the "eval" stage of the main run
        if is_main and os.path.exists(stored) and load_json(stored).get("meta") == json.loads(json.dumps(want)):
            res = load_json(stored)["metrics"]                  # the same state, split and settings: not measured twice
            self.log.info(f"[cell] test scores taken from {stored}")
        else:
            model, _ = load_model(cfg, tr.data, tr.device, os.path.join(cfg.run_dir, "checkpoints", "best.pt"))
            res = Evaluator(cfg, tr.data, tr.device).evaluate(model, "test", self.test_items, quick=False, step=best,
                                                             save_path=os.path.join(out_dir, "eval_test.json"))
        res["checkpoint_step"] = best
        res["params_total"] = tr.param_report["total"]
        res["per_language_pct"] = tr.param_report["per_language_pct_of_spokes"]
        save_json(res, res_path)
        return res

    def _score_again(self, group: str, name: str, cfg: Config, out_dir: str, seed: int) -> Dict[str, float]:
        """An evaluation-only variant: the full model's selected state, scored on the test split with ``cfg``'s
        test-time settings.  The full model is trained first if it is not there yet."""
        from .data import GeoBridgeData
        from .inferencer import load_model
        from .metrics import Evaluator
        from .utils import resolve_device

        full = self.run_cell(group, "the full model", {}, seed)
        ck = os.path.join(cfg.run_dir, "checkpoints", "best.pt")
        self.log.info(f"[cell] {group} / {name} / seed {seed}  (no training: {ck} scored with other test-time settings)")
        device = resolve_device(cfg.device)
        data = GeoBridgeData(cfg, self.log, device)
        model, _ = load_model(cfg, data, device, ck)
        os.makedirs(out_dir, exist_ok=True)
        res = Evaluator(cfg, data, device).evaluate(model, "test", self.test_items, quick=False, step=full.get("checkpoint_step"),
                                                    save_path=os.path.join(out_dir, "eval_test.json"))
        res.update({k: full[k] for k in ("checkpoint_step", "params_total", "per_language_pct") if k in full})
        save_json(res, os.path.join(out_dir, "results.json"))
        return res

    # ------------------------------------------------------------------ whole study
    def run(self, groups: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None) -> Dict:
        """Run every variant of the given groups (all groups by default).  ``only`` restricts the run to
        variants whose name contains one of the given substrings; the full model is always included."""
        groups = list(groups) if groups else list(GROUPS)
        cells = [(g, name, over, s) for g in groups for name, over in GROUPS[g]["variants"]
                 if not only or self._is_full(over) or any(o.lower() in name.lower() for o in only) for s in self.seeds]
        cells = self._cheapest_first(cells)
        dirs = {self._cell(*c)[1] for c in cells}
        todo = [d for d in dirs if not os.path.exists(os.path.join(d, "results.json"))]
        own_a = len({self._cell(*c)[0].signature(PHASE_A_KEYS) for c in cells if self._cost(*c) == 2})
        self.log.info(f"[ablation] {len(dirs)} runs in {len(groups)} group(s), {len(todo)} without results yet.  Order: the "
                      f"full model, then variants that reuse a representation that exists (one transport phase each), "
                      f"then the others ({own_a} more representation phase(s) to train).")
        try:
            for c in cells:
                self.run_cell(*c)
        except OutOfTime:
            self.collect(groups)   # tables of what is finished; the interrupted cell continues on the next run
            raise
        return self.collect(groups)

    def collect(self, groups: Optional[Sequence[str]] = None) -> Dict:
        """Rebuild tables from finished cells only (no training)."""
        summary: Dict = {}
        for g in (list(groups) if groups else list(GROUPS)):
            rows: Dict[str, List[Dict]] = {}
            used = set()
            # the full model first, whatever its place in the list; a second variant that is the same run under
            # these settings is left out (two rows of one run are not a comparison)
            for name, over in sorted(GROUPS[g]["variants"], key=lambda v: not self._is_full(v[1])):
                paths = tuple(os.path.join(self._cell(g, name, over, s)[1], "results.json") for s in self.seeds)
                done = [load_json(p) for p in paths if os.path.exists(p)]
                if done and paths not in used:
                    rows[name + (" (the full model)" if self._is_full(over) and "(ours)" not in name else "")] = done
                    used.add(paths)
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
              "through the hub, with no image and no text of the island language at test time, at the state each "
              "run's own validation score selected.  The first row of every table is the full model.", ""]
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
