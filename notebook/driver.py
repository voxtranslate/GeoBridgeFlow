def describe_runtime() -> None:
    log = get_logger()
    where = "Kaggle" if IN_KAGGLE else "Colab" if IN_COLAB else "local"
    log.info(f"[setup] {where} | torch {torch.__version__} | runs -> {WORK_DIR} | data -> {DATA_DIR}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            log.info(f"[setup] cuda:{i} {p.name} {p.total_memory / 2 ** 30:.1f} GB")
        log.info(f"[setup] {torch.cuda.device_count()} GPU(s): one batch per GPU in every training step")
    else:
        log.info("[setup] no GPU visible: this will run, slowly, on the CPU (pick a 'synth' preset to try it out)")


def show_images(paths: Sequence[str], width: int = 1100) -> None:
    """Display saved figures below the cell (no-op outside a notebook)."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return
    if get_ipython() is None:
        return
    for p in paths:
        if os.path.exists(p):
            print(os.path.relpath(p, WORK_DIR))
            display(Image(filename=p, width=width))


def main_notebook() -> Dict[str, Any]:
    describe_runtime()
    results: Dict[str, Any] = {}
    cfg = build_config(PRESET, {"out_root": os.path.join(WORK_DIR, "runs"), "data_root": DATA_DIR, **OVERRIDES}, EXP_NAME)
    if STAGES:
        results = run_stages(cfg, STAGES, split=EVAL_SPLIT, groups=ABLATION_GROUPS or None, seeds=ABLATION_SEEDS)
        if EVAL_SPLIT in results:
            keep = {k: round(v, 2) for k, v in results[EVAL_SPLIT].items()
                    if k.endswith(("/bleu", "/chrf", "/wer", "/cer", "/r1", "content_acc")) or k.startswith("tone/") or k == "score"}
            print(json.dumps(keep, indent=1))
        if SHOW_FIGURES:
            show_images(sorted(glob.glob(os.path.join(cfg.run_dir, "curves", "*.png"))))
            show_images(sorted(glob.glob(os.path.join(cfg.run_dir, "inference", "*.png"))))
    if ABLATION_PRESET and "ablate" not in STAGES:
        a_cfg = build_config(ABLATION_PRESET, {"out_root": os.path.join(WORK_DIR, "runs"), "data_root": DATA_DIR,
                                               **ABLATION_OVERRIDES})
        results["ablation"] = run_stages(a_cfg, ("ablate",), groups=ABLATION_GROUPS or None, seeds=ABLATION_SEEDS)["ablation"]
        root = os.path.join(a_cfg.out_root, f"{a_cfg.exp_name}_ablation")
        summary = os.path.join(root, "SUMMARY.md")
        if os.path.exists(summary):
            print(open(summary, encoding="utf8").read())
        if SHOW_FIGURES:
            show_images(sorted(glob.glob(os.path.join(root, "*.png"))))
    print(f"\nEverything is in {WORK_DIR}")
    return results


RESULTS = main_notebook()
