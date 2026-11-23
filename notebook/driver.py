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
    log = get_logger()
    set_time_budget(TIME_BUDGET_HOURS, SESSION_START)
    if TIME_BUDGET_HOURS:
        log.info(f"[setup] session started {(time.time() - SESSION_START) / 60:.0f} min ago; training stops cleanly "
                 f"in {time_left() / 3600:.1f} h (TIME_BUDGET_HOURS = {TIME_BUDGET_HOURS})")
    results: Dict[str, Any] = {}
    paths = {"out_root": os.path.join(WORK_DIR, "runs"), "data_root": DATA_DIR}
    cfg = build_config(PRESET, {**paths, **OVERRIDES}, EXP_NAME)
    # the ablation of PRESET is the ablation of the run above: same settings, same folder name
    same = ABLATION_PRESET == PRESET
    a_cfg = cfg if "ablate" in STAGES or not ABLATION_PRESET else build_config(
        ABLATION_PRESET, {**paths, **(OVERRIDES if same else {}), **ABLATION_OVERRIDES}, EXP_NAME if same else "")
    a_root = os.path.join(a_cfg.out_root, f"{a_cfg.exp_name}_ablation")
    ablation = dict(groups=ABLATION_GROUPS or None, only=ABLATION_ONLY or None, seeds=ABLATION_SEEDS)
    try:
        if STAGES:
            results = run_stages(cfg, STAGES, split=EVAL_SPLIT, **ablation)
        if ABLATION_PRESET and "ablate" not in STAGES:
            results["ablation"] = run_stages(a_cfg, ("ablate",), **ablation)["ablation"]
    except OutOfTime as e:
        log.info(f"[time] {e}")
        log.info("[time] stopped before the session limit: run this cell again (same settings) to continue")
    if EVAL_SPLIT in results:
        keep = {k: round(v, 2) for k, v in results[EVAL_SPLIT].items()
                if k.endswith(("/bleu", "/chrf", "/wer", "/cer", "/r1", "content_acc")) or k.startswith("tone/") or k == "score"}
        print(json.dumps(keep, indent=1))
    if STAGES and SHOW_FIGURES:
        show_images(sorted(glob.glob(os.path.join(cfg.run_dir, "curves", "*.png"))))
        show_images(sorted(glob.glob(os.path.join(cfg.run_dir, "inference", "*.png"))))
    if ABLATION_PRESET or "ablate" in STAGES:
        summary = os.path.join(a_root, "SUMMARY.md")
        if os.path.exists(summary):
            print(open(summary, encoding="utf8").read())
        if SHOW_FIGURES:
            show_images(sorted(glob.glob(os.path.join(a_root, "*.png"))))
    print(f"\nEverything is in {WORK_DIR}")
    return results


RESULTS = main_notebook()
