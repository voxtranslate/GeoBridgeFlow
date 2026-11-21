# Results measured so far

Everything below was produced by the code in this repository on a machine with **4 CPU cores and no GPU**.
The runs are therefore small: a few thousand optimisation steps, models of 5–10 M parameters, **one seed**.
They show that the pipeline runs end to end and which mechanisms have an effect in a controlled setting.
They are **not** the numbers of the paper: those need the GPU schedule in `scripts/paper_experiments.sh`
(full data, three seeds, significance tests).

How to regenerate this file:

```bash
python scripts/make_results.py --synth runs/sf_main --ablation runs/sf_ablation --real runs/f1200
```
