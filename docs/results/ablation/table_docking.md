### docking: How should unpaired island speech be attached to the hub?  (N3: Gromov->Wasserstein docking)

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST BLEU (multi-ref) ↑ | island ST chrF++ ↑ | island ST content acc ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->text hub R@1 ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text codec CER (written lang) ↓ | gate (island) ↑ | gate (hub lang) ↓ | tone reliance (island) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G->W anneal, balanced marginals, pinned (ours) | **22.41** | **29.29** | **57.27** | 28.33 | 81.25 | 47.50 | 25.42 | 52.92 | **35.58** | 25.79 | 43.06 | **51.02** | **10.43** | 15.00 | 11.62 | 0.58 | 0.00 | 0.29 |
| no docking (bridge pairs only) | 16.87 | 21.83 | 56.89 | **33.33** | **85.83** | 48.33 | **29.58** | **55.83** | 37.30 | 22.36 | 41.39 | 49.18 | 9.60 | **29.17** | 11.62 | 0.58 | 0.00 | 0.32 |
| semi-relaxed marginals | 1.02 | 2.01 | 15.88 | 2.08 | 12.50 | 14.17 | 1.67 | 2.08 | 45.63 | 25.02 | 41.81 | 44.84 | 1.40 | 2.08 | 11.62 | 0.58 | 0.00 | 0.41 |
| semi-relaxed, pseudo-pairs only (no docking loss) | 14.23 | 21.28 | 52.66 | 19.58 | 77.92 | 46.67 | 22.50 | 48.75 | 37.96 | **26.59** | **44.05** | 48.74 | 7.58 | 8.33 | 11.62 | 0.58 | 0.00 | 0.36 |
| pure Gromov-Wasserstein | 3.91 | 6.36 | 23.78 | 2.92 | 22.08 | 12.50 | 2.50 | 2.92 | 38.49 | 25.86 | 41.65 | 45.27 | 2.08 | 1.67 | 11.62 | 0.58 | 0.00 | **0.46** |
| pure Wasserstein | 21.34 | 29.01 | 55.59 | 23.33 | 85.83 | **55.42** | 20.83 | 43.75 | 38.36 | 25.25 | 43.73 | 49.54 | 6.64 | 9.58 | 11.62 | 0.58 | 0.00 | 0.26 |
