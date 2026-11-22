### private: Do the spokes need private (non-hub) degrees of freedom?  (N2: hub (+) private flows)

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST BLEU (multi-ref) ↑ | island ST chrF++ ↑ | island ST content acc ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->text hub R@1 ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text codec CER (written lang) ↓ | gate (island) ↑ | gate (hub lang) ↓ | tone reliance (island) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hub + private Gaussian tokens (ours) | **22.41** | **29.29** | **57.27** | **28.33** | **81.25** | 47.50 | **25.42** | **52.92** | **35.58** | **25.79** | **43.06** | **51.02** | **10.43** | **15.00** | 11.62 | 0.58 | 0.00 | 0.29 |
| private tokens fixed to zero (deterministic spokes) | 7.01 | 19.69 | 39.57 | 18.33 | 79.17 | **57.50** | 15.83 | 20.42 | 64.95 | 10.54 | 8.54 | 14.77 | 5.20 | 3.33 | 11.62 | 0.58 | 0.00 | **0.40** |
