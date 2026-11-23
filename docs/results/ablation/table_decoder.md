### decoder: Should the byte decoder be trained on the latents the text writer actually writes?

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST chrF++ ↑ | island ST content acc ↑ |   ... two-object sentences ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->written image R@1 ↑ | hub ASR content acc ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written content acc ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | text->written image R@1 ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text read+write, two-object sentences ↑ | both objects covered (text) ↑ | both objects covered (island speech) ↑ | slots on unmentioned objects (text) ↓ | gate (island) ↑ | gate (hub lang) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decoder frozen after the representation phase (reference) | **48.06** | 77.05 | **56.67** | 26.09 | 87.50 | 50.83 | **42.08** | 10.83 | 75.42 | 13.49 | **23.06** | 76.67 | 66.76 | 79.91 | **13.33** | 14.40 | **36.67** | 67.39 | 92.39 | 59.78 | 7.29 | 0.87 | 0.01 |
| decoder trained on written latents (write-latent exposure) | 47.65 | **77.28** | 55.00 | **27.17** | **89.58** | 50.83 | 41.67 | **11.25** | **82.92** | **8.60** | 22.34 | **92.08** | **91.26** | **91.20** | 9.58 | **17.72** | 36.25 | **89.13** | 92.39 | 59.78 | 7.29 | 0.87 | 0.01 |
