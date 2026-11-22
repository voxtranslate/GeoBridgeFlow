### transitive: What makes two spokes agree on the hub at test time?  (N4: transitive hub-swap)

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST BLEU (multi-ref) ↑ | island ST chrF++ ↑ | island ST content acc ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->text hub R@1 ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text codec CER (written lang) ↓ | gate (island) ↑ | gate (hub lang) ↓ | tone reliance (island) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hub-swap CFM (ours) | **22.41** | **29.29** | 57.27 | 28.33 | 81.25 | 47.50 | **25.42** | **52.92** | **35.58** | 25.79 | 43.06 | **51.02** | **10.43** | **15.00** | 11.62 | 0.58 | 0.00 | 0.29 |
| no cross-spoke term | 21.08 | 27.61 | **58.13** | **30.42** | **88.33** | **51.25** | 25.00 | 50.00 | 53.44 | **30.02** | **48.68** | 48.58 | 5.19 | 10.42 | 11.62 | 0.58 | 0.00 | **0.31** |
