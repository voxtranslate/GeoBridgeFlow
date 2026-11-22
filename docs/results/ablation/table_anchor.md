### anchor: What should the hub be anchored to?  (N1: barycentric visual interlingua)

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST BLEU (multi-ref) ↑ | island ST chrF++ ↑ | island ST content acc ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->text hub R@1 ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text codec CER (written lang) ↓ | gate (island) ↑ | gate (hub lang) ↓ | tone reliance (island) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bary (ours) | **22.41** | **29.29** | 57.27 | 28.33 | 81.25 | 47.50 | **25.42** | 52.92 | 35.58 | 25.79 | **43.06** | **51.02** | 10.43 | 15.00 | 11.62 | 0.58 | 0.00 | 0.29 |
| global visual pivot | 6.80 | 15.53 | 40.28 | 11.67 | 58.33 | 36.25 | 8.33 | 13.33 | 51.32 | 11.30 | 27.77 | 37.29 | 4.51 | 2.50 | 11.62 | 0.58 | 0.00 | **0.31** |
| text anchor (image-pivoted) | 21.75 | 27.63 | **59.79** | **41.25** | **90.83** | **55.00** | 1.25 | **60.42** | **30.16** | **30.35** | 13.81 | 28.58 | **16.07** | **18.75** | 11.62 | 0.58 | 0.00 | 0.24 |
