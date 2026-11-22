### tone: Does lexical tone need its own gated path into the hub?  (N5: lexical-tonality gate)

_1 seed(s); test split; mean; bold = best in column (arrows give the direction)._

| variant | island ST BLEU ↑ | island ST BLEU (multi-ref) ↑ | island ST chrF++ ↑ | island ST content acc ↑ | island ST tone-word acc ↑ |   ... with pitch removed ↑ | island speech->image R@1 ↑ | island speech->text hub R@1 ↑ | hub ASR WER ↓ | captioning BLEU (multi-ref) ↑ | MT hub->written BLEU ↑ | MT written->hub BLEU ↑ | TTS into island (round-trip BLEU) ↑ | TTS into island (round-trip content) ↑ | text codec CER (written lang) ↓ | gate (island) ↑ | gate (hub lang) ↓ | tone reliance (island) ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gated tone stream (ours) | **22.41** | **29.29** | **57.27** | **28.33** | **81.25** | 47.50 | **25.42** | **52.92** | **35.58** | **25.79** | 43.06 | **51.02** | **10.43** | **15.00** | 11.62 | **0.58** | 0.00 | **0.29** |
| no tone stream (single stream) | 15.49 | 21.32 | 48.08 | 23.33 | 56.25 | **56.25** | 13.33 | 38.33 | 36.77 | 19.05 | **43.30** | 46.15 | 4.27 | 11.67 | **11.59** | 0.00 | **0.00** | 0.00 |
