"""Tokenizer-free text units.

``grapheme``  (ours)  one unit per extended grapheme cluster; the unit is embedded from its UTF-8 bytes.
``byte``              one unit per UTF-8 byte (ByT5-style baseline).
``bpe``               byte-level BPE learned on the hub (high-resource) language only, i.e. the situation
                      real pretrained vocabularies are in when they meet a new script.

All three share one tensor layout so the text encoder is identical across the ablation:
``units``  LongTensor [L, M]  byte ids of each unit (PAD = 256), or the BPE id in column 0
``target`` LongTensor [T]     BOS + UTF-8 bytes + EOS  (the decoder is byte-level in every variant)
"""
from __future__ import annotations

import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PAD, BOS, EOS = 256, 257, 258
BYTE_VOCAB = 259

try:  # the `regex` package implements UAX #29 extended grapheme clusters (\X)
    import regex as _regex

    _GRAPHEME = _regex.compile(r"\X")
except Exception:  # pragma: no cover - fallback keeps the package importable without `regex`
    _regex = None
    _GRAPHEME = None

_EXTEND_CATS = {"Mn", "Mc", "Me"}
_VIRAMAS = {0x094D, 0x09CD, 0x0A4D, 0x0ACD, 0x0B4D, 0x0BCD, 0x0C4D, 0x0CCD, 0x0D4D, 0x0DCA}


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.strip().split()))


def graphemes(s: str) -> List[str]:
    """Extended grapheme clusters.  Indic conjuncts (C + virama + C) are kept together."""
    if _GRAPHEME is not None:
        parts = _GRAPHEME.findall(s)
    else:
        parts = []
        for ch in s:
            if parts and (unicodedata.category(ch) in _EXTEND_CATS or ch in "‍‌"):
                parts[-1] += ch
            else:
                parts.append(ch)
    out: List[str] = []
    for p in parts:  # merge "…virama" + next consonant (UAX #29 GB9c, absent from older tables)
        if out and ord(out[-1][-1]) in _VIRAMAS:
            out[-1] += p
        else:
            out.append(p)
    return out


def _chunk(b: bytes, m: int) -> List[bytes]:
    return [b[i:i + m] for i in range(0, len(b), m)] or [b]


class TextUnitizer:
    """Turns strings into unit tensors and reports script-fairness statistics."""

    def __init__(self, mode: str = "grapheme", max_units: int = 128, max_bytes_per_unit: int = 8,
                 max_text_bytes: int = 160, bpe_path: Optional[str] = None) -> None:
        assert mode in ("grapheme", "byte", "bpe")
        self.mode = mode
        self.max_units = max_units
        self.m = max_bytes_per_unit if mode == "grapheme" else 1
        self.max_text_bytes = max_text_bytes
        self.bpe = None
        if mode == "bpe":
            from tokenizers import Tokenizer

            if not (bpe_path and os.path.exists(bpe_path)):
                raise FileNotFoundError("BPE tokenizer not trained yet; call TextUnitizer.train_bpe first")
            self.bpe = Tokenizer.from_file(bpe_path)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ BPE baseline
    @staticmethod
    def train_bpe(texts: Iterable[str], path: str, vocab_size: int = 2000) -> None:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator((normalize_text(t) for t in texts), trainer)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tok.save(path)

    @property
    def unit_vocab(self) -> int:
        return self.bpe.get_vocab_size() + 1 if self.mode == "bpe" else BYTE_VOCAB

    @property
    def unit_pad(self) -> int:
        return self.bpe.get_vocab_size() if self.mode == "bpe" else PAD

    # ------------------------------------------------------------------ segmentation
    def segment(self, s: str) -> List[bytes]:
        s = normalize_text(s)
        if self.mode == "grapheme":
            out: List[bytes] = []
            for g in graphemes(s):
                out.extend(_chunk(g.encode("utf8"), self.m))  # over-long clusters are split, never truncated
            return out
        if self.mode == "byte":
            return [bytes([b]) for b in s.encode("utf8")]
        return [t.encode("utf8") for t in self.bpe.encode(s).tokens]  # only used for statistics

    def encode(self, s: str) -> Tuple[torch.Tensor, torch.Tensor]:
        hit = self._cache.get(s)
        if hit is not None:
            return hit
        s_n = normalize_text(s)
        raw = s_n.encode("utf8")[: self.max_text_bytes]
        s_n = raw.decode("utf8", errors="ignore")  # never cut inside a code point
        raw = s_n.encode("utf8")
        if self.mode == "bpe":
            ids = self.bpe.encode(s_n).ids[: self.max_units] or [self.unit_pad]
            units = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
        else:
            segs = self.segment(s_n)[: self.max_units] or [b" "]
            units = torch.full((len(segs), self.m), PAD, dtype=torch.long)
            for i, seg in enumerate(segs):
                units[i, : len(seg)] = torch.tensor(list(seg), dtype=torch.long)
        target = torch.tensor([BOS] + list(raw) + [EOS], dtype=torch.long)
        out = (units, target)
        if len(self._cache) < 500_000:
            self._cache[s] = out
        return out

    def collate(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = [self.encode(t) for t in texts]
        L = max(u.shape[0] for u, _ in enc)
        T = max(t.shape[0] for _, t in enc)
        units = torch.full((len(enc), L, self.m), self.unit_pad, dtype=torch.long)
        mask = torch.zeros(len(enc), L, dtype=torch.bool)
        target = torch.full((len(enc), T), PAD, dtype=torch.long)
        for i, (u, t) in enumerate(enc):
            units[i, : u.shape[0]] = u
            mask[i, : u.shape[0]] = True
            target[i, : t.shape[0]] = t
        return {"units": units, "unit_mask": mask, "bytes": target}

    # ------------------------------------------------------------------ statistics (fertility)
    def fertility(self, texts: Sequence[str]) -> Dict[str, float]:
        """Units per word / per grapheme / per byte: the script-fairness numbers of the tokenizer study."""
        n_units = n_words = n_graph = n_bytes = 0
        for t in texts:
            t = normalize_text(t)
            n_units += len(self.segment(t))
            n_words += max(1, len(t.split()))
            n_graph += max(1, len(graphemes(t)))
            n_bytes += max(1, len(t.encode("utf8")))
        return {"units_per_word": n_units / max(1, n_words), "units_per_grapheme": n_units / max(1, n_graph),
                "units_per_byte": n_units / max(1, n_bytes), "bytes_per_grapheme": n_bytes / max(1, n_graph)}


def bytes_to_text(ids: Sequence[int]) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i < 256:
            out.append(i)
    return bytes(out).decode("utf8", errors="ignore")
