"""SynthWorld: a procedural tri-modal benchmark in which every factor of the problem is known exactly.

It exists for two reasons.  (1) The whole pipeline can be exercised in minutes on a CPU, with no downloads.
(2) It gives *controlled* evidence for claims real corpora can only support indirectly:

  * tonal minimal pairs   - in language ``sb`` colour words come in pairs that share every segment and differ
                            only in their tone pattern, so colour is unrecoverable without pitch;
  * a disjoint island     - ``sb`` is speech-only (its orthography is held out for scoring);
  * script fragmentation  - ``sc`` is written in Devanagari syllables (6 UTF-8 bytes per grapheme cluster);
  * sentence-level meaning - a caption describes a *subset* of the objects in its image, so a scene-level
                            visual pivot cannot translate it, and grounding can be scored by a pointing game.

Languages
  sa  hub language      Latin script, non-tonal, text + speech, several speakers, adjective-noun order
  sb  island language   tonal, speech only (2 speakers), noun-adjective order
  sc  written language  Devanagari, text only

Pitch moves by a similar amount in ``sa`` and ``sb``; only in ``sb`` is the movement lexical.
Object types follow a Zipf law (``synth_zipf``), as content words do in real captions.
"""
from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

CONS = list("ptkbdgmnslfv")
VOWS = list("aeiou")
FORMANTS = {"a": (750, 1250), "e": (450, 2000), "i": (300, 2400), "o": (500, 900), "u": (330, 800)}
CONS_SPEC = {  # (noise centre Hz, noise bandwidth Hz, voiced)
    "p": (800, 400, False), "t": (3500, 700, False), "k": (2000, 500, False),
    "b": (700, 350, True), "d": (3000, 700, True), "g": (1800, 500, True),
    "m": (1000, 250, True), "n": (1500, 250, True), "s": (5200, 900, False),
    "l": (1200, 300, True), "f": (3000, 1500, False), "v": (2600, 1200, True),
}
DEVA_C = dict(zip(CONS, "पतकबदगमनसलफव"))
DEVA_V = {"a": "ा", "e": "े", "i": "ि", "o": "ो", "u": "ु"}
ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
GRAVE = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
TONE_ST = {"H": 3.0, "L": -3.0, "M": 0.0}  # semitones relative to the speaker's register
COLOR_RGB = [(220, 40, 40), (40, 170, 60), (50, 90, 220), (235, 200, 40), (200, 60, 200), (40, 200, 210),
             (240, 140, 30), (120, 80, 40), (250, 250, 250), (90, 90, 90)]
SHAPES = ["circle", "square", "triangle", "diamond", "cross", "bar", "ring", "star", "hex", "tee"]
COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "brown", "white", "grey"]

SYNTH_LANGS = {
    "sa": {"tonal": False, "script": "Latin", "written": True, "spoken": True},
    "sb": {"tonal": True, "script": "Latin+tone marks (held out)", "written": False, "spoken": True},
    "sc": {"tonal": False, "script": "Devanagari", "written": True, "spoken": False},
}

Word = List[Tuple[str, str, str]]  # syllables as (consonant, vowel, tone)


class SynthWorld:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.nc, self.ns, self.g = cfg.synth_n_colors, cfg.synth_n_shapes, cfg.synth_grid
        assert self.nc % 2 == 0 and self.nc <= len(COLOR_RGB) and self.ns <= len(SHAPES)
        rng = np.random.default_rng(20261121)  # the *world* is fixed; cfg.seed only affects training
        self.lex = self._make_lexicons(rng)
        d = cfg.synth_vision_dim
        self.e_color = rng.standard_normal((self.nc, d)) / math.sqrt(d) * 2.0
        self.e_shape = rng.standard_normal((self.ns, d)) / math.sqrt(d) * 2.0
        self.e_bg = rng.standard_normal(d) / math.sqrt(d) * 0.6
        n_sa = cfg.synth_n_speakers
        self.speakers = {
            "sa": [(float(rng.uniform(95, 240)), float(rng.uniform(0.9, 1.15))) for _ in range(n_sa)],
            "sb": [(float(rng.uniform(110, 210)), float(rng.uniform(0.92, 1.1))) for _ in range(2)],
        }
        self.scenes, self.caps = self._make_scenes(rng, cfg.synth_n_images)
        self.x_scenes: List = []   # extra scenes (hub-language text only), appended after the benchmark's own

    # ------------------------------------------------------------------ lexicons
    def _stems(self, rng, n: int, used: set) -> List[List[Tuple[str, str]]]:
        out = []
        while len(out) < n:
            st = tuple((str(rng.choice(CONS)), str(rng.choice(VOWS))) for _ in range(2))
            if st not in used:
                used.add(st)
                out.append(list(st))
        return out

    def _make_lexicons(self, rng) -> Dict[str, Dict]:
        lex: Dict[str, Dict] = {}
        for lang in ("sa", "sb", "sc"):
            used: set = set()
            if lang == "sb":  # tonal minimal pairs: colours 2i and 2i+1 share a stem
                stems = self._stems(rng, self.nc // 2, used)
                colors = []
                for st in stems:
                    colors.append([(c, v, t) for (c, v), t in zip(st, "HL")])
                    colors.append([(c, v, t) for (c, v), t in zip(st, "LH")])
                shapes = [[(c, v, str(rng.choice(["H", "L"]))) for c, v in st]
                          for st in self._stems(rng, self.ns, used)]
                conj = [(c, v, "M") for c, v in self._stems(rng, 1, used)[0][:1]]
            else:
                colors = [[(c, v, "M") for c, v in st] for st in self._stems(rng, self.nc, used)]
                shapes = [[(c, v, "M") for c, v in st] for st in self._stems(rng, self.ns, used)]
                conj = [(c, v, "M") for c, v in self._stems(rng, 1, used)[0][:1]]
            lex[lang] = {"color": colors, "shape": shapes, "conj": conj,
                         "order": "noun_adj" if lang == "sb" else "adj_noun"}
        return lex

    def spell(self, word: Word, lang: str) -> str:
        if lang == "sc":
            return "".join(DEVA_C[c] + DEVA_V[v] for c, v, _ in word)
        if lang == "sb":
            return "".join(c + {"H": ACUTE[v], "L": GRAVE[v], "M": v}[t] for c, v, t in word)
        return "".join(c + v for c, v, _ in word)

    def words(self, content: Sequence[Tuple[int, int]], lang: str) -> List[Word]:
        lx, out = self.lex[lang], []
        for i, (c, s) in enumerate(content):
            if i:
                out.append(lx["conj"])
            pair = [lx["color"][c], lx["shape"][s]]
            out.extend(pair if lx["order"] == "adj_noun" else pair[::-1])
        return out

    def text(self, content: Sequence[Tuple[int, int]], lang: str) -> str:
        return " ".join(self.spell(w, lang) for w in self.words(content, lang))

    def parse(self, text: str, lang: str) -> Optional[List[Tuple[int, int]]]:
        """Inverse of :meth:`text`; ``None`` when the string is not a well-formed caption."""
        lx = self.lex[lang]
        col = {self.spell(w, lang): i for i, w in enumerate(lx["color"])}
        shp = {self.spell(w, lang): i for i, w in enumerate(lx["shape"])}
        conj = self.spell(lx["conj"], lang)
        toks = [t for t in text.strip().split() if t != conj]
        if len(toks) == 0 or len(toks) % 2:
            return None
        out = []
        for a, b in zip(toks[0::2], toks[1::2]):
            if lx["order"] != "adj_noun":
                a, b = b, a
            if a not in col or b not in shp:
                return None
            out.append((col[a], shp[b]))
        return out

    # ------------------------------------------------------------------ scenes & captions
    def _make_scenes(self, rng, n: int):
        cfg, scenes, caps = self.cfg, [], []
        cells = self.g * self.g
        # Zipfian object types: like real captions, a few (colour, shape) combinations are very frequent and
        # most are rare, and colour is not independent of shape.  This is what gives unpaired data a shape.
        n_types = self.nc * self.ns
        rank = np.random.default_rng(4242).permutation(n_types)
        p_type = 1.0 / (1.0 + rank) ** cfg.synth_zipf
        self.type_prob = p_type / p_type.sum()
        for i in range(n):
            n_obj = int(rng.integers(2, 4))
            pos = rng.choice(cells, size=n_obj, replace=False)
            types = rng.choice(n_types, size=n_obj, p=self.type_prob)
            objs = [(int(p), int(t) // self.ns, int(t) % self.ns) for p, t in zip(pos, types)]
            scenes.append(objs)
            for k in range(cfg.synth_caps_per_image):
                two = rng.random() < cfg.synth_two_obj_prob
                pick = rng.choice(n_obj, size=2 if two else 1, replace=False)
                caps.append({"img": i, "cap": k, "objs": [objs[int(j)] for j in pick],
                             "spk_sa": int(rng.integers(len(self.speakers["sa"]))),
                             "spk_sb": int(rng.integers(len(self.speakers["sb"]))),
                             "seed": int(rng.integers(1 << 30))})
        return scenes, caps

    # ------------------------------------------------------------------ manifest
    def manifest(self) -> Dict:
        images = [{"id": f"im{i:05d}", "scene": sc} for i, sc in enumerate(self.scenes)]
        utts = []
        for ci, c in enumerate(self.caps):
            content = [(o[1], o[2]) for o in c["objs"]]
            cells = [o[0] for o in c["objs"]]
            for lang in ("sa", "sb", "sc"):
                info = SYNTH_LANGS[lang]
                utts.append({
                    "uid": f"{lang}/im{c['img']:05d}/{c['cap']}", "img": c["img"], "cap": c["cap"], "lang": lang,
                    "text": self.text(content, lang), "has_audio": info["spoken"],
                    "spk": f"{lang}{c.get('spk_' + lang, 0)}" if info["spoken"] else "",
                    "content": content, "cells": cells, "cap_row": ci,
                })
        return {"name": "synthworld", "langs": SYNTH_LANGS, "images": images, "utts": utts,
                "meta": {"grid": self.g, "colors": COLOR_NAMES[: self.nc], "shapes": SHAPES[: self.ns]}}

    def extra_manifest(self, n: int) -> Dict:
        """``n`` more scenes with captions in the hub language only, as text: what a high-resource language has
        plenty of.  Drawn from the same world with their own random stream, so the benchmark itself is untouched."""
        n = 4 * len(self.scenes) if n < 0 else n
        H = self.cfg.hub_langs[0]
        self.x_scenes, caps = self._make_scenes(np.random.default_rng(20261122), n)
        images = [{"id": f"xim{i:05d}", "scene": sc, "official_split": "train", "extra": True} for i, sc in enumerate(self.x_scenes)]
        utts = [{"uid": f"{H}/xim{c['img']:05d}/{c['cap']}", "img": c["img"], "cap": c["cap"], "lang": H,
                 "text": self.text([(o[1], o[2]) for o in c["objs"]], H), "has_audio": False, "spk": "",
                 "content": [(o[1], o[2]) for o in c["objs"]], "cells": [o[0] for o in c["objs"]]} for c in caps]
        return {"name": "synthworld_extra", "images": images, "utts": utts, "meta": {"scenes": n}}

    def _scene(self, img: int) -> List:
        """Scene of an image index: the benchmark's own images first, then the extra ones."""
        return self.scenes[img] if img < len(self.scenes) else self.x_scenes[img - len(self.scenes)]

    # ------------------------------------------------------------------ vision
    def vision_tokens(self, img: int) -> np.ndarray:
        """Analytic 'frozen encoder' output: one token per grid cell + a global token (last)."""
        rng = np.random.default_rng(7919 * (img + 1))
        d, cells = self.cfg.synth_vision_dim, self.g * self.g
        tok = np.tile(self.e_bg, (cells, 1))
        for p, c, s in self._scene(img):
            tok[p] = self.e_color[c] + self.e_shape[s]
        tok = tok + self.cfg.synth_feat_noise * rng.standard_normal((cells, d)) / math.sqrt(d) * 2.0
        return np.concatenate([tok, tok.mean(0, keepdims=True)], 0).astype(np.float32)

    def render(self, img: int, size: int = 128):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (size, size), (24, 24, 28))
        dr = ImageDraw.Draw(im)
        cell = size / self.g
        for p, c, s in self._scene(img):
            r, q = divmod(p, self.g)
            x0, y0, x1, y1 = q * cell + 3, r * cell + 3, (q + 1) * cell - 3, (r + 1) * cell - 3
            cx, cy, col = (x0 + x1) / 2, (y0 + y1) / 2, COLOR_RGB[c]
            name = SHAPES[s]
            if name == "circle":
                dr.ellipse([x0, y0, x1, y1], fill=col)
            elif name == "square":
                dr.rectangle([x0, y0, x1, y1], fill=col)
            elif name == "triangle":
                dr.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=col)
            elif name == "diamond":
                dr.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=col)
            elif name == "cross":
                w = (x1 - x0) / 3
                dr.rectangle([cx - w / 2, y0, cx + w / 2, y1], fill=col)
                dr.rectangle([x0, cy - w / 2, x1, cy + w / 2], fill=col)
            elif name == "bar":
                dr.rectangle([x0, cy - (y1 - y0) / 6, x1, cy + (y1 - y0) / 6], fill=col)
            elif name == "ring":
                dr.ellipse([x0, y0, x1, y1], outline=col, width=max(2, int(cell / 7)))
            elif name == "star":
                pts = []
                for k in range(10):
                    rad = (x1 - x0) / 2 * (1.0 if k % 2 == 0 else 0.45)
                    a = -math.pi / 2 + k * math.pi / 5
                    pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
                dr.polygon(pts, fill=col)
            elif name == "hex":
                pts = [(cx + (x1 - x0) / 2 * math.cos(k * math.pi / 3), cy + (y1 - y0) / 2 * math.sin(k * math.pi / 3))
                       for k in range(6)]
                dr.polygon(pts, fill=col)
            else:  # tee
                w = (x1 - x0) / 3
                dr.rectangle([x0, y0, x1, y0 + w], fill=col)
                dr.rectangle([cx - w / 2, y0, cx + w / 2, y1], fill=col)
        return im

    # ------------------------------------------------------------------ speech synthesis
    def synth(self, cap_row: int, lang: str, sr: int = 16000) -> np.ndarray:
        c = self.caps[cap_row]
        rng = np.random.default_rng(c["seed"] + (0 if lang == "sa" else 77))
        f_base, fscale = self.speakers[lang][c["spk_" + lang]]
        words = self.words([(o[1], o[2]) for o in c["objs"]], lang)
        tonal = SYNTH_LANGS[lang]["tonal"]
        rate = float(rng.uniform(0.88, 1.15))
        n_syl = sum(len(w) for w in words)
        chunks, k = [np.zeros(int(0.06 * sr))], 0
        for w in words:
            for cons, vow, tone in w:
                # lexical tone (sb) vs. free, uninformative intonation of the same magnitude (sa)
                st = TONE_ST[tone] if tonal else float(rng.uniform(-3.0, 3.0))
                st += -1.0 * k / max(1, n_syl - 1) + float(rng.normal(0, 0.25))  # declination + jitter
                f0 = f_base * 2 ** (st / 12)
                chunks.append(_consonant(cons, f0, int(0.05 * sr / rate), sr, rng))
                chunks.append(_vowel(vow, f0, fscale, int(0.13 * sr / rate), sr, rng))
                k += 1
            chunks.append(np.zeros(int(0.04 * sr / rate)))
        chunks.append(np.zeros(int(0.06 * sr)))
        wav = np.concatenate(chunks)
        wav = wav / (np.abs(wav).max() + 1e-6) * 0.7
        return (wav + 0.004 * rng.standard_normal(len(wav))).astype(np.float32)

    def audio_groups(self, utts: List[Dict], chunk: int = 400) -> List[Dict]:
        rows = [u for u in utts if u["has_audio"]]
        groups = []
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]

            def loader(part=part) -> Iterator[Tuple[str, np.ndarray, int]]:
                for u in part:
                    yield u["uid"], self.synth(u["cap_row"], u["lang"]), 16000

            groups.append({"key": f"synth_{i // chunk:05d}", "uids": [u["uid"] for u in part], "loader": loader})
        return groups


def _vowel(v: str, f0: float, fscale: float, n: int, sr: int, rng) -> np.ndarray:
    t = np.arange(n) / sr
    glide = 1.0 + 0.015 * np.linspace(-1, 1, n) * rng.choice([-1, 1])
    phase = 2 * np.pi * np.cumsum(f0 * glide) / sr
    out = np.zeros(n)
    f1, f2 = (f * fscale for f in FORMANTS[v])
    for h in range(1, int(3800 / f0) + 1):
        fh = h * f0
        amp = math.exp(-((fh - f1) ** 2) / (2 * 110 ** 2)) + 0.7 * math.exp(-((fh - f2) ** 2) / (2 * 150 ** 2)) + 0.02
        out += amp * np.sin(h * phase)
    env = np.minimum(1.0, np.minimum(t / 0.015, (t[-1] - t) / 0.02 + 1e-3))
    return out * env


def _consonant(c: str, f0: float, n: int, sr: int, rng) -> np.ndarray:
    centre, bw, voiced = CONS_SPEC[c]
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    noise = np.fft.irfft(spec * np.exp(-((freqs - centre) ** 2) / (2 * bw ** 2)), n)
    noise = noise / (np.abs(noise).max() + 1e-6)
    env = np.concatenate([np.zeros(n // 3), np.hanning(2 * (n - n // 3))[: n - n // 3]])
    out = 0.5 * noise * env
    if voiced:
        t = np.arange(n) / sr
        out = out + 0.25 * np.sin(2 * np.pi * f0 * t) * np.hanning(n)
    return out
