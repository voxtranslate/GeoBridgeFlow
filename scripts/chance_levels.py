#!/usr/bin/env python
"""How far are the outputs of a finished evaluation from chance?

    python scripts/chance_levels.py runs/<exp>/eval_test.json

Works on any ``eval_*.json``, also from runs made before the evaluator reported chance levels itself: the stored
sample outputs (at most 60 per direction) are scored against their own reference and against the references of
other images (mean and spread over 200 random re-pairings).  A caption model that ignores its source still writes
fluent captions; what it scores is the chance level, and only the distance from it is evidence of content.

For English targets one more column: how often the output names the right kind of subject (dog, man, woman, boy,
girl, child, people), the coarsest thing a caption can get right.  The word list is English and ad hoc; the other
columns need no language knowledge.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geobridge.metrics import bleu_chrf, content_f1, words  # noqa: E402

SUBJECTS = {"dog": {"dog", "dogs", "puppy", "puppies", "lab", "terrier", "poodle", "retriever", "pug"},
            "man": {"man", "men", "guy", "male"}, "woman": {"woman", "women", "lady", "female"},
            "boy": {"boy", "boys"}, "girl": {"girl", "girls"},
            "child": {"child", "children", "kid", "kids", "baby", "toddler", "infant"},
            "people": {"people", "person", "crowd", "group", "couple", "family"}}


def subjects(s: str) -> set:
    w = set(words(s))
    return {k for k, v in SUBJECTS.items() if w & v}


def subject_match(hyps, refs) -> float:
    hits = [bool(subjects(h) & subjects(r)) for h, r in zip(hyps, refs) if subjects(r)]
    return 100.0 * sum(hits) / max(1, len(hits))


def chance_table(samples: dict, pairings: int = 200, seed: int = 0) -> list:
    """One row per direction: (name, n, [(own, chance mean, chance sd) for content F1, BLEU, chrF++, right subject]);
    the last entry is None where the target is not English."""
    rng = random.Random(seed)
    out = []
    for name, rows in samples.items():
        hy, rf, img = [r["hyp"] for r in rows], [r["ref"] for r in rows], [r["img"] for r in rows]
        if len(set(img)) < 2:
            continue
        english = all(ord(c) < 0x250 for c in "".join(rf))
        own = [content_f1(hy, rf), *bleu_chrf(hy, [[r] for r in rf]).values(), subject_match(hy, rf)]
        chance = [[], [], [], []]
        for _ in range(pairings):
            while True:   # every output gets the reference of another image
                p = list(range(len(rows)))
                rng.shuffle(p)
                if all(img[i] != img[j] for i, j in enumerate(p)):
                    break
            wrong = [rf[j] for j in p]
            b = bleu_chrf(hy, [[r] for r in wrong])
            for c, v in zip(chance, (content_f1(hy, wrong, pool=rf), b["bleu"], b["chrf"], subject_match(hy, wrong))):
                c.append(v)
        cells = [(o, statistics.mean(c), statistics.pstdev(c)) for o, c in zip(own, chance)]
        out.append((name, len(rows), cells[:3] + [cells[3] if english else None]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json")
    ap.add_argument("--pairings", type=int, default=200)
    a = ap.parse_args()
    samples = json.load(open(a.eval_json, encoding="utf8"))["samples"]
    head = f"{'direction':28s} {'n':>3s} | {'content F1':>18s} | {'BLEU':>18s} | {'chrF++':>18s} | {'right subject, %':>18s}"
    print(head + "\n" + "-" * len(head) + "\n" + " " * 34 + "own  chance ± sd" + "      own  chance ± sd" * 3)
    for name, n, cells in chance_table(samples, a.pairings):
        print(f"{name[:28]:28s} {n:3d} | " + " | ".join(f"{c[0]:6.1f} {c[1]:6.1f} ± {c[2]:3.1f}" if c else " " * 18 for c in cells))


if __name__ == "__main__":
    main()
