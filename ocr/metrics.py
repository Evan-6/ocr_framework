"""Sequence accuracy and character accuracy (normalized edit distance)."""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compute_metrics(preds: list[str], labels: list[str]) -> dict:
    assert len(preds) == len(labels)
    correct = sum(p == t for p, t in zip(preds, labels))
    edits = sum(levenshtein(p, t) for p, t in zip(preds, labels))
    total_chars = sum(len(t) for t in labels)
    return {
        "seq_acc": correct / len(labels),
        "char_acc": max(0.0, 1.0 - edits / max(1, total_chars)),
        "n": len(labels),
    }


def detailed_metrics(preds: list[str], labels: list[str]) -> dict:
    """Per-position accuracy and a character confusion map.

    Positional alignment is only well-defined when pred and gt have equal length
    (they always do for fixed-length captchas; for CTC length mismatches the sample
    is counted as unaligned and skipped for the positional breakdown).
    """
    from collections import Counter, defaultdict

    pos_correct: Counter = Counter()
    pos_total: Counter = Counter()
    confusion: dict = defaultdict(Counter)  # gt_char -> {pred_char: count}
    aligned = 0
    for p, t in zip(preds, labels):
        if len(p) != len(t):
            continue
        aligned += 1
        for i, (cp, ct) in enumerate(zip(p, t)):
            pos_total[i] += 1
            if cp == ct:
                pos_correct[i] += 1
            else:
                confusion[ct][cp] += 1
    return {
        "per_position_acc": {str(i): pos_correct[i] / pos_total[i] for i in sorted(pos_total)},
        "confusion": {gt: dict(c) for gt, c in confusion.items()},
        "aligned": aligned,
        "unaligned": len(labels) - aligned,
    }
