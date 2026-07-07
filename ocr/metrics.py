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
