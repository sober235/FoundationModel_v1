"""One-to-one lesion instance matching: maximize eligible pairs first, then IoU."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def match_instances(pred: list[dict], truth: list[dict], policy: dict) -> list[tuple[int, int]]:
    if not pred or not truth:
        return []
    threshold = float(policy["iou_match_threshold"])
    ious = np.zeros((len(pred), len(truth)))
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            ious[i, j] = _iou(np.asarray(p["mask"], dtype=bool), np.asarray(t["mask"], dtype=bool))
    allowed = ious >= threshold
    cost = np.where(allowed, -(2.0 + ious), 0.0)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if allowed[i, j]]
    pairs.sort(key=lambda pair: (-ious[pair], pair[0], pair[1]))
    return pairs


def detection_counts(pred: list[dict], truth: list[dict], matches: list[tuple[int, int]]) -> dict:
    return {"tp": len(matches), "fp": len(pred) - len(matches), "fn": len(truth) - len(matches)}
