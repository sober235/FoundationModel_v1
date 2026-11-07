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


# ---------------------------------------------------------------------------
# v5.1 cardinality-first matching (Proposal_v5 §8.1). v41 match_instances above is unchanged.

import math  # noqa: E402

from ..contracts_v5 import V5Error  # noqa: E402


def _check_threshold_v5(policy: dict) -> float:
    threshold = policy.get("iou_match_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) \
            or not 0 < threshold <= 1:
        raise V5Error("INVALID_IOU_THRESHOLD", "$.policy.iou_match_threshold")
    return float(threshold)


def match_instances_v5(pred: list[dict], truth: list[dict], policy: dict) -> list[tuple[int, int]]:
    """Max eligible one-to-one pairs first, then max total IoU; ties resolved by sorted instance_uid.

    Eligible edges weigh M + IoU with M = min(n_pred, n_truth) + 1: M exceeds any IoU-sum difference,
    so adding one eligible pair always outweighs any IoU gain among fewer pairs.
    """
    threshold = _check_threshold_v5(policy)
    class_aware = policy.get("class_aware")
    if not isinstance(class_aware, bool):
        raise V5Error("INVALID_POLICY", "$.policy.class_aware")
    shapes = {np.shape(item["mask"]) for item in pred + truth}
    if len(shapes) > 1:
        raise V5Error("MASK_SHAPE_MISMATCH", "$.mask")
    for side, items in (("pred", pred), ("truth", truth)):
        uids = [item["instance_uid"] for item in items]
        if len(set(uids)) != len(uids):
            raise V5Error("DUPLICATE_INSTANCE_ID", f"$.{side}.instance_uid")
    if not pred or not truth:
        return []
    pred_order = sorted(range(len(pred)), key=lambda i: pred[i]["instance_uid"])
    truth_order = sorted(range(len(truth)), key=lambda j: truth[j]["instance_uid"])
    big_m = min(len(pred), len(truth)) + 1
    weights = np.zeros((len(pred), len(truth)))
    eligible = np.zeros((len(pred), len(truth)), dtype=bool)
    for r, i in enumerate(pred_order):
        p_mask = np.asarray(pred[i]["mask"], dtype=bool)
        for c, j in enumerate(truth_order):
            if class_aware and pred[i]["class_name"] != truth[j]["class_name"]:
                continue
            overlap = _iou(p_mask, np.asarray(truth[j]["mask"], dtype=bool))
            if overlap >= threshold:
                eligible[r, c] = True
                weights[r, c] = big_m + overlap
    rows, cols = linear_sum_assignment(weights, maximize=True)
    pairs = [(pred_order[r], truth_order[c]) for r, c in zip(rows, cols) if eligible[r, c]]
    return sorted(pairs)
