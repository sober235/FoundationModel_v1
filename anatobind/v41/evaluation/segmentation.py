"""Segmentation metrics with explicit empty/undefined conventions.

Dice over an empty-versus-empty pair is NaN and must be aggregated by the
caller, never silently averaged. Surface distances are undefined when either
side is empty; callers count those cases as failures instead of dropping
them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class MetricPolicyError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


FROZEN_KEYS = ("surface_tolerance_mm", "iou_match_threshold", "fp_budget_per_scan",
               "j_target_range", "a_noninferiority_margin")


def check_policy(policy: dict, strict: bool) -> list[str]:
    missing = [key for key in FROZEN_KEYS if policy.get(key) is None]
    if strict and missing:
        raise MetricPolicyError("METRIC_POLICY_NULL")
    return missing


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def dice_score(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray | None = None) -> float:
    pred, truth = _as_bool(pred), _as_bool(truth)
    if eligible is not None:
        eligible = _as_bool(eligible)
        pred, truth = pred & eligible, truth & eligible
    denominator = pred.sum() + truth.sum()
    if denominator == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, truth).sum() / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    return mask & ~ndimage.binary_erosion(mask)


def surface_distances(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray | None:
    pred, truth = _as_bool(pred), _as_bool(truth)
    if not pred.any() or not truth.any():
        return None
    pred_surface, truth_surface = _surface(pred), _surface(truth)
    to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[truth_surface]
    return np.concatenate([to_truth, to_pred])


def hd95(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...]) -> float | None:
    distances = surface_distances(pred, truth, spacing)
    if distances is None:
        return None
    return float(np.percentile(distances, 95))


def nsd(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...], tolerance_mm: float) -> float | None:
    distances = surface_distances(pred, truth, spacing)
    if distances is None:
        return None
    return float((distances <= tolerance_mm).mean())


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray,
                         spacing: tuple[float, ...], policy: dict, strict: bool = False) -> dict:
    missing = check_policy(policy, strict)
    tolerance = policy.get("surface_tolerance_mm")
    eligible = _as_bool(eligible)
    pred, truth = _as_bool(pred) & eligible, _as_bool(truth) & eligible
    metrics = {
        "dice": dice_score(pred, truth),
        "hd95_mm": hd95(pred, truth, spacing),
        "nsd": None if tolerance is None else nsd(pred, truth, spacing, tolerance),
        "surface_tolerance_mm": tolerance,
        "truth_empty": not _as_bool(truth).any(),
        "pred_empty": not _as_bool(pred).any(),
        "policy_blocked": bool(missing),
        "missing_policy_keys": missing,
    }
    metrics["surface_defined"] = metrics["hd95_mm"] is not None
    return metrics
