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


# ---------------------------------------------------------------------------
# v5.1 segmentation metrics (Proposal_v5 §8.3). v41 functions above are unchanged.
# Empty or undefined values are None with a named reason, never 0, 1, NaN or inf.

import math  # noqa: E402

from ..contracts_v5 import V5Error  # noqa: E402
from .policy import validate_metric_policy_v5  # noqa: E402
from .surface_v5 import SURFACE_SOURCE_COMMIT, hd95_v5, nsd_v5, surface_distances_v5  # noqa: E402

METRIC_VERSION_V5 = f"seg_v5.1/surface-distance@{SURFACE_SOURCE_COMMIT[:7]}"


def _physical_spacing_v5(geometry: dict, ndim: int) -> tuple[float, ...] | None:
    spacing = geometry.get("spacing_mm")
    if geometry.get("physical_geometry_status") != "verified" or spacing is None:
        return None
    if len(spacing) != ndim or not all(isinstance(s, (int, float)) and not isinstance(s, bool)
                                       and math.isfinite(s) and s > 0 for s in spacing):
        raise V5Error("INVALID_SPACING", "$.geometry.spacing_mm")
    return tuple(float(s) for s in spacing)


def _touches_unreviewed_v5(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray) -> bool:
    """True when a mask reaches the border of the non-evaluable region (its surface may be an artificial cut)."""
    if eligible.all():
        return False
    structure = ndimage.generate_binary_structure(eligible.ndim, eligible.ndim)
    ring = ndimage.binary_dilation(~eligible, structure=structure)
    return bool(((pred | truth) & ring).any())


def segmentation_metrics_v5(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray, geometry: dict,
                            policy: dict) -> dict:
    pred, truth, eligible = (np.asarray(a, dtype=bool) for a in (pred, truth, eligible))
    if pred.ndim not in (2, 3):
        raise V5Error("UNSUPPORTED_NDIM", "$.pred")
    if not (pred.shape == truth.shape == eligible.shape):
        raise V5Error("GRID_MISMATCH", "$.pred")
    requested = policy.get("metrics") or []
    blocked = validate_metric_policy_v5(policy, "development")["blocked_metrics"]
    pred, truth = pred & eligible, truth & eligible
    truth_empty, pred_empty = not truth.any(), not pred.any()
    result = {"dice": None, "hd95_mm": None, "nsd": None, "reasons": [], "surface_failure": False,
              "truth_empty": truth_empty, "pred_empty": pred_empty, "metric_version": METRIC_VERSION_V5}
    reasons = result["reasons"]
    if not eligible.any():
        reasons.append("NO_EVALUABLE_REGION")
        return result
    if truth_empty and pred_empty:
        reasons.append("BOTH_EMPTY")
        return result
    result["dice"] = 2.0 * float((pred & truth).sum()) / float(pred.sum() + truth.sum())
    if not ("hd95" in requested or "nsd" in requested):
        return result
    surface_ok = True
    if truth_empty or pred_empty:
        reasons.append("ONE_SIDED_EMPTY")
        result["surface_failure"] = True
        surface_ok = False
    spacing = _physical_spacing_v5(geometry, pred.ndim)
    if spacing is None:
        reasons.append("PHYSICAL_GEOMETRY_UNKNOWN")
        surface_ok = False
    if _touches_unreviewed_v5(pred, truth, eligible):
        reasons.append("SURFACE_REFERENCE_INCOMPLETE")
        surface_ok = False
    if "nsd" in blocked:
        reasons.append("POLICY_VALUE_UNSET")
    if not surface_ok:
        return result
    distances = surface_distances_v5(truth, pred, spacing)
    if "hd95" in requested:
        result["hd95_mm"] = hd95_v5(distances)
    if "nsd" in requested and "nsd" not in blocked:
        result["nsd"] = nsd_v5(distances, float(policy["surface_tolerance_mm"]))
    return result
