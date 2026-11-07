from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from anatobind.v41._vendor.surface_distance import metrics as sd
from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.evaluation.policy import validate_metric_policy_v5
from anatobind.v41.evaluation.segmentation import segmentation_metrics_v5

# The reference calls below hit the pinned source's deprecated scipy.ndimage aliases directly.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

VERIFIED = {"physical_geometry_status": "verified"}
POLICY = {"metrics": ["dice", "hd95", "nsd"], "surface_tolerance_mm": 1.0}


def _cube(shape, lo, hi):
    mask = np.zeros(shape, bool)
    mask[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return mask


def test_unknown_geometry_does_not_block_dice_or_invent_mm():
    mask = _cube((8, 8, 8), (2, 2, 2), (6, 6, 6))
    m = segmentation_metrics_v5(mask, mask, np.ones_like(mask), {"spacing_mm": None, "physical_geometry_status": "unknown"}, POLICY)
    assert m["dice"] == 1.0 and m["hd95_mm"] is None and m["nsd"] is None
    assert "PHYSICAL_GEOMETRY_UNKNOWN" in m["reasons"]


@pytest.mark.parametrize("shape, lo, hi, spacing", [
    ((12, 12, 10), (2, 2, 2), (8, 8, 6), (0.3125, 0.3125, 0.8)),
    ((16, 16), (3, 3), (11, 9), (1.0, 0.5)),
])

def test_anisotropic_matches_reference(shape, lo, hi, spacing):
    truth = _cube(shape, lo, hi)
    pred = np.roll(truth, 1, axis=-1)
    m = segmentation_metrics_v5(pred, truth, np.ones_like(truth), {**VERIFIED, "spacing_mm": list(spacing)}, POLICY)
    d = sd.compute_surface_distances(truth, pred, spacing)
    assert math.isclose(m["hd95_mm"], sd.compute_robust_hausdorff(d, 95), rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(m["nsd"], sd.compute_surface_dice_at_tolerance(d, 1.0), rel_tol=1e-10, abs_tol=1e-10)


def test_one_sided_empty_is_failure_not_inf():
    truth = _cube((8, 8), (2, 2), (5, 5))
    m = segmentation_metrics_v5(np.zeros_like(truth), truth, np.ones_like(truth), {**VERIFIED, "spacing_mm": [1.0, 1.0]}, POLICY)
    assert m["hd95_mm"] is None and m["surface_failure"] is True and "ONE_SIDED_EMPTY" in m["reasons"]


def test_both_empty_dice_is_none():
    empty = np.zeros((8, 8), bool)
    m = segmentation_metrics_v5(empty, empty, np.ones_like(empty), {**VERIFIED, "spacing_mm": [1.0, 1.0]}, POLICY)
    assert m["dice"] is None and "BOTH_EMPTY" in m["reasons"]


def test_partial_review_touching_edge_blocks_surface_not_dice():
    truth = _cube((8, 8), (2, 4), (6, 7))
    eligible = np.ones_like(truth)
    eligible[:, 7] = False
    m = segmentation_metrics_v5(truth, truth, eligible, {**VERIFIED, "spacing_mm": [1.0, 1.0]}, POLICY)
    assert m["dice"] == 1.0 and m["hd95_mm"] is None and "SURFACE_REFERENCE_INCOMPLETE" in m["reasons"]


def test_policy_checks_only_requested_metrics():
    assert validate_metric_policy_v5({"metrics": ["dice"], "surface_tolerance_mm": None}, "locked")["blocked_metrics"] == []
    assert validate_metric_policy_v5({"metrics": ["nsd"], "surface_tolerance_mm": None}, "development")["blocked_metrics"] == ["nsd"]
    with pytest.raises(V5Error) as error:
        validate_metric_policy_v5({"metrics": ["dice", "nsd"], "surface_tolerance_mm": None}, "locked")
    assert error.value.code == "METRIC_POLICY_NULL"


def test_vendored_sources_match_recorded_hashes():
    root = Path("anatobind/v41/_vendor/surface_distance")
    source = json.loads((root / "SOURCE.json").read_text())
    assert source["commit"] == "1f805cea446805e559badc80dd27f34175e4173d"
    for name, digest in source["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
