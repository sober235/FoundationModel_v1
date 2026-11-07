from __future__ import annotations

from anatobind.v41.evaluation.instances import lesion_detection_metrics_v5, summarize_lesion_detection_v5
from tests.v5.test_matching_cardinality import PRED, TRUTH

POLICY = {"iou_match_threshold": 0.1, "class_aware": True, "size_strata_voxels": [50, 150]}


def test_detection_counts_use_cardinality_first_matching():
    result = lesion_detection_metrics_v5(PRED, TRUTH, POLICY, voxel_volume_mm3=None)
    assert (result["tp"], result["fp"], result["fn"]) == (6, 0, 0)


def test_no_truth_sensitivity_none_fp_counted():
    result = lesion_detection_metrics_v5(PRED[:2], [], POLICY, voxel_volume_mm3=None)
    assert result["sensitivity"] is None and result["fp"] == 2


def test_volume_error_needs_physical_volume():
    without = lesion_detection_metrics_v5(PRED[:1], TRUTH[:1], POLICY, voxel_volume_mm3=None)
    assert without["matched_volume_error_mm3"] is None and without["matched_volume_error_voxels"] == [0]
    with_mm = lesion_detection_metrics_v5(PRED[:1], TRUTH[:1], POLICY, voxel_volume_mm3=0.078125)
    assert with_mm["matched_volume_error_mm3"] == [0.0]


def test_size_strata_follow_policy_boundaries():
    result = lesion_detection_metrics_v5([], TRUTH, POLICY, voxel_volume_mm3=None)
    assert result["size_strata"] == {"<50": {"tp": 0, "fn": 0}, "50-150": {"tp": 0, "fn": 6},
                                     ">=150": {"tp": 0, "fn": 0}}


def test_summary_reports_fp_per_scan():
    rows = [lesion_detection_metrics_v5(PRED[:1], TRUTH[:1], POLICY, None),
            lesion_detection_metrics_v5(PRED[1:3], [], POLICY, None)]
    summary = summarize_lesion_detection_v5(rows)
    assert summary["n_scans"] == 2 and summary["fp_per_scan"] == 1.0 and summary["pooled_sensitivity"] == 1.0
