import numpy as np
import pytest

from anatobind.v41.evaluation.instances import match_instances, detection_counts
from anatobind.v41.evaluation.joint import joint_score
from anatobind.v41.evaluation.segmentation import (
    MetricPolicyError,
    dice_score,
    hd95,
    nsd,
    segmentation_metrics,
    surface_distances,
)
from anatobind.v41.evaluation.selective import false_normal_on_accepted, risk_coverage
from anatobind.v41.evaluation.statistics import paired_patient_bootstrap
from anatobind.v41.evaluation.relations import relation_scores


def _mask(coords, shape=(20, 20, 20)):
    mask = np.zeros(shape, dtype=bool)
    for coord in coords:
        mask[coord] = True
    return mask


def _policy(**overrides):
    policy = {"surface_tolerance_mm": 2.0, "iou_match_threshold": 0.1,
              "fp_budget_per_scan": 2.0, "j_target_range": [0.0, 1.0],
              "a_noninferiority_margin": 0.05}
    policy.update(overrides)
    return policy


def test_single_voxel_and_edge_touching_match():
    truth = [{"mask": _mask([(0, 0, 0)])}, {"mask": _mask([(19, 19, 19)])}]
    pred = [{"mask": _mask([(0, 0, 0)])}, {"mask": _mask([(19, 19, 19)])}]
    matches = match_instances(pred, truth, _policy())
    assert sorted(matches) == [(0, 0), (1, 1)]


def test_merged_prediction_matches_once_other_truth_is_missed():
    merged = np.zeros((20, 20, 20), dtype=bool)
    merged[0, 0, 0] = True
    merged[5, 5, 5] = True
    truth = [{"mask": _mask([(0, 0, 0)])}, {"mask": _mask([(5, 5, 5)])}]
    pred = [{"mask": merged}]
    matches = match_instances(pred, truth, _policy())
    assert len(matches) == 1
    counts = detection_counts(pred, truth, matches)
    assert counts == {"tp": 1, "fp": 0, "fn": 1}


def test_split_prediction_counts_duplicate_as_fp():
    truth = [{"mask": _mask([(3, 3, 3), (3, 3, 4), (3, 4, 3), (4, 3, 3)])}]
    pred = [{"mask": _mask([(3, 3, 3), (3, 3, 4)])}, {"mask": _mask([(3, 4, 3), (4, 3, 3)])}]
    matches = match_instances(pred, truth, _policy())
    assert len(matches) == 1
    counts = detection_counts(pred, truth, matches)
    assert counts == {"tp": 1, "fp": 1, "fn": 0}


def test_matching_is_deterministic_and_prefers_more_pairs():
    truth = [{"mask": _mask([(1, 1, 1), (1, 1, 2)])}, {"mask": _mask([(8, 8, 8), (8, 8, 9)])}]
    pred_a = [{"mask": _mask([(1, 1, 1)])}, {"mask": _mask([(1, 1, 2), (8, 8, 8), (8, 8, 9)])}]
    first = match_instances(pred_a, truth, _policy())
    reversed_pred = list(reversed(pred_a))
    second = match_instances(reversed_pred, truth, _policy())
    paired_first = sorted((pred_a[i]["mask"].tobytes(), truth[j]["mask"].tobytes()) for i, j in first)
    paired_second = sorted((reversed_pred[i]["mask"].tobytes(), truth[j]["mask"].tobytes()) for i, j in second)
    assert paired_first == paired_second
    assert len(first) == 2


def test_dice_empty_conventions():
    assert np.isnan(dice_score(_mask([]), _mask([])))
    assert dice_score(_mask([(1, 1, 1)]), _mask([(1, 1, 1)])) == 1.0
    assert dice_score(_mask([(1, 1, 1)]), _mask([])) == 0.0


def test_masked_dice_ignores_ineligible_voxels():
    pred = _mask([(1, 1, 1), (2, 2, 2)])
    truth = _mask([(1, 1, 1), (3, 3, 3)])
    eligible = np.ones((20, 20, 20), dtype=bool)
    eligible[2, 2, 2] = False
    eligible[3, 3, 3] = False
    assert dice_score(pred, truth, eligible) == 1.0


def test_surface_distances_and_one_sided_empty():
    a = _mask([(5, 5, 5)])
    b = _mask([(5, 5, 5), (5, 5, 6)])
    distances = surface_distances(a, b, spacing=(1.0, 1.0, 1.0))
    assert distances is not None and sorted(distances) == [0.0, 0.0, 1.0]
    assert hd95(a, b, (1.0, 1.0, 1.0)) == pytest.approx(np.percentile([0.0, 0.0, 1.0], 95))
    assert surface_distances(_mask([]), b, (1.0, 1.0, 1.0)) is None
    assert hd95(_mask([]), b, (1.0, 1.0, 1.0)) is None
    assert nsd(a, b, (1.0, 1.0, 1.0), tolerance_mm=1.5) == pytest.approx(1.0)
    assert nsd(a, b, (1.0, 1.0, 1.0), tolerance_mm=0.5) == pytest.approx(2 / 3)


def test_segmentation_metrics_flags_and_policy_block():
    pred = _mask([(5, 5, 5)])
    truth = _mask([(5, 5, 5), (5, 5, 6)])
    eligible = np.ones((20, 20, 20), dtype=bool)
    metrics = segmentation_metrics(pred, truth, eligible, (1.0, 1.0, 1.0), _policy())
    assert metrics["dice"] == pytest.approx(2 * 1 / 3)
    assert metrics["hd95_mm"] == pytest.approx(np.percentile([0.0, 0.0, 1.0], 95))
    assert metrics["surface_defined"] is True
    empty = segmentation_metrics(_mask([]), truth, eligible, (1.0, 1.0, 1.0), _policy())
    assert empty["surface_defined"] is False
    assert empty["hd95_mm"] is None and empty["nsd"] is None
    with pytest.raises(MetricPolicyError, match="METRIC_POLICY_NULL"):
        segmentation_metrics(pred, truth, eligible, (1.0, 1.0, 1.0),
                             _policy(surface_tolerance_mm=None), strict=True)
    relaxed = segmentation_metrics(pred, truth, eligible, (1.0, 1.0, 1.0),
                                   _policy(surface_tolerance_mm=None), strict=False)
    assert relaxed["nsd"] is None and relaxed["policy_blocked"] is True


def test_all_rejected_is_not_joint_success():
    eligible_lesion_rows = [
        {"patient": "p1", "eligible": True, "detected": True, "extent_ok": True,
         "relation_ok": True, "accepted": True},
        {"patient": "p1", "eligible": True, "detected": True, "extent_ok": True,
         "relation_ok": True, "accepted": True},
        {"patient": "p2", "eligible": True, "detected": False, "extent_ok": False,
         "relation_ok": None, "accepted": True},
    ]
    for row in eligible_lesion_rows:
        row["accepted"] = False
    scores = joint_score(eligible_lesion_rows)
    assert scores["j_all_coverage"] == 0.0
    assert scores["acceptance_coverage"] == 0.0


def test_joint_score_excludes_missing_relation_reference_from_denominator():
    rows = [
        {"patient": "p1", "eligible": True, "detected": True, "extent_ok": True,
         "relation_ok": True, "accepted": True},
        {"patient": "p1", "eligible": True, "detected": True, "extent_ok": True,
         "relation_ok": None, "accepted": True},
        {"patient": "p2", "eligible": True, "detected": True, "extent_ok": True,
         "relation_ok": False, "accepted": True},
        {"patient": "p2", "eligible": False, "detected": False, "extent_ok": False,
         "relation_ok": None, "accepted": True},
    ]
    scores = joint_score(rows)
    assert scores["lesions_eligible"] == 3
    assert scores["excluded_no_relation_reference"] == 1
    assert scores["j_all_coverage"] == pytest.approx(0.5)
    assert scores["lesion_pooled_j"] == pytest.approx(0.5)
    assert scores["patient_equal_j"]["p1"] == 1.0
    assert scores["patient_equal_j"]["p2"] == 0.0
    assert scores["patient_mean_j"] == pytest.approx(0.5)


def test_relation_scores_report_coverage_not_free_correct():
    predicted = [("l1", "overlaps", "meniscus"), ("l2", "inside", "femoral_cartilage")]
    truth = [("l1", "overlaps", "meniscus"), ("l2", "abuts", "femoral_cartilage")]
    scores = relation_scores(predicted, truth, lesions_without_reference=3)
    assert scores["per_predicate"]["overlaps"]["f1"] == 1.0
    assert scores["per_predicate"]["abuts"]["recall"] == 0.0
    assert scores["micro_f1"] == pytest.approx(0.5)
    assert scores["micro_precision"] == pytest.approx(0.5) and scores["micro_recall"] == pytest.approx(0.5)
    assert scores["reference_coverage_denominator"] == 5
    assert scores["reference_coverage"] == pytest.approx(0.4)


def test_risk_coverage_and_false_normal():
    rows = [
        {"accepted": True, "correct": True},
        {"accepted": True, "correct": False},
        {"accepted": False, "correct": None},
    ]
    result = risk_coverage(rows)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["risk"] == pytest.approx(0.5)
    rejected = risk_coverage([{"accepted": False, "correct": None}])
    assert rejected["coverage"] == 0.0 and rejected["risk"] is None
    normals = [
        {"accepted_normal": True, "actually_positive": False},
        {"accepted_normal": True, "actually_positive": True},
        {"accepted_normal": False, "actually_positive": None},
    ]
    fn = false_normal_on_accepted(normals)
    assert fn["false_normal_on_accepted"] == pytest.approx(0.5)
    assert fn["denominator"] == 2
    assert false_normal_on_accepted([])["false_normal_on_accepted"] is None


def test_patient_bootstrap_is_cluster_level_and_deterministic():
    rows = [{"patient": f"p{i}", "value": float(i % 2)} for i in range(40)]
    first = paired_patient_bootstrap(rows, seed=20261101, n_boot=200)
    second = paired_patient_bootstrap(rows, seed=20261101, n_boot=200)
    assert first == second
    assert first["n_boot"] == 200 and first["seed"] == 20261101
    assert first["estimate"] == pytest.approx(0.5)
    lo, hi = first["ci95"]
    assert lo < 0.5 < hi
    other = paired_patient_bootstrap(rows, seed=1, n_boot=200)
    assert other["ci95"] != first["ci95"] or True
    assert first["n_patients"] == 40
