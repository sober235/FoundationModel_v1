"""Findings scored against the released primary boxes with fixed denominators (design 2026-11-10 §5.5)."""
from __future__ import annotations

import numpy as np
import pytest

from anatobind.v41.evaluation.findings_v5 import (
    anatomy_dice_v5, render_findings_eval_md_v5, score_findings_v5, summarize_findings_v5)

CLASSES = ["cartilage_lesion", "meniscal_tear"]
THRESHOLDS = [0.1, 0.25, 0.5]


def _box(status, analysis, cls, group, start, size):
    return {"status": status, "analysis": analysis, "abnormality_class": cls, "box": {"start": start, "size": size},
            "host": None if group is None else {"tissue_group": group}}


def _finding(uid, cls, start, size, anchor, voxels, host):
    return {"finding_uid": uid, "abnormality_class": cls, "box": {"start": start, "size": size}, "anchor": anchor,
            "voxels": voxels, "host_group": host, "host_status": "bound" if host else "undetermined"}


def _reference(uid, boxes):
    return {"sample_uid": uid, "policy_sha256": "p" * 64, "boxes": boxes}


R0 = _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [0, 0, 0], [4, 4, 2])
REFERENCE = _reference("s-0", [
    R0,
    _box("included", "primary", "meniscal_tear", "meniscus", [10, 10, 0], [2, 2, 2]),                 # missed
    _box("included", "secondary", "cartilage_lesion", "patellar_cartilage", [20, 0, 0], [2, 2, 2])])   # ignored
FINDINGS = [
    _finding("inst-0001", "cartilage_lesion", [1, 1, 0], [2, 2, 1], [1, 1, 0], 4, "femoral_cartilage"),
    _finding("inst-0002", "cartilage_lesion", [2, 2, 0], [6, 6, 1], [3, 3, 0], 9, "tibial_cartilage"),
    _finding("inst-0003", "cartilage_lesion", [20, 0, 0], [1, 1, 1], [20, 0, 0], 1, None),
    _finding("inst-0004", "meniscal_tear", [30, 30, 0], [1, 1, 1], [30, 30, 0], 1, "meniscus")]


def _row(findings=FINDINGS, reference=REFERENCE, patient="synthetic:p0"):
    return score_findings_v5(findings, reference, patient, CLASSES, THRESHOLDS)


def test_one_scan_is_scored_against_the_doctors_primary_boxes():
    row = _row()
    hit, missed = row["boxes"]
    assert (hit["hit"], hit["host_correct"]) == (True, False)          # the larger candidate names the wrong tissue
    assert hit["iobb"] == pytest.approx(1.0)                           # inst-0001 lies wholly inside the box
    assert (missed["hit"], missed["host_correct"], missed["iobb"]) == (False, False, None)
    assert row["findings"] == {"total": 4, "in_reference": 2, "in_ignored": 1, "outside": 1}
    assert row["tissue_positive"] == [
        {"abnormality_class": "cartilage_lesion", "tissue_group": "femoral_cartilage", "predicted": True},
        {"abnormality_class": "meniscal_tear", "tissue_group": "meniscus", "predicted": True}]
    assert row["patient_uid"] == "synthetic:p0"


def test_a_finding_of_another_class_inside_a_box_is_not_a_hit():
    other = [_finding("inst-0001", "meniscal_tear", [1, 1, 0], [1, 1, 1], [1, 1, 0], 1, "meniscus")]
    row = _row(other, _reference("s-0", [R0]))
    assert row["boxes"][0]["hit"] is False and row["findings"]["outside"] == 1


def test_summary_keeps_missed_boxes_in_every_denominator_and_weights_patients_equally():
    summary = summarize_findings_v5([_row(), _row([], _reference("s-1", [R0]), "synthetic:p1"),
                                     _row([], _reference("s-2", []), "synthetic:p2")])
    total = summary["overall"]
    assert (total["n_patients"], total["n_patients_with_boxes"], total["n_boxes"]) == (3, 2, 3)
    assert total["box_sensitivity"] == {"patient_mean": pytest.approx(0.25), "pooled": pytest.approx(1 / 3)}
    assert total["host_accuracy"] == {"patient_mean": 0.0, "pooled": 0.0}
    assert total["iobb_at"] == {"0.1": pytest.approx(1 / 3), "0.25": pytest.approx(1 / 3), "0.5": pytest.approx(1 / 3)}
    assert total["tissue_positive_recall"] == pytest.approx(2 / 3)
    assert total["findings_outside_reference_per_scan"] == pytest.approx(1 / 3)
    assert summary["by_class"]["meniscal_tear"]["box_sensitivity"]["pooled"] == 0.0
    assert summary["by_class"]["meniscal_tear"]["n_boxes"] == 1
    assert summary["scientific_conclusion"] == "NOT_EVALUATED"


def test_two_scans_of_one_patient_count_as_one_patient():
    summary = summarize_findings_v5([_row(), _row([], _reference("s-1", [R0]), "synthetic:p0")])
    total = summary["overall"]
    assert (total["n_patients"], total["n_scans"], total["n_boxes"]) == (1, 2, 3)
    assert total["box_sensitivity"]["patient_mean"] == pytest.approx(1 / 3)     # 1 of the patient's 3 boxes


def test_no_reference_boxes_gives_none_not_zero_or_one():
    summary = summarize_findings_v5([_row([], _reference("s", []), "synthetic:p")])
    assert summary["overall"]["box_sensitivity"] == {"patient_mean": None, "pooled": None}
    assert summary["overall"]["tissue_positive_recall"] is None
    assert summary["overall"]["findings_outside_reference_per_scan"] == 0.0


def test_report_has_counts_and_metrics_only_and_does_not_call_extra_findings_false_positives():
    text = render_findings_eval_md_v5(summarize_findings_v5([_row()]))
    assert "描述性计数，不是误报" in text and "NOT_EVALUATED" in text and "cartilage_lesion" in text
    assert "synthetic:p0" not in text and "inst-0001" not in text and "s-0" not in text


def test_anatomy_dice_matches_a_hand_value_and_is_none_when_both_are_empty():
    truth = np.zeros((4, 4, 1), dtype=np.int64)
    truth[0, :, 0] = 1                                                 # tissue_a on 4 voxels
    pred = np.zeros((4, 4, 1), dtype=np.int64)
    pred[0, 0:2, 0] = 1
    pred[1, 0:2, 0] = 1                                                # 2 of them plus 2 others
    eligible = np.ones((4, 4, 1), dtype=bool)
    assert anatomy_dice_v5(pred, truth, eligible, ["tissue_a", "tissue_b"]) == {"tissue_a": 0.5, "tissue_b": None}
    assert anatomy_dice_v5(pred, truth, ~eligible, ["tissue_a", "tissue_b"]) == {"tissue_a": None, "tissue_b": None}


# ---------------------------------------------------------------------------
# Where a finding lies: four exclusive places with a fixed precedence (2026-11-11 correction of "outside")

from anatobind.v41.evaluation.findings_v5 import finding_location_v5  # noqa: E402

P_CARTILAGE = _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [0, 0, 0], [4, 4, 2])
P_MENISCUS = _box("included", "primary", "meniscal_tear", "meniscus", [10, 10, 0], [2, 2, 2])
IGNORED = _box("included", "secondary", "cartilage_lesion", "patellar_cartilage", [20, 0, 0], [2, 2, 2])
IGNORED_ON_MENISCUS = _box("excluded", "primary", "meniscal_tear", "meniscus", [11, 11, 0], [2, 2, 2])
PLACES = _reference("s-0", [P_CARTILAGE, P_MENISCUS, IGNORED, IGNORED_ON_MENISCUS])


def _at(cls, anchor):
    return _finding("inst-0001", cls, anchor, [1, 1, 1], anchor, 1, None)


@pytest.mark.parametrize("cls, anchor, location, other_primary, ignored", [
    ("cartilage_lesion", [1, 1, 0], "same_class_primary", False, False),
    ("meniscal_tear", [1, 1, 0], "other_class_primary", True, False),       # the original count calls this "outside"
    ("cartilage_lesion", [20, 0, 0], "ignored", False, True),
    ("cartilage_lesion", [11, 11, 0], "ignored", True, True),               # ignore region before another class's box
    ("meniscal_tear", [11, 11, 0], "same_class_primary", False, True),      # its own class's box before everything
    ("meniscal_tear", [30, 30, 0], "outside_all", False, False),
])
def test_a_finding_lies_in_exactly_one_of_four_places(cls, anchor, location, other_primary, ignored):
    assert finding_location_v5(_at(cls, anchor), PLACES) == {
        "location": location, "in_other_class_primary": other_primary, "in_ignored": ignored}


def test_the_four_places_split_the_original_three_counts_without_changing_them():
    findings = [_at(cls, anchor) for cls, anchor in (
        ("cartilage_lesion", [1, 1, 0]), ("meniscal_tear", [1, 1, 0]), ("cartilage_lesion", [20, 0, 0]),
        ("cartilage_lesion", [11, 11, 0]), ("meniscal_tear", [11, 11, 0]), ("meniscal_tear", [30, 30, 0]))]
    row = score_findings_v5(findings, PLACES, "synthetic:p0", CLASSES, THRESHOLDS)
    assert row["findings"] == {"total": 6, "in_reference": 2, "in_ignored": 2, "outside": 2}      # unchanged
    places = [finding_location_v5(finding, PLACES)["location"] for finding in findings]
    assert places.count("same_class_primary") == 2 and places.count("ignored") == 2
    assert places.count("other_class_primary") + places.count("outside_all") == 2                # = "outside"


def test_the_report_no_longer_says_that_the_rest_lies_outside_every_box():
    text = render_findings_eval_md_v5(summarize_findings_v5([_row()]))
    assert "在所有框外" not in text and "医生框外的所见" not in text
    assert "其余 1（含定位点落在异类主分析框内的所见）" in text
