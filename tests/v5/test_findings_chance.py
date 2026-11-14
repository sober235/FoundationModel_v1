"""Chance level of the box hit rate: the findings of one scan scored against the primary boxes of another patient's
scan (proposal of 2026-11-13; hypothesis H3 of the diagnostics of 2026-11-12)."""
from __future__ import annotations

import numpy as np
import pytest

from anatobind.v41.evaluation.findings_chance_v5 import (
    chance_hit_control_v5, relocation_rows_v5, render_chance_md_v5, summarize_relocation_v5)
from anatobind.v41.evaluation.findings_v5 import score_findings_v5, summarize_findings_v5

CLASSES = ["cartilage_lesion", "meniscal_tear"]


def _box(cls, group, start, size, analysis="primary"):
    return {"status": "included", "analysis": analysis, "abnormality_class": cls, "box": {"start": start, "size": size},
            "host": {"tissue_group": group}}


def _finding(uid, cls, anchor, voxels, host):
    return {"finding_uid": uid, "abnormality_class": cls, "box": {"start": anchor, "size": [1, 1, 1]},
            "anchor": anchor, "voxels": voxels, "host_group": host}


def _scan(patient, findings, boxes):
    return {"patient_uid": patient, "findings": findings, "reference": {"boxes": boxes}}


def _specific():
    """Three patients: every scan's finding sits in its own box and in nobody else's."""
    return [_scan(f"p{i}", [_finding("inst-0001", "cartilage_lesion", [10 * i + 1, 1, 1], 60, "femoral_cartilage")],
                  [_box("cartilage_lesion", "femoral_cartilage", [10 * i, 0, 0], [4, 4, 4])]) for i in range(3)]


def _everywhere():
    """Three patients: the same findings on every scan, wherever its box is (an always-on response)."""
    findings = [_finding(f"inst-{i:04d}", "cartilage_lesion", [10 * i + 1, 1, 1], 60 + i, "femoral_cartilage")
                for i in range(3)]
    return [_scan(f"p{i}", list(findings), [_box("cartilage_lesion", "femoral_cartilage", [10 * i, 0, 0], [4, 4, 4])])
            for i in range(3)]


def test_the_observed_counts_are_those_of_the_official_scoring():
    scans = _specific() + [_scan("p9", [], [_box("meniscal_tear", "meniscus", [50, 0, 0], [2, 2, 2])])]
    control = chance_hit_control_v5(scans, CLASSES, n_permutations=20, seed=7)
    official = summarize_findings_v5([score_findings_v5(s["findings"], s["reference"], s["patient_uid"], CLASSES, [0.1])
                                      for s in scans])
    overall = control["overall"]
    assert overall["n_boxes"] == official["overall"]["n_boxes"] == 4
    assert overall["hit"]["observed"] == 3 and overall["host_correct"]["observed"] == 3
    assert overall["hit"]["observed_rate"] == pytest.approx(official["overall"]["box_sensitivity"]["pooled"])
    assert control["by_class"]["meniscal_tear"]["hit"]["observed"] == 0
    assert control["by_class"]["cartilage_lesion"]["n_boxes"] == 3


def test_findings_tied_to_their_own_scan_are_far_above_chance():
    control = chance_hit_control_v5(_specific(), CLASSES, n_permutations=50, seed=7)
    hit = control["overall"]["hit"]
    assert hit["observed_rate"] == 1.0
    assert hit["chance_rate"] == 0.0                    # no other patient's findings reach a box
    assert hit["permutation"]["max"] == 0 and hit["permutation"]["n_at_least_observed"] == 0
    assert hit["permutation"]["p_at_least_observed"] == pytest.approx(1 / 51)
    assert control["n_cross_patient_pairs"] == 6 and control["n_scans"] == 3 and control["n_patients"] == 3


def test_an_always_on_response_is_at_chance():
    control = chance_hit_control_v5(_everywhere(), CLASSES, n_permutations=50, seed=7)
    hit = control["overall"]["hit"]
    assert hit["observed_rate"] == 1.0 and hit["chance_rate"] == 1.0
    assert hit["permutation"]["n_at_least_observed"] == 50 and hit["permutation"]["p_at_least_observed"] == 1.0
    assert control["overall"]["host_correct"]["chance_rate"] == 1.0


def test_the_host_of_the_largest_candidate_decides_host_correct_also_across_scans():
    scans = _everywhere()
    scans[0]["findings"] = [dict(f, host_group="tibial_cartilage") for f in scans[0]["findings"]]
    control = chance_hit_control_v5(scans, CLASSES, n_permutations=10, seed=7)
    host = control["overall"]["host_correct"]
    assert host["observed"] == 2                        # scan 0 names the wrong tissue on its own box
    assert host["chance_rate"] == pytest.approx(4 / 6)  # its findings also name it on the two other boxes


def test_scans_of_one_patient_are_never_paired_and_the_ignore_regions_are_no_boxes():
    scans = _everywhere()
    scans[1]["patient_uid"] = "p0"                      # two scans of one patient
    scans[2]["reference"]["boxes"].append(_box("cartilage_lesion", "femoral_cartilage", [0, 0, 0], [9, 9, 9],
                                               analysis="secondary"))
    control = chance_hit_control_v5(scans, CLASSES, n_permutations=10, seed=7)
    assert control["n_patients"] == 2 and control["n_cross_patient_pairs"] == 4
    assert control["overall"]["n_boxes"] == 3


def test_the_control_is_reproducible_and_says_when_it_cannot_be_computed():
    first = chance_hit_control_v5(_specific(), CLASSES, n_permutations=30, seed=11)
    assert first == chance_hit_control_v5(_specific(), CLASSES, n_permutations=30, seed=11)
    assert (first["n_permutations"], first["seed"]) == (30, 11)
    alone = chance_hit_control_v5(_specific()[:1], CLASSES, n_permutations=30, seed=11)
    assert alone["overall"]["hit"]["observed"] == 1
    assert alone["overall"]["hit"]["chance_rate"] is None and alone["overall"]["hit"]["permutation"] is None
    empty = chance_hit_control_v5([], CLASSES, n_permutations=30, seed=11)
    assert empty["overall"]["n_boxes"] == 0 and empty["overall"]["hit"]["observed_rate"] is None


# ---------------------------------------------------------------------------
# Relocation on the host tissue, within the scan

GROUP_IDS = {"femoral_cartilage": [1], "meniscus": [2, 3]}


def _slab(shape=(20, 12, 4)):
    """Femoral cartilage in the low half of axis 0, meniscus (two classes of one group) in the high half."""
    anatomy = np.zeros(shape, dtype=np.int64)
    anatomy[2:9, 2:10, 1:3] = 1
    anatomy[11:15, 2:10, 1:3] = 2
    anatomy[15:18, 2:10, 1:3] = 3
    return anatomy


def _brute(findings, box, anatomy, group_ids):
    """Every placement tried one by one with the official scoring."""
    shape, size = anatomy.shape, box["box"]["size"]
    tissue = np.isin(anatomy, group_ids[box["host"]["tissue_group"]])

    def count(start):
        return int(tissue[tuple(slice(a, a + n) for a, n in zip(start, size))].sum())

    own = count(box["box"]["start"])
    octave = int(np.floor(np.log2(own)))
    total = hit = host = 0
    for start in np.ndindex(*[n - m + 1 for n, m in zip(shape, size)]):
        c = count(start)
        if c == 0 or int(np.floor(np.log2(c))) != octave:
            continue
        moved = dict(box, box={"start": list(start), "size": size})
        row = score_findings_v5(findings, {"boxes": [moved]}, "p", CLASSES, [0.1])["boxes"][0]
        total, hit, host = total + 1, hit + row["hit"], host + row["host_correct"]
    return total, hit / total, host / total


def test_the_relocation_probabilities_equal_a_placement_by_placement_count():
    anatomy = _slab()
    rng = np.random.default_rng(3)
    findings = []
    for i in range(12):
        cls = CLASSES[i % 2]
        anchor = [int(rng.integers(0, n)) for n in anatomy.shape]
        host = [None, "femoral_cartilage", "meniscus"][int(rng.integers(0, 3))]
        findings.append(_finding(f"inst-{i:04d}", cls, anchor, int(rng.integers(50, 60)), host))
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [3, 3, 0], [4, 3, 2]),
             _box("meniscal_tear", "meniscus", [12, 1, 1], [5, 4, 3]),
             _box("cartilage_lesion", "femoral_cartilage", [7, 8, 2], [3, 4, 2])]
    rows = relocation_rows_v5(findings=findings, reference={"boxes": boxes}, anatomy_reference=anatomy,
                              group_ids=GROUP_IDS)
    official = score_findings_v5(findings, {"boxes": boxes}, "p", CLASSES, [0.1])["boxes"]
    for row, box, scored in zip(rows, boxes, official):
        total, p_hit, p_host = _brute(findings, box, anatomy, GROUP_IDS)
        assert row["status"] == "ok" and row["comparable_placements"] == total
        assert row["p_hit"] == pytest.approx(p_hit) and row["p_host_correct"] == pytest.approx(p_host)
        assert (row["hit"], row["host_correct"]) == (scored["hit"], scored["host_correct"])


def test_a_box_reaching_out_of_the_volume_is_cut_and_a_box_off_its_tissue_is_left_out():
    anatomy = _slab()
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [-2, 2, 1], [6, 3, 2]),       # cut to [0, 4)
             _box("meniscal_tear", "meniscus", [0, 0, 0], [2, 2, 1]),                      # no meniscus voxel inside
             _box("cartilage_lesion", "femoral_cartilage", [3, 3, 1], [2, 2, 2], analysis="secondary")]
    findings = [_finding("inst-0001", "cartilage_lesion", [3, 3, 1], 60, "femoral_cartilage")]
    rows = relocation_rows_v5(findings=findings, reference={"boxes": boxes}, anatomy_reference=anatomy,
                              group_ids=GROUP_IDS)
    assert len(rows) == 2                                           # the ignore region is no box
    assert rows[0]["status"] == "ok" and rows[0]["hit"] is True and rows[0]["host_voxels"] == 2 * 3 * 2
    assert rows[1]["status"] == "no_host_tissue" and rows[1]["p_hit"] is None
    summary = summarize_relocation_v5(rows, CLASSES)
    assert (summary["overall"]["n_boxes"], summary["overall"]["n_usable"], summary["overall"]["excluded"]) == (2, 1, 1)
    assert summary["by_class"]["meniscal_tear"]["hit"]["chance_rate"] is None


def test_findings_at_the_lesions_beat_the_tissue_and_findings_all_over_the_tissue_do_not():
    anatomy = np.zeros((60, 12, 4), dtype=np.int64)
    anatomy[:, 2:10, 1:3] = 1
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [10 * i + 2, 4, 1], [3, 3, 2]) for i in range(6)]
    at_lesions = [_finding(f"inst-{i:04d}", "cartilage_lesion", [10 * i + 3, 5, 1], 60, "femoral_cartilage")
                  for i in range(6)]
    specific = summarize_relocation_v5(relocation_rows_v5(
        findings=at_lesions, reference={"boxes": boxes}, anatomy_reference=anatomy, group_ids=GROUP_IDS), CLASSES)
    hit = specific["overall"]["hit"]
    assert hit["observed_rate"] == 1.0 and hit["chance_rate"] < 0.35 and hit["p_at_least_observed"] < 0.01
    everywhere = [_finding(f"inst-{i:04d}", "cartilage_lesion", [x, y, 1], 60, "femoral_cartilage")
                  for i, (x, y) in enumerate((x, y) for x in range(0, 60, 2) for y in range(2, 10, 2))]
    always_on = summarize_relocation_v5(relocation_rows_v5(
        findings=everywhere, reference={"boxes": boxes}, anatomy_reference=anatomy, group_ids=GROUP_IDS), CLASSES)
    hit = always_on["overall"]["hit"]
    assert hit["observed_rate"] == 1.0 and hit["chance_rate"] > 0.95 and hit["p_at_least_observed"] > 0.7


def test_under_the_null_the_p_value_is_not_small_more_often_than_it_says():
    """Boxes drawn uniformly over the placements that hold tissue (hence uniform within each octave class), with
    fixed findings: P(p <= a) must not exceed a (up to the noise of 400 simulated splits)."""
    anatomy = _slab((24, 14, 4))
    rng = np.random.default_rng(5)
    tissue = np.argwhere(anatomy == 1)
    findings = [_finding(f"inst-{i:04d}", "cartilage_lesion", [int(v) for v in tissue[rng.integers(len(tissue))]], 60,
                         "femoral_cartilage") for i in range(3)]
    size = [3, 3, 2]
    starts = [start for start in np.ndindex(*[n - m + 1 for n, m in zip(anatomy.shape, size)])
              if (anatomy[tuple(slice(a, a + n) for a, n in zip(start, size))] == 1).any()]
    p_values, observed, expected = [], 0, 0.0
    for _ in range(400):
        boxes = [_box("cartilage_lesion", "femoral_cartilage", list(starts[rng.integers(len(starts))]), size)
                 for _ in range(8)]
        rows = relocation_rows_v5(findings=findings, reference={"boxes": boxes}, anatomy_reference=anatomy,
                                  group_ids=GROUP_IDS)
        hit = summarize_relocation_v5(rows, CLASSES)["overall"]["hit"]
        p_values.append(hit["p_at_least_observed"])
        observed, expected = observed + hit["observed"], expected + hit["chance_rate"] * 8
    p_values = np.asarray(p_values)
    for alpha in (0.01, 0.05, 0.1, 0.25):
        assert (p_values <= alpha).mean() <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / 400)
    assert observed == pytest.approx(expected, rel=0.1)


def test_the_report_carries_counts_and_no_identifier():
    anatomy = _slab()
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [3, 3, 0], [4, 3, 2])]
    findings = [_finding("inst-0001", "cartilage_lesion", [4, 4, 1], 60, "femoral_cartilage")]
    relocation = summarize_relocation_v5(relocation_rows_v5(
        findings=findings, reference={"boxes": boxes}, anatomy_reference=anatomy, group_ids=GROUP_IDS), CLASSES)
    text = render_chance_md_v5({
        "train": {"relocation": relocation,
                  "cross_patient": chance_hit_control_v5(_specific(), CLASSES, n_permutations=30, seed=11)},
        "development_val": {"relocation": relocation,
                            "cross_patient": chance_hit_control_v5(_everywhere(), CLASSES, n_permutations=30,
                                                                   seed=11)}})
    assert "inst-" not in text and "p0" not in text
    for phrase in ("偶然命中", "train", "development_val", "cartilage_lesion", "不是误报率", "30", "仅探索性",
                   "沿所在组织挪动框", "泊松二项"):
        assert phrase in text
