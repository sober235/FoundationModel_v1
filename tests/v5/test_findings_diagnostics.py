"""Diagnostics of the findings chain, one primary box at a time: response in the box, components, the size filter,
the anchor, the host tissue (design 2026-11-10 §5.4–§5.5; diagnostics of 2026-11-11). Synthetic volumes."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from anatobind.v41.evaluation.findings_diagnostics_v5 import (
    diagnose_scan_v5, render_diagnostics_md_v5, summarize_diagnostics_v5)
from anatobind.v41.evaluation.findings_v5 import score_findings_v5
from anatobind.v41.findings_v5 import findings_from_output_v5, host_relation_fn_v5, load_protocols_v5
from anatobind.v41.perception import extract_instances_v5

CONFIG = Path("configs/v5")
KNEE = load_protocols_v5(CONFIG / "protocols_v5.yaml", CONFIG)["knee_qdess"]
SCOPE, GROUPS, GROUPS_BY_CLASS = KNEE["scope"], KNEE["groups"], KNEE["groups_by_class"]
PLAN = yaml.safe_load((CONFIG / "proposals" / "unified_knee_brain.yaml").read_text())
POLICY = copy.deepcopy(PLAN["members"]["knee_qdess"]["acceptance_policy"])
POLICY["instances"]["min_voxels"] = 3
FEMORAL = SCOPE.anatomy_classes.index("femoral_cartilage") + 1
MENISCUS = SCOPE.anatomy_classes.index("meniscus_medial") + 1
SHAPE = (12, 8, 4)
CARTILAGE, TEAR = 0, 1                      # SCOPE.abnormality_classes order


def _box(cls, group, start, size, analysis="primary"):
    return {"status": "included", "analysis": analysis, "abnormality_class": cls, "box": {"start": start, "size": size},
            "host": {"tissue_group": group}}


REFERENCE = {"sample_uid": "s-0", "policy_sha256": "p" * 64, "boxes": [
    _box("cartilage_lesion", "femoral_cartilage", [0, 0, 0], [3, 3, 2]),     # B1: hit, host lost by predicted anatomy
    _box("cartilage_lesion", "femoral_cartilage", [3, 4, 0], [2, 2, 2]),     # B2: response below min_voxels
    _box("meniscal_tear", "meniscus", [6, 0, 0], [2, 2, 1]),                 # B3: a kept component, anchor outside
    _box("meniscal_tear", "meniscus", [10, 6, 2], [2, 2, 2]),                # B4: no response at all
    _box("cartilage_lesion", "patellar_cartilage", [8, 6, 0], [2, 2, 1], analysis="secondary")]}   # ignore region


def _scan():
    reference_anatomy = np.zeros(SHAPE, dtype=np.int64)
    reference_anatomy[:6], reference_anatomy[6:] = FEMORAL, MENISCUS
    predicted_anatomy = reference_anatomy.astype(np.int16)
    predicted_anatomy[:3] = 0                                 # the model missed the cartilage under B1
    positive = np.zeros((2, *SHAPE), dtype=bool)
    positive[CARTILAGE, 0:3, 0:2, 0] = True                   # A: 6 voxels inside B1
    positive[CARTILAGE, 3:5, 4, 0] = True                     # B: 2 voxels inside B2 (under min_voxels 3)
    for voxel in ((6, 0, 0), (6, 1, 0), (7, 0, 0)):           # D: 3 cartilage voxels inside the meniscal box B3
        positive[(CARTILAGE, *voxel)] = True
    for voxel in ((8, 6, 0), (9, 6, 0), (8, 7, 0)):           # F: 3 voxels inside the ignore region
        positive[(CARTILAGE, *voxel)] = True
    positive[TEAR, 7, 1:6, 0] = True                          # C: 5 voxels, one inside B3, centroid (7, 3, 0) outside
    for voxel in ((10, 0, 3), (11, 0, 3), (11, 1, 3)):        # E: 3 voxels outside every box
        positive[(TEAR, *voxel)] = True
    probability = np.where(positive, 0.95, 0.2).astype(np.float32)
    probability[TEAR, 10, 6, 2] = 0.3                         # the strongest response inside B4
    accepted = np.ones((2, *SHAPE), dtype=bool)
    anatomy_accepted = np.ones(SHAPE, dtype=bool)
    instances = extract_instances_v5(positive, accepted, SCOPE, POLICY)
    relations, _ = host_relation_fn_v5(GROUPS, GROUPS_BY_CLASS)(instances, predicted_anatomy, anatomy_accepted, SCOPE)
    output = SimpleNamespace(output_kind="joint", abnormal_instances=instances, tissue_relations=relations)
    findings = findings_from_output_v5(output, SCOPE, POLICY, GROUPS)
    return {"findings": findings, "instances": instances, "reference": REFERENCE, "scope": SCOPE,
            "probability": probability, "positive": positive, "accepted": accepted,
            "anatomy_predicted": predicted_anatomy, "anatomy_accepted": anatomy_accepted,
            "anatomy_reference": reference_anatomy, "anatomy_eligible": np.ones(SHAPE, dtype=bool),
            "groups": GROUPS, "groups_by_class": GROUPS_BY_CLASS, "acceptance_policy": POLICY,
            "topk_fraction": 0.5, "patient_uid": "synthetic:p0"}


@pytest.fixture(scope="module")
def row():
    return diagnose_scan_v5(**_scan())


def test_each_primary_box_is_followed_from_the_response_to_the_host(row):
    b1, b2, b3, b4 = row["boxes"]                              # the ignore region is not a row
    assert (b1["abnormality_class"], b1["tissue_group"], b1["box_voxels"], b1["bag_voxels"]) == (
        "cartilage_lesion", "femoral_cartilage", 18, 18)
    assert (b1["positive_voxels_box"], b1["components_touching"], b1["largest_touching_voxels"],
            b1["kept_touching"]) == (6, 1, 6, 1)
    assert (b1["hit"], b1["host_correct"], b1["host_correct_reference_anatomy"]) == (True, False, True)
    assert b1["chosen"] == {"voxels": 6, "inside_fraction": 1.0, "host_group": None,
                            "host_group_reference_anatomy": "femoral_cartilage"}
    assert b1["stage"] == "hit_host_wrong" and b1["largest_kept_touching"] is None

    assert (b2["box_voxels"], b2["positive_voxels_box"], b2["components_touching"], b2["largest_touching_voxels"],
            b2["kept_touching"], b2["hit"], b2["stage"]) == (8, 2, 1, 2, 0, False, "below_min_voxels")

    assert (b3["box_voxels"], b3["bag_voxels"], b3["positive_voxels_box"], b3["components_touching"],
            b3["kept_touching"], b3["hit"], b3["stage"]) == (4, 4, 1, 1, 1, False, "anchor_outside")
    assert b3["largest_kept_touching"] == {"voxels": 5, "inside_fraction": 0.2, "anchor_distance_voxels": 2.0}
    assert b3["chosen"] is None

    assert (b4["positive_voxels_box"], b4["components_touching"], b4["largest_touching_voxels"], b4["kept_touching"],
            b4["hit"], b4["stage"]) == (0, 0, 0, 0, False, "no_response")


def test_the_response_in_a_box_is_read_from_the_probabilities(row):
    b1, _, _, b4 = row["boxes"]
    assert b1["probability"]["max_box"] == pytest.approx(0.95) and b1["probability"]["max_bag"] == pytest.approx(0.95)
    # top half of the 18 bag voxels: the 6 marked ones (0.95) and 3 unmarked ones (0.2)
    assert b1["probability"]["topk_mean_bag"] == pytest.approx((6 * 0.95 + 3 * 0.2) / 9)
    assert b1["probability"]["positive_fraction_bag"] == pytest.approx(6 / 18)
    assert b4["probability"]["max_box"] == pytest.approx(0.3) and b4["probability"]["positive_fraction_bag"] == 0.0


def test_the_box_rows_agree_with_the_official_scores(row):
    scan = _scan()
    official = score_findings_v5(scan["findings"], REFERENCE, "synthetic:p0", list(SCOPE.abnormality_classes),
                                 [0.1, 0.25, 0.5])
    assert [box["hit"] for box in row["boxes"]] == [box["hit"] for box in official["boxes"]] == [
        True, False, False, False]
    assert [box["host_correct"] for box in row["boxes"]] == [box["host_correct"] for box in official["boxes"]]
    assert official["findings"] == {"total": 5, "in_reference": 1, "in_ignored": 1, "outside": 3}


def test_every_finding_gets_one_place_and_both_host_readings(row):
    found = {(f["abnormality_class"], f["voxels"]): f for f in row["findings"]}
    assert len(row["findings"]) == 5
    a = found[("cartilage_lesion", 6)]
    assert (a["location"], a["host_status"], a["host_group"]) == ("same_class_primary", "undetermined", None)
    assert (a["host_status_reference_anatomy"], a["host_group_reference_anatomy"]) == ("bound", "femoral_cartilage")
    assert (a["on_tissue_fraction_predicted"], a["on_tissue_fraction_reference"]) == (0.0, 1.0)
    assert a["probability_mean"] == pytest.approx(0.95) and a["probability_max"] == pytest.approx(0.95)
    places = sorted((f["abnormality_class"], f["voxels"], f["location"]) for f in row["findings"])
    assert places == [("cartilage_lesion", 3, "ignored"), ("cartilage_lesion", 3, "other_class_primary"),
                      ("cartilage_lesion", 6, "same_class_primary"), ("meniscal_tear", 3, "outside_all"),
                      ("meniscal_tear", 5, "outside_all")]
    tear = found[("meniscal_tear", 5)]
    assert (tear["host_group"], tear["host_group_reference_anatomy"]) == ("meniscus", "meniscus")
    # the cartilage finding on the meniscus has no allowed tissue under either anatomy
    on_meniscus = [f for f in row["findings"] if f["location"] == "other_class_primary"][0]
    assert (on_meniscus["host_status"], on_meniscus["host_status_reference_anatomy"]) == (
        "undetermined", "undetermined")


def test_components_are_counted_before_and_after_the_size_filter(row):
    assert row["components"]["cartilage_lesion"] == {
        "positive_voxels": 14, "count": 4, "kept": 3,
        "size_histogram": {"1": 0, "2-9": 4, "10-49": 0, "50-99": 0, "100-999": 0, "1000-9999": 0, ">=10000": 0}}
    assert (row["components"]["meniscal_tear"]["positive_voxels"], row["components"]["meniscal_tear"]["count"],
            row["components"]["meniscal_tear"]["kept"]) == (8, 2, 2)
    assert row["patient_uid"] == "synthetic:p0"


def test_a_summary_counts_the_failure_point_of_every_box(row):
    other = diagnose_scan_v5(**dict(_scan(), patient_uid="synthetic:p1"))
    summary = summarize_diagnostics_v5([row, other])
    assert summary["n_scans"] == 2
    overall = summary["funnel"]["overall"]
    assert {key: overall[key] for key in ("n_boxes", "responded", "kept_touching", "hit", "host_correct",
                                          "host_correct_reference_anatomy")} == {
        "n_boxes": 8, "responded": 6, "kept_touching": 4, "hit": 2, "host_correct": 0,
        "host_correct_reference_anatomy": 2}
    assert overall["stages"] == {"no_response": 2, "below_min_voxels": 2, "anchor_outside": 2, "hit_host_wrong": 2,
                                 "hit_host_correct": 0}
    assert summary["funnel"]["by_class"]["meniscal_tear"]["stages"]["anchor_outside"] == 2
    assert summary["locations"]["overall"] == {
        "total": 10, "same_class_primary": 2, "ignored": 2, "other_class_primary": 2, "outside_all": 4,
        "in_other_class_primary": 2, "in_ignored": 2}
    assert summary["locations"]["by_class"]["meniscal_tear"]["outside_all"] == 4
    assert summary["hosts"]["overall"]["status"] == {
        "bound_bound": 4, "bound_undetermined": 0, "undetermined_bound": 2, "undetermined_undetermined": 4}
    assert summary["instance_sizes"]["by_class"]["meniscal_tear"]["voxels"]["max"] == 5
    assert summary["supervision_sizes"]["overall"]["box_voxels"]["min"] == 4
    assert summary["supervision_sizes"]["overall"]["box_voxels"]["max"] == 18
    assert summary["response"]["overall"]["boxes_with_max_at_least"] == {"0.1": 8, "0.25": 8, "0.5": 6, "0.75": 6,
                                                                         "0.9": 6}
    assert summary["components"]["cartilage_lesion"]["size_histogram"]["2-9"] == 8


def test_the_report_carries_counts_and_no_identifier(row):
    text = render_diagnostics_md_v5({"train": summarize_diagnostics_v5([row])})
    assert "synthetic:p0" not in text and "s-0" not in text
    assert "| 主分析框 | 4 |" in text and "| 框内有达到阈值的响应 | 3 |" in text
    assert "| 定位点在同类主分析框内 | 1 |" in text and "| 定位点在异类主分析框内 | 1 |" in text
    assert "诊断" in text and "参照解剖" in text


# ---------------------------------------------------------------------------
# Pilot of 2026-11-12: positives on the host tissue by zone, and the components that lie off every host tissue

ZONES = ("same_class_primary", "ignored", "outside")


def test_positive_voxels_on_the_host_tissue_are_counted_by_zone(row):
    """Zones follow the supervision: a primary box of the class, else an ignore region, else the rest (the
    class's negative support, which holds the primary boxes of the other class). Host tissue: the reference anatomy
    of the groups the class may sit on."""
    volume = SHAPE[0] * SHAPE[1] * SHAPE[2]
    assert set(row["zones"]) == {"cartilage_lesion", "meniscal_tear"}
    assert row["zones"]["cartilage_lesion"] == {
        # B1 (18) and B2 (8), all femoral cartilage, with components A (6) and B (2)
        "same_class_primary": {"voxels": 26, "host_voxels": 26, "positive_voxels": 8, "positive_on_host": 8},
        # the ignore region lies on the meniscus: no cartilage there; component F (3) is inside it
        "ignored": {"voxels": 4, "host_voxels": 0, "positive_voxels": 3, "positive_on_host": 0},
        # the rest of the femoral cartilage has no positive; component D (3) lies on the meniscus, in a tear box
        "outside": {"voxels": volume - 30, "host_voxels": 192 - 26, "positive_voxels": 3, "positive_on_host": 0}}
    assert row["zones"]["meniscal_tear"] == {
        "same_class_primary": {"voxels": 12, "host_voxels": 12, "positive_voxels": 1, "positive_on_host": 1},
        "ignored": {"voxels": 4, "host_voxels": 4, "positive_voxels": 0, "positive_on_host": 0},
        "outside": {"voxels": volume - 16, "host_voxels": 192 - 16, "positive_voxels": 7, "positive_on_host": 7}}
    for name, zones in row["zones"].items():
        assert sum(zone["voxels"] for zone in zones.values()) == volume
        assert sum(zone["positive_voxels"] for zone in zones.values()) == row["components"][name]["positive_voxels"]


def test_a_voxel_in_a_primary_box_and_an_ignore_region_belongs_to_the_box():
    scan = _scan()
    overlapping = _box("meniscal_tear", "meniscus", [2, 0, 0], [2, 2, 1], analysis="secondary")    # 2 voxels in B1
    scan["reference"] = dict(REFERENCE, boxes=[*REFERENCE["boxes"], overlapping])
    zones = diagnose_scan_v5(**scan)["zones"]
    assert zones["cartilage_lesion"]["same_class_primary"] == {"voxels": 26, "host_voxels": 26, "positive_voxels": 8,
                                                               "positive_on_host": 8}
    assert (zones["cartilage_lesion"]["ignored"]["voxels"], zones["cartilage_lesion"]["ignored"]["host_voxels"]) == (6, 2)
    assert zones["meniscal_tear"]["ignored"]["voxels"] == 8    # for the other class the whole region is ignored


def test_host_tissue_is_only_counted_where_the_reference_anatomy_is_eligible():
    scan = _scan()
    scan["anatomy_eligible"] = scan["anatomy_eligible"].copy()
    scan["anatomy_eligible"][0:3, 0:3, 0:2] = False                 # the anatomy under B1 cannot be read
    zones = diagnose_scan_v5(**scan)["zones"]["cartilage_lesion"]
    assert zones["same_class_primary"] == {"voxels": 26, "host_voxels": 8, "positive_voxels": 8, "positive_on_host": 2}


def test_components_off_every_host_tissue_are_counted_with_their_sizes(row):
    assert row["off_tissue"]["cartilage_lesion"] == {          # D and F: three voxels each, on the meniscus
        "count": 2, "kept": 2, "voxels": 6, "kept_voxels": 6,
        "size_histogram": {"1": 0, "2-9": 2, "10-49": 0, "50-99": 0, "100-999": 0, "1000-9999": 0, ">=10000": 0}}
    assert row["off_tissue"]["meniscal_tear"]["count"] == 0 and row["off_tissue"]["meniscal_tear"]["voxels"] == 0
    scan = _scan()
    scan["acceptance_policy"] = copy.deepcopy(POLICY)
    scan["positive"] = scan["positive"].copy()
    scan["positive"][CARTILAGE, 11, 7, 3] = True               # one more cartilage voxel on the meniscus, under min_voxels
    scan["probability"] = np.where(scan["positive"], 0.95, 0.2).astype(np.float32)
    off = diagnose_scan_v5(**scan)["off_tissue"]["cartilage_lesion"]
    assert (off["count"], off["kept"], off["voxels"], off["kept_voxels"]) == (3, 2, 7, 6)
    assert off["size_histogram"]["1"] == 1 and off["size_histogram"]["2-9"] == 2


def test_a_summary_pools_the_zones_and_the_components_off_tissue(row):
    other = diagnose_scan_v5(**dict(_scan(), patient_uid="synthetic:p1"))
    summary = summarize_diagnostics_v5([row, other])
    cartilage = summary["zones"]["cartilage_lesion"]
    assert {key: cartilage["same_class_primary"][key] for key in ("host_voxels", "positive_on_host", "positive_voxels")} \
        == {"host_voxels": 52, "positive_on_host": 16, "positive_voxels": 16}
    assert cartilage["same_class_primary"]["fraction_pooled"] == pytest.approx(8 / 26)
    assert cartilage["same_class_primary"]["fraction_per_scan"]["n"] == 2
    assert cartilage["same_class_primary"]["fraction_per_scan"]["p50"] == pytest.approx(8 / 26)
    assert cartilage["outside"]["fraction_pooled"] == 0.0 and cartilage["outside"]["scans_with_host"] == 2
    assert cartilage["ignored"]["fraction_pooled"] is None and cartilage["ignored"]["scans_with_host"] == 0
    assert cartilage["ignored"]["fraction_per_scan"]["n"] == 0
    tear = summary["zones"]["meniscal_tear"]
    assert tear["outside"]["fraction_pooled"] == pytest.approx(7 / 176)
    assert cartilage["outside"]["positive_off_host"] == 6 and tear["outside"]["positive_off_host"] == 0
    off = summary["off_tissue"]["cartilage_lesion"]
    assert {key: off[key] for key in ("count", "kept", "voxels", "kept_voxels")} == {
        "count": 4, "kept": 4, "voxels": 12, "kept_voxels": 12}
    assert off["size_histogram"]["2-9"] == 4
    assert off["share_of_components"] == pytest.approx(4 / 8) and off["share_of_positive_voxels"] == pytest.approx(12 / 28)
    assert summary["off_tissue"]["meniscal_tear"]["share_of_components"] == 0.0


def test_the_report_shows_the_zones_and_the_components_off_tissue(row):
    text = render_diagnostics_md_v5({"train": summarize_diagnostics_v5([row])})
    assert "## 宿主组织上的阳性体素（按区域）：cartilage_lesion" in text
    assert "| 同类主分析框内：宿主组织上达到阈值的体素比例（合并） | 0.308 |" in text          # 8 / 26
    assert "| 所有框之外（该类别的负监督区）：宿主组织上达到阈值的体素比例（合并） | 0.000 |" in text
    assert "| 忽略区内：宿主组织体素 | 0 |" in text
    assert "## 不在可归属组织上的连通块：cartilage_lesion" in text
    assert "| 不在可归属组织上的连通块 | 2 |" in text and "| 其中体素数 2-9 的 | 2 |" in text
    assert "synthetic:p0" not in text
