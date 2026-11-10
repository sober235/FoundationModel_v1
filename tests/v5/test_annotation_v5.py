"""T11 early phase: SKM-TEA released boxes and tissue_id as reference candidates (synthetic only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anatobind.v41.annotation_v5 import (
    EXCLUSION_CASCADE, import_skm_boxes_v5, load_skm_box_policy_v5, skm_release_tables_v5, summarize_skm_import_v5)
from anatobind.v41.contracts_v5 import V5Error

POLICY_PATH = Path("configs/v5/skm_box_policy_v1.yaml")
SHAPE = (12, 10, 8)
CATEGORIES = ["Meniscal Tear (Myxoid)", "Meniscal Tear (Horizontal)", "Meniscal Tear (Radial)",
              "Meniscal Tear (Vertical/Longitudinal)", "Meniscal Tear (Oblique)", "Meniscal Tear (Complex)",
              "Meniscal Tear (Flap)", "Meniscal Tear (Extrusion)", "Ligament Tear (Low-Grade Sprain)",
              "Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration)",
              "Ligament Tear (Full Thickness/Complete Tear)", "Cartilage Lesion (1)", "Cartilage Lesion (2A)",
              "Cartilage Lesion (2B)", "Cartilage Lesion (3)", "Effusion"]
TISSUES = ["Meniscus", "ACL", "PCL", "Femoral Cartilage", "Patellar Cartilage", "Tibial Cartilage"]


def _release(tissue_order=TISSUES) -> dict:
    return {"categories": [{"id": i + 1, "name": name, "supercategory": name.split(" (")[0]}
                           for i, name in enumerate(CATEGORIES)],
            "tissues": [{"id": i + 1, "name": name} for i, name in enumerate(tissue_order)]}


def _reference_a() -> dict:
    a = {name: np.zeros(SHAPE, bool) for name in ("patellar_cartilage", "femoral_cartilage", "tibial_cartilage_medial",
                                                  "tibial_cartilage_lateral", "meniscus_medial", "meniscus_lateral")}
    a["femoral_cartilage"][0:4, 0:5, :] = True
    a["patellar_cartilage"][0:4, 5:10, :] = True
    a["tibial_cartilage_medial"][6:9, 0:5, 0:4] = True
    a["tibial_cartilage_lateral"][6:9, 0:5, 4:8] = True
    a["meniscus_medial"][9:12, 0:5, 0:4] = True
    a["meniscus_lateral"][9:12, 0:5, 4:8] = True
    return a                                   # rows 4–5 and columns 5–9 below row 6 hold no tissue


def _box(box_id, category, tissue, bbox, confidence=3.0, release=None):
    release = release or _release()
    tissue_id = -1 if tissue is None else next(t["id"] for t in release["tissues"] if t["name"] == tissue)
    return {"id": box_id, "category_id": CATEGORIES.index(category) + 1, "tissue_id": tissue_id,
            "bbox": [float(v) for v in bbox], "confidence": confidence, "labeler": 1}


def _import(boxes, release=None, reference=None):
    release = release or _release()
    return import_skm_boxes_v5(sample_uid="skm:synthetic-1", shape=SHAPE, annotations=boxes,
                               tables=skm_release_tables_v5(release, POLICY), reference_a=reference or _reference_a(),
                               policy=POLICY)


POLICY = load_skm_box_policy_v5(POLICY_PATH)


def _by_id(result):
    return {box["annotation_id"]: box for box in result["boxes"]}


def test_included_box_is_box_level_u_with_tissue_level_r_and_positive_only_s():
    result = _import([_box(1, "Cartilage Lesion (2A)", "Femoral Cartilage", [0, 0, 0, 3, 3, 3])])
    box = _by_id(result)[1]
    assert box["status"] == "included" and box["analysis"] == "primary" and box["reason"] is None
    assert box["abnormality_class"] == "cartilage_lesion" and box["reference_granularity"] == "box"
    assert box["host"] == {"tissue_group": "femoral_cartilage", "relation": "bound", "reference_granularity": "tissue",
                           "assignment": "released_tissue_id", "side": None}
    assert box["origin"] == {"source_id": "skm_tea", "source_version": "skm_tea_annotations_v1.0.0", "labeler": 1,
                             "category": "Cartilage Lesion (2A)", "tissue": "Femoral Cartilage"}
    state = result["tissue_state"]
    assert state["reference_granularity"] == "tissue"
    assert state["states"]["cartilage_lesion"] == {"femoral_cartilage": "positive",
                                                   "patellar_cartilage": "annotation_unknown",
                                                   "tibial_cartilage": "annotation_unknown"}
    assert state["states"]["meniscal_tear"] == {"meniscus": "annotation_unknown"}


def test_a_scan_without_boxes_is_annotation_unknown_never_negative():
    result = _import([])
    values = {v for groups in result["tissue_state"]["states"].values() for v in groups.values()}
    assert values == {"annotation_unknown"} and result["boxes"] == []
    assert "negative" not in json.dumps(result)


def test_box_references_never_become_voxel_truth():
    result = _import([_box(1, "Meniscal Tear (Radial)", "Meniscus", [9, 0, 0, 3, 5, 4])])
    text = json.dumps(result)                       # plain JSON: no arrays, masks or eligibility maps
    for word in ("e_pos", "e_neg", "mask", "voxel_truth", "negative"):
        assert word not in text
    assert result["policy_sha256"] == POLICY["policy_sha256"]


@pytest.mark.parametrize("box, reason", [
    (_box(1, "Ligament Tear (Low-Grade Sprain)", "Meniscus", [9, 0, 0, 2, 2, 2]), "category_tissue_mismatch"),
    (_box(1, "Cartilage Lesion (2B)", "ACL", [0, 0, 0, 2, 2, 2]), "category_tissue_mismatch"),
    (_box(1, "Cartilage Lesion (3)", "Femoral Cartilage", [0, 0, 0, 2, 2, 2], confidence=0.0), "confidence_zero"),
    (_box(1, "Cartilage Lesion (3)", "Femoral Cartilage", [0, 0, 0, 2, 0, 2]), "degenerate_box"),
    (_box(1, "Cartilage Lesion (3)", "Femoral Cartilage", [0.5, 0, 0, 2, 2, 2]), "non_integer_box"),
    (_box(1, "Cartilage Lesion (3)", "Femoral Cartilage", [10, 0, 0, 4, 2, 2]), "out_of_bounds"),
    (_box(1, "Cartilage Lesion (3)", "Femoral Cartilage", [4, 0, 0, 2, 5, 8]), "no_host_tissue_in_box"),
])
def test_invalid_boxes_are_excluded_counted_and_stay_unknown(box, reason):
    result = _import([box])
    assert _by_id(result)[1]["status"] == "excluded" and _by_id(result)[1]["reason"] == reason
    assert all(v == "annotation_unknown" for groups in result["tissue_state"]["states"].values()
               for v in groups.values())
    assert summarize_skm_import_v5([result])["excluded"] == {reason: 1}


def test_confidence_one_is_kept():
    result = _import([_box(1, "Cartilage Lesion (2A)", "Patellar Cartilage", [0, 5, 0, 2, 2, 2], confidence=1.0)])
    assert _by_id(result)[1]["status"] == "included"


@pytest.mark.parametrize("category, tissue, relation", [
    ("Meniscal Tear (Extrusion)", "Meniscus", "bound"),
    ("Ligament Tear (Full Thickness/Complete Tear)", "ACL", "out_of_scope"),
    ("Ligament Tear (Low-Grade Sprain)", "PCL", "out_of_scope"),
    ("Effusion", None, "no_host"),
])
def test_s3_classes_are_out_of_scope_not_targets(category, tissue, relation):
    result = _import([_box(1, category, tissue, [9, 0, 0, 3, 5, 4])])
    box = _by_id(result)[1]
    assert box["status"] == "out_of_scope" and box["abnormality_class"] is None and box["analysis"] is None
    assert box["host"]["relation"] == relation
    assert result["tissue_state"]["states"]["meniscal_tear"]["meniscus"] == "annotation_unknown"


def test_grade_one_is_secondary_and_never_a_primary_positive():
    result = _import([_box(1, "Cartilage Lesion (1)", "Patellar Cartilage", [0, 5, 0, 2, 2, 2])])
    box = _by_id(result)[1]
    assert box["status"] == "included" and box["analysis"] == "secondary"
    assert result["tissue_state"]["states"]["cartilage_lesion"]["patellar_cartilage"] == "annotation_unknown"
    assert result["tissue_state"]["secondary"]["cartilage_lesion"]["patellar_cartilage"] == "positive"


def test_tissue_assignment_uses_the_release_name_table_not_integer_ids():
    shuffled = _release(tissue_order=["Femoral Cartilage", "Meniscus", "Tibial Cartilage", "ACL", "PCL",
                                      "Patellar Cartilage"])
    box = _box(1, "Meniscal Tear (Complex)", "Meniscus", [9, 0, 0, 3, 5, 4], release=shuffled)
    assert box["tissue_id"] == 2
    result = _import([box], release=shuffled)
    assert _by_id(result)[1]["host"]["tissue_group"] == "meniscus"
    assert result["tissue_state"]["states"]["meniscal_tear"]["meniscus"] == "positive"


def test_sides_are_derived_from_the_box_and_reference_a():
    result = _import([_box(1, "Meniscal Tear (Flap)", "Meniscus", [9, 0, 0, 3, 5, 3]),
                      _box(2, "Cartilage Lesion (2B)", "Tibial Cartilage", [6, 0, 5, 3, 5, 3]),
                      _box(3, "Cartilage Lesion (2B)", "Femoral Cartilage", [0, 0, 0, 2, 2, 2])])
    boxes = _by_id(result)
    assert boxes[1]["host"]["side"] == "medial" and boxes[2]["host"]["side"] == "lateral"
    assert boxes[3]["host"]["side"] is None


def test_summary_counts_the_cascade_by_split_and_reason():
    first = _import([_box(1, "Cartilage Lesion (2A)", "Femoral Cartilage", [0, 0, 0, 3, 3, 3]),
                     _box(2, "Cartilage Lesion (1)", "Patellar Cartilage", [0, 5, 0, 2, 2, 2]),
                     _box(3, "Meniscal Tear (Extrusion)", "Meniscus", [9, 0, 0, 3, 5, 4]),
                     _box(4, "Cartilage Lesion (3)", "Femoral Cartilage", [0, 0, 0, 2, 2, 2], confidence=0.0),
                     _box(5, "Effusion", None, [0, 0, 0, 5, 5, 5])])
    second = _import([])
    summary = summarize_skm_import_v5([first, second], splits=["train", "development_val"])
    assert summary["scans"] == 2 and summary["boxes"] == 5
    assert summary["cascade"] == {"target_supercategory_consistent": 4, "confidence_nonzero": 3, "nonzero_size": 3,
                                  "integer_coordinates": 3, "in_bounds": 3, "host_tissue_in_box": 3, "primary": 1,
                                  "secondary": 1, "out_of_scope_extrusion": 1}
    assert summary["excluded"] == {"confidence_zero": 1}
    assert summary["out_of_scope"] == {"Effusion": 1, "Meniscal Tear (Extrusion)": 1}
    assert summary["positive_scans_by_split"] == {"train": {"cartilage_lesion:femoral_cartilage": 1},
                                                  "development_val": {}}
    assert summary["scans_without_primary_box"] == {"development_val": 1}
    assert list(EXCLUSION_CASCADE) == ["category_tissue_mismatch", "confidence_zero", "degenerate_box",
                                       "non_integer_box", "out_of_bounds", "no_host_tissue_in_box"]


def test_malformed_release_or_reference_raises():
    release = _release()
    release["categories"][0]["name"] = "Meniscal Tear (Bucket Handle)"
    with pytest.raises(V5Error) as err:
        skm_release_tables_v5(release, POLICY)
    assert err.value.code == "SKM_CATEGORY_UNKNOWN"
    release = _release(tissue_order=["Meniscus", "ACL", "PCL", "Femoral Cartilage", "Patellar Cartilage", "Bone"])
    with pytest.raises(V5Error) as err:
        skm_release_tables_v5(release, POLICY)
    assert err.value.code == "SKM_TISSUE_UNKNOWN"
    with pytest.raises(V5Error) as err:
        _import([_box(1, "Cartilage Lesion (2A)", "Femoral Cartilage", [0, 0, 0, 3, 3])])
    assert err.value.code == "INVALID_BOX"
    reference = _reference_a()
    reference["meniscus_lateral"] = np.zeros((2, 2, 2), bool)
    with pytest.raises(V5Error) as err:
        _import([], reference=reference)
    assert err.value.code == "GRID_MISMATCH"
    with pytest.raises(V5Error) as err:
        _import([{**_box(1, "Cartilage Lesion (2A)", "Femoral Cartilage", [0, 0, 0, 3, 3, 3]), "tissue_id": 99}])
    assert err.value.code == "SKM_TISSUE_UNKNOWN"


def test_policy_encodes_the_adopted_s1_to_s7_values():
    assert POLICY["policy_id"] == "skm_box_reference_v1"
    assert POLICY["source"] == {"source_id": "skm_tea", "source_version": "skm_tea_annotations_v1.0.0"}
    primary = sorted(name for name, t in POLICY["targets"].items() if t["analysis"] == "primary")
    assert primary == sorted(["Cartilage Lesion (2A)", "Cartilage Lesion (2B)", "Cartilage Lesion (3)",
                              "Meniscal Tear (Myxoid)", "Meniscal Tear (Horizontal)", "Meniscal Tear (Radial)",
                              "Meniscal Tear (Vertical/Longitudinal)", "Meniscal Tear (Oblique)",
                              "Meniscal Tear (Complex)", "Meniscal Tear (Flap)"])                       # S1
    assert POLICY["targets"]["Cartilage Lesion (1)"]["analysis"] == "secondary"                         # S2
    assert "Meniscal Tear (Extrusion)" in POLICY["out_of_scope_categories"]                              # S3
    assert POLICY["min_confidence"] == 1                                                                 # S4
    assert sorted(POLICY["host_tissues"]) == ["Femoral Cartilage", "Meniscus", "Patellar Cartilage",
                                              "Tibial Cartilage"]                                        # S6
    bad = dict(POLICY, min_confidence=3)
    bad.pop("policy_sha256")
    with pytest.raises(V5Error):
        load_skm_box_policy_v5(bad)


def test_joint_early_scope_has_its_own_binding_and_cannot_complete_the_core():
    import yaml

    from anatobind.v41.contracts_v5 import validate_scope_v5
    from anatobind.v41.gates_v5 import evaluate_gate_v5
    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

    path = Path("configs/v5/scope_skm_joint_early.yaml")              # approved by the researcher 2026-11-10
    assert "approved by the researcher 2026-11-10" in path.read_text()
    assert not list(Path("configs/v5/proposals").glob("*joint_early*"))
    scope = validate_scope_v5(yaml.safe_load(path.read_text()))
    skm_a = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_skm_a.yaml").read_text()))
    assert scope.scope_hash != skm_a.scope_hash and scope.anatomy_classes == skm_a.anatomy_classes
    assert scope.reference_policy_id == POLICY["policy_id"]
    assert list(scope.abnormality_classes) == list(POLICY["tissue_groups_by_class"])
    assert scope.relation_contract["completion_role"] == "auxiliary_spatial"
    policy = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))[
        "skm_tea_joint_early"]
    assert (policy.scope_id, policy.reference_policy_id, list(policy.model_modalities)) == (
        scope.scope_id, scope.reference_policy_id, list(scope.input_modalities))
    assert set(policy.annotation_modalities) == {"A", "U"} and policy.index_evidence is None
    source = source_binding_v5(policy)
    assert source == {"source_id": "skm_tea_joint_early", "source_version": POLICY["source"]["source_version"]}
    gate = evaluate_gate_v5("G-JOINT", scope, {}, {}, source)
    assert "CORE_RELATION_REQUIRED" in gate["blockers"]
