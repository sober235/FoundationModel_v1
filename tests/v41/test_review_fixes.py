"""Regression tests from the final branch review (findings 1-12)."""
import copy
import json

import numpy as np
import pytest

from anatobind.v41.registry import (
    REQUIRED_AUDITS,
    ManifestError,
    assemble_audit_index,
    evaluate_gate,
    publish_manifest,
)
from anatobind.v41.schema import SchemaError, validate_prediction, validate_sample


@pytest.fixture
def good_audits():
    return [{"name": name, "scope": "synthetic", "status": "pass", "sha256": "0" * 64}
            for name in REQUIRED_AUDITS]


def _index(tmp_path, status="pass", scope="synthetic"):
    p = tmp_path / "audit.txt"
    p.write_text("x\n")
    return [{"name": n, "scope": scope, "status": status, "path": str(p)} for n in REQUIRED_AUDITS]


# 1. audit status vocabulary
@pytest.mark.parametrize("status", ["pending", "not_checked", "FAIL", "quarantined", "PASS"])
def test_unknown_audit_status_is_rejected_everywhere(tmp_path, status, sample_dict):
    with pytest.raises(ManifestError, match="INVALID_AUDIT_STATUS"):
        assemble_audit_index(_index(tmp_path, status))
    audits = [{"name": n, "scope": "synthetic", "status": status, "sha256": "0" * 64} for n in REQUIRED_AUDITS]
    with pytest.raises(ManifestError, match="INVALID_AUDIT_STATUS"):
        publish_manifest([validate_sample(sample_dict)], audits, tmp_path / "m")
    with pytest.raises(ManifestError, match="INVALID_AUDIT_STATUS"):
        evaluate_gate("G0-A", audits, {}, source="synthetic")


# 7. scope is mandatory
def test_audit_index_requires_scope(tmp_path):
    entries = _index(tmp_path)
    del entries[0]["scope"]
    with pytest.raises(ManifestError, match="AUDIT_SCOPE_MISSING"):
        assemble_audit_index(entries)


# 2. expert_voxel qualification
@pytest.mark.parametrize("override", [
    {"representation": "bbox"},
    {"quality_status": "rejected"},
    {"completeness": "unknown"},
    {"completeness": "unreviewed"},
    {"completeness": "positive_only"},
])
def test_expert_voxel_requires_voxel_accepted_complete(sample_dict, good_audits, tmp_path, override):
    sample_dict["annotation_records"][0].update(override)
    sample_dict["annotation_records"][0]["reference_role"] = "expert_voxel"
    with pytest.raises(ManifestError, match="UNQUALIFIED_REFERENCE"):
        publish_manifest([validate_sample(sample_dict)], good_audits, tmp_path)


def test_isles_release_masks_register_as_unverified():
    from anatobind.v41.audits import build_registration_records

    assets = [{"source": "ISLES_ltr", "kind": "label", "case_id": "sub-0001",
               "relpath": "ISLES_ltr/ISLES-2022/derivatives/sub-0001/ses-0001/sub-0001_msk.nii.gz"}]
    for sub, role in (("dwi", "dwi"), ("dwi", "adc")):
        assets.append({"source": "ISLES_ltr", "kind": "image", "case_id": "sub-0001",
                       "relpath": f"ISLES_ltr/ISLES-2022/sub-0001/ses-0001/{sub}/sub-0001_ses-0001_{role}.nii.gz"})
    records = build_registration_records("isles_u", assets, None)
    record = validate_sample(records[0])
    assert record.annotation_records[0].reference_role == "unverified"
    # 12. partial modality sets stay schema-valid and aligned
    assert len(record.image_paths) == len(record.modality_names) == len(record.available_modality_mask) == 2


def test_skm_case_without_h5_is_not_registered():
    from anatobind.v41.audits import build_registration_records

    assets = [{"source": "SKM-TEA_ltr", "kind": "label", "case_id": "CASE_001",
               "relpath": "SKM-TEA_ltr/segmentation_masks/dicom-track/CASE_001.nii.gz"}]
    assert build_registration_records("skm_a", assets, {}) == []


# 6. gate bound to audit artefacts
def test_gate_uses_assembled_index_hashes_and_scope(tmp_path):
    entries = assemble_audit_index(_index(tmp_path, "pass", "src_a") + _index(tmp_path, "fail", "src_b"))
    approvals = {"target_abnormality_definition": {"approved_by": "x", "approved_at": "t"}}
    ok = evaluate_gate("G0-A", entries, approvals, source="src_a")
    assert ok.engineering_status == "PASS"
    assert sorted(ok.evidence_hashes) == sorted({e["sha256"] for e in entries if e["scope"] == "src_a"})
    bad = evaluate_gate("G0-A", entries, approvals, source="src_b")
    assert bad.engineering_status == "FAIL"
    missing = evaluate_gate("G0-A", entries, approvals, source="src_c")
    assert missing.engineering_status == "BLOCKED"


# 3. prediction nested objects are closed
def test_prediction_nested_objects_reject_ground_truth_keys():
    pred = {
        "schema_version": "v41.0", "sample_uid": "s", "observation_uid": "s:3d", "mode": "3d",
        "geometry": {"shape": [8, 8, 8], "spacing_mm": [1.0, 1.0, 1.0],
                     "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     "axis_codes": ["R", "A", "S"], "units": "mm"},
        "a_prob": None, "u_prob": None,
        "instances": [{"instance_id": 1, "class_name": "lesion", "score": 0.5, "voxel_count": 3,
                       "bbox_voxel": [0, 0, 0, 1, 1, 1], "host_structure": None}],
        "relations": [{"subject_instance_id": 1, "predicate": "within", "object_structure": "femoral_cartilage",
                       "score": 0.5}],
        "acceptance": {"accepted": True, "reason": None},
        "lineage": {"checkpoint_sha256": "0" * 64, "code_commit": "0" * 40, "config_sha256": None},
    }
    validate_prediction(pred)
    for where, key in (("instances", "gt_tissue_id"), ("instances", "a_target"), ("relations", "truth"),
                       ("acceptance", "gt_mask"), ("lineage", "a_target_path"), ("lineage", "reader_ids")):
        bad = copy.deepcopy(pred)
        target = bad[where][0] if isinstance(bad[where], list) else bad[where]
        target[key] = "x"
        with pytest.raises(SchemaError, match="UNKNOWN_FIELD"):
            validate_prediction(bad)


# 4. build_eligibility honours roles
def test_build_eligibility_excludes_pseudo_unverified_and_excluded_by_default(sample_dict):
    from anatobind.v41.eligibility import build_eligibility

    shape = (2, 2, 2)
    base = sample_dict["annotation_records"][0]
    for override in ({"provenance_type": "automated", "reference_role": "pseudo_only", "intended_role": "pseudo_only"},
                     {"reference_role": "unverified"},
                     {"intended_role": "excluded"}):
        record = copy.deepcopy(sample_dict)
        record["annotation_records"][0] = {**base, **override}
        assert not build_eligibility(record, "U", ["cartilage_lesion"], shape).any()
    record = copy.deepcopy(sample_dict)
    record["annotation_records"][0] = {**base, "provenance_type": "automated", "reference_role": "pseudo_only"}
    assert build_eligibility(record, "U", ["cartilage_lesion"], shape, roles=("pseudo",)).all()
    assert build_eligibility(sample_dict, "U", ["cartilage_lesion"], shape).all()


# 5. surface metrics respect eligibility
def test_surface_metrics_are_masked_by_eligibility():
    from anatobind.v41.evaluation.segmentation import segmentation_metrics

    truth = np.zeros((12, 12, 12), bool)
    truth[2:5, 2:5, 2:5] = True
    pred = truth.copy()
    pred[9:11, 9:11, 9:11] = True
    eligible = np.zeros_like(truth)
    eligible[:6, :6, :6] = True
    m = segmentation_metrics(pred, truth, eligible, (1.0, 1.0, 1.0), {"surface_tolerance_mm": 1.0})
    assert m["dice"] == 1.0 and m["hd95_mm"] == 0.0 and m["nsd"] == 1.0


# 8. leaking assignment is not frozen
def test_freeze_splits_refuses_leaking_assignment(make_sample, tmp_path):
    from anatobind.v41.splits import freeze_splits

    exposed = make_sample(sample_uid="x", patient_uid="src:x", split="locked_test",
                         exposure_history=[{"experiment_id": "legacy", "role": "test",
                                            "recorded_at": "2026-01-01T00:00:00Z"}])
    policy = {"seed": 1, "fractions": {"train": 0.6, "development_val": 0.2, "locked_test": 0.2}}
    with pytest.raises(ValueError, match="SPLIT_LEAKAGE"):
        freeze_splits([validate_sample(exposed)], policy, tmp_path / "s")
    assert not (tmp_path / "s" / "splits.json").exists()
    assert (tmp_path / "s" / "split_leakage_report.json").exists()


# 9. relation coverage counts lesions
def test_reference_coverage_counts_lesions_not_tuples():
    from anatobind.v41.evaluation.relations import relation_scores

    truth = [("l1", "within", "a"), ("l1", "adjacent", "b")]
    out = relation_scores(truth, truth, lesions_without_reference=1)
    assert out["reference_coverage"] == 0.5


# 10. unknown correctness is not a non-error
def test_risk_coverage_excludes_unknown_correctness():
    from anatobind.v41.evaluation.selective import risk_coverage

    out = risk_coverage([{"accepted": True, "correct": None}, {"accepted": True, "correct": False}])
    assert out["risk"] == 1.0 and out["unknown_correctness"] == 1


# 11. geometry: no pseudo-mm, schema-shaped output
def test_geometry_from_nifti_unknown_units_flagged_and_schema_compatible():
    import nibabel as nib

    from anatobind.v41.geometry import geometry_from_nifti, to_schema_geometry

    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.eye(4))
    img.header.set_xyzt_units("unknown", "unknown")
    geometry = geometry_from_nifti(img)
    assert geometry["units"] == "unknown"
    assert geometry["units_declared"] == "unknown"
    img.header.set_xyzt_units("mm", "unknown")
    schema_geometry = to_schema_geometry(geometry_from_nifti(img))
    assert set(schema_geometry) == {"shape", "spacing_mm", "affine", "axis_codes", "units"}
    with pytest.raises(ValueError, match="UNITS_NOT_MM"):
        to_schema_geometry(geometry)
