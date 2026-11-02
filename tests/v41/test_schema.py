import json
from pathlib import Path

import pytest

from anatobind.v41.schema import (
    SCHEMA_VERSION,
    SchemaError,
    AnnotationRecord,
    GateDecision,
    SampleRecord,
    require_assigned_split,
    validate_annotation,
    validate_gate_decision,
    validate_prediction,
    validate_sample,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas" / "v41"


def test_full_synthetic_sample_validates(sample_dict):
    record = validate_sample(sample_dict)
    assert isinstance(record, SampleRecord)
    assert record.sample_uid == "synth-0001"
    assert isinstance(record.annotation_records[0], AnnotationRecord)
    assert record.annotation_records[0].provenance_type == "manual"


def test_missing_provenance_is_rejected(annotation_dict):
    annotation_dict.pop("provenance_type")
    with pytest.raises(ValueError):
        validate_annotation(annotation_dict)


def test_missing_completeness_is_not_defaulted(annotation_dict):
    annotation_dict.pop("completeness")
    with pytest.raises(SchemaError, match="MISSING_FIELD"):
        validate_annotation(annotation_dict)


def test_unknown_enum_values_rejected(annotation_dict, sample_dict):
    annotation_dict["provenance_type"] = "probably_fine"
    with pytest.raises(SchemaError, match="INVALID_ENUM"):
        validate_annotation(annotation_dict)
    sample_dict["split"] = "test"
    with pytest.raises(SchemaError, match="INVALID_ENUM"):
        validate_sample(sample_dict)


def test_unknown_fields_rejected(sample_dict):
    sample_dict["normal"] = True
    with pytest.raises(SchemaError, match="UNKNOWN_FIELD"):
        validate_sample(sample_dict)


def test_unassigned_split_allowed_at_registration_only(sample_dict):
    record = validate_sample(sample_dict)
    assert record.split == "unassigned"
    with pytest.raises(SchemaError, match="SPLIT_UNASSIGNED"):
        require_assigned_split([record])
    sample_dict["split"] = "train"
    assert require_assigned_split([validate_sample(sample_dict)]) == 1


def test_schema_version_mismatch_rejected(sample_dict):
    sample_dict["schema_version"] = "v40.0"
    with pytest.raises(SchemaError, match="SCHEMA_VERSION"):
        validate_sample(sample_dict)


def test_prediction_rejects_ground_truth_fields():
    pred = {
        "schema_version": SCHEMA_VERSION,
        "sample_uid": "synth-0001",
        "observation_uid": "synth-0001:3d",
        "mode": "3d",
        "geometry": {"shape": [8, 8, 8], "spacing_mm": [1.0, 1.0, 1.0],
                     "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     "axis_codes": ["R", "A", "S"], "units": "mm"},
        "a_prob": "pred/a.npy",
        "u_prob": "pred/u.npy",
        "instances": [],
        "relations": [],
        "acceptance": {"accepted": True, "reason": None},
        "lineage": {"checkpoint_sha256": "0" * 64},
    }
    assert validate_prediction(pred).mode == "3d"
    for forbidden in ("a_target", "u_target", "gt_boxes", "tissue_id", "annotation_records"):
        bad = dict(pred)
        bad[forbidden] = "x"
        with pytest.raises(SchemaError, match="UNKNOWN_FIELD"):
            validate_prediction(bad)


def test_gate_decision_requires_both_statuses():
    gate = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": "G0-A",
        "scope_hash": "0" * 64,
        "engineering_status": "BLOCKED",
        "scientific_conclusion": "NOT_EVALUATED",
        "evidence_hashes": [],
        "blockers": ["no expert reference"],
        "approved_by": None,
        "approved_at": None,
    }
    decision = validate_gate_decision(gate)
    assert isinstance(decision, GateDecision)
    gate["engineering_status"] = "PASSED"
    with pytest.raises(SchemaError, match="INVALID_ENUM"):
        validate_gate_decision(gate)
    gate["engineering_status"] = "PASS"
    gate.pop("scientific_conclusion")
    with pytest.raises(SchemaError, match="MISSING_FIELD"):
        validate_gate_decision(gate)


def test_error_text_contains_no_private_values(sample_dict, secret_values):
    secret_path, secret_case = secret_values
    sample_dict["sample_uid"] = secret_case
    sample_dict["image_paths"] = [secret_path]
    sample_dict["annotation_records"][0]["path"] = secret_path
    sample_dict["annotation_records"][0].pop("provenance_type")
    with pytest.raises(SchemaError) as info:
        validate_sample(sample_dict)
    text = str(info.value)
    assert secret_path not in text and secret_case not in text
    assert "annotation_records[0].provenance_type" in text


def test_schema_files_exist_and_are_versioned():
    for name in ("sample", "annotation", "gate", "experiment"):
        document = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())
        assert document["properties"]["schema_version"]["const"] == SCHEMA_VERSION
        assert document["additionalProperties"] is False
