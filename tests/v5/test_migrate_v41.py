from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import SampleV5, V5Error, validate_scope_v5
from anatobind.v41.migrate_v5 import migrate_v41_record_v5, migrate_v41_records_v5
from anatobind.v41.schema import validate_sample
from anatobind.v41.source_policy_v5 import load_source_policies_v5
from tests.v41.conftest import _sample
from tests.v5.factories import make_module_scope, make_source_policy

U_SCOPE = validate_scope_v5(make_module_scope("U"))


def _record(**overrides):
    body = _sample()
    body.update(overrides)
    return validate_sample(body)


def test_migrated_record_validates_and_keeps_v41_fields():
    migrated = migrate_v41_record_v5(_record(), make_source_policy(), U_SCOPE)
    assert isinstance(migrated, SampleV5) and migrated.schema_version == "v5.1"
    assert (migrated.active_scope_id, migrated.reference_policy_id) == ("synthetic_u", "synthetic_policy_v1")
    assert migrated.model_modalities == ["qdess_e1"]
    assert (migrated.patient_uid, migrated.split, migrated.sample_uid) == ("synthetic:p0001", "unassigned", "synth-0001")
    annotation = migrated.annotation_records[0]
    assert annotation.annotation_modalities == ["qdess_e1"] and annotation.visibility_region_path is None
    assert annotation.input_visibility == "visible"


@pytest.mark.parametrize("geometry_status,index_evidence,units_policy,units,expected", [
    ("fail", "synthetic", "trust_header", "mm", ("failed", "failed")),
    ("pass", "synthetic", "trust_header", "mm", ("verified", "verified")),
    ("pass", None, "trust_header", "mm", ("unknown", "verified")),
    ("pass", "synthetic", None, "mm", ("verified", "unknown")),
    ("not_checked", "synthetic", "trust_header", "mm", ("unknown", "unknown")),
    ("needs_review", "synthetic", "trust_header", "mm", ("unknown", "unknown")),
])
def test_geometry_status_never_upgraded(geometry_status, index_evidence, units_policy, units, expected):
    geometry = {**_sample()["image_geometry"], "units": units}
    record = _record(geometry_status=geometry_status, image_geometry=geometry)
    policy = make_source_policy(index_evidence=index_evidence, physical_units_policy=units_policy)
    migrated = migrate_v41_record_v5(record, policy, U_SCOPE)
    assert (migrated.index_alignment_status, migrated.physical_geometry_status) == expected
    assert migrated.image_geometry["units"] == units


def test_limited_visibility_without_region_is_not_silently_migrated():
    body = _sample()
    body["annotation_records"][0]["input_visibility"] = "limited"
    with pytest.raises(V5Error) as error:
        migrate_v41_record_v5(validate_sample(body), make_source_policy(), U_SCOPE)
    assert error.value.code == "MIGRATION_NEEDS_VISIBILITY_REGION"


@pytest.mark.parametrize("record_overrides,policy_overrides,code", [
    ({"source_dataset": "other"}, {}, "SOURCE_MISMATCH"),
    ({}, {"scope_id": "other_scope"}, "SCOPE_MISMATCH"),
    ({}, {"reference_policy_id": "other_policy"}, "SCOPE_MISMATCH"),
])
def test_source_or_scope_mismatch_rejected(record_overrides, policy_overrides, code):
    with pytest.raises(V5Error) as error:
        migrate_v41_record_v5(_record(**record_overrides), make_source_policy(**policy_overrides), U_SCOPE)
    assert error.value.code == code


def test_v41_record_not_mutated():
    record = _record()
    before = copy.deepcopy(dataclasses.asdict(record))
    migrate_v41_record_v5(record, make_source_policy(), U_SCOPE)
    assert dataclasses.asdict(record) == before and record.schema_version == "v41.0"


def test_batch_failures_report_index_not_ids():
    bad = _record(source_dataset="other")
    migrated, failures = migrate_v41_records_v5([_record(), bad], make_source_policy(), U_SCOPE)
    assert len(migrated) == 1
    assert failures == [{"index": 1, "code": "SOURCE_MISMATCH", "path": "$.source_dataset"}]
    assert "synth-0001" not in str(failures) and "p0001" not in str(failures)


def test_template_unknowns_fail_closed():
    policy = load_source_policies_v5(Path("configs/v5/sources_v5.template.yaml"))["skm_tea"]
    scope = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_skm_a.yaml").read_text()))
    body = _sample()
    body.update(source_dataset="SKM-TEA_ltr", image_paths=["e1.nii.gz", "e2.nii.gz"],
                modality_names=["qdess_e1", "qdess_e2"], modality_is_synthetic=[False, False],
                available_modality_mask=[True, True])
    body["annotation_records"][0].update(task="A", class_scope=["femoral_cartilage"])
    with pytest.raises(V5Error) as error:
        migrate_v41_record_v5(validate_sample(body), policy, scope)
    assert error.value.code == "ANNOTATION_MODALITY_UNKNOWN"


def test_only_v41_records_are_migrated():
    migrated = migrate_v41_record_v5(_record(), make_source_policy(), U_SCOPE)
    with pytest.raises(V5Error) as error:
        migrate_v41_record_v5(migrated, make_source_policy(), U_SCOPE)
    assert (error.value.code, error.value.path) == ("SCHEMA_VERSION_MISMATCH", "$.schema_version")
    with pytest.raises(V5Error) as error:
        migrate_v41_record_v5(_sample(), make_source_policy(), U_SCOPE)
    assert (error.value.code, error.value.path) == ("INVALID_TYPE", "$")


@pytest.mark.parametrize("geometry_status,index_evidence,units_policy,expected", [
    ("pass", "synthetic-index", "trust_header", {"index_alignment": "synthetic-index", "physical_geometry": "synthetic-units"}),
    ("fail", "synthetic-index", "trust_header", {"index_alignment": "v41_geometry_status:fail",
                                                 "physical_geometry": "v41_geometry_status:fail"}),
    ("pass", None, None, {"index_alignment": None, "physical_geometry": None}),
])
def test_migration_records_geometry_evidence(geometry_status, index_evidence, units_policy, expected):
    policy = make_source_policy(index_evidence=index_evidence, physical_units_policy=units_policy,
                                physical_units_evidence="synthetic-units" if units_policy else None)
    migrated = migrate_v41_record_v5(_record(geometry_status=geometry_status), policy, U_SCOPE)
    assert migrated.geometry_evidence == expected


@pytest.mark.parametrize("field,key", [("index_alignment_status", "index_alignment"),
                                       ("physical_geometry_status", "physical_geometry")])
def test_verified_geometry_needs_evidence(field, key):
    from tests.v5.factories import make_sample_v5
    from anatobind.v41.contracts_v5 import validate_sample_v5
    with pytest.raises(V5Error) as error:
        validate_sample_v5(make_sample_v5(**{field: "verified"}))
    assert (error.value.code, error.value.path) == ("GEOMETRY_EVIDENCE_MISSING", f"$.geometry_evidence.{key}")
