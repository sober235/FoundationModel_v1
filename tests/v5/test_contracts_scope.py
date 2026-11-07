from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5, validate_scope_v5
from tests.v5.factories import make_module_scope, make_sample_v5, make_scope


@pytest.fixture
def secret_values():
    return ("/private/zz_patient_root/case_000777",)


def test_scope_hash_is_canonical_and_content_bound():
    a, b = validate_scope_v5(make_scope()), validate_scope_v5(make_scope())
    assert a.scope_hash == b.scope_hash and len(a.scope_hash) == 64
    assert validate_scope_v5(make_scope(abnormality_classes=["target_a"])).scope_hash != a.scope_hash


@pytest.mark.parametrize("override, code", [
    ({"state_anatomy_classes": []}, "STATE_CLASSES_REQUIRED"),
    ({"state_anatomy_classes": ["tissue_x"]}, "STATE_CLASSES_NOT_SUBSET"),
    ({"abnormality_classes": []}, "JOINT_REQUIRES_U"),
    ({"relation_contract": None}, "RELATION_CONTRACT_REQUIRED"),
    ({"relation_predicate": "abuts"}, "PREDICATE_MISMATCH"),
    ({"anatomy_classes": ["tissue_a", "tissue_a"]}, "DUPLICATE_CLASS"),
    ({"observation_mode": "4d"}, "INVALID_ENUM"),
])
def test_joint_scope_rules(override, code):
    with pytest.raises(V5Error) as error:
        validate_scope_v5(make_scope(**override))
    assert error.value.code == code


def test_module_scope_cannot_claim_state_or_relation():
    with pytest.raises(V5Error) as error:
        validate_scope_v5(make_module_scope("A", state_anatomy_classes=["tissue_a"]))
    assert error.value.code == "MODULE_SCOPE_CANNOT_CLAIM_STATE"
    assert validate_scope_v5(make_module_scope("U")).anatomy_classes == []


def test_tissue_ids_start_at_one():
    scope = validate_scope_v5(make_scope())
    assert (scope.tissue_id("tissue_a"), scope.tissue_id("tissue_b")) == (1, 2)
    assert scope.state_tissue_ids == (1, 2)


def test_errors_never_echo_values(secret_values):
    record = make_scope(extra_field=secret_values[0])
    with pytest.raises(V5Error) as error:
        validate_scope_v5(record)
    assert error.value.code == "UNKNOWN_FIELD" and secret_values[0] not in str(error.value)


@pytest.mark.parametrize("annotation, code", [
    ({"input_visibility": "visible", "visibility_evidence": None}, "VISIBILITY_EVIDENCE_REQUIRED"),
    ({"input_visibility": "limited", "visibility_evidence": "synthetic"}, "LIMITED_VISIBILITY_REGION_REQUIRED"),
    ({"adjudication_status": "accepted"}, "INVALID_ENUM"),
])
def test_sample_visibility_and_adjudication_rules(annotation, code):
    with pytest.raises(V5Error) as error:
        validate_sample_v5(make_sample_v5(annotation=annotation))
    assert error.value.code == code


def test_model_modalities_must_be_declared():
    with pytest.raises(V5Error) as error:
        validate_sample_v5(make_sample_v5(model_modalities=["qdess_e2"]))
    assert error.value.code == "MODEL_MODALITY_UNKNOWN"


def test_joint_template_is_not_a_valid_scope_until_filled():
    template = yaml.safe_load(Path("configs/v5/scope_skm_joint.template.yaml").read_text())
    with pytest.raises(V5Error):
        validate_scope_v5(template)
    for name in ("scope_isles_u", "scope_skm_a"):
        validate_scope_v5(yaml.safe_load(Path(f"configs/v5/{name}.yaml").read_text()))
