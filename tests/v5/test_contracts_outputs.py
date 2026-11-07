from __future__ import annotations

import inspect

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import (
    V5Error,
    validate_module_output_v5,
    validate_perception_output_v5,
    validate_scope_v5,
)
from anatobind.v41.state_rule_v5 import derive_state_map_v5
from tests.v5.factories import make_joint_output, make_module_output, make_module_scope, make_scope

SCOPE = validate_scope_v5(make_scope())


def test_fixture_is_valid_and_keeps_u_under_anatomy_rejection():
    out = validate_perception_output_v5(make_joint_output(SCOPE), SCOPE)
    assert out.tissue_state_map[0, 1, 0] == 2 and out.reason["state"][0, 1, 0] == 4
    assert bool(out.abnormality_map[0, 0, 1, 0]) is True


def test_rejected_class_blocks_negative_but_not_accepted_positive():
    out = validate_perception_output_v5(make_joint_output(SCOPE), SCOPE)
    assert out.tissue_state_map[0, 1, 1] == 2
    assert out.tissue_state_map[0, 0, 0] == 1


def _mutate(path, value):
    out = make_joint_output(SCOPE)
    target = out
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return out


@pytest.mark.parametrize("path, value, code", [
    (("scope_hash",), "e" * 64, "SCOPE_MISMATCH"),
    (("mode",), "2d", "MODE_MISMATCH"),
    (("tissue_state_map",), np.zeros((1, 2, 3), np.uint8), "GRID_MISMATCH"),
    (("reference_mask",), np.zeros((1, 2, 2), bool), "UNKNOWN_FIELD"),
    (("provenance", "model_sha256"), "f" * 64, "PROVENANCE_MISMATCH"),
    (("tissue_state_map",), np.array([[[0, 0], [2, 2]]], np.uint8), "STATE_COMPOSITION_VIOLATION"),
])
def test_joint_output_rejects(path, value, code):
    with pytest.raises(V5Error) as error:
        validate_perception_output_v5(_mutate(path, value), SCOPE)
    assert error.value.code == code


def test_joint_output_requires_state_map():
    out = make_joint_output(SCOPE)
    out.pop("tissue_state_map")
    with pytest.raises(V5Error) as error:
        validate_perception_output_v5(out, SCOPE)
    assert error.value.code == "MISSING_FIELD"


@pytest.mark.parametrize("status, tissue_ids", [("bound", []), ("unknown", [1])])
def test_relation_status_rules(status, tissue_ids):
    out = make_joint_output(SCOPE)
    out["tissue_relations"][0].update(status=status, tissue_ids=tissue_ids)
    with pytest.raises(V5Error) as error:
        validate_perception_output_v5(out, SCOPE)
    assert error.value.code == "RELATION_STATUS_INVALID"


def test_module_output_cannot_carry_state():
    scope = validate_scope_v5(make_module_scope("A"))
    out = make_module_output(scope, "A")
    validate_module_output_v5(out, scope)
    out["tissue_state_map"] = np.zeros((1, 2, 2), np.uint8)
    with pytest.raises(V5Error) as error:
        validate_module_output_v5(out, scope)
    assert error.value.code == "UNKNOWN_FIELD"
    bad = make_module_output(scope, "A", disabled_tasks=["U", "R"])
    with pytest.raises(V5Error) as error:
        validate_module_output_v5(bad, scope)
    assert error.value.code == "MODULE_TASKS_INVALID"


def test_state_rule_signature_has_no_reference_inputs():
    params = set(inspect.signature(derive_state_map_v5).parameters)
    assert not params & {"truth", "reference", "review", "visibility", "eligible", "gt"}


def test_unknown_output_key_is_not_echoed():
    secret = "/private/zz_patient_root/case_000777"
    out = make_joint_output(SCOPE)
    out[secret] = 1
    with pytest.raises(V5Error) as error:
        validate_perception_output_v5(out, SCOPE)
    assert error.value.code == "UNKNOWN_FIELD" and secret not in str(error.value)
