from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5, gate_task_v5
from tests.v5.factories import SYNTH_SOURCE, make_approvals, make_evidence, make_module_scope, make_scope

A_SCOPE = validate_scope_v5(make_module_scope("A"))
U_SCOPE = validate_scope_v5(make_module_scope("U"))
BASE_KEYS = ["g_data", "supervision_policy", "input_policy", "frozen_dev_split", "training_smoke"]


def _gate(gate_id, scope, evidence, approvals):
    return evaluate_gate_v5(gate_id, scope, evidence, approvals, SYNTH_SOURCE, fixture_mode=True)


def test_a_only_gate_ignores_unrelated_nulls():
    evidence = {**make_evidence(BASE_KEYS, A_SCOPE), "near_mm": None}
    approvals = {**make_approvals(A_SCOPE, ["resources"]), "noninferiority_margins": None,
                 "u_clinical_signoff": None}
    decision = _gate("G-BASE-A", A_SCOPE, evidence, approvals)
    assert decision["engineering_status"] == "PASS" and decision["blockers"] == []
    assert decision["scientific_conclusion"] == "NOT_EVALUATED"


def test_missing_evidence_and_null_approval_are_named():
    evidence = make_evidence(BASE_KEYS[:-1], A_SCOPE)
    decision = _gate("G-BASE-A", A_SCOPE, evidence, {"resources": None})
    assert decision["engineering_status"] == "BLOCKED"
    assert decision["blockers"] == ["APPROVAL_MISSING:resources", "EVIDENCE_MISSING:training_smoke"]


def test_approval_is_bound_to_scope_hash():
    other = validate_scope_v5(make_module_scope("A", scope_id="other_a"))
    decision = _gate("G-BASE-A", A_SCOPE, make_evidence(BASE_KEYS, A_SCOPE), make_approvals(other, ["resources"]))
    assert "APPROVAL_SCOPE_MISMATCH:resources" in decision["blockers"]


def test_evidence_needs_pass_and_valid_hash():
    evidence = make_evidence(BASE_KEYS, A_SCOPE)
    evidence["training_smoke"] = {**evidence["training_smoke"], "sha256": "x"}
    decision = _gate("G-BASE-A", A_SCOPE, evidence, make_approvals(A_SCOPE, ["resources"]))
    assert "EVIDENCE_MISSING:training_smoke" in decision["blockers"]


def test_u_gate_rejects_anatomy_only_scope():
    decision = _gate("G-BASE-U", A_SCOPE, make_evidence(BASE_KEYS, A_SCOPE), make_approvals(A_SCOPE, ["resources"]))
    assert "SCOPE_TASK_MISMATCH" in decision["blockers"]


def test_joint_gate_requires_core_relation():
    contract = {**make_scope()["relation_contract"], "completion_role": "auxiliary_spatial"}
    scope = validate_scope_v5(make_scope(relation_contract=contract))
    keys = ["qualified_same_case_cohort", "unified_output_chain", "state_counterexamples", "corrected_evaluators"]
    decision = _gate("G-JOINT", scope, make_evidence(keys, scope), make_approvals(scope, ["resources"]))
    assert "CORE_RELATION_REQUIRED" in decision["blockers"]


def test_unknown_gate_and_null_template():
    with pytest.raises(V5Error) as error:
        evaluate_gate_v5("G-ALL", A_SCOPE, {}, {}, SYNTH_SOURCE)
    assert error.value.code == "UNKNOWN_GATE"
    template = yaml.safe_load(Path("configs/v5/approvals.template.yaml").read_text())
    assert template and all(value is None for gate in template.values() for value in gate.values())


@pytest.mark.parametrize("gate_id", ["G-DATA", "G-BASE-U"])
def test_evidence_from_other_scope_is_rejected(gate_id):
    keys = list(GATE_REQUIREMENTS[gate_id]["evidence"])
    a_evidence = make_evidence(keys, A_SCOPE)
    decision = _gate(gate_id, U_SCOPE, a_evidence, make_approvals(U_SCOPE, GATE_REQUIREMENTS[gate_id]["approvals"]))
    assert decision["engineering_status"] == "BLOCKED"
    assert {f"EVIDENCE_BINDING_MISMATCH:{k}" for k in keys} <= set(decision["blockers"])


@pytest.mark.parametrize("field", ["source_id", "source_version"])
def test_evidence_from_other_source_or_version_is_rejected(field):
    other = {**SYNTH_SOURCE, field: "other"}
    decision = _gate("G-BASE-A", A_SCOPE, make_evidence(BASE_KEYS, A_SCOPE, other),
                     make_approvals(A_SCOPE, ["resources"]))
    assert decision["blockers"] == sorted(f"EVIDENCE_BINDING_MISMATCH:{k}" for k in BASE_KEYS)


def test_evidence_without_binding_is_rejected():
    legacy = {k: {"status": "PASS", "sha256": hashlib.sha256(k.encode()).hexdigest()} for k in BASE_KEYS}
    decision = _gate("G-BASE-A", A_SCOPE, legacy, make_approvals(A_SCOPE, ["resources"]))
    assert decision["blockers"] == sorted(f"EVIDENCE_BINDING_MISMATCH:{k}" for k in BASE_KEYS)


def test_fixture_approver_rejected_outside_fixture_mode():
    args = ("G-BASE-A", A_SCOPE, make_evidence(BASE_KEYS, A_SCOPE), make_approvals(A_SCOPE, ["resources"]),
            SYNTH_SOURCE)
    assert evaluate_gate_v5(*args)["blockers"] == ["APPROVAL_FIXTURE_NOT_ALLOWED:resources"]
    assert evaluate_gate_v5(*args, fixture_mode=True)["engineering_status"] == "PASS"


def test_decision_records_source_and_task():
    decision = _gate("G-BASE-A", A_SCOPE, make_evidence(BASE_KEYS, A_SCOPE), make_approvals(A_SCOPE, ["resources"]))
    assert (decision["source_id"], decision["source_version"], decision["task"]) == ("synthetic_source", "v0", "A")
    assert gate_task_v5(U_SCOPE) == "U" and gate_task_v5(validate_scope_v5(make_scope())) == "joint"


@pytest.mark.parametrize("source", [None, {}, {"source_id": "", "source_version": "v0"}, {"source_id": "s"}])
def test_invalid_source_raises(source):
    with pytest.raises(V5Error) as error:
        evaluate_gate_v5("G-DATA", A_SCOPE, {}, {}, source)
    assert error.value.code == "INVALID_SOURCE"
