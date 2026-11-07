from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.gates_v5 import evaluate_gate_v5
from tests.v5.factories import make_approvals, make_evidence, make_module_scope, make_scope

A_SCOPE = validate_scope_v5(make_module_scope("A"))
BASE_KEYS = ["g_data", "supervision_policy", "input_policy", "frozen_dev_split", "training_smoke"]


def test_a_only_gate_ignores_unrelated_nulls():
    evidence = {**make_evidence(BASE_KEYS), "near_mm": None}
    approvals = {**make_approvals(A_SCOPE, ["resources"]), "noninferiority_margins": None,
                 "u_clinical_signoff": None}
    decision = evaluate_gate_v5("G-BASE-A", A_SCOPE, evidence, approvals)
    assert decision["engineering_status"] == "PASS" and decision["blockers"] == []
    assert decision["scientific_conclusion"] == "NOT_EVALUATED"


def test_missing_evidence_and_null_approval_are_named():
    evidence = make_evidence(BASE_KEYS[:-1])
    decision = evaluate_gate_v5("G-BASE-A", A_SCOPE, evidence, {"resources": None})
    assert decision["engineering_status"] == "BLOCKED"
    assert decision["blockers"] == ["APPROVAL_MISSING:resources", "EVIDENCE_MISSING:training_smoke"]


def test_approval_is_bound_to_scope_hash():
    other = validate_scope_v5(make_module_scope("A", scope_id="other_a"))
    decision = evaluate_gate_v5("G-BASE-A", A_SCOPE, make_evidence(BASE_KEYS), make_approvals(other, ["resources"]))
    assert "APPROVAL_SCOPE_MISMATCH:resources" in decision["blockers"]


def test_evidence_needs_pass_and_valid_hash():
    evidence = make_evidence(BASE_KEYS)
    evidence["training_smoke"] = {"status": "PASS", "sha256": "x"}
    decision = evaluate_gate_v5("G-BASE-A", A_SCOPE, evidence, make_approvals(A_SCOPE, ["resources"]))
    assert "EVIDENCE_MISSING:training_smoke" in decision["blockers"]


def test_u_gate_rejects_anatomy_only_scope():
    decision = evaluate_gate_v5("G-BASE-U", A_SCOPE, make_evidence(BASE_KEYS), make_approvals(A_SCOPE, ["resources"]))
    assert "SCOPE_TASK_MISMATCH" in decision["blockers"]


def test_joint_gate_requires_core_relation():
    contract = {**make_scope()["relation_contract"], "completion_role": "auxiliary_spatial"}
    scope = validate_scope_v5(make_scope(relation_contract=contract))
    keys = ["qualified_same_case_cohort", "unified_output_chain", "state_counterexamples", "corrected_evaluators"]
    decision = evaluate_gate_v5("G-JOINT", scope, make_evidence(keys), make_approvals(scope, ["resources"]))
    assert "CORE_RELATION_REQUIRED" in decision["blockers"]


def test_unknown_gate_and_null_template():
    with pytest.raises(V5Error) as error:
        evaluate_gate_v5("G-ALL", A_SCOPE, {}, {})
    assert error.value.code == "UNKNOWN_GATE"
    template = yaml.safe_load(Path("configs/v5/approvals.template.yaml").read_text())
    assert template and all(value is None for gate in template.values() for value in gate.values())
