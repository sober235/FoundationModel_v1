"""Task- and scope-specific admission gates (Proposal_v5 §9).

A gate is blocked only by its own evidence, approvals and scope rule; keys
outside its requirement row are ignored. Approvals are bound to scope_hash;
evidence is bound to scope_hash and also to the data source (id, version) and
task, so evidence produced for one source or scope cannot open another gate
(G-DATA is per (source, task), §9). Test-only approvers are rejected unless the
caller explicitly runs in fixture mode; decisions record fixture_mode and the approvals a fixture granted,
and evidence marked fixture_mode is rejected outside fixture mode. G-DATA approvals (data use is a
per-source sign-off) are also bound to source_id and source_version. This module neither reads nor writes files.
"""
from __future__ import annotations

from .contracts_v5 import ActiveScope, V5Error

_BASE = ("g_data", "supervision_policy", "input_policy", "frozen_dev_split", "training_smoke")

GATE_REQUIREMENTS: dict[str, dict] = {
    "G-DATA": {"evidence": ("readability", "semantics", "identity", "pairing", "task_geometry", "provenance"),
               "approvals": ("data_use",), "scope_rule": None, "source_bound_approvals": True},
    "G-BASE-A": {"evidence": _BASE, "approvals": ("resources",), "scope_rule": "anatomy"},
    "G-BASE-U": {"evidence": _BASE, "approvals": ("resources",), "scope_rule": "abnormality"},
    "G-ANNOTATION": {"evidence": ("readers", "quality", "review_regions", "visibility_regions", "voxel_u_reference",
                                  "state_reference", "core_relation_basis"),
                     "approvals": ("clinical_definition", "reader_policy"), "scope_rule": "joint"},
    "G-JOINT": {"evidence": ("qualified_same_case_cohort", "unified_output_chain", "state_counterexamples",
                             "corrected_evaluators"),
                "approvals": ("resources",), "scope_rule": "joint_core"},
    "G-H2": {"evidence": ("complete_dev_reference", "public_information_hash", "negative_support", "isolation"),
             "approvals": ("resources",), "scope_rule": "abnormality"},
    "G-H3-ENG": {"evidence": ("single_slice_boundary", "shared_weights", "synthetic_and_module_data"),
                 "approvals": (), "scope_rule": None},
    "G-H3-SCI": {"evidence": ("two_mode_references", "shared_weights", "dedicated_baselines", "capability_standards",
                              "noninferiority_policy"),
                 "approvals": ("capability_standards", "noninferiority_margins"), "scope_rule": "joint_core"},
    "G-LOCKED": {"evidence": ("frozen_models", "frozen_metrics", "frozen_thresholds", "evaluator_validation",
                              "independence"),
                 "approvals": ("independence", "unseal"), "scope_rule": None},
}

FIXTURE_APPROVERS = ("synthetic-fixture",)
_BINDING_FIELDS = ("scope_hash", "source_id", "source_version", "task")
_HEX = set("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _scope_blockers(rule: str | None, scope: ActiveScope) -> list[str]:
    if rule == "anatomy" and not scope.anatomy_classes:
        return ["SCOPE_TASK_MISMATCH"]
    if rule == "abnormality" and not scope.abnormality_classes:
        return ["SCOPE_TASK_MISMATCH"]
    if rule in ("joint", "joint_core") and scope.scope_kind != "joint":
        return ["SCOPE_TASK_MISMATCH"]
    if rule == "joint_core" and scope.relation_contract["completion_role"] != "core_host_binding":
        return ["CORE_RELATION_REQUIRED"]
    return []


def gate_task_v5(scope: ActiveScope) -> str:
    """Task a scope's gate evidence is issued for: "joint", or "A" / "U" / "A+U" for module scopes."""
    if scope.scope_kind == "joint":
        return "joint"
    tasks = [task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes)) if classes]
    return "+".join(tasks)


def _check_source(source: object) -> None:
    if not (isinstance(source, dict) and _non_empty_string(source.get("source_id"))
            and _non_empty_string(source.get("source_version"))):
        raise V5Error("INVALID_SOURCE", "$.source")


def evaluate_gate_v5(gate_id: str, scope: ActiveScope, evidence: dict, approvals: dict, source: dict,
                     fixture_mode: bool = False) -> dict:
    if gate_id not in GATE_REQUIREMENTS:
        raise V5Error("UNKNOWN_GATE", "$.gate_id")
    _check_source(source)
    if not isinstance(fixture_mode, bool):
        raise V5Error("INVALID_TYPE", "$.fixture_mode")
    requirement = GATE_REQUIREMENTS[gate_id]
    task = gate_task_v5(scope)
    expected = {"scope_hash": scope.scope_hash, "source_id": source["source_id"],
                "source_version": source["source_version"], "task": task}
    blockers = []
    hashes = []
    fixture_approvals = []
    for key in requirement["evidence"]:
        item = evidence.get(key)
        if not (isinstance(item, dict) and item.get("status") == "PASS" and _is_sha256(item.get("sha256"))):
            blockers.append(f"EVIDENCE_MISSING:{key}")
        elif any(item.get(field) != expected[field] for field in _BINDING_FIELDS):
            blockers.append(f"EVIDENCE_BINDING_MISMATCH:{key}")
        elif item.get("fixture_mode") is True and not fixture_mode:
            blockers.append(f"EVIDENCE_FIXTURE_NOT_ALLOWED:{key}")
        else:
            hashes.append(item["sha256"])
    for key in requirement["approvals"]:
        item = approvals.get(key)
        if not (isinstance(item, dict) and _non_empty_string(item.get("approved_by"))
                and _non_empty_string(item.get("approved_at"))):
            blockers.append(f"APPROVAL_MISSING:{key}")
        elif item["approved_by"] in FIXTURE_APPROVERS and not fixture_mode:
            blockers.append(f"APPROVAL_FIXTURE_NOT_ALLOWED:{key}")
        elif item.get("scope_hash") != scope.scope_hash:
            blockers.append(f"APPROVAL_SCOPE_MISMATCH:{key}")
        elif requirement.get("source_bound_approvals") and (item.get("source_id") != source["source_id"]
                                                            or item.get("source_version") != source["source_version"]):
            blockers.append(f"APPROVAL_SOURCE_MISMATCH:{key}")
        elif item["approved_by"] in FIXTURE_APPROVERS:
            fixture_approvals.append(key)
    blockers.extend(_scope_blockers(requirement["scope_rule"], scope))
    return {
        "gate_id": gate_id,
        "scope_id": scope.scope_id,
        "scope_hash": scope.scope_hash,
        "source_id": source["source_id"],
        "source_version": source["source_version"],
        "task": task,
        "fixture_mode": fixture_mode,
        "fixture_approvals": sorted(fixture_approvals),
        "engineering_status": "BLOCKED" if blockers else "PASS",
        "scientific_conclusion": "NOT_EVALUATED",
        "blockers": sorted(blockers),
        "evidence_hashes": sorted(hashes),
    }
