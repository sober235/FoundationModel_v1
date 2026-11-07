"""Synthetic fixtures for v5.1 tests. Names are tissue_a/tissue_b and target_a/target_b only."""
from __future__ import annotations

import copy

from tests.v41.conftest import _sample

FIXTURE_APPROVER = "synthetic-fixture"

_RELATION_CONTRACT = {
    "target_question": "which tissue hosts the lesion",
    "predicate": "overlaps",
    "reference_method": "independent_expert_host",
    "display_name": "located_in",
    "completion_role": "core_host_binding",
}

_SCOPE = {
    "schema_version": "v5.1",
    "scope_id": "synthetic_joint",
    "scope_kind": "joint",
    "anatomy_classes": ["tissue_a", "tissue_b"],
    "state_anatomy_classes": ["tissue_a", "tissue_b"],
    "abnormality_classes": ["target_a", "target_b"],
    "relation_type": "host_binding",
    "relation_predicate": "overlaps",
    "relation_contract": _RELATION_CONTRACT,
    "input_modalities": ["synthetic_e1", "synthetic_e2"],
    "observation_mode": "3d",
    "normal_claim_scope": "no_enabled_target_in_reviewed_state_tissues",
    "reference_policy_id": "synthetic_policy_v1",
}

_MODULE_SCOPES = {
    "A": {"scope_id": "synthetic_a", "anatomy_classes": ["tissue_a", "tissue_b"], "abnormality_classes": []},
    "U": {"scope_id": "synthetic_u", "anatomy_classes": [], "abnormality_classes": ["target_a"]},
}


def make_scope(**overrides) -> dict:
    record = copy.deepcopy(_SCOPE)
    record.update(copy.deepcopy(overrides))
    return record


def make_module_scope(task: str, **overrides) -> dict:
    record = make_scope(scope_kind="module", state_anatomy_classes=[], relation_type=None,
                        relation_predicate=None, relation_contract=None, **_MODULE_SCOPES[task])
    record.update(copy.deepcopy(overrides))
    return record


def make_sample_v5(annotation: dict | None = None, **overrides) -> dict:
    record = _sample()
    record.update({
        "schema_version": "v5.1",
        "active_scope_id": "synthetic_joint",
        "reference_policy_id": "synthetic_policy_v1",
        "model_modalities": ["qdess_e1"],
        "index_alignment_status": "unknown",
        "physical_geometry_status": "unknown",
    })
    record["annotation_records"][0].update({
        "annotation_uid": "ann-v5-0001",
        "class_scope": ["target_a"],
        "annotation_modalities": ["qdess_e1"],
        "visibility_region_path": None,
    })
    record.update(copy.deepcopy(overrides))
    if annotation is not None:
        record["annotation_records"][0].update(copy.deepcopy(annotation))
    return record
