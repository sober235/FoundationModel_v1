"""Synthetic fixtures for v5.1 tests. Names are tissue_a/tissue_b and target_a/target_b only."""
from __future__ import annotations

import copy

import numpy as np

from anatobind.v41.state_rule_v5 import REASON_BITS, derive_state_map_v5
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


# Fixed joint-output sample, shape [1,2,2] (plan_v5 T01b table).
_ANATOMY = np.array([[[1, 1], [-1, 2]]])
_ANATOMY_ACCEPTED = np.array([[[True, True], [False, True]]])
_ABNORMALITY = np.array([[[[True, False], [True, False]]], [[[False, False], [False, False]]]])
_ABNORMALITY_ACCEPTED = np.array([[[[True, True], [True, True]]], [[[True, True], [True, False]]]])
_INSTANCE_MASK = np.array([[[True, False], [True, False]]])


def _spatial(array: np.ndarray, mode: str, leading: int = 0) -> np.ndarray:
    """The fixed sample is 3d; a 2d scope gets its single z-slice."""
    if mode == "2d":
        index = (slice(None),) * leading + (0,)
        return array[index]
    return array


def _reasons(accepted: np.ndarray, bit: str) -> np.ndarray:
    return np.where(accepted, 0, REASON_BITS[bit]).astype(np.uint8)


def _geometry(shape: tuple) -> dict:
    return {"shape": list(shape), "axis_order": "zyx"[-len(shape):], "spacing_mm": None, "affine": None,
            "units_reliable": False, "transform_chain": []}


def _provenance() -> dict:
    return {"input_sha256": "a" * 64, "model_sha256": "b" * 64, "output_policy_version": "synthetic_output_policy_v1"}


def make_joint_output(scope, *, anatomy_map=None, abnormality_map=None, anatomy_accepted=None,
                      abnormality_accepted=None, **fields) -> dict:
    mode = scope.observation_mode
    n_classes = len(scope.abnormality_classes)
    if anatomy_map is None and abnormality_map is None:
        anatomy_map = _spatial(_ANATOMY, mode)
        abnormality_map = _spatial(_ABNORMALITY[:n_classes], mode, 1)
        anatomy_accepted = _spatial(_ANATOMY_ACCEPTED, mode)
        abnormality_accepted = _spatial(_ABNORMALITY_ACCEPTED[:n_classes], mode, 1)
    else:
        if anatomy_map is None:
            anatomy_map = np.ones(np.asarray(abnormality_map).shape[1:], dtype=np.int64)
        anatomy_map = np.asarray(anatomy_map)
        if abnormality_map is None:
            abnormality_map = np.zeros((n_classes, *anatomy_map.shape), dtype=bool)
        if anatomy_accepted is None:
            if (anatomy_map == -1).any():
                raise ValueError("anatomy_map contains -1: pass anatomy_accepted explicitly")
            anatomy_accepted = np.ones(anatomy_map.shape, dtype=bool)
        if abnormality_accepted is None:
            abnormality_accepted = np.ones(np.asarray(abnormality_map).shape, dtype=bool)
    anatomy_map = np.asarray(anatomy_map)
    abnormality_map = np.asarray(abnormality_map, dtype=bool)
    anatomy_accepted = np.asarray(anatomy_accepted, dtype=bool)
    abnormality_accepted = np.asarray(abnormality_accepted, dtype=bool)
    anatomy_reason = _reasons(anatomy_accepted, "anatomy_uncertain")
    abnormality_reason = _reasons(abnormality_accepted, "abnormality_uncertain")
    state, state_reason = derive_state_map_v5(anatomy_map, anatomy_accepted, anatomy_reason, abnormality_map,
                                              abnormality_accepted, abnormality_reason, scope.state_tissue_ids)
    mask = _spatial(_INSTANCE_MASK, mode)
    bbox = [0, 0, 0, 1, 2, 1] if mode == "3d" else [0, 0, 2, 1]
    output = {
        "output_kind": "joint",
        "output_version": "v5.1",
        "observation_uid": "obs-synth-1",
        "mode": mode,
        "scope_id": scope.scope_id,
        "scope_hash": scope.scope_hash,
        "checkpoint_sha256": "b" * 64,
        "state_policy_hash": "d" * 64,
        "anatomy_map": anatomy_map,
        "abnormality_map": abnormality_map,
        "tissue_state_map": state,
        "abnormal_instances": [{"instance_uid": "inst-1", "class_name": scope.abnormality_classes[0], "mask": mask,
                                "bbox": bbox, "candidate_set_hash": "c" * 64, "accepted": True}],
        "tissue_relations": [{"instance_uid": "inst-1", "predicate": scope.relation_contract["predicate"],
                              "tissue_ids": [1], "status": "bound", "accepted": True,
                              "completion_role": scope.relation_contract["completion_role"]}],
        "acceptance": {"anatomy": anatomy_accepted, "abnormality": abnormality_accepted, "state": state != 2,
                       "relations": {"inst-1": True}},
        "reason": {"anatomy": anatomy_reason, "abnormality": abnormality_reason, "state": state_reason,
                   "relations": {"inst-1": 0}},
        "geometry": _geometry(anatomy_map.shape),
        "provenance": _provenance(),
    }
    output.update(fields)
    return output


def make_module_output(scope, task: str, **fields) -> dict:
    mode = scope.observation_mode
    output = {
        "output_kind": "module",
        "output_version": "v5.1",
        "observation_uid": "obs-synth-1",
        "mode": mode,
        "scope_id": scope.scope_id,
        "scope_hash": scope.scope_hash,
        "checkpoint_sha256": "b" * 64,
        "enabled_tasks": [task],
        "disabled_tasks": sorted({"A", "U", "S", "R"} - {task}),
        "anatomy_map": None,
        "abnormality_map": None,
        "acceptance": {},
        "reason": {},
        "geometry": _geometry(_spatial(_ANATOMY, mode).shape),
        "provenance": _provenance(),
    }
    if task == "A":
        accepted = _spatial(_ANATOMY_ACCEPTED, mode)
        output.update(anatomy_map=_spatial(_ANATOMY, mode))
        output["acceptance"]["anatomy"] = accepted
        output["reason"]["anatomy"] = _reasons(accepted, "anatomy_uncertain")
    else:
        n_classes = len(scope.abnormality_classes)
        abnormality = np.zeros((n_classes, *_spatial(_ANATOMY, mode).shape), dtype=bool)
        abnormality[0] = _spatial(_ABNORMALITY[0], mode)
        accepted = np.ones(abnormality.shape, dtype=bool)
        output.update(abnormality_map=abnormality)
        output["acceptance"]["abnormality"] = accepted
        output["reason"]["abnormality"] = _reasons(accepted, "abnormality_uncertain")
    output.update(fields)
    return output
