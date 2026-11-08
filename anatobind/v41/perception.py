"""v5.1 deployment chain: acceptance, state composition and the formal inference entry (plan_v5 T09c; Proposal §2.1).

Probabilities are restored to the original grid before any threshold. A voxel outside the represented region or
the FOV is never accepted (out_of_fov). A is the argmax with a confidence floor; U is per-class thresholding with
an optional reject band, and a rejected positive stays in the voxel map and its reason but never forms an
instance. Joint outputs compose S with ``derive_state_map_v5`` (rules 1-5). Nothing here takes a reference,
eligibility, review or visibility input (GC8).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .contracts_v5 import (
    GEOMETRY_KEYS, MODULE_TASK_KEYS, OUTPUT_VERSION_V5, UNKNOWN_KEY, ActiveScope, V5Error, array_sha256,
    canonical_sha256, validate_module_output_v5, validate_perception_output_v5)
from .engine.infer_v5 import predict_volume_v5, restore_probabilities_v5
from .models.interface_v5 import PerceptionModelV5
from .paths import PrivatePathError, resolve_private_output
from .perception_io import output_prefix_v5, save_output_v5
from .state_rule_v5 import NOT_ASSESSABLE, REASON_BITS, derive_state_map_v5

POLICY_ROLES = ("engineering", "calibrated")
CONNECTIVITY = {6: 1, 26: 3}     # instance connectivity -> scipy structuring-element rank
_FOV_TOLERANCE = 1e-6
_POLICY_KEYS = {"policy_id": None, "policy_role": None, "anatomy": ("min_confidence",),
                "abnormality": ("thresholds", "reject_band"), "instances": ("connectivity", "min_voxels")}


# ---------------------------------------------------------------------------
# Acceptance policy


def _probability(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _class_map(value: object, scope: ActiveScope, path: str, check) -> None:
    if not isinstance(value, dict) or set(value) != set(scope.abnormality_classes):
        raise V5Error("INVALID_ACCEPTANCE_VALUE", path)
    for name in scope.abnormality_classes:
        if not check(value[name]):
            raise V5Error("INVALID_ACCEPTANCE_VALUE", f"{path}.{name}")


def _band(value: object) -> bool:
    return value == "none" or (isinstance(value, list) and len(value) == 2 and all(_probability(v) for v in value)
                               and value[0] < value[1])


def validate_acceptance_policy_v5(policy: dict, scope: ActiveScope) -> dict:
    """Every value is registered (no defaults); values for a task the scope lacks must stay null."""
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not isinstance(policy, dict):
        raise V5Error("INVALID_TYPE", "$")
    if any(key not in _POLICY_KEYS for key in policy):
        raise V5Error("UNKNOWN_FIELD", f"$.{UNKNOWN_KEY}")
    for key, inner in _POLICY_KEYS.items():
        if policy.get(key) is None:
            raise V5Error("ACCEPTANCE_VALUE_MISSING", f"$.{key}")
        if inner is not None and (not isinstance(policy[key], dict) or set(policy[key]) - set(inner)):
            raise V5Error("UNKNOWN_FIELD", f"$.{key}.{UNKNOWN_KEY}")
    if not (isinstance(policy["policy_id"], str) and policy["policy_id"].strip()):
        raise V5Error("INVALID_ACCEPTANCE_VALUE", "$.policy_id")
    if policy["policy_role"] not in POLICY_ROLES:
        raise V5Error("INVALID_ACCEPTANCE_VALUE", "$.policy_role")
    checks = {
        ("anatomy", "min_confidence"): (bool(scope.anatomy_classes), _probability),
        ("abnormality", "thresholds"): (bool(scope.abnormality_classes),
                                        lambda v: _class_map(v, scope, "$.abnormality.thresholds",
                                                             lambda p: _probability(p) and p > 0) is None),
        ("abnormality", "reject_band"): (bool(scope.abnormality_classes),
                                         lambda v: _class_map(v, scope, "$.abnormality.reject_band", _band) is None),
        ("instances", "connectivity"): (bool(scope.abnormality_classes),
                                        lambda v: isinstance(v, int) and not isinstance(v, bool) and v in CONNECTIVITY),
        ("instances", "min_voxels"): (bool(scope.abnormality_classes),
                                      lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1),
    }
    for (section, key), (needed, check) in checks.items():
        value, path = policy[section].get(key), f"$.{section}.{key}"
        if not needed:
            if value is not None:
                raise V5Error("INVALID_ACCEPTANCE_VALUE", path)
        elif value is None:
            raise V5Error("ACCEPTANCE_VALUE_MISSING", path)
        elif not check(value):
            raise V5Error("INVALID_ACCEPTANCE_VALUE", path)
    return policy


# ---------------------------------------------------------------------------
# Acceptance, instances, state


def apply_acceptance_v5(probabilities: dict, represented: np.ndarray, valid_fov: np.ndarray, policy: dict,
                        scope: ActiveScope) -> dict:
    """{"anatomy": {map, accepted, reason}, "abnormality": {...}} on the original grid, for the tasks given."""
    validate_acceptance_policy_v5(policy, scope)
    if not isinstance(probabilities, dict) or not probabilities or set(probabilities) - {"anatomy", "abnormality"}:
        raise V5Error("INVALID_TYPE", "$.probabilities")
    represented, valid_fov = np.asarray(represented), np.asarray(valid_fov)
    if represented.dtype != bool or valid_fov.dtype != bool or represented.shape != valid_fov.shape:
        raise V5Error("GRID_MISMATCH", "$.valid_fov")
    spatial = represented.shape
    inside = represented & valid_fov
    channels = {"anatomy": 1 + len(scope.anatomy_classes), "abnormality": len(scope.abnormality_classes)}
    out = {}
    for key, prob in probabilities.items():
        if not isinstance(prob, np.ndarray) or not np.issubdtype(prob.dtype, np.floating) \
                or prob.shape != (channels[key], *spatial):
            raise V5Error("GRID_MISMATCH", f"$.probabilities.{key}")
        if key == "anatomy":
            confident = prob.max(axis=0) >= policy["anatomy"]["min_confidence"]
            accepted = inside & confident
            reason = np.where(~inside, REASON_BITS["out_of_fov"],
                              np.where(confident, 0, REASON_BITS["anatomy_uncertain"])).astype(np.uint8)
            label = np.where(accepted, prob.argmax(axis=0), -1).astype(np.int16)
            out[key] = {"map": label, "accepted": accepted, "reason": reason}
            continue
        positive = np.zeros(prob.shape, dtype=bool)
        uncertain = np.zeros(prob.shape, dtype=bool)
        for c, name in enumerate(scope.abnormality_classes):
            positive[c] = prob[c] >= policy["abnormality"]["thresholds"][name]
            band = policy["abnormality"]["reject_band"][name]
            if band != "none":
                uncertain[c] = (prob[c] >= band[0]) & (prob[c] <= band[1])
        accepted = inside[None] & ~uncertain
        reason = np.where(~inside[None], REASON_BITS["out_of_fov"],
                          np.where(uncertain, REASON_BITS["abnormality_uncertain"], 0)).astype(np.uint8)
        out[key] = {"map": positive & inside[None], "accepted": accepted, "reason": reason}
    return out


def extract_instances_v5(abnormality_map: np.ndarray, abnormality_accepted: np.ndarray, scope: ActiveScope,
                         policy: dict) -> list[dict]:
    """Connected components of accepted positives per class, sorted by class then bbox; uids follow that order."""
    validate_acceptance_policy_v5(policy, scope)
    if scope.observation_mode != "3d":
        raise V5Error("MODE_NOT_SUPPORTED", "$.observation_mode")
    positive, accepted = np.asarray(abnormality_map), np.asarray(abnormality_accepted)
    if positive.dtype != bool or accepted.dtype != bool or positive.shape != accepted.shape \
            or positive.ndim != 4 or positive.shape[0] != len(scope.abnormality_classes):
        raise V5Error("GRID_MISMATCH", "$.abnormality_map")
    structure = ndimage.generate_binary_structure(3, CONNECTIVITY[policy["instances"]["connectivity"]])
    found = []
    for c, name in enumerate(scope.abnormality_classes):
        labels, count = ndimage.label(positive[c] & accepted[c], structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        for k, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None or sizes[k] < policy["instances"]["min_voxels"]:
                continue
            mask = np.zeros(labels.shape, dtype=bool)
            mask[box] = labels[box] == k
            bbox = [int(s.start) for s in box] + [int(s.stop) for s in box]
            found.append((c, bbox, name, mask))
    found.sort(key=lambda item: (item[0], item[1]))
    candidate_set_hash = canonical_sha256([{"class_name": name, "bbox": bbox, "mask_sha256": array_sha256(mask)}
                                           for _, bbox, name, mask in found])
    return [{"instance_uid": f"inst-{i:04d}", "class_name": name, "mask": mask, "bbox": bbox,
             "candidate_set_hash": candidate_set_hash, "accepted": True}
            for i, (_, bbox, name, mask) in enumerate(found, start=1)]


def state_policy_hash_v5(scope: ActiveScope, acceptance_policy: dict) -> str:
    """Identity of what composes S: the state rule, its reason bits, the state tissues and the acceptance policy."""
    return canonical_sha256({"state_rule": "derive_state_map_v5", "reason_bits": REASON_BITS,
                             "state_tissue_ids": list(scope.state_tissue_ids), "acceptance_policy": acceptance_policy})


# ---------------------------------------------------------------------------
# The chain


def run_perception_v5(model: PerceptionModelV5, batch, scope: ActiveScope, *, observation_uid: str,
                      checkpoint_sha256: str, original_geometry: dict, acceptance_policy: dict, window: tuple,
                      stride: tuple, precision: str, relation_fn=None):
    """predict → restore → accept, then a validated ModuleOutput (module scope) or PerceptionOutput (joint)."""
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not isinstance(model, PerceptionModelV5):
        raise V5Error("INVALID_TYPE", "$.model")
    validate_acceptance_policy_v5(acceptance_policy, scope)
    if scope.observation_mode not in model.supported_modes:
        raise V5Error("MODE_NOT_SUPPORTED", "$.observation_mode")
    if model.scope.scope_hash != scope.scope_hash:
        raise V5Error("SCOPE_MISMATCH", "$.scope_hash")
    joint = scope.scope_kind == "joint"
    if joint and relation_fn is None:
        raise V5Error("RELATION_FN_REQUIRED", "$.relation_fn")
    if joint and set(model.tasks) != {"A", "U"}:
        raise V5Error("MODULE_TASKS_INVALID", "$.model.tasks")
    if not isinstance(original_geometry, dict) or set(original_geometry) != set(GEOMETRY_KEYS):
        raise V5Error("GRID_MISMATCH", "$.original_geometry")
    original_shape = tuple(int(s) for s in original_geometry["shape"])

    probabilities = predict_volume_v5(model, batch, window, stride, precision)
    chain = list(batch.to_original[0])
    restored, represented = {}, None
    for key, prob in probabilities.items():
        restored[key], represented = restore_probabilities_v5(prob, chain, original_shape)
    # The model-grid FOV maps back conservatively: in FOV only where every coarse voxel used is in FOV.
    fov = batch.valid_fov[0].detach().cpu().numpy().astype(np.float32)[None]
    valid_fov = restore_probabilities_v5(fov, chain, original_shape)[0][0] >= 1.0 - _FOV_TOLERANCE
    decided = apply_acceptance_v5(restored, represented, valid_fov, acceptance_policy, scope)

    image = batch.image[0].detach().cpu().numpy()
    common = {"output_version": OUTPUT_VERSION_V5, "observation_uid": observation_uid,
              "mode": scope.observation_mode, "scope_id": scope.scope_id, "scope_hash": scope.scope_hash,
              "checkpoint_sha256": checkpoint_sha256, "geometry": copy.deepcopy(original_geometry),
              "provenance": {"input_sha256": canonical_sha256([array_sha256(channel) for channel in image]),
                             "model_sha256": checkpoint_sha256,
                             "output_policy_version": acceptance_policy["policy_id"]}}
    if not joint:
        enabled = [task for task in ("A", "U") if task in model.tasks]
        keys = [MODULE_TASK_KEYS[task] for task in enabled]
        output = dict(common, output_kind="module", enabled_tasks=enabled,
                      disabled_tasks=[task for task in ("A", "U", "S", "R") if task not in enabled],
                      anatomy_map=decided["anatomy"]["map"] if "A" in enabled else None,
                      abnormality_map=decided["abnormality"]["map"] if "U" in enabled else None,
                      acceptance={key: decided[key]["accepted"] for key in keys},
                      reason={key: decided[key]["reason"] for key in keys})
        return validate_module_output_v5(output, scope)

    a, u = decided["anatomy"], decided["abnormality"]
    state, state_reason = derive_state_map_v5(a["map"], a["accepted"], a["reason"], u["map"], u["accepted"],
                                              u["reason"], scope.state_tissue_ids)
    instances = extract_instances_v5(u["map"], u["accepted"], scope, acceptance_policy)
    result = relation_fn(instances, a["map"], a["accepted"], scope)
    if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list)
            and isinstance(result[1], dict)
            and all(isinstance(r, dict) and "instance_uid" in r and "accepted" in r for r in result[0])):
        raise V5Error("INVALID_TYPE", "$.tissue_relations")
    relations, relation_reasons = result
    output = dict(common, output_kind="joint", state_policy_hash=state_policy_hash_v5(scope, acceptance_policy),
                  anatomy_map=a["map"], abnormality_map=u["map"], tissue_state_map=state,
                  abnormal_instances=instances, tissue_relations=relations,
                  acceptance={"anatomy": a["accepted"], "abnormality": u["accepted"],
                              "state": state != NOT_ASSESSABLE,
                              "relations": {r["instance_uid"]: r["accepted"] for r in relations}},
                  reason={"anatomy": a["reason"], "abnormality": u["reason"], "state": state_reason,
                          "relations": dict(relation_reasons)})
    return validate_perception_output_v5(output, scope)


def infer_and_save_v5(model: PerceptionModelV5, batch, scope: ActiveScope, *, observation_uid: str,
                      checkpoint_sha256: str, original_geometry: dict, acceptance_policy: dict, window: tuple,
                      stride: tuple, precision: str, out_dir: Path, private_root: Path, repo: Path,
                      relation_fn=None) -> tuple:
    """The formal inference entry: run_perception_v5, then save_output_v5. The destination is checked first."""
    if not isinstance(observation_uid, str) or not observation_uid:
        raise V5Error("INVALID_TYPE", "$.observation_uid")
    record = resolve_private_output(Path(out_dir) / f"{output_prefix_v5(observation_uid)}.record.json",
                                    private_root, repo)
    if record.exists():
        raise PrivatePathError("PRIVATE_FILE_EXISTS")
    output = run_perception_v5(model, batch, scope, observation_uid=observation_uid,
                               checkpoint_sha256=checkpoint_sha256, original_geometry=original_geometry,
                               acceptance_policy=acceptance_policy, window=window, stride=stride,
                               precision=precision, relation_fn=relation_fn)
    return output, save_output_v5(output, out_dir, private_root, repo)
