"""Synthetic fixtures for v5.1 tests. Names are tissue_a/tissue_b and target_a/target_b only."""
from __future__ import annotations

import copy
import hashlib
import zlib
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import yaml

from anatobind.v41.contracts_v5 import validate_module_output_v5, validate_sample_v5, validate_scope_v5
from anatobind.v41.evaluation.reference import ReferenceLesion
from anatobind.v41.gates_v5 import GATE_REQUIREMENTS, evaluate_gate_v5, gate_task_v5
from anatobind.v41.models.interface_v5 import ModelOutputV5, PerceptionModelV5, build_model_input_v5
from anatobind.v41.splits_v5 import freeze_splits_v5
from anatobind.v41.source_policy_v5 import validate_source_policy_v5
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
        "geometry_evidence": {"index_alignment": None, "physical_geometry": None},
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


SYNTH_SOURCE = {"source_id": "synthetic_source", "source_version": "v0"}


def make_evidence(keys, scope, source=SYNTH_SOURCE) -> dict:
    """PASS evidence bound to scope_hash, source and task (FX1)."""
    binding = {"scope_hash": scope.scope_hash, "source_id": source["source_id"],
               "source_version": source["source_version"], "task": gate_task_v5(scope)}
    return {key: {"status": "PASS", "sha256": hashlib.sha256(key.encode()).hexdigest(), **binding} for key in keys}


def make_approvals(scope, keys, source=None) -> dict:
    """Fixture approvals bound to scope_hash; with source, also to source_id/source_version (G-DATA)."""
    binding = {} if source is None else {"source_id": source["source_id"], "source_version": source["source_version"]}
    return {key: {"approved_by": FIXTURE_APPROVER, "approved_at": "2026-11-06T00:00:00Z",
                  "scope_hash": scope.scope_hash, **binding} for key in keys}


def make_reference_lesion(**overrides) -> ReferenceLesion:
    fields = {"patient_uid": "p1", "instance_uid": "l1", "a_reference_available": True,
              "u_reference_available": True, "r_reference_available": True, "eligible": True,
              "visibility_qualified": True, "exclusion_reason": None, "truth_relation_set": ("tissue_a",)}
    fields.update(overrides)
    return ReferenceLesion(**fields)


def make_state_reference_inputs(scope, **overrides) -> dict:
    """Default: every voxel reference tissue_a, target_a positive at (0,0,0) and (0,1,0), all reviewed/visible/in FOV.

    Overrides: review (array; makes the U annotation region_complete on path "review/r1"), u_completeness,
    u_visibility, u_reference_role, a_reference_role, a_class_scope, single_slice_readable, a_visibility,
    a_visibility_region (array; registered as "visibility/a1" and used when a_visibility is "limited"),
    a_tissue_visibility ({tissue: visibility}; replaces the single A annotation with one per tissue).
    """
    shape = (2, 2) if scope.observation_mode == "2d" else (1, 2, 2)
    review = overrides.get("review")
    visibility = overrides.get("u_visibility", "visible")
    u_annotation = {
        "class_scope": list(scope.abnormality_classes),
        "completeness": "region_complete" if review is not None else overrides.get("u_completeness", "target_complete"),
        "review_region_path": "review/r1" if review is not None else None,
        "input_visibility": visibility,
        "visibility_evidence": None if visibility == "unknown" else "synthetic",
        "reference_role": overrides.get("u_reference_role", "expert_voxel"),
    }
    sample = make_sample_v5(annotation=u_annotation)
    a_visibility = overrides.get("a_visibility", "visible")
    anatomy_annotation = copy.deepcopy(_sample()["annotation_records"][0])
    anatomy_annotation.update(annotation_uid="ann-v5-a", task="A",
                              class_scope=list(overrides.get("a_class_scope", scope.anatomy_classes)),
                              completeness="target_complete", annotation_modalities=["qdess_e1"],
                              input_visibility=a_visibility,
                              visibility_evidence=None if a_visibility == "unknown" else "synthetic",
                              visibility_region_path="visibility/a1" if a_visibility == "limited" else None,
                              reference_role=overrides.get("a_reference_role", "expert_voxel"))
    if "a_tissue_visibility" in overrides:
        # One A annotation per tissue, each with its own visibility (per-tissue A visibility, FX2).
        for name, tissue_visibility in reversed(list(overrides["a_tissue_visibility"].items())):
            per_tissue = copy.deepcopy(anatomy_annotation)
            per_tissue.update(annotation_uid=f"ann-v5-a-{name}", class_scope=[name], input_visibility=tissue_visibility,
                              visibility_evidence=None if tissue_visibility == "unknown" else "synthetic",
                              visibility_region_path=None)
            sample["annotation_records"].insert(0, per_tissue)
    else:
        sample["annotation_records"].insert(0, anatomy_annotation)
    abnormality = np.zeros((len(scope.abnormality_classes), *shape), dtype=bool)
    positive = abnormality[0].reshape(-1, 2, 2)
    positive[0, 0, 0] = positive[0, 1, 0] = True
    arrays = {
        "anatomy": np.ones(shape, dtype=np.int64),
        "abnormality": abnormality,
        "valid_fov": np.ones(shape, dtype=bool),
        "single_slice_readable": overrides.get("single_slice_readable", True),
        "review_regions": {"review/r1": review} if review is not None else {},
        "visibility_regions": ({"visibility/a1": overrides["a_visibility_region"]}
                               if "a_visibility_region" in overrides else {}),
    }
    return {"record": validate_sample_v5(sample), "arrays": arrays}


def make_source_policy(**overrides):
    """Synthetic SourcePolicy matching tests.v41.conftest._sample() and make_module_scope("U")."""
    entry = {"source_version": "v0", "source_release": "unknown", "source_dataset": "synthetic", "scope_id": "synthetic_u",
             "reference_policy_id": "synthetic_policy_v1", "model_modalities": ["qdess_e1"],
             "annotation_modalities": {"U": ["qdess_e1"]}, "index_evidence": "synthetic",
             "physical_units_policy": "trust_header", "physical_units_evidence": "synthetic"}
    entry.update(overrides)
    return validate_source_policy_v5("synthetic_source", entry)


# ---------------------------------------------------------------------------
# Run registration fixtures (plan_v5 T09a). Every value here is synthetic.

CONFIG_DIR_V5 = Path(__file__).resolve().parents[2] / "configs" / "v5"
SYNTH_ENVIRONMENT = {"python": "3.9.6", "torch": "0.0.0", "numpy": "0.0.0", "cuda": None, "devices": []}
SYNTH_CODE = {"git_commit": "0" * 40, "dirty": False}
SYNTH_CREATED_AT = "2026-11-07T00:00:00Z"

_RECIPE_VALUES = {
    "recipe_id": "synthetic_recipe_v1",
    "model": {"architecture": "resunet3d", "tasks": ["A"], "widths": [4, 8], "strides": [[1, 1, 1], [1, 2, 2]],
              "blocks_per_stage": 1},
    "seeds": {"init": 0, "sampler": 1},
    "patch_size": [4, 8, 8],
    "batch_size": 2,
    "precision": "fp32",
    "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
    "scheduler": {"name": "poly", "exponent": 0.9},
    "downsample_factor": [1, 1, 1],
    "grid_tolerance": {"atol_affine": 1e-4, "atol_spacing": 1e-4},
    "normalization": "zscore_volume",
    "sampling": {"positive_fraction": 0.5},
    "readable_roles": ["train"],
    "selection_rule": "last",
    "inference": {"window": [4, 8, 8], "stride": [2, 4, 4]},
    "checkpoint_every": 2,
    "budget": {"max_steps": 4, "max_wall_hours": 0.1},
}
_SUPERVISION_VALUES = {"policy_id": "synthetic_supervision_v1", "track": "released_reference",
                       "allow_unknown_visibility": False, "channel": "expert"}
_TASK_LOSS = {"A": {"kind": "softmax", "ce_weight": 1.0, "dice_weight": 1.0, "task_weight": 1.0},
              "U": {"kind": "sigmoid", "ce_weight": 1.0, "dice_weight": 1.0, "task_weight": 1.0}}
_OFFICIAL_NAMES = {"train": "train", "development_val": "val", "locked_test": "test"}


def load_template_v5(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR_V5 / name).read_text())


def _fill(template, values, path: str = "$"):
    """Fill every null leaf of a template; template and values must have the same keys."""
    if isinstance(template, dict):
        if not isinstance(values, dict) or set(template) != set(values):
            raise AssertionError(f"template keys differ at {path}")
        return {key: _fill(template[key], values[key], f"{path}.{key}") for key in template}
    if template is not None:
        raise AssertionError(f"template leaf is not null at {path}")
    return copy.deepcopy(values)


def _merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _trainable(scope) -> list:
    return [task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes)) if classes]


def make_recipe(**overrides) -> dict:
    """The null recipe template filled with synthetic values (A-only); dict overrides merge into nested keys."""
    return _merge(_fill(load_template_v5("run_recipe.template.yaml"), _RECIPE_VALUES), overrides)


def make_supervision(scope, **overrides) -> dict:
    """The null supervision template filled for every trainable task of the scope; dict overrides merge."""
    template = load_template_v5("supervision.template.yaml")
    values = dict(_SUPERVISION_VALUES, scope_id=scope.scope_id, reference_policy_id=scope.reference_policy_id)
    policy = _fill({key: value for key, value in template.items() if key != "tasks"}, values)
    policy["tasks"] = {task: _fill(template["tasks"][task], _TASK_LOSS[task]) for task in _trainable(scope)}
    return _merge(policy, overrides)


def make_split_records(splits) -> list:
    """One SampleV5 per entry, one patient each; the entry is the record's own split field, not the registry."""
    return [validate_sample_v5(make_sample_v5(sample_uid=f"s-{i}", split=split, patient_uid=f"synthetic:p{i:04d}",
                                              leakage_group_uid=f"synthetic:p{i:04d}"))
            for i, split in enumerate(splits)]


def make_split_registry(records, **overrides) -> dict:
    """Official-kind frozen registry grouped by leakage group.

    By default a group's official split mirrors its records' split field; other groups stay unassigned.
    Overrides: official_split, exposed_groups, lineage_id, ancestor_lineages.
    """
    groups = {record.sample_uid: record.leakage_group_uid for record in records}
    official = {r.leakage_group_uid: _OFFICIAL_NAMES[r.split] for r in records if r.split in _OFFICIAL_NAMES}
    policy = {"kind": "official", "official_split": overrides.pop("official_split", official),
              "exposed_groups": overrides.pop("exposed_groups", "excluded")}
    lineage_id = overrides.pop("lineage_id", "lin-a")
    ancestors = overrides.pop("ancestor_lineages", [])
    if overrides:
        raise TypeError(f"unknown overrides: {sorted(overrides)}")
    return freeze_splits_v5(records, groups, policy, lineage_id, ancestor_lineages=ancestors)


def make_gate_decisions(scope, run_kind: str, source=SYNTH_SOURCE) -> dict:
    """Fixture PASS decisions a run needs: G-DATA for smoke, G-BASE-A/-U per trainable task otherwise."""
    gate_ids = ["G-DATA"] if run_kind == "smoke" else [f"G-BASE-{task}" for task in _trainable(scope)]
    decisions = {}
    for gate_id in gate_ids:
        requirement = GATE_REQUIREMENTS[gate_id]
        bound = source if requirement.get("source_bound_approvals") else None
        decisions[gate_id] = evaluate_gate_v5(gate_id, scope, make_evidence(requirement["evidence"], scope, source),
                                              make_approvals(scope, requirement["approvals"], bound), source,
                                              fixture_mode=True)
    return decisions


def make_run(**overrides) -> dict:
    """A registered fixture run: an A-only baseline on the synthetic A module scope unless overridden.

    Any register_run_v5 keyword can be overridden. `scope`, `run_kind`, `lineage_id`, `parents` and `records`
    also drive the other defaults; the default split registry is frozen with the inherited ancestors.
    """
    from anatobind.v41.engine.registration_v5 import inherit_lineage_v5, register_run_v5

    scope = overrides.get("scope") or validate_scope_v5(make_module_scope("A"))
    run_kind = overrides.get("run_kind", "baseline")
    lineage_id = overrides.get("lineage_id", "lin-a")
    parents = overrides.get("parents", [])
    records = overrides.get("records") or make_split_records(["train", "development_val", "locked_test"])
    kwargs = {"run_id": "run-synthetic-0001", "lineage_id": lineage_id, "run_kind": run_kind,
              "recipe": make_recipe(model={"tasks": _trainable(scope)}), "supervision": make_supervision(scope),
              "scope": scope, "source": dict(SYNTH_SOURCE), "records": records,
              "gate_decisions": make_gate_decisions(scope, run_kind),
              "approvals": make_approvals(scope, ["resources"]) if run_kind == "smoke" else {},
              "parents": parents, "environment": dict(SYNTH_ENVIRONMENT), "code": dict(SYNTH_CODE),
              "created_at": SYNTH_CREATED_AT, "fixture_mode": True}
    kwargs.update(overrides)
    if "split_registry" not in overrides:
        kwargs["split_registry"] = make_split_registry(records, lineage_id=lineage_id,
                                                       ancestor_lineages=inherit_lineage_v5(parents, lineage_id))
    return register_run_v5(**kwargs)


def make_parent_entry(run, records, checkpoint_sha256: str = "c" * 64):
    """(parent entry with the run's exposure file, `records` with that exposure merged) for a run that make_run
    registered on `records` with its default split registry."""
    from anatobind.v41.engine.registration_v5 import merge_run_exposure_v5, run_exposure_file_v5

    registry = make_split_registry(records, lineage_id=run["lineage_id"], ancestor_lineages=run["ancestor_lineages"])
    exposure = run_exposure_file_v5(run, registry)
    return ({"run": run, "checkpoint_sha256": checkpoint_sha256, "exposure": exposure},
            merge_run_exposure_v5(records, [exposure]))


def make_dev_split(scope, sample_uids, roles=None, readable_roles=("train", "development_val")):
    """(registry, run): a frozen registry assigning each uid development_val unless `roles` maps it to another
    split (one train sample "s-train" is always added), and a baseline run on `scope` that reads `readable_roles`.
    The same uids and roles give the same split_sha256."""
    roles = dict(roles or {})
    entries = [("s-train", "train")] + [(uid, roles.get(uid, "development_val")) for uid in sample_uids]
    records = [validate_sample_v5(make_sample_v5(sample_uid=uid, split=split, patient_uid=f"synthetic:dev{i:04d}",
                                                 leakage_group_uid=f"synthetic:dev{i:04d}"))
               for i, (uid, split) in enumerate(entries)]
    registry = make_split_registry(records)
    run = make_run(scope=scope, records=records, split_registry=registry,
                   recipe=make_recipe(model={"tasks": _trainable(scope)}, readable_roles=list(readable_roles)))
    return registry, run


# ---------------------------------------------------------------------------
# Volume, sample and model fixtures (plan_v5 T09b). Arrays are (x, y, z) voxel order.

SYNTH_LABEL_MAPS = {
    "synthetic_a_values_v1": {"encoding": "values", "classes": {"tissue_a": 1, "tissue_b": 2}},
    "synthetic_a_channels_v1": {"encoding": "channels", "classes": {"tissue_a": 0, "tissue_b": 1}},
    "synthetic_u_values_v1": {"encoding": "values", "classes": {"target_a": 1}},
}
_MET_TYPES = {"uint8": "MET_UCHAR", "int8": "MET_CHAR", "uint16": "MET_USHORT", "int16": "MET_SHORT",
              "uint32": "MET_UINT", "int32": "MET_INT", "float32": "MET_FLOAT", "float64": "MET_DOUBLE"}


def write_nifti(path, array, affine, slope=None) -> None:
    array = np.asarray(array)
    image = nib.Nifti1Image(array, np.asarray(affine, dtype=float))
    if slope is not None:
        image.header.set_slope_inter(slope, 0.0)
        image.header.set_data_dtype(array.dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def write_skm_h5(path, echo1, echo2, seg) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("echo1", data=echo1)
        handle.create_dataset("echo2", data=echo2)
        handle.create_dataset("seg", data=seg)


def write_mha(path, array, spacing, direction, offset, compressed, msb=False, **header) -> None:
    """MetaImage writer: `array` is (x, y, z); row j of `direction` is the direction of axis j (MetaIO).

    Keyword `header` entries replace or add keys; a value of None drops the key.
    """
    array = np.asarray(array)
    order = ">" if msb else "<"
    raw = np.ascontiguousarray(array.transpose(2, 1, 0)).astype(array.dtype.newbyteorder(order)).tobytes()
    data = zlib.compress(raw) if compressed else raw
    fields = {"ObjectType": "Image", "NDims": "3", "BinaryData": "True",
              "BinaryDataByteOrderMSB": "True" if msb else "False",
              "CompressedData": "True" if compressed else "False",
              "TransformMatrix": " ".join(repr(float(v)) for v in np.asarray(direction, dtype=float).ravel()),
              "Offset": " ".join(repr(float(v)) for v in offset),
              "ElementSpacing": " ".join(repr(float(v)) for v in spacing),
              "DimSize": " ".join(str(n) for n in array.shape),
              "ElementType": _MET_TYPES[array.dtype.name], "ElementDataFile": "LOCAL"}
    if compressed:
        fields["CompressedDataSize"] = str(len(data))
    for key, value in header.items():
        if value is None:
            fields.pop(key, None)
        else:
            fields[key] = str(value)
    datafile = fields.pop("ElementDataFile", None)
    lines = [f"{key} = {value}" for key, value in fields.items()]
    if datafile is not None:
        lines.append(f"ElementDataFile = {datafile}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(("\n".join(lines) + "\n").encode("ascii") + data)


def write_volume_case(root, scope, uid: str = "case-1", shape=(8, 8, 8), label=None, **record_overrides):
    """Synthetic NIfTI case under `root` for a module scope; returns its validated SampleV5.

    Channel i is filled with i+1 plus a ramp so each modality is recognisable. The default label puts
    tissue_a / target_a in one block (A) or one voxel (U). `label=False` writes no label file.
    """
    root = Path(root)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    ramp = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 100.0
    modalities = list(scope.input_modalities)
    paths = []
    for i, name in enumerate(modalities):
        paths.append(f"img/{uid}_{name}.nii.gz")
        write_nifti(root / paths[-1], ramp + float(i + 1) * 10.0, affine)
    task = "A" if scope.anatomy_classes else "U"
    if label is None:
        label = np.zeros(shape, dtype=np.int16)
        if task == "A":
            label[:4] = 1
            label[4:6] = 2
        else:
            label[2, 2, 2] = 1
    label_path = f"lab/{uid}.nii.gz"
    if label is not False:
        write_nifti(root / label_path, np.asarray(label, dtype=np.int16), affine)
    annotation = {"annotation_uid": f"ann-{uid}", "task": task,
                  "class_scope": list(scope.anatomy_classes if task == "A" else scope.abnormality_classes),
                  "path": label_path, "annotation_modalities": modalities[:1],
                  "label_map_version": "synthetic_a_values_v1" if task == "A" else "synthetic_u_values_v1"}
    fields = {"sample_uid": uid, "patient_uid": f"synthetic:{uid}", "leakage_group_uid": f"synthetic:{uid}",
              "image_paths": paths, "modality_names": modalities, "modality_is_synthetic": [False] * len(paths),
              "available_modality_mask": [True] * len(paths), "model_modalities": modalities,
              "active_scope_id": scope.scope_id, "reference_policy_id": scope.reference_policy_id,
              "image_geometry": {"shape": list(shape), "spacing_mm": [1.0, 1.0, 2.0], "affine": affine.tolist(),
                                 "axis_codes": ["R", "A", "S"], "units": "mm"}}
    fields.update(record_overrides)
    return validate_sample_v5(make_sample_v5(annotation=annotation, **fields))


def make_training_sample_v5(scope, uid: str, shape=(6, 6, 6), seed: int = 0):
    """In-memory TrainingSampleV5 with random channels and labels, eligible everywhere."""
    from anatobind.v41.engine.data_v5 import ObservationV5, TrainingSampleV5

    rng = np.random.default_rng(seed)
    channels = {name: rng.standard_normal(shape).astype(np.float32) for name in scope.input_modalities}
    grid = {"shape": list(shape), "axis_order": "xyz", "spacing_mm": [1.0, 1.0, 1.0],
            "affine": np.eye(4).tolist(), "units_reliable": True, "transform_chain": []}
    observation = ObservationV5(sample_uid=uid, patient_uid=f"synthetic:{uid}", channels=channels,
                                valid_fov=np.ones(shape, dtype=bool), grid=grid, to_original=[],
                                original_shape=tuple(shape), original_header=None)
    targets, e_pos, e_neg = {}, {}, {}
    if scope.anatomy_classes:
        k = len(scope.anatomy_classes)
        targets["A"] = rng.integers(0, k + 1, size=shape).astype(np.int64)
        e_pos["A"] = np.ones((k, *shape), dtype=bool)
        e_neg["A"] = np.ones((k, *shape), dtype=bool)
    if scope.abnormality_classes:
        c = len(scope.abnormality_classes)
        targets["U"] = rng.random((c, *shape)) < 0.2
        e_pos["U"] = np.ones((c, *shape), dtype=bool)
        e_neg["U"] = np.ones((c, *shape), dtype=bool)
    return TrainingSampleV5(observation=observation, targets=targets, e_pos=e_pos, e_neg=e_neg)


class TinyPerceptionModel(PerceptionModelV5):
    """One 3x3x3 convolution per enabled head (A: 1+K channels, U: C_U channels); counts forward calls."""

    def __init__(self, scope, tasks):
        super().__init__(scope, tasks)
        channels = len(scope.input_modalities)
        self.heads = torch.nn.ModuleDict()
        if "A" in self.tasks:
            self.heads["A"] = torch.nn.Conv3d(channels, 1 + len(scope.anatomy_classes), 3, padding=1)
        if "U" in self.tasks:
            self.heads["U"] = torch.nn.Conv3d(channels, len(scope.abnormality_classes), 3, padding=1)
        self.calls = 0

    def _forward_v5(self, batch):
        self.calls += 1
        return ModelOutputV5(anatomy_logits=self.heads["A"](batch.image) if "A" in self.heads else None,
                             abnormality_logits=self.heads["U"](batch.image) if "U" in self.heads else None)


# ---------------------------------------------------------------------------
# Inference and output fixtures (plan_v5 T09c)


def make_acceptance_policy(scope, **overrides) -> dict:
    """A synthetic engineering acceptance policy for `scope`; tasks the scope lacks keep null values."""
    has_a, has_u = bool(scope.anatomy_classes), bool(scope.abnormality_classes)
    policy = {"policy_id": "synthetic_acceptance_v1", "policy_role": "engineering",
              "anatomy": {"min_confidence": 0.5 if has_a else None},
              "abnormality": {"thresholds": {c: 0.5 for c in scope.abnormality_classes} if has_u else None,
                              "reject_band": {c: "none" for c in scope.abnormality_classes} if has_u else None},
              "instances": {"connectivity": 26 if has_u else None, "min_voxels": 1 if has_u else None}}
    return _merge(policy, overrides)


def make_model_input_v5(scope, shape, channels=None, to_original=()):
    """A B=1 ModelInputV5 on an identity (x, y, z) grid; channels default to zeros in scope order."""
    shape = tuple(shape)
    channels = channels or {name: np.zeros(shape, dtype=np.float32) for name in scope.input_modalities}
    grid = make_original_geometry(shape)
    grid["transform_chain"] = list(to_original)
    return build_model_input_v5(channels, scope, np.ones(shape, dtype=bool), grid, list(to_original))


def make_original_geometry(shape) -> dict:
    ndim = len(shape)
    return {"shape": list(shape), "axis_order": "xyz"[:ndim], "spacing_mm": [1.0] * ndim,
            "affine": np.eye(4).tolist() if ndim == 3 else None, "units_reliable": True, "transform_chain": []}


class StubPerceptionModel(PerceptionModelV5):
    """Logits are fixed functions of each window batch; records every window's transform chain."""

    def __init__(self, scope, tasks, anatomy_fn=None, abnormality_fn=None):
        super().__init__(scope, tasks)
        self.anatomy_fn, self.abnormality_fn = anatomy_fn, abnormality_fn
        self.chains = []
        self.calls = 0

    def _forward_v5(self, batch):
        self.calls += 1
        self.chains.append(batch.to_original[0])
        return ModelOutputV5(anatomy_logits=None if self.anatomy_fn is None else self.anatomy_fn(batch),
                             abnormality_logits=None if self.abnormality_fn is None else self.abnormality_fn(batch))


# ---------------------------------------------------------------------------
# Module evaluation fixtures (plan_v5 T10)

_METRIC_VALUES = {"metrics": ["dice", "hd95", "nsd", "lesion_detection"], "surface_tolerance_mm": 1.0,
                  "iou_match_threshold": 0.3, "fp_budget_per_scan": None, "class_aware": True,
                  "size_strata_voxels": [10]}


def make_metric_policy(**overrides) -> dict:
    """The null metric template filled with synthetic development values; dict overrides merge."""
    return _merge(_fill(load_template_v5("metrics.template.yaml"), _METRIC_VALUES), overrides)


def make_module_reference(scope, patient_uid: str, shape, anatomy_truth=None, abnormality_truth=None,
                          units_reliable: bool = True, **overrides) -> dict:
    """A reference on the identity grid of make_original_geometry, evaluable everywhere; its split comes from the
    frozen registry it is frozen against (make_dev_split), never from the reference."""
    shape = tuple(shape)
    geometry = make_original_geometry(shape)
    geometry["units_reliable"] = units_reliable
    tasks = {}
    if anatomy_truth is not None:
        tasks["A"] = {"truth": np.array(anatomy_truth, dtype=np.int16),
                      "eligible": np.ones((len(scope.anatomy_classes), *shape), dtype=bool)}
    if abnormality_truth is not None:
        tasks["U"] = {"truth": np.array(abnormality_truth, dtype=bool),
                      "eligible": np.ones((len(scope.abnormality_classes), *shape), dtype=bool)}
    reference = {"patient_uid": patient_uid, "geometry": geometry, "tasks": tasks}
    reference.update(overrides)
    return reference


def make_scored_output(scope, uid: str, shape, anatomy_map=None, abnormality_map=None, abnormality_accepted=None,
                       policy_id: str = "synthetic_acceptance_v1", units_reliable: bool = True):
    """A validated ModuleOutput on that grid; anatomy -1 marks rejected voxels, U is accepted unless given."""
    geometry = make_original_geometry(tuple(shape))
    geometry["units_reliable"] = units_reliable
    enabled = [task for task, value in (("A", anatomy_map), ("U", abnormality_map)) if value is not None]
    acceptance, reason = {}, {}
    if anatomy_map is not None:
        anatomy_map = np.array(anatomy_map, dtype=np.int16)
        acceptance["anatomy"] = anatomy_map != -1
        reason["anatomy"] = _reasons(acceptance["anatomy"], "anatomy_uncertain")
    if abnormality_map is not None:
        abnormality_map = np.array(abnormality_map, dtype=bool)
        accepted = np.ones(abnormality_map.shape, dtype=bool) if abnormality_accepted is None \
            else np.array(abnormality_accepted, dtype=bool)
        acceptance["abnormality"] = accepted
        reason["abnormality"] = _reasons(accepted, "abnormality_uncertain")
    output = {"output_kind": "module", "output_version": "v5.1", "observation_uid": uid,
              "mode": scope.observation_mode, "scope_id": scope.scope_id, "scope_hash": scope.scope_hash,
              "checkpoint_sha256": "b" * 64, "enabled_tasks": enabled,
              "disabled_tasks": [task for task in ("A", "U", "S", "R") if task not in enabled],
              "anatomy_map": anatomy_map, "abnormality_map": abnormality_map, "acceptance": acceptance,
              "reason": reason, "geometry": geometry,
              "provenance": {"input_sha256": "a" * 64, "model_sha256": "b" * 64, "output_policy_version": policy_id}}
    return validate_module_output_v5(output, scope)


# ---------------------------------------------------------------------------
# Unified model (Proposal_v5 §0.6): a synthetic plan and its registered member runs

_UNIFIED_MODEL = {"architecture": "resunet3d", "widths": [4, 8], "strides": [[1, 1, 1], [1, 2, 2]],
                  "blocks_per_stage": 1, "stem_channels": 3}
SYNTH_BOX_SPEC = {"kind": "box_mil", "topk_fraction": 0.5, "positive_weight": 1.0, "negative_weight": 1.0,
                  "box_policy_sha256": "p" * 64}


def make_unified_plan(scopes: dict, **overrides) -> dict:
    """A synthetic unified plan with one member per scope; a joint scope's member is box-supervised and reports
    hosts. Dict overrides merge into nested keys."""
    members = {}
    for pid, scope in scopes.items():
        joint = scope.scope_kind == "joint"
        members[pid] = {
            "scope_id": scope.scope_id, "scope_hash": scope.scope_hash, "source": dict(SYNTH_SOURCE),
            "lineage_id": f"lin-{pid}", "tasks": _trainable(scope), "patch_size": [4, 8, 8], "batch_size": 1,
            "downsample_factor": [1, 1, 1], "grid_tolerance": {"atol_affine": 1e-4, "atol_spacing": 1e-4},
            "normalization": "zscore_volume", "sampling": {"positive_fraction": 0.5},
            "inference": {"window": [4, 8, 8], "stride": [4, 8, 8]}, "loss_weight": 1.0,
            "supervision": make_supervision(scope), "box_supervision": dict(SYNTH_BOX_SPEC) if joint else None,
            "acceptance_policy": make_acceptance_policy(scope), "metric_policy": make_metric_policy(),
            "finding_policy": {"host": "majority_accepted_anatomy" if joint else "none",
                               "iobb_thresholds": [0.1, 0.25, 0.5] if joint else None}}
    plan = {"plan_id": "synthetic_unified_v1",
            "shared": {"model": copy.deepcopy(_UNIFIED_MODEL), "seeds": {"init": 0, "sampler": 1},
                       "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
                       "scheduler": {"name": "poly", "exponent": 0.9}, "precision": "fp32", "selection_rule": "last",
                       "checkpoint_every": 2, "budget": {"max_steps": 4, "max_wall_hours": 1.0}},
            "members": members,
            "smoke": {"steps": 4, "pause_at": 2, "checkpoint_every": 2, "subset_count": {pid: 1 for pid in scopes},
                      "subset_salt": "synthetic-unified",
                      "tolerances": {"loss_atol": 1e-4, "loss_rtol": 1e-3, "param_rtol": 1e-3}, "max_wall_hours": 1.0},
            "budget": {"max_wall_hours": 2.0}}
    return _merge(plan, overrides)


def make_unified_members(plan: dict, scopes: dict, run_id: str, run_kind: str = "baseline"):
    """(member_runs, registries, records), each keyed by protocol: every member gets its own records
    (two train, one development_val), frozen registry and registered fixture run `<run_id>.<protocol>`."""
    from anatobind.v41.engine.registration_v5 import register_run_v5
    from anatobind.v41.engine.unified_registration_v5 import member_recipe_v5

    member_runs, registries, records = {}, {}, {}
    for pid, scope in scopes.items():
        member = plan["members"][pid]
        records[pid] = make_split_records(["train", "train", "development_val"])
        registries[pid] = make_split_registry(records[pid], lineage_id=member["lineage_id"])
        member_runs[pid] = register_run_v5(
            run_id=f"{run_id}.{pid}", lineage_id=member["lineage_id"], run_kind=run_kind,
            recipe=member_recipe_v5(plan, pid, run_kind), supervision=member["supervision"], scope=scope,
            source=dict(SYNTH_SOURCE), split_registry=registries[pid], records=records[pid],
            gate_decisions=make_gate_decisions(scope, run_kind),
            approvals=make_approvals(scope, ["resources"]) if run_kind == "smoke" else {}, parents=[],
            environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE), created_at=SYNTH_CREATED_AT, fixture_mode=True)
    return member_runs, registries, records
