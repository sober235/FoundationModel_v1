"""v5.1 record contracts: ActiveScope, SampleV5/AnnotationV5 (Proposal_v5 §2.2, §4.1).

Errors are V5Error(code, path); the message never contains input values,
paths or case identifiers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .schema import AnnotationRecord, SampleRecord, SchemaError, _resolve_ref, load_schema, validate_against

SCHEMA_VERSION_V5 = "v5.1"
SCHEMA_DIR_V5 = Path(__file__).resolve().parents[2] / "schemas" / "v5"


class V5Error(ValueError):
    def __init__(self, code: str, path: str = "$"):
        self.code = code
        self.path = path
        super().__init__(f"{code} at {path}")


def canonical_sha256(value: object) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        raise V5Error("NON_JSON_VALUE") from None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}|{list(array.shape)}|".encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


UNKNOWN_KEY = "<unknown>"


def _raise_redacted_unknown(instance: object, schema: dict, root: dict, path: str) -> None:
    """Re-walk the schema to report an unknown key at its parent path without the key text (it may be a case ID)."""
    if "$ref" in schema:
        schema = {**_resolve_ref(schema["$ref"], root), **{k: v for k, v in schema.items() if k != "$ref"}}
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties", True) is False and any(key not in properties for key in instance):
            raise V5Error("UNKNOWN_FIELD", f"{path}.{UNKNOWN_KEY}")
        for name, sub in properties.items():
            if name in instance:
                _raise_redacted_unknown(instance[name], sub, root, f"{path}.{name}")
    elif isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(instance):
            _raise_redacted_unknown(item, schema["items"], root, f"{path}[{i}]")


def _check_schema(record: object, name: str, path: str = "$") -> None:
    try:
        schema = load_schema(name, SCHEMA_DIR_V5)
        validate_against(record, schema, path=path)
    except SchemaError as error:
        if error.code == "UNKNOWN_FIELD":
            _raise_redacted_unknown(record, schema, schema, path)
            raise V5Error("UNKNOWN_FIELD", path) from None
        raise V5Error(error.code, error.path) from None


def _require_dict(record: object, path: str = "$") -> None:
    if not isinstance(record, dict):
        raise V5Error("INVALID_TYPE", path)


@dataclass(frozen=True)
class ActiveScope:
    schema_version: str
    scope_id: str
    scope_kind: str
    anatomy_classes: list[str]
    state_anatomy_classes: list[str]
    abnormality_classes: list[str]
    relation_type: str | None
    relation_predicate: str | None
    relation_contract: dict | None
    input_modalities: list[str]
    observation_mode: str
    normal_claim_scope: str
    reference_policy_id: str

    @property
    def scope_hash(self) -> str:
        return canonical_sha256(asdict(self))

    def tissue_id(self, name: str) -> int:
        if name not in self.anatomy_classes:
            raise V5Error("UNKNOWN_CLASS", "$.anatomy_classes")
        return self.anatomy_classes.index(name) + 1

    @property
    def state_tissue_ids(self) -> tuple[int, ...]:
        return tuple(self.tissue_id(name) for name in self.state_anatomy_classes)


def validate_scope_v5(record: dict) -> ActiveScope:
    _require_dict(record)
    _check_schema(record, "scope")
    for name in ("anatomy_classes", "state_anatomy_classes", "abnormality_classes"):
        if len(set(record[name])) != len(record[name]):
            raise V5Error("DUPLICATE_CLASS", f"$.{name}")
    if record["scope_kind"] == "joint":
        if not record["abnormality_classes"]:
            raise V5Error("JOINT_REQUIRES_U", "$.abnormality_classes")
        if not record["state_anatomy_classes"]:
            raise V5Error("STATE_CLASSES_REQUIRED", "$.state_anatomy_classes")
        if not set(record["state_anatomy_classes"]) <= set(record["anatomy_classes"]):
            raise V5Error("STATE_CLASSES_NOT_SUBSET", "$.state_anatomy_classes")
        if record["relation_contract"] is None:
            raise V5Error("RELATION_CONTRACT_REQUIRED", "$.relation_contract")
        if record["relation_predicate"] != record["relation_contract"]["predicate"]:
            raise V5Error("PREDICATE_MISMATCH", "$.relation_predicate")
    else:
        if record["state_anatomy_classes"]:
            raise V5Error("MODULE_SCOPE_CANNOT_CLAIM_STATE", "$.state_anatomy_classes")
        for name in ("relation_contract", "relation_predicate", "relation_type"):
            if record[name] is not None:
                raise V5Error("MODULE_SCOPE_CANNOT_CLAIM_RELATION", f"$.{name}")
        if not record["anatomy_classes"] and not record["abnormality_classes"]:
            raise V5Error("EMPTY_SCOPE", "$")
    return ActiveScope(**record)


@dataclass(frozen=True)
class AnnotationV5(AnnotationRecord):
    annotation_modalities: list[str]
    visibility_region_path: str | None


@dataclass(frozen=True)
class SampleV5(SampleRecord):
    annotation_records: list[AnnotationV5]
    active_scope_id: str
    reference_policy_id: str
    model_modalities: list[str]
    index_alignment_status: str
    physical_geometry_status: str
    geometry_evidence: dict


def _build(cls, record: dict):
    names = set(cls.__dataclass_fields__)
    return cls(**{key: value for key, value in record.items() if key in names})


def _validate_annotation_v5(record: dict, path: str) -> AnnotationV5:
    _require_dict(record, path)
    _check_schema(record, "annotation", path)
    if record["input_visibility"] != "unknown" and not record["visibility_evidence"]:
        raise V5Error("VISIBILITY_EVIDENCE_REQUIRED", f"{path}.visibility_evidence")
    if record["input_visibility"] == "limited" and not record["visibility_region_path"]:
        raise V5Error("LIMITED_VISIBILITY_REGION_REQUIRED", f"{path}.visibility_region_path")
    return _build(AnnotationV5, record)


def validate_sample_v5(record: dict) -> SampleV5:
    _require_dict(record)
    if record.get("schema_version") != SCHEMA_VERSION_V5:
        raise V5Error("SCHEMA_VERSION_MISMATCH", "$.schema_version")
    _check_schema(record, "sample")
    lists = ("modality_names", "modality_is_synthetic", "available_modality_mask")
    if {len(record[name]) for name in lists} != {len(record["image_paths"])}:
        raise V5Error("MODALITY_LENGTH_MISMATCH", "$.image_paths")
    if not set(record["model_modalities"]) <= set(record["modality_names"]):
        raise V5Error("MODEL_MODALITY_UNKNOWN", "$.model_modalities")
    for status, key in (("index_alignment_status", "index_alignment"), ("physical_geometry_status", "physical_geometry")):
        if record[status] == "verified" and not record["geometry_evidence"][key]:
            raise V5Error("GEOMETRY_EVIDENCE_MISSING", f"$.geometry_evidence.{key}")
    body = dict(record)
    body["annotation_records"] = [
        _validate_annotation_v5(item, f"$.annotation_records[{i}]")
        for i, item in enumerate(record["annotation_records"])
    ]
    return _build(SampleV5, body)


# ---------------------------------------------------------------------------
# PerceptionOutput / ModuleOutput (Proposal_v5 §2.1)

OUTPUT_VERSION_V5 = "v5.1"
SPATIAL_NDIM = {"2d": 2, "3d": 3}
INSTANCE_KEYS = ("instance_uid", "class_name", "mask", "bbox", "candidate_set_hash", "accepted")
RELATION_KEYS = ("instance_uid", "predicate", "tissue_ids", "status", "accepted", "completion_role")
RELATION_STATUSES = ("bound", "no_host", "unknown", "out_of_scope")
GEOMETRY_KEYS = ("shape", "axis_order", "spacing_mm", "affine", "units_reliable", "transform_chain")
PROVENANCE_KEYS = ("input_sha256", "model_sha256", "output_policy_version")
ACCEPTANCE_KEYS = ("anatomy", "abnormality", "state", "relations")
MODULE_TASK_KEYS = {"A": "anatomy", "U": "abnormality"}
_REASON_LIMIT = 32
_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class PerceptionOutput:
    output_kind: str
    output_version: str
    observation_uid: str
    mode: str
    scope_id: str
    scope_hash: str
    checkpoint_sha256: str
    state_policy_hash: str
    anatomy_map: np.ndarray
    abnormality_map: np.ndarray
    tissue_state_map: np.ndarray
    abnormal_instances: list[dict]
    tissue_relations: list[dict]
    acceptance: dict
    reason: dict
    geometry: dict
    provenance: dict


@dataclass(frozen=True)
class ModuleOutput:
    output_kind: str
    output_version: str
    observation_uid: str
    mode: str
    scope_id: str
    scope_hash: str
    checkpoint_sha256: str
    enabled_tasks: list[str]
    disabled_tasks: list[str]
    anatomy_map: np.ndarray | None
    abnormality_map: np.ndarray | None
    acceptance: dict
    reason: dict
    geometry: dict
    provenance: dict


def _check_keys(record: object, expected, path: str) -> None:
    _require_dict(record, path)
    missing = [key for key in expected if key not in record]
    if missing:
        raise V5Error("MISSING_FIELD", f"{path}.{missing[0]}")
    if any(key not in expected for key in record):
        raise V5Error("UNKNOWN_FIELD", f"{path}.{UNKNOWN_KEY}")


def _check_hash(value: object, path: str) -> None:
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
        raise V5Error("INVALID_HASH", path)


def _check_array(value: object, path: str, kind: str, shape: tuple) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise V5Error("ARRAY_REQUIRED", path)
    if kind == "bool" and value.dtype != np.bool_:
        raise V5Error("DTYPE_MISMATCH", path)
    if kind == "int" and (value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.integer)):
        raise V5Error("DTYPE_MISMATCH", path)
    if kind == "uint8" and value.dtype != np.uint8:
        raise V5Error("DTYPE_MISMATCH", path)
    if value.shape != shape:
        raise V5Error("GRID_MISMATCH", path)
    return value


def _check_identity(output: dict, scope: ActiveScope, kind: str) -> None:
    if output["output_kind"] != kind:
        raise V5Error("OUTPUT_KIND_MISMATCH", "$.output_kind")
    if output["output_version"] != OUTPUT_VERSION_V5:
        raise V5Error("OUTPUT_VERSION_MISMATCH", "$.output_version")
    expected_kind = "joint" if kind == "joint" else "module"
    if scope.scope_kind != expected_kind or output["scope_id"] != scope.scope_id \
            or output["scope_hash"] != scope.scope_hash:
        raise V5Error("SCOPE_MISMATCH", "$.scope_hash")
    if output["mode"] != scope.observation_mode:
        raise V5Error("MODE_MISMATCH", "$.mode")
    if not isinstance(output["observation_uid"], str) or not output["observation_uid"]:
        raise V5Error("INVALID_TYPE", "$.observation_uid")
    _check_hash(output["checkpoint_sha256"], "$.checkpoint_sha256")


def _spatial_shape(anatomy_or_abnormality: object, mode: str, path: str, leading: int) -> tuple:
    if not isinstance(anatomy_or_abnormality, np.ndarray):
        raise V5Error("ARRAY_REQUIRED", path)
    if anatomy_or_abnormality.ndim != SPATIAL_NDIM[mode] + leading:
        raise V5Error("GRID_MISMATCH", path)
    return tuple(anatomy_or_abnormality.shape[leading:])


def _check_geometry(geometry: object, spatial: tuple) -> None:
    _check_keys(geometry, GEOMETRY_KEYS, "$.geometry")
    if list(geometry["shape"]) != list(spatial):
        raise V5Error("GRID_MISMATCH", "$.geometry.shape")


def _check_anatomy_values(anatomy_map: np.ndarray, accepted: np.ndarray, n_tissues: int) -> None:
    if anatomy_map.size and (anatomy_map.min() < -1 or anatomy_map.max() > n_tissues):
        raise V5Error("ANATOMY_ID_OUT_OF_RANGE", "$.anatomy_map")
    if not np.array_equal(anatomy_map == -1, ~accepted):
        raise V5Error("ANATOMY_REJECTION_MISMATCH", "$.anatomy_map")


def _check_reason_pair(accepted: np.ndarray, reason: np.ndarray, path: str) -> None:
    if reason.size and reason.max() >= _REASON_LIMIT:
        raise V5Error("REASON_CODE_INVALID", f"$.reason.{path}")
    if not np.array_equal(accepted, reason == 0):
        raise V5Error("REASON_ACCEPTANCE_MISMATCH", f"$.reason.{path}")


def _check_provenance(provenance: object, checkpoint_sha256: str) -> None:
    _check_keys(provenance, PROVENANCE_KEYS, "$.provenance")
    _check_hash(provenance["input_sha256"], "$.provenance.input_sha256")
    _check_hash(provenance["model_sha256"], "$.provenance.model_sha256")
    if provenance["model_sha256"] != checkpoint_sha256:
        raise V5Error("PROVENANCE_MISMATCH", "$.provenance.model_sha256")


def _check_instances(instances: object, scope: ActiveScope, spatial: tuple) -> set:
    if not isinstance(instances, list):
        raise V5Error("INVALID_TYPE", "$.abnormal_instances")
    uids = set()
    for i, item in enumerate(instances):
        path = f"$.abnormal_instances[{i}]"
        _check_keys(item, INSTANCE_KEYS, path)
        if item["instance_uid"] in uids:
            raise V5Error("DUPLICATE_INSTANCE_ID", f"{path}.instance_uid")
        uids.add(item["instance_uid"])
        if item["class_name"] not in scope.abnormality_classes:
            raise V5Error("UNKNOWN_CLASS", f"{path}.class_name")
        _check_array(item["mask"], f"{path}.mask", "bool", spatial)
        bbox = item["bbox"]
        n = len(spatial)
        if not (isinstance(bbox, list) and len(bbox) == 2 * n
                and all(isinstance(v, int) and not isinstance(v, bool) for v in bbox)
                and all(0 <= bbox[d] < bbox[d + n] <= spatial[d] for d in range(n))):
            raise V5Error("INVALID_BBOX", f"{path}.bbox")
        _check_hash(item["candidate_set_hash"], f"{path}.candidate_set_hash")
        if not isinstance(item["accepted"], bool):
            raise V5Error("INVALID_TYPE", f"{path}.accepted")
    return uids


def _check_relations(relations: object, instance_uids: set, scope: ActiveScope,
                     acceptance: dict, reason: dict) -> None:
    if not isinstance(relations, list):
        raise V5Error("INVALID_TYPE", "$.tissue_relations")
    contract = scope.relation_contract
    seen = {}
    for i, item in enumerate(relations):
        path = f"$.tissue_relations[{i}]"
        _check_keys(item, RELATION_KEYS, path)
        uid = item["instance_uid"]
        if uid not in instance_uids:
            raise V5Error("UNKNOWN_INSTANCE", f"{path}.instance_uid")
        if uid in seen:
            raise V5Error("DUPLICATE_INSTANCE_ID", f"{path}.instance_uid")
        if item["predicate"] != contract["predicate"] or item["completion_role"] != contract["completion_role"]:
            raise V5Error("PREDICATE_MISMATCH", f"{path}.predicate")
        tissue_ids = item["tissue_ids"]
        if item["status"] not in RELATION_STATUSES or not isinstance(tissue_ids, list) \
                or (item["status"] == "bound") != bool(tissue_ids):
            raise V5Error("RELATION_STATUS_INVALID", f"{path}.status")
        if any(not isinstance(t, int) or isinstance(t, bool) or not 1 <= t <= len(scope.anatomy_classes)
               for t in tissue_ids):
            raise V5Error("ANATOMY_ID_OUT_OF_RANGE", f"{path}.tissue_ids")
        if not isinstance(item["accepted"], bool):
            raise V5Error("INVALID_TYPE", f"{path}.accepted")
        seen[uid] = item["accepted"]
    accepted, reasons = acceptance["relations"], reason["relations"]
    if not (isinstance(accepted, dict) and isinstance(reasons, dict)) or set(accepted) != set(seen) \
            or set(reasons) != set(seen):
        raise V5Error("REASON_ACCEPTANCE_MISMATCH", "$.acceptance.relations")
    for uid, flag in seen.items():
        code = reasons[uid]
        if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code < _REASON_LIMIT:
            raise V5Error("REASON_CODE_INVALID", "$.reason.relations")
        if accepted[uid] is not flag or flag != (code == 0):
            raise V5Error("REASON_ACCEPTANCE_MISMATCH", "$.acceptance.relations")


def validate_perception_output_v5(output: dict, scope: ActiveScope) -> PerceptionOutput:
    from .state_rule_v5 import derive_state_map_v5

    _check_keys(output, tuple(PerceptionOutput.__dataclass_fields__), "$")
    _check_identity(output, scope, "joint")
    _check_hash(output["state_policy_hash"], "$.state_policy_hash")

    spatial = _spatial_shape(output["anatomy_map"], scope.observation_mode, "$.anatomy_map", 0)
    channels = (len(scope.abnormality_classes), *spatial)
    anatomy = _check_array(output["anatomy_map"], "$.anatomy_map", "int", spatial)
    abnormality = _check_array(output["abnormality_map"], "$.abnormality_map", "bool", channels)
    state = _check_array(output["tissue_state_map"], "$.tissue_state_map", "uint8", spatial)
    _check_keys(output["acceptance"], ACCEPTANCE_KEYS, "$.acceptance")
    _check_keys(output["reason"], ACCEPTANCE_KEYS, "$.reason")
    shapes = {"anatomy": spatial, "abnormality": channels, "state": spatial}
    accepted = {k: _check_array(output["acceptance"][k], f"$.acceptance.{k}", "bool", s) for k, s in shapes.items()}
    reasons = {k: _check_array(output["reason"][k], f"$.reason.{k}", "uint8", s) for k, s in shapes.items()}
    _check_geometry(output["geometry"], spatial)

    _check_anatomy_values(anatomy, accepted["anatomy"], len(scope.anatomy_classes))
    if state.size and state.max() > 2:
        raise V5Error("STATE_CODE_INVALID", "$.tissue_state_map")
    for key in shapes:
        _check_reason_pair(accepted[key], reasons[key], key)
    if not np.array_equal(accepted["state"], state != 2):
        raise V5Error("REASON_ACCEPTANCE_MISMATCH", "$.acceptance.state")

    expected_state, expected_reason = derive_state_map_v5(
        anatomy, accepted["anatomy"], reasons["anatomy"], abnormality, accepted["abnormality"],
        reasons["abnormality"], scope.state_tissue_ids)
    if not (np.array_equal(state, expected_state) and np.array_equal(reasons["state"], expected_reason)):
        raise V5Error("STATE_COMPOSITION_VIOLATION", "$.tissue_state_map")

    instance_uids = _check_instances(output["abnormal_instances"], scope, spatial)
    _check_relations(output["tissue_relations"], instance_uids, scope, output["acceptance"], output["reason"])
    _check_provenance(output["provenance"], output["checkpoint_sha256"])
    return PerceptionOutput(**output)


def validate_module_output_v5(output: dict, scope: ActiveScope) -> ModuleOutput:
    _check_keys(output, tuple(ModuleOutput.__dataclass_fields__), "$")
    _check_identity(output, scope, "module")
    enabled, disabled = output["enabled_tasks"], output["disabled_tasks"]
    if not (isinstance(enabled, list) and isinstance(disabled, list)) or not enabled \
            or len(set(enabled)) != len(enabled) or not set(enabled) <= set(MODULE_TASK_KEYS) \
            or len(set(disabled)) != len(disabled) or set(disabled) != {"A", "U", "S", "R"} - set(enabled) \
            or ("A" in enabled and not scope.anatomy_classes) or ("U" in enabled and not scope.abnormality_classes):
        raise V5Error("MODULE_TASKS_INVALID", "$.enabled_tasks")
    keys = tuple(MODULE_TASK_KEYS[task] for task in ("A", "U") if task in enabled)
    _check_keys(output["acceptance"], keys, "$.acceptance")
    _check_keys(output["reason"], keys, "$.reason")
    for task, key in MODULE_TASK_KEYS.items():
        if task not in enabled and output[f"{key}_map"] is not None:
            raise V5Error("MODULE_TASKS_INVALID", f"$.{key}_map")

    if "A" in enabled:
        spatial = _spatial_shape(output["anatomy_map"], scope.observation_mode, "$.anatomy_map", 0)
    else:
        spatial = _spatial_shape(output["abnormality_map"], scope.observation_mode, "$.abnormality_map", 1)
    shapes = {"anatomy": spatial, "abnormality": (len(scope.abnormality_classes), *spatial)}
    kinds = {"anatomy": "int", "abnormality": "bool"}
    for key in keys:
        _check_array(output[f"{key}_map"], f"$.{key}_map", kinds[key], shapes[key])
        accepted = _check_array(output["acceptance"][key], f"$.acceptance.{key}", "bool", shapes[key])
        reason = _check_array(output["reason"][key], f"$.reason.{key}", "uint8", shapes[key])
        if key == "anatomy":
            _check_anatomy_values(output["anatomy_map"], accepted, len(scope.anatomy_classes))
        _check_reason_pair(accepted, reason, key)
    _check_geometry(output["geometry"], spatial)
    _check_provenance(output["provenance"], output["checkpoint_sha256"])
    return ModuleOutput(**output)
