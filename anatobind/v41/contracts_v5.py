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

from .schema import AnnotationRecord, SampleRecord, SchemaError, load_schema, validate_against

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


def _check_schema(record: object, name: str, path: str = "$") -> None:
    try:
        validate_against(record, load_schema(name, SCHEMA_DIR_V5), path=path)
    except SchemaError as error:
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
    body = dict(record)
    body["annotation_records"] = [
        _validate_annotation_v5(item, f"$.annotation_records[{i}]")
        for i, item in enumerate(record["annotation_records"])
    ]
    return _build(SampleV5, body)
