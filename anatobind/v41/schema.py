"""Versioned v4.1 record types and a small JSON-Schema-subset validator.

Validation errors carry only a code and a field path; they never echo input
values, because sample identifiers and paths may identify a case.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v41.0"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas" / "v41"

SPLITS = ("train", "development_val", "locked_test", "unassigned", "excluded")
TASK_STATUSES = ("TODO", "IN_PROGRESS", "BLOCKED", "PASS", "FAIL", "DEFERRED")
ENGINEERING_STATUSES = ("PASS", "BLOCKED", "FAIL")
SCIENTIFIC_CONCLUSIONS = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_EVALUATED")

_SUPPORTED_KEYWORDS = {
    "$schema", "$id", "title", "description", "type", "properties", "required",
    "additionalProperties", "enum", "const", "items", "minItems", "maxItems",
    "minimum", "maximum", "definitions", "$ref",
}
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float), "null": type(None),
}


class SchemaError(ValueError):
    def __init__(self, code: str, path: str = "$"):
        self.code = code
        self.path = path
        super().__init__(f"{code} at {path}")


_SCHEMA_CACHE: dict[tuple[Path, str], dict] = {}


def load_schema(name: str, schema_dir: Path | None = None) -> dict:
    directory = SCHEMA_DIR if schema_dir is None else Path(schema_dir)
    key = (directory, name)
    if key not in _SCHEMA_CACHE:
        document = json.loads((directory / f"{name}.schema.json").read_text())
        _check_keywords(document, name)
        _SCHEMA_CACHE[key] = document
    return _SCHEMA_CACHE[key]


def _check_keywords(node: dict, where: str) -> None:
    unknown = set(node) - _SUPPORTED_KEYWORDS
    if unknown:
        raise SchemaError("UNSUPPORTED_SCHEMA_KEYWORD", f"{where}:{sorted(unknown)[0]}")
    for key in ("properties", "definitions"):
        for name, sub in node.get(key, {}).items():
            _check_keywords(sub, f"{where}/{key}/{name}")
    if isinstance(node.get("items"), dict):
        _check_keywords(node["items"], f"{where}/items")


def _resolve_ref(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise SchemaError("UNSUPPORTED_SCHEMA_KEYWORD", ref)
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else expected
    for name in names:
        py = _TYPES[name]
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def validate_against(instance: Any, schema: dict, root: dict | None = None, path: str = "$") -> None:
    root = schema if root is None else root
    if "$ref" in schema:
        schema = {**_resolve_ref(schema["$ref"], root), **{k: v for k, v in schema.items() if k != "$ref"}}
    if "type" in schema and not _type_ok(instance, schema["type"]):
        raise SchemaError("INVALID_TYPE", path)
    if "const" in schema and instance != schema["const"]:
        raise SchemaError("INVALID_CONST", path)
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaError("INVALID_ENUM", path)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaError("BELOW_MINIMUM", path)
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaError("ABOVE_MAXIMUM", path)
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                raise SchemaError("MISSING_FIELD", f"{path}.{name}")
        if schema.get("additionalProperties", True) is False:
            for name in instance:
                if name not in properties:
                    raise SchemaError("UNKNOWN_FIELD", f"{path}.{name}")
        for name, sub in properties.items():
            if name in instance:
                validate_against(instance[name], sub, root, f"{path}.{name}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise SchemaError("TOO_FEW_ITEMS", path)
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaError("TOO_MANY_ITEMS", path)
        if "items" in schema:
            for i, item in enumerate(instance):
                validate_against(item, schema["items"], root, f"{path}[{i}]")


def _check_version(record: dict, path: str = "$") -> None:
    if not isinstance(record, dict):
        raise SchemaError("INVALID_TYPE", path)
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError("SCHEMA_VERSION_MISMATCH", f"{path}.schema_version")


def _build(cls, record: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in record.items() if k in names})


@dataclass(frozen=True)
class AnnotationRecord:
    annotation_uid: str
    task: str
    class_scope: list[str]
    representation: str
    path: str | None
    label_map_version: str
    provenance_type: str
    creator_or_model: str
    creator_version: str
    parent_annotation_uid: str | None
    completeness: str
    review_region_path: str | None
    input_visibility: str
    quality_status: str
    intended_role: str
    reference_role: str
    adjudication_status: str
    reader_ids: list[str]
    protocol_hash: str | None
    visibility_evidence: str | None


@dataclass(frozen=True)
class SampleRecord:
    schema_version: str
    sample_uid: str
    source_dataset: str
    source_release: str
    cohort: str
    patient_uid: str
    study_uid: str | None
    session_uid: str | None
    parent_scan_uid: str | None
    derivative_uid: str | None
    leakage_group_uid: str
    exposure_history: list[dict]
    split: str
    native_acquisition_dim: int
    observation_mode: str
    image_paths: list[str]
    modality_names: list[str]
    modality_is_synthetic: list[bool]
    available_modality_mask: list[bool]
    image_geometry: dict
    original_geometry: dict | None
    transform_chain: list[dict]
    geometry_status: str
    label_value_qc: str
    anatomy_ontology_version: str
    abnormality_ontology_version: str
    relation_semantics_version: str
    annotation_records: list[AnnotationRecord]
    review_regions: list[dict]
    visibility_records: list[dict]
    quality_records: list[dict]
    provenance: dict


@dataclass(frozen=True)
class Prediction:
    schema_version: str
    sample_uid: str
    observation_uid: str
    mode: str
    geometry: dict
    a_prob: str | None
    u_prob: str | None
    instances: list[dict]
    relations: list[dict]
    acceptance: dict
    lineage: dict


@dataclass(frozen=True)
class GateDecision:
    schema_version: str
    gate_id: str
    scope_hash: str
    engineering_status: str
    scientific_conclusion: str
    evidence_hashes: list[str]
    blockers: list[str]
    approved_by: str | None
    approved_at: str | None


@dataclass(frozen=True)
class RunRegistration:
    schema_version: str
    run_id: str
    code_sha256: str
    config_sha256: str
    data_manifest_sha256: str
    split_sha256: str
    ontology_sha256: str
    supervision_sha256: str
    initialization: dict
    readable_roles: list[str]
    budget: dict
    hardware: dict
    seeds: dict
    numerical_settings: dict
    evaluation_rules: dict
    approvals: dict
    status: str = "TODO"


def validate_annotation(record: dict, path: str = "$") -> AnnotationRecord:
    if not isinstance(record, dict):
        raise SchemaError("INVALID_TYPE", path)
    validate_against(record, load_schema("annotation"), path=path)
    return _build(AnnotationRecord, record)


def validate_sample(record: dict) -> SampleRecord:
    _check_version(record)
    validate_against(record, load_schema("sample"))
    lists = ("modality_names", "modality_is_synthetic", "available_modality_mask")
    lengths = {len(record[name]) for name in lists}
    if len(lengths) != 1 or lengths != {len(record["image_paths"])}:
        raise SchemaError("MODALITY_LENGTH_MISMATCH", "$.image_paths")
    annotations = [
        validate_annotation(item, f"$.annotation_records[{i}]")
        for i, item in enumerate(record["annotation_records"])
    ]
    body = dict(record)
    body["annotation_records"] = annotations
    return _build(SampleRecord, body)


def validate_prediction(record: dict) -> Prediction:
    _check_version(record)
    validate_against(record, load_schema("prediction"))
    return _build(Prediction, record)


def validate_gate_decision(record: dict) -> GateDecision:
    _check_version(record)
    validate_against(record, load_schema("gate"))
    return _build(GateDecision, record)


def validate_experiment(record: dict) -> RunRegistration:
    _check_version(record)
    validate_against(record, load_schema("experiment"))
    return _build(RunRegistration, record)


def require_assigned_split(records: list[SampleRecord]) -> int:
    for i, record in enumerate(records):
        if record.split == "unassigned":
            raise SchemaError("SPLIT_UNASSIGNED", f"$[{i}].split")
    return len(records)
