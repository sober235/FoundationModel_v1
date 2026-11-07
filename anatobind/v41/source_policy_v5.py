"""Per-source data policies for v5.1 (Proposal_v5 §3.1-3.2).

A SourcePolicy names one data source (id + version) and the facts migration
needs about it. Facts that need real-data evidence (annotation modalities,
index-alignment evidence, unit policy and the evidence for it) stay null until verified; migration of
real records then fails closed instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .contracts_v5 import V5Error

POLICY_FIELDS = ("source_version", "source_dataset", "scope_id", "reference_policy_id", "model_modalities",
                 "annotation_modalities", "index_evidence", "physical_units_policy", "physical_units_evidence")
UNITS_POLICIES = ("trust_header",)
_TASKS = ("A", "U")


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    source_version: str
    source_dataset: str
    scope_id: str
    reference_policy_id: str
    model_modalities: tuple
    annotation_modalities: dict
    index_evidence: str | None
    physical_units_policy: str | None
    physical_units_evidence: str | None


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _names(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_name(v) for v in value) and len(set(value)) == len(value)


def validate_source_policy_v5(source_id: str, entry: dict) -> SourcePolicy:
    path = f"$.{source_id}"
    if not _is_name(source_id) or not isinstance(entry, dict) or set(entry) != set(POLICY_FIELDS):
        raise V5Error("INVALID_SOURCE_POLICY", path)
    for name in ("source_version", "source_dataset", "scope_id", "reference_policy_id"):
        if not _is_name(entry[name]):
            raise V5Error("INVALID_SOURCE_POLICY", f"{path}.{name}")
    if not _names(entry["model_modalities"]):
        raise V5Error("INVALID_SOURCE_POLICY", f"{path}.model_modalities")
    modalities = entry["annotation_modalities"]
    if not isinstance(modalities, dict) or not modalities or not set(modalities) <= set(_TASKS) \
            or not all(v is None or _names(v) for v in modalities.values()):
        raise V5Error("INVALID_SOURCE_POLICY", f"{path}.annotation_modalities")
    if entry["index_evidence"] is not None and not _is_name(entry["index_evidence"]):
        raise V5Error("INVALID_SOURCE_POLICY", f"{path}.index_evidence")
    if entry["physical_units_policy"] is not None and entry["physical_units_policy"] not in UNITS_POLICIES:
        raise V5Error("INVALID_SOURCE_POLICY", f"{path}.physical_units_policy")
    evidence = entry["physical_units_evidence"]
    if (evidence is not None and not _is_name(evidence)) or (entry["physical_units_policy"] and evidence is None):
        raise V5Error("INVALID_SOURCE_POLICY", f"{path}.physical_units_evidence")
    return SourcePolicy(
        source_id=source_id, source_version=entry["source_version"], source_dataset=entry["source_dataset"],
        scope_id=entry["scope_id"], reference_policy_id=entry["reference_policy_id"],
        model_modalities=tuple(entry["model_modalities"]),
        annotation_modalities={task: None if v is None else tuple(v) for task, v in modalities.items()},
        index_evidence=entry["index_evidence"], physical_units_policy=entry["physical_units_policy"],
        physical_units_evidence=evidence,
    )


def load_source_policies_v5(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise V5Error("INVALID_SOURCE_POLICY", "$")
    return {source_id: validate_source_policy_v5(source_id, entry) for source_id, entry in data.items()}


def source_binding_v5(policy: SourcePolicy) -> dict:
    """The {source_id, source_version} binding that gate evidence and G-DATA approvals carry (FX1)."""
    return {"source_id": policy.source_id, "source_version": policy.source_version}
