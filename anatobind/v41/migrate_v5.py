"""Migrate v41 SampleRecords to the v5.1 sample contract (Proposal_v5 §3).

Migration adds the v5.1 fields, never upgrades evidence: geometry status can
only stay unknown or fail unless the source policy registers the evidence,
headers and units are not edited, and v41 "limited" visibility (which has no
region) is refused rather than rewritten. v41 exposure roles are mapped to the
v5 roles with no lineage. A record is accepted only under the policy for its
own release string, and only when the policy's model modalities are the
scope's input modalities. Each status records its evidence (the policy's
evidence when verified, the v41 status when failed). Only v41 records are
accepted, so a migrated record cannot be migrated again. The input record is
not modified.
"""
from __future__ import annotations

import dataclasses

from .contracts_v5 import SCHEMA_VERSION_V5, ActiveScope, SampleV5, V5Error, validate_sample_v5
from .exposure_v5 import migrate_exposure_v41
from .schema import SCHEMA_VERSION as SCHEMA_VERSION_V41
from .schema import SampleRecord
from .source_policy_v5 import SourcePolicy


def _index_status(record: dict, policy: SourcePolicy) -> str:
    if record["geometry_status"] == "fail":
        return "failed"
    if record["geometry_status"] == "pass" and policy.index_evidence is not None:
        return "verified"
    return "unknown"


def _physical_status(record: dict, policy: SourcePolicy) -> str:
    if record["geometry_status"] == "pass" and not isinstance(record["image_geometry"], dict):
        raise V5Error("GEOMETRY_MISSING", "$.image_geometry")
    if record["geometry_status"] == "fail":
        return "failed"
    if record["geometry_status"] == "pass" and policy.physical_units_policy == "trust_header" \
            and record["image_geometry"].get("units") == "mm":
        return "verified"
    return "unknown"


def _geometry_evidence(body: dict, policy: SourcePolicy, index_status: str, physical_status: str) -> dict:
    failed = f"v41_geometry_status:{body['geometry_status']}"
    return {"index_alignment": policy.index_evidence if index_status == "verified" else
            failed if index_status == "failed" else None,
            "physical_geometry": policy.physical_units_evidence if physical_status == "verified" else
            failed if physical_status == "failed" else None}


def _annotation(annotation: dict, policy: SourcePolicy, path: str) -> dict:
    modalities = policy.annotation_modalities.get(annotation["task"])
    if modalities is None:
        raise V5Error("ANNOTATION_MODALITY_UNKNOWN", f"{path}.task")
    if annotation["input_visibility"] == "limited":
        raise V5Error("MIGRATION_NEEDS_VISIBILITY_REGION", f"{path}.input_visibility")
    return {**annotation, "annotation_modalities": list(modalities), "visibility_region_path": None}


def migrate_v41_record_v5(record: SampleRecord, policy: SourcePolicy, scope: ActiveScope) -> SampleV5:
    if isinstance(record, SampleV5):
        raise V5Error("SCHEMA_VERSION_MISMATCH", "$.schema_version")
    if not isinstance(record, SampleRecord):
        raise V5Error("INVALID_TYPE", "$")
    if record.schema_version != SCHEMA_VERSION_V41:
        raise V5Error("SCHEMA_VERSION_MISMATCH", "$.schema_version")
    body = dataclasses.asdict(record)
    if body["source_dataset"] != policy.source_dataset:
        raise V5Error("SOURCE_MISMATCH", "$.source_dataset")
    if body["source_release"] != policy.source_release:
        raise V5Error("SOURCE_VERSION_MISMATCH", "$.source_release")
    if policy.scope_id != scope.scope_id or policy.reference_policy_id != scope.reference_policy_id:
        raise V5Error("SCOPE_MISMATCH", "$.active_scope_id")
    if list(policy.model_modalities) != list(scope.input_modalities):
        raise V5Error("MODALITY_SCOPE_MISMATCH", "$.model_modalities")
    body["annotation_records"] = [_annotation(a, policy, f"$.annotation_records[{i}]")
                                  for i, a in enumerate(body["annotation_records"])]
    body["exposure_history"] = [migrate_exposure_v41(e, f"$.exposure_history[{i}]")
                                for i, e in enumerate(body["exposure_history"])]
    index_status, physical_status = _index_status(body, policy), _physical_status(body, policy)
    body.update(schema_version=SCHEMA_VERSION_V5, active_scope_id=scope.scope_id,
                reference_policy_id=policy.reference_policy_id, model_modalities=list(policy.model_modalities),
                index_alignment_status=index_status, physical_geometry_status=physical_status,
                geometry_evidence=_geometry_evidence(body, policy, index_status, physical_status))
    return validate_sample_v5(body)


def migrate_v41_records_v5(records: list, policy: SourcePolicy, scope: ActiveScope) -> tuple:
    """Migrate a batch; failures carry only the record index, error code and field path (no IDs)."""
    migrated, failures = [], []
    for index, record in enumerate(records):
        try:
            migrated.append(migrate_v41_record_v5(record, policy, scope))
        except V5Error as error:
            failures.append({"index": index, "code": error.code, "path": error.path})
    return migrated, failures
