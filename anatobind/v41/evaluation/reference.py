"""Frozen lesion reference sets, kept apart from prediction outcomes (Proposal_v5 §4.1, §8.2).

The reference set is fixed and hashed before any prediction is scored. A
ReferenceLesion carries no prediction field; misses, rejections and missing
relation predictions are LesionOutcome rows and never shrink the denominator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..contracts_v5 import V5Error, canonical_sha256

PREDICTION_STATUSES = ("matched", "missed", "rejected", "no_relation")
# Exclusion reasons for lesions outside the joint set G, in priority order.
EXCLUSION_ORDER = (
    ("r_reference_available", "R_REFERENCE_UNAVAILABLE"),
    ("a_reference_available", "A_REFERENCE_UNAVAILABLE"),
    ("u_reference_available", "U_REFERENCE_UNAVAILABLE"),
    ("eligible", "NOT_ELIGIBLE"),
    ("visibility_qualified", "VISIBILITY_UNQUALIFIED"),
)


@dataclass(frozen=True)
class ReferenceLesion:
    patient_uid: str
    instance_uid: str
    a_reference_available: bool
    u_reference_available: bool
    r_reference_available: bool
    eligible: bool
    visibility_qualified: bool
    exclusion_reason: str | None
    truth_relation_set: tuple[str, ...]


@dataclass(frozen=True)
class LesionOutcome:
    reference_instance_uid: str
    detected: bool
    extent_ok: bool
    relation_correct: bool | None
    accepted: bool
    prediction_status: str

    def __post_init__(self):
        if self.prediction_status not in PREDICTION_STATUSES:
            raise V5Error("INVALID_PREDICTION_STATUS", "$.prediction_status")


def exclusion_reason_v5(lesion: dict) -> str | None:
    """None when the lesion belongs to the joint set G, else the first failing reason."""
    for field, reason in EXCLUSION_ORDER:
        if not lesion[field]:
            return reason
    return None


def _reference_hash(frozen: dict) -> str:
    return canonical_sha256({"policy_id": frozen["policy_id"], "lesions": frozen["lesions"],
                             "patient_uids": frozen["patient_uids"]})


def freeze_reference_set_v5(refs: list[ReferenceLesion], patient_uids: list[str], policy: dict) -> dict:
    if "policy_id" not in policy:
        raise V5Error("MISSING_FIELD", "$.policy.policy_id")
    if len(set(patient_uids)) != len(patient_uids):
        raise V5Error("DUPLICATE_PATIENT", "$.patient_uids")
    patients = sorted(patient_uids)
    seen = set()
    lesions = []
    for i, ref in enumerate(refs):
        if ref.patient_uid not in patient_uids:
            raise V5Error("UNKNOWN_PATIENT", f"$.refs[{i}].patient_uid")
        if ref.instance_uid in seen:
            raise V5Error("DUPLICATE_REFERENCE", f"$.refs[{i}].instance_uid")
        seen.add(ref.instance_uid)
        lesion = asdict(ref)
        lesion["truth_relation_set"] = sorted(ref.truth_relation_set)
        lesions.append(lesion)
    lesions.sort(key=lambda lesion: (lesion["patient_uid"], lesion["instance_uid"]))
    frozen = {"policy_id": policy["policy_id"], "lesions": lesions, "patient_uids": patients}
    frozen["reference_set_hash"] = _reference_hash(frozen)
    frozen["joint_eligible_uids"] = [lesion["instance_uid"] for lesion in lesions
                                     if exclusion_reason_v5(lesion) is None]
    return frozen


def verify_reference_set_v5(frozen: dict) -> dict:
    """Recompute the hash and the joint set G; return lesions keyed by instance_uid."""
    expected_g = [lesion["instance_uid"] for lesion in frozen["lesions"] if exclusion_reason_v5(lesion) is None]
    if _reference_hash(frozen) != frozen["reference_set_hash"] or frozen["joint_eligible_uids"] != expected_g:
        raise V5Error("REFERENCE_HASH_MISMATCH", "$.reference_set_hash")
    return {lesion["instance_uid"]: lesion for lesion in frozen["lesions"]}
