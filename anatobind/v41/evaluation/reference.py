"""Frozen lesion reference sets, kept apart from prediction outcomes (Proposal_v5 §4.1, §8.2).

The reference set is fixed and hashed before any prediction is scored. A
ReferenceLesion carries no prediction field; misses, rejections and missing
relation predictions are LesionOutcome rows and never shrink the denominator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

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


def _as_bool(value: object, code: str, path: str, allow_none: bool = False) -> bool | None:
    """Python or numpy bool -> bool; anything else (1, "False", ...) is rejected, never coerced by truthiness."""
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise V5Error(code, path)


_REFERENCE_FLAGS = ("a_reference_available", "u_reference_available", "r_reference_available", "eligible",
                    "visibility_qualified")


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

    def __post_init__(self):
        for name in ("patient_uid", "instance_uid"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise V5Error("INVALID_REFERENCE", f"$.{name}")
        for name in _REFERENCE_FLAGS:
            object.__setattr__(self, name, _as_bool(getattr(self, name), "INVALID_REFERENCE", f"$.{name}"))
        if self.exclusion_reason is not None and not isinstance(self.exclusion_reason, str):
            raise V5Error("INVALID_REFERENCE", "$.exclusion_reason")
        tissues = self.truth_relation_set
        if not isinstance(tissues, (tuple, list)) or not all(isinstance(t, str) for t in tissues):
            raise V5Error("INVALID_REFERENCE", "$.truth_relation_set")
        object.__setattr__(self, "truth_relation_set", tuple(tissues))


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
        for name in ("detected", "extent_ok", "accepted"):
            object.__setattr__(self, name, _as_bool(getattr(self, name), "INVALID_OUTCOME", f"$.{name}"))
        object.__setattr__(self, "relation_correct",
                           _as_bool(self.relation_correct, "INVALID_OUTCOME", "$.relation_correct", allow_none=True))
        # The status must agree with the flags, so a row cannot be both "missed" and a success.
        status = self.prediction_status
        consistent = {
            "matched": self.detected and self.accepted and self.relation_correct is not None,
            "missed": not self.detected,
            "rejected": self.detected and not self.accepted,
            "no_relation": self.detected and self.relation_correct is None,
        }[status]
        if not consistent:
            raise V5Error("INVALID_OUTCOME", "$.prediction_status")


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
        if ref.exclusion_reason is not None and exclusion_reason_v5(lesion) is None:
            raise V5Error("EXCLUSION_REASON_CONFLICT", f"$.refs[{i}].exclusion_reason")
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
