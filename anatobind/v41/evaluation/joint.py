"""Joint A/U/R lesion score with pre-frozen denominators.

A lesion counts toward j_all only when it is detected, extent-qualified,
relation-correct and not rejected. Lesions without a relation reference are
excluded from the J denominator and disclosed, never auto-corrected.
"""
from __future__ import annotations


def joint_score(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["eligible"]]
    with_reference = [row for row in eligible if row["relation_ok"] is not None]
    excluded = len(eligible) - len(with_reference)

    def passed(row) -> bool:
        return bool(row["detected"] and row["extent_ok"] and row["relation_ok"] and row["accepted"])

    accepted = sum(1 for row in eligible if row["accepted"])
    per_patient: dict[str, list[bool]] = {}
    for row in with_reference:
        per_patient.setdefault(row["patient"], []).append(passed(row))
    patient_j = {patient: (sum(flags) / len(flags) if flags else 0.0)
                 for patient, flags in per_patient.items()}
    return {
        "lesions_total": len(rows),
        "lesions_eligible": len(eligible),
        "excluded_no_relation_reference": excluded,
        "j_denominator": len(with_reference),
        "j_all_coverage": (sum(1 for row in with_reference if passed(row)) / len(with_reference)) if with_reference else 0.0,
        "lesion_pooled_j": (sum(1 for row in with_reference if passed(row)) / len(with_reference)) if with_reference else 0.0,
        "acceptance_coverage": accepted / len(eligible) if eligible else 0.0,
        "accepted": accepted,
        "patient_equal_j": patient_j,
        "patient_mean_j": sum(patient_j.values()) / len(patient_j) if patient_j else 0.0,
    }


# ---------------------------------------------------------------------------
# v5.1 joint score on a frozen reference set (Proposal_v5 §8.2). v41 joint_score above is unchanged.

from ..contracts_v5 import V5Error  # noqa: E402
from .reference import LesionOutcome, exclusion_reason_v5, verify_reference_set_v5  # noqa: E402


def _outcomes_by_uid_v5(outcomes: list[LesionOutcome], lesions: dict) -> dict:
    by_uid = {}
    for i, outcome in enumerate(outcomes):
        uid = outcome.reference_instance_uid
        if uid not in lesions:
            raise V5Error("UNKNOWN_REFERENCE_INSTANCE", f"$.outcomes[{i}]")
        if uid in by_uid:
            raise V5Error("DUPLICATE_OUTCOME", f"$.outcomes[{i}]")
        by_uid[uid] = outcome
    return by_uid


def _success_v5(outcome: LesionOutcome | None) -> bool:
    return outcome is not None and bool(outcome.detected and outcome.extent_ok and outcome.accepted) \
        and outcome.relation_correct is True


def joint_score_v5(frozen: dict, outcomes: list[LesionOutcome]) -> dict:
    """J_i = successes / |G_i| over the frozen joint set; a G lesion without an outcome is a failure."""
    lesions = verify_reference_set_v5(frozen)
    by_uid = _outcomes_by_uid_v5(outcomes, lesions)
    per_patient = {}
    excluded = {}
    n_negative = 0
    for patient in frozen["patient_uids"]:
        own = [lesion for lesion in frozen["lesions"] if lesion["patient_uid"] == patient]
        if not own:
            n_negative += 1
            continue
        joint_set = []
        for lesion in own:
            reason = exclusion_reason_v5(lesion)
            if reason is None:
                joint_set.append(lesion["instance_uid"])
            else:
                excluded[reason] = excluded.get(reason, 0) + 1
        if joint_set:
            numerator = sum(_success_v5(by_uid.get(uid)) for uid in joint_set)
            per_patient[patient] = {"numerator": numerator, "denominator": len(joint_set),
                                    "j": numerator / len(joint_set)}
    j_numerator = sum(row["numerator"] for row in per_patient.values())
    j_denominator = sum(row["denominator"] for row in per_patient.values())
    return {
        "j_numerator": j_numerator,
        "j_denominator": j_denominator,
        "patient_mean_j": (sum(row["j"] for row in per_patient.values()) / len(per_patient)) if per_patient else None,
        "lesion_pooled_j": (j_numerator / j_denominator) if j_denominator else None,
        "per_patient": per_patient,
        "excluded_counts": excluded,
        "n_negative_patients": n_negative,
        "reference_set_hash": frozen["reference_set_hash"],
    }
