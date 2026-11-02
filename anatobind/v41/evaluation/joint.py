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
