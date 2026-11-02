"""Selective prediction: risk-coverage and false-normal rates with fixed denominators."""
from __future__ import annotations


def risk_coverage(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    errors = sum(1 for row in accepted if row["correct"] is False)
    return {
        "total": len(rows),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "risk": errors / len(accepted) if accepted else None,
        "rejected": len(rows) - len(accepted),
    }


def false_normal_on_accepted(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["accepted_normal"]]
    false_normal = sum(1 for row in accepted if row["actually_positive"])
    return {
        "false_normal_on_accepted": false_normal / len(accepted) if accepted else None,
        "denominator": len(accepted),
        "false_normal": false_normal,
    }
