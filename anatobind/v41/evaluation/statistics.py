"""Patient-cluster bootstrap: patients are the resampling unit, never lesions or seeds."""
from __future__ import annotations

import numpy as np


def _patient_means(rows: list[dict]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        values.setdefault(row["patient"], []).append(float(row["value"]))
    return values


def paired_patient_bootstrap(rows: list[dict], seed: int, n_boot: int) -> dict:
    groups = _patient_means(rows)
    patients = sorted(groups)
    pooled = [v for row in rows for v in [float(row["value"])]]
    estimate = float(np.mean(pooled)) if pooled else float("nan")
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        draw = rng.choice(len(patients), size=len(patients), replace=True)
        sample = [v for index in draw for v in groups[patients[index]]]
        reps.append(float(np.mean(sample)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"estimate": estimate, "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "seed": seed, "n_patients": len(patients), "n_rows": len(rows)}


# ---------------------------------------------------------------------------
# v5.1 patient-equal statistics (Proposal_v5 §8.4). v41 paired_patient_bootstrap above is unchanged.
# No function here has a default seed, n_boot or ci_level: candidate values are registered inputs.

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from ..contracts_v5 import V5Error  # noqa: E402

STATISTIC_VERSION_V5 = "patient_bootstrap_v5.1"


@dataclass(frozen=True)
class PatientContrast:
    patient_uid: str
    method_value: float
    comparator_value: float
    delta: float
    seed_policy_id: str


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def patient_bootstrap_v5(patient_values: dict, seed: int, n_boot: int, ci_level: float) -> dict:
    """Mean of one value per patient; percentile CI from resampling patients with replacement."""
    if not _is_int(n_boot) or n_boot <= 0:
        raise V5Error("INVALID_N_BOOT", "$.n_boot")
    if not _finite(ci_level) or not 0 < ci_level < 1:
        raise V5Error("INVALID_CI_LEVEL", "$.ci_level")
    if not _is_int(seed):
        raise V5Error("INVALID_SEED", "$.seed")
    if not all(_finite(value) for value in patient_values.values()):
        raise V5Error("NON_FINITE_VALUE", "$.patient_values")
    result = {"estimate": None, "ci": None, "ci_level": ci_level, "n_patients": len(patient_values),
              "n_boot": n_boot, "seed": seed, "statistic_version": STATISTIC_VERSION_V5}
    if not patient_values:
        return result
    values = np.array([float(patient_values[uid]) for uid in sorted(patient_values)])
    rng = np.random.default_rng(seed)
    replicates = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    bounds = np.percentile(replicates, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2])
    result.update(estimate=float(values.mean()), ci=[float(b) for b in bounds])
    return result


def _seed_means_v5(rows: list[dict], policy: dict, side: str) -> dict:
    """{patient: {seed: mean over that patient's scans}}; every seed must cover the same scans."""
    seeds = policy.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(_is_int(s) for s in seeds) or len(set(seeds)) != len(seeds):
        raise V5Error("INVALID_SEED_POLICY", "$.policy.seeds")
    scans: dict = {}
    seen = set()
    for i, row in enumerate(rows):
        key = (row["patient_uid"], row["scan_uid"], row["seed"])
        if key in seen:
            raise V5Error("DUPLICATE_ROW", f"$.{side}[{i}]")
        seen.add(key)
        if row["seed"] not in seeds:
            raise V5Error("UNREGISTERED_SEED", f"$.{side}[{i}].seed")
        if not _finite(row["value"]):
            raise V5Error("NON_FINITE_VALUE", f"$.{side}[{i}].value")
        scans.setdefault(row["patient_uid"], {}).setdefault(row["seed"], {})[row["scan_uid"]] = float(row["value"])
    for patient, by_seed in scans.items():
        if set(by_seed) != set(seeds):
            raise V5Error("MISSING_SEED", f"$.{side}")
        if len({frozenset(scan_set) for scan_set in by_seed.values()}) != 1:
            raise V5Error("MISSING_SCAN", f"$.{side}")
    return {patient: {s: sum(v.values()) / len(v) for s, v in by_seed.items()} for patient, by_seed in scans.items()}


def _scan_sets_v5(rows: list[dict]) -> dict:
    sets: dict = {}
    for row in rows:
        sets.setdefault(row["patient_uid"], set()).add(row["scan_uid"])
    return sets


def make_patient_contrasts_v5(method_rows: list[dict], comparator_rows: list[dict], policy: dict) -> list:
    """Scans -> per-seed mean, seeds -> equal-weight mean, then one delta row per patient."""
    method = _seed_means_v5(method_rows, policy, "method_rows")
    comparator = _seed_means_v5(comparator_rows, policy, "comparator_rows")
    if set(method) != set(comparator):
        raise V5Error("UNPAIRED_PATIENT", "$.patient_uid")
    if _scan_sets_v5(method_rows) != _scan_sets_v5(comparator_rows):
        raise V5Error("UNPAIRED_SCAN", "$.scan_uid")
    seeds = policy["seeds"]
    contrasts = []
    for patient in sorted(method):
        m = sum(method[patient][s] for s in seeds) / len(seeds)
        c = sum(comparator[patient][s] for s in seeds) / len(seeds)
        contrasts.append(PatientContrast(patient, m, c, m - c, policy["seed_policy_id"]))
    return contrasts


def paired_patient_bootstrap_v5(rows: list, seed: int, n_boot: int, ci_level: float) -> dict:
    deltas = {}
    for i, row in enumerate(rows):
        if row.patient_uid in deltas:
            raise V5Error("DUPLICATE_ROW", f"$.rows[{i}]")
        deltas[row.patient_uid] = row.delta
    return patient_bootstrap_v5(deltas, seed=seed, n_boot=n_boot, ci_level=ci_level)


def seed_variation_v5(rows: list[dict], policy: dict) -> dict:
    """Patient-equal mean per registered seed, reported apart from the patient-level estimate."""
    means = _seed_means_v5(rows, policy, "rows")
    per_seed = {}
    for s in policy["seeds"]:
        values = [means[patient][s] for patient in sorted(means)]
        per_seed[s] = sum(values) / len(values) if values else None
    present = [v for v in per_seed.values() if v is not None]
    return {"per_seed": per_seed, "range": (max(present) - min(present)) if present else None}
