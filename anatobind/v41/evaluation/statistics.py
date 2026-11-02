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
