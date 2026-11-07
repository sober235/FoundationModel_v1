from __future__ import annotations

import math

import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.evaluation.joint import joint_score_v5
from anatobind.v41.evaluation.reference import LesionOutcome, freeze_reference_set_v5
from anatobind.v41.evaluation.statistics import (
    make_patient_contrasts_v5,
    paired_patient_bootstrap_v5,
    patient_bootstrap_v5,
    seed_variation_v5,
)
from tests.v5.factories import make_reference_lesion


def test_unequal_lesion_counts_preserve_patient_mean():
    refs = [make_reference_lesion(patient_uid="p1", instance_uid=f"a{i}") for i in range(100)]
    refs.append(make_reference_lesion(patient_uid="p2", instance_uid="b0"))
    frozen = freeze_reference_set_v5(refs, ["p1", "p2"], {"policy_id": "synthetic_ref_v1"})
    outcomes = [LesionOutcome(f"a{i}", True, True, True, True, "matched") for i in range(100)]
    joint = joint_score_v5(frozen, outcomes)
    values = {uid: row["j"] for uid, row in joint["per_patient"].items()}
    result = patient_bootstrap_v5(values, seed=7, n_boot=200, ci_level=0.9)
    assert result["estimate"] == 0.5 and result["n_patients"] == 2
    assert math.isclose(joint["lesion_pooled_j"], 100 / 101)
    assert result["ci_level"] == 0.9 and "ci95" not in result


def _rows(values):
    return [{"patient_uid": p, "scan_uid": s, "seed": k, "value": v} for (p, s, k), v in values.items()]


METHOD = _rows({("p1", "s1", 17): 1.0, ("p1", "s2", 17): 0.0, ("p1", "s1", 29): 1.0, ("p1", "s2", 29): 1.0,
                ("p2", "s1", 17): 0.0, ("p2", "s1", 29): 0.0})
COMPARATOR = _rows({("p1", "s1", 17): 0.5, ("p1", "s2", 17): 0.5, ("p1", "s1", 29): 0.5, ("p1", "s2", 29): 0.5,
                    ("p2", "s1", 17): 0.0, ("p2", "s1", 29): 0.0})
POLICY = {"seeds": [17, 29], "seed_policy_id": "synthetic_seeds"}


def test_contrasts_average_scans_then_seeds_one_row_per_patient():
    contrasts = make_patient_contrasts_v5(METHOD, COMPARATOR, POLICY)
    assert [c.patient_uid for c in contrasts] == ["p1", "p2"]
    assert math.isclose(contrasts[0].delta, 0.25) and contrasts[1].delta == 0.0
    result = paired_patient_bootstrap_v5(contrasts, seed=3, n_boot=100, ci_level=0.95)
    assert result["n_patients"] == 2 and math.isclose(result["estimate"], 0.125)
    assert result == paired_patient_bootstrap_v5(contrasts, seed=3, n_boot=100, ci_level=0.95)


@pytest.mark.parametrize("method, code", [
    (METHOD + METHOD[:1], "DUPLICATE_ROW"),
    ([r for r in METHOD if not (r["patient_uid"] == "p2" and r["seed"] == 29)], "MISSING_SEED"),
    ([r for r in METHOD if r["patient_uid"] == "p1"] + _rows({("p3", "s1", 17): 0.0, ("p3", "s1", 29): 0.0}),
     "UNPAIRED_PATIENT"),
])

def test_duplicate_row_rejected(method, code):
    with pytest.raises(V5Error) as error:
        make_patient_contrasts_v5(method, COMPARATOR, POLICY)
    assert error.value.code == code


@pytest.mark.parametrize("kwargs, code", [
    ({"n_boot": 0, "ci_level": 0.95}, "INVALID_N_BOOT"),
    ({"n_boot": 10, "ci_level": 1.0}, "INVALID_CI_LEVEL"),
])

def test_invalid_interval_policy_rejected(kwargs, code):
    with pytest.raises(V5Error) as error:
        patient_bootstrap_v5({"p1": 1.0}, seed=0, **kwargs)
    assert error.value.code == code


def test_non_finite_value_rejected():
    with pytest.raises(V5Error) as error:
        patient_bootstrap_v5({"p1": float("nan")}, seed=0, n_boot=10, ci_level=0.95)
    assert error.value.code == "NON_FINITE_VALUE"


def test_empty_returns_none():
    result = patient_bootstrap_v5({}, seed=0, n_boot=10, ci_level=0.95)
    assert result["estimate"] is None and result["ci"] is None and result["n_patients"] == 0


def test_seed_variation_is_reported_separately():
    variation = seed_variation_v5(METHOD, POLICY)
    assert variation["per_seed"] == {17: 0.25, 29: 0.5} and variation["range"] == 0.25
