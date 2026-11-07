from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.evaluation.joint import joint_score_v5
from anatobind.v41.evaluation.reference import LesionOutcome, ReferenceLesion, freeze_reference_set_v5
from anatobind.v41.evaluation.relations import relation_scores_v5
from tests.v5.factories import make_reference_lesion


def _frozen(refs, patients):
    return freeze_reference_set_v5(refs, patients, {"policy_id": "synthetic_ref_v1"})


REFS = [make_reference_lesion(patient_uid="p1", instance_uid="l1"),
        make_reference_lesion(patient_uid="p1", instance_uid="l2")]
OK = LesionOutcome("l1", True, True, True, True, "matched")


@pytest.mark.parametrize("second", [
    [], [LesionOutcome("l2", False, False, None, False, "missed")],
    [LesionOutcome("l2", True, True, True, False, "rejected")],
    [LesionOutcome("l2", True, True, None, True, "no_relation")],
])

def test_missing_outcome_keeps_frozen_denominator(second):
    frozen = _frozen(REFS, ["p1"])
    result = joint_score_v5(frozen, [OK, *second])
    assert result["j_denominator"] == 2
    assert result["patient_mean_j"] == 0.5
    assert result["reference_set_hash"] == frozen["reference_set_hash"]


def test_empty_relation_set_differs_from_missing_reference():
    refs = [make_reference_lesion(instance_uid="l1", truth_relation_set=()),
            make_reference_lesion(instance_uid="l2", r_reference_available=False)]
    result = joint_score_v5(_frozen(refs, ["p1"]), [])
    assert result["j_denominator"] == 1
    assert result["excluded_counts"] == {"R_REFERENCE_UNAVAILABLE": 1}


def test_negative_patient_not_in_positive_j():
    result = joint_score_v5(_frozen(REFS, ["p1", "p2"]), [OK])
    assert result["n_negative_patients"] == 1 and set(result["per_patient"]) == {"p1"}


def test_no_positive_patients_returns_none():
    result = joint_score_v5(_frozen([], ["p1"]), [])
    assert result["patient_mean_j"] is None and result["lesion_pooled_j"] is None


@pytest.mark.parametrize("outcomes, code", [
    ([LesionOutcome("l9", True, True, True, True, "matched")], "UNKNOWN_REFERENCE_INSTANCE"),
    ([OK, OK], "DUPLICATE_OUTCOME"),
])

def test_unknown_or_duplicate_outcome_rejected(outcomes, code):
    with pytest.raises(V5Error) as error:
        joint_score_v5(_frozen(REFS, ["p1"]), outcomes)
    assert error.value.code == code


def test_tampered_reference_rejected():
    frozen = _frozen(REFS, ["p1"])
    frozen["lesions"] = frozen["lesions"][:1]
    with pytest.raises(V5Error) as error:
        joint_score_v5(frozen, [OK])
    assert error.value.code == "REFERENCE_HASH_MISMATCH"


def test_reference_lesion_has_no_prediction_fields():
    names = {f.name for f in dataclasses.fields(ReferenceLesion)}
    assert not names & {"detected", "score", "confidence", "probability", "accepted"}


def test_relation_exact_match_and_missing_prediction():
    refs = [make_reference_lesion(instance_uid="l1", truth_relation_set=("tissue_a",)),
            make_reference_lesion(instance_uid="l2", truth_relation_set=())]
    result = relation_scores_v5(_frozen(refs, ["p1"]), {"l1": ["tissue_a"], "l2": None})
    assert result["exact_match_rate"] == 0.5
    assert result["n_empty_truth_sets"] == 1 and result["n_missing_prediction"] == 1
    assert result["per_tissue"]["tissue_a"]["recall"] == 1.0


# Final review fixes (I1, M5, I2, M6).

def test_numpy_bool_outcome_counts_as_success():
    frozen = _frozen(REFS[:1], ["p1"])
    outcome = LesionOutcome("l1", np.bool_(True), np.bool_(True), np.bool_(True), np.bool_(True), "matched")
    assert joint_score_v5(frozen, [outcome])["patient_mean_j"] == 1.0


@pytest.mark.parametrize("args", [
    ("l1", True, True, 1, True, "matched"),
    ("l1", True, True, True, True, "missed"),
    ("l1", True, True, True, False, "matched"),
    ("l1", True, True, None, True, "matched"),
])
def test_outcome_flags_must_be_bool_and_match_status(args):
    with pytest.raises(V5Error) as error:
        LesionOutcome(*args)
    assert error.value.code == "INVALID_OUTCOME"


def test_reference_flags_must_be_bool():
    with pytest.raises(V5Error) as error:
        make_reference_lesion(eligible="False")
    assert error.value.code == "INVALID_REFERENCE"


def test_exclusion_reason_on_joint_lesion_rejected():
    with pytest.raises(V5Error) as error:
        _frozen([make_reference_lesion(exclusion_reason="QUALITY_FAIL")], ["p1"])
    assert error.value.code == "EXCLUSION_REASON_CONFLICT"


def test_never_predicted_tissue_has_zero_f1():
    refs = [make_reference_lesion(instance_uid="l1", truth_relation_set=("tissue_a",)),
            make_reference_lesion(instance_uid="l2", truth_relation_set=("tissue_b",))]
    tissue_b = relation_scores_v5(_frozen(refs, ["p1"]), {"l1": ["tissue_a"], "l2": []})["per_tissue"]["tissue_b"]
    assert tissue_b["f1"] == 0.0 and tissue_b["precision"] is None and tissue_b["recall"] == 0.0


def test_string_predicted_set_rejected():
    with pytest.raises(V5Error) as error:
        relation_scores_v5(_frozen(REFS, ["p1"]), {"l1": "tissue_a"})
    assert error.value.code == "INVALID_PREDICTED_SET"
