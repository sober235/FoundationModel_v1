from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_perception_output_v5, validate_scope_v5
from anatobind.v41.evaluation.state import (
    STATE_ELIGIBILITY_BITS,
    build_state_reference_v5,
    summarize_state_by_patient_v5,
    tissue_state_metrics_v5,
)
from tests.v5.factories import make_joint_output, make_scope, make_state_reference_inputs


def reference_u() -> np.ndarray:
    """Same [2,1,2,2] abnormality as the default state reference: target_a at (0,0,0) and (0,1,0)."""
    u = np.zeros((2, 1, 2, 2), bool)
    u[0, 0, 0, 0] = u[0, 0, 1, 0] = True
    return u


SCOPE = validate_scope_v5(make_scope())


def _reference(**overrides):
    inputs = make_state_reference_inputs(SCOPE, **overrides)
    return build_state_reference_v5(inputs["record"], inputs["arrays"], SCOPE, "obs-synth-1")


def test_correct_u_wrong_a_fails_joint_state():
    reference = _reference()             # 全部体素参照为 tissue_a，(0,0,0)、(0,1,0) 为阳性
    anatomy = np.full((1, 2, 2), 2)       # 预测全部为 tissue_b
    out = validate_perception_output_v5(
        make_joint_output(SCOPE, anatomy_map=anatomy, abnormality_map=reference_u()), SCOPE)
    result = tissue_state_metrics_v5(out, reference, SCOPE)
    assert result["u_dice_in_reference_tissue"] == 1.0
    assert result["per_tissue_state"]["tissue_a"]["target_positive"]["recall"] == 0.0
    assert result["reference_count"] == 4


def test_rejection_keeps_state_denominator():
    reference = _reference()
    out = validate_perception_output_v5(make_joint_output(
        SCOPE, anatomy_map=np.full((1, 2, 2), -1), anatomy_accepted=np.zeros((1, 2, 2), bool),
        abnormality_map=reference_u()), SCOPE)
    result = tissue_state_metrics_v5(out, reference, SCOPE)
    assert result["coverage"] == 0.0 and result["false_normal_on_accepted"] is None
    assert result["reference_count"] == 4 and result["rejection_reasons"] == {"anatomy_uncertain": 4}


def test_positive_only_and_unreviewed_are_never_normal():
    assert not _reference(u_completeness="positive_only").state_eligible.any()
    review = np.zeros((1, 2, 2), bool)
    review[0, 0, 0] = True
    assert _reference(review=review).state_eligible.sum() == 1


def test_2d_reference_needs_single_slice_readability():
    scope_2d = validate_scope_v5(make_scope(observation_mode="2d"))
    inputs = make_state_reference_inputs(scope_2d, single_slice_readable=None)
    reference = build_state_reference_v5(inputs["record"], inputs["arrays"], scope_2d, "obs-2d")
    assert not reference.state_eligible.any()
    assert reference.eligibility_reason.max() & STATE_ELIGIBILITY_BITS["single_slice_unreadable"]


def test_reference_hash_is_deterministic_and_bound_to_reference_inputs():
    assert _reference().reference_set_hash == _reference().reference_set_hash
    review = np.ones((1, 2, 2), bool)
    review[0, 1, 1] = False
    assert _reference(review=review).reference_set_hash != _reference().reference_set_hash


def test_identity_mismatch_rejected():
    reference = dataclasses.replace(_reference(), mode="2d")
    with pytest.raises(V5Error) as error:
        tissue_state_metrics_v5(validate_perception_output_v5(make_joint_output(SCOPE), SCOPE), reference, SCOPE)
    assert error.value.code == "IDENTITY_MISMATCH"


def test_empty_state_class_returns_none():
    result = tissue_state_metrics_v5(validate_perception_output_v5(make_joint_output(SCOPE), SCOPE), _reference(), SCOPE)
    assert result["per_tissue_state"]["tissue_b"]["target_positive"]["recall"] is None


def test_patient_summary_is_patient_equal():
    rows = [("p1", {"per_tissue_state": {"tissue_a": {"target_positive": {"recall": 1.0}}}}),
            ("p1", {"per_tissue_state": {"tissue_a": {"target_positive": {"recall": 0.0}}}}),
            ("p2", {"per_tissue_state": {"tissue_a": {"target_positive": {"recall": 1.0}}}}),
            ("p3", {"per_tissue_state": {"tissue_a": {"target_positive": {"recall": None}}}})]
    summary = summarize_state_by_patient_v5(rows, ("tissue_a", "target_positive", "recall"), seed=1, n_boot=50, ci_level=0.9)
    assert summary["estimate"] == 0.75 and summary["n_patients"] == 2 and summary["n_patients_without_support"] == 1


def test_capability_template_is_all_null():
    template = yaml.safe_load(Path("configs/v5/capability_policy.template.yaml").read_text())
    assert set(template) == {"standard_kind", "required_tissues", "required_states", "min_reference_support",
                             "metric_bounds", "min_coverage", "uncertainty_rule", "ci_level", "n_boot", "seed"}
    assert all(value is None for value in template.values())


# Final review fixes (I4, I5, I6).

@pytest.mark.parametrize("visibility", ["unknown", "unobservable"])
def test_unknown_or_unobservable_visibility_is_never_normal(visibility):
    reference = _reference(u_visibility=visibility)
    assert not reference.state_eligible.any()
    assert (reference.eligibility_reason & STATE_ELIGIBILITY_BITS["not_visible"]).all()


def test_region_complete_without_review_region_rejected():
    inputs = make_state_reference_inputs(SCOPE, review=np.ones((1, 2, 2), bool))
    inputs["arrays"]["review_regions"] = {}
    with pytest.raises(V5Error) as error:
        build_state_reference_v5(inputs["record"], inputs["arrays"], SCOPE, "obs-synth-1")
    assert error.value.code == "REVIEW_REGION_MISSING"


@pytest.mark.parametrize("override", [{"u_reference_role": "pseudo_only"}, {"u_reference_role": "weak_localization"},
                                      {"a_reference_role": "pseudo_only"}, {"a_class_scope": ["tissue_a"]}])
def test_reference_needs_expert_voxel_a_and_u_covering_scope(override):
    assert not _reference(**override).state_eligible.any()


def test_reference_arrays_are_copied_and_read_only():
    inputs = make_state_reference_inputs(SCOPE)
    reference = build_state_reference_v5(inputs["record"], inputs["arrays"], SCOPE, "obs-synth-1")
    inputs["arrays"]["anatomy"][:] = 2
    assert (reference.anatomy_truth == 1).all() and not reference.anatomy_truth.flags.writeable


def test_tampered_state_reference_rejected():
    reference = dataclasses.replace(_reference(), state_truth=np.zeros((1, 2, 2), np.uint8))
    with pytest.raises(V5Error) as error:
        tissue_state_metrics_v5(validate_perception_output_v5(make_joint_output(SCOPE), SCOPE), reference, SCOPE)
    assert error.value.code == "REFERENCE_HASH_MISMATCH"
