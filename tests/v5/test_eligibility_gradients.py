from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5
from anatobind.v41.eligibility import build_eligibility_v5
from anatobind.v41.losses import masked_objective_v5
from tests.v5.factories import make_sample_v5


OBS = {"track": "observable_target", "reference_policy_id": "synthetic_policy_v1"}
FOV = np.ones(8, bool)


def _elig(annotation, policy=OBS, fov=FOV, targets=None, vis=None, channel="expert", classes=("target_a",)):
    record = validate_sample_v5(make_sample_v5(annotation=annotation))
    return build_eligibility_v5(record, "U", list(classes), targets or {}, {}, vis or {}, fov, policy, channel)


@pytest.mark.parametrize("visibility, evidence", [("unknown", None), ("unobservable", "synthetic")])

def test_observable_track_unknown_or_unobservable_has_no_support(visibility, evidence):
    e_pos, e_neg = _elig({"input_visibility": visibility, "visibility_evidence": evidence})
    assert not e_pos.any() and not e_neg.any()


def test_limited_uses_only_visibility_region():
    region = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)
    e_pos, e_neg = _elig({"input_visibility": "limited", "visibility_evidence": "synthetic",
                          "visibility_region_path": "vis/r1"}, vis={"vis/r1": region})
    assert e_pos[0].tolist() == region.tolist() and e_neg[0].tolist() == region.tolist()


def test_padding_outside_fov_has_no_support_even_if_complete():
    fov = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    e_pos, e_neg = _elig({}, fov=fov)
    assert not e_pos[0, 6:].any() and not e_neg[0, 6:].any() and e_pos[0, :6].all()


def test_positive_only_is_class_specific():
    policy = {**OBS, "class_values": {"target_a": 1, "target_b": 2}}
    targets = {"ann-v5-0001": np.array([1, 2, 0, 0, 0, 0, 0, 0])}
    e_pos, e_neg = _elig({"completeness": "positive_only", "class_scope": ["target_a", "target_b"]},
                         policy=policy, targets=targets, classes=("target_a", "target_b"))
    assert e_pos[0, :2].tolist() == [True, False] and e_pos[1, :2].tolist() == [False, True]
    assert not e_pos[:, 2:].any() and not e_neg.any()


def test_positive_only_requires_class_map():
    with pytest.raises(V5Error) as error:
        _elig({"completeness": "positive_only"}, targets={"ann-v5-0001": np.ones(8, int)})
    assert error.value.code == "POSITIVE_ONLY_CLASS_MAP_REQUIRED"


def test_pseudo_only_needs_pseudo_channel():
    pseudo = {"provenance_type": "automated", "intended_role": "pseudo_only", "reference_role": "pseudo_only"}
    assert not _elig(pseudo)[0].any()
    assert _elig(pseudo, channel="pseudo")[0].all()


def test_released_reference_unknown_needs_bound_policy():
    released = {"track": "released_reference", "reference_policy_id": "synthetic_policy_v1",
                "allow_unknown_visibility": True}
    unknown = {"input_visibility": "unknown", "visibility_evidence": None}
    assert _elig(unknown, policy=released)[0].all()
    assert not _elig(unknown, policy={**released, "reference_policy_id": "other"})[0].any()


def test_box_never_gives_dense_support():
    assert not _elig({"representation": "bbox"})[0].any()


SIG = {"kind": "sigmoid", "ce_weight": 1.0, "dice_weight": 0.0}


def test_signed_bce_ignores_unsupported_sign():
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    e_pos = torch.tensor([[[True, False, False, False]]])
    e_neg = torch.zeros_like(e_pos)
    base = masked_objective_v5(torch.zeros(1, 1, 4), target, e_pos, e_neg, SIG)
    moved = torch.zeros(1, 1, 4); moved[0, 0, 1:] = 50.0
    assert torch.isclose(base, torch.tensor(math.log(2.0)))
    assert torch.isclose(masked_objective_v5(moved, target, e_pos, e_neg, SIG), base)


def test_all_ignored_is_finite_graph_connected_zero():
    logits = torch.tensor([[[float("inf"), -float("inf"), 0.0]]], requires_grad=True)
    none = torch.zeros(1, 1, 3, dtype=torch.bool)
    loss = masked_objective_v5(logits, torch.zeros(1, 1, 3), none, none, SIG)
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(logits.grad).all() and not logits.grad.any()


def test_nonfinite_supported_logit_raises():
    support = torch.ones(1, 1, 2, dtype=torch.bool)
    with pytest.raises(V5Error) as error:
        masked_objective_v5(torch.tensor([[[float("nan"), 0.0]]]), torch.zeros(1, 1, 2), support, support, SIG)
    assert error.value.code == "NON_FINITE_LOGITS"


def test_invalid_weights_raise():
    support = torch.ones(1, 1, 2, dtype=torch.bool)
    with pytest.raises(V5Error) as error:
        masked_objective_v5(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), support, support,
                            {"kind": "sigmoid", "ce_weight": 0.0, "dice_weight": 0.0})
    assert error.value.code == "INVALID_LOSS_WEIGHTS"


def test_softmax_requires_complete_class_support():
    spec = {"kind": "softmax", "ce_weight": 1.0, "dice_weight": 1.0}
    target = torch.tensor([[1, 2]])
    e_pos = torch.ones(1, 2, 2, dtype=torch.bool)
    e_neg = torch.tensor([[[True, True], [True, False]]])
    base = masked_objective_v5(torch.zeros(1, 3, 2), target, e_pos, e_neg, spec)
    moved = torch.zeros(1, 3, 2); moved[0, :, 1] = torch.tensor([9.0, -9.0, 3.0])
    assert torch.isclose(masked_objective_v5(moved, target, e_pos, e_neg, spec), base)
