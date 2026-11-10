"""Box multiple-instance objective (design 2026-11-10 §5.2). Hand-computed values."""
from __future__ import annotations

import math

import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.losses import box_mil_objective_v5

SPEC = {"kind": "box_mil", "topk_fraction": 0.5, "positive_weight": 1.0, "negative_weight": 1.0,
        "box_policy_sha256": "p" * 64}


def _case(values, bag_ids, negative):
    logits = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1).requires_grad_()
    bags = torch.tensor(bag_ids, dtype=torch.int32).reshape(1, 1, 1, 1, -1)
    e_neg = torch.tensor(negative, dtype=torch.bool).reshape(1, 1, 1, 1, -1)
    return logits, bags, bags > 0, e_neg


def test_value_matches_the_hand_computation():
    logits, bags, e_pos, e_neg = _case([2.0, -1.0, 0.0, 3.0], [1, 1, 0, 0], [False, False, True, True])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)
    positive = math.log1p(math.exp(-2.0))                                  # top 1 of the 2 bag voxels
    negative = (math.log(2.0) + math.log1p(math.exp(3.0))) / 2
    assert float(loss.detach()) == pytest.approx(positive + negative, abs=1e-6)     # 1.997795
    loss.backward()
    assert logits.grad.reshape(-1)[1] == 0                                 # the bag voxel outside the top-k


def test_a_bag_cut_by_the_patch_border_still_gives_a_finite_loss():
    logits, bags, e_pos, e_neg = _case([-4.0, 0.5, 0.2, 0.1], [7, 0, 0, 0], [False, False, False, False])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)          # one bag voxel left, no negatives
    assert float(loss.detach()) == pytest.approx(math.log1p(math.exp(4.0)), abs=1e-6)


def test_no_support_gives_a_differentiable_zero_even_with_infinite_logits():
    logits, bags, e_pos, e_neg = _case([float("inf"), 1.0], [0, 0], [False, False])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)
    loss.backward()
    assert float(loss.detach()) == 0.0 and not logits.grad.any()


def test_two_bags_average_and_weights_scale_each_part():
    logits, bags, e_pos, e_neg = _case([0.0, 0.0, 0.0], [1, 2, 0], [False, False, True])
    spec = dict(SPEC, topk_fraction=1.0, positive_weight=2.0, negative_weight=0.5)
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, spec)
    assert float(loss.detach()) == pytest.approx(2.5 * math.log(2.0), abs=1e-6)


@pytest.mark.parametrize("change,code,path", [
    ({"kind": "sigmoid"}, "UNKNOWN_LOSS_KIND", "$.spec.kind"),
    ({"topk_fraction": 0.0}, "INVALID_LOSS_WEIGHTS", "$.spec"),
    ({"positive_weight": 0.0, "negative_weight": 0.0}, "INVALID_LOSS_WEIGHTS", "$.spec")])
def test_invalid_specs_are_refused(change, code, path):
    logits, bags, e_pos, e_neg = _case([0.0], [0], [True])
    with pytest.raises(V5Error) as err:
        box_mil_objective_v5(logits, bags, e_pos, e_neg, dict(SPEC, **change))
    assert (err.value.code, err.value.path) == (code, path)
