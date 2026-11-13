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


# ---------------------------------------------------------------------------
# Hard-negative part (pilot of 2026-11-12): negative_topk_fraction keeps, per class, the top ceil(fraction·N) of the
# N eligible negative voxels pooled over the whole batch. Without the key the objective is the earlier one.

from anatobind.v41.losses import box_mil_terms_v5, box_terms_record_v5  # noqa: E402


def _softplus(value: float) -> float:
    return math.log1p(math.exp(value))


def _grid(values, negative, bag_ids=None):
    """[batch, classes, 1, 1, n] tensors from nested lists; without bag ids there is no bag."""
    logits = torch.tensor(values, dtype=torch.float32)[:, :, None, None, :].requires_grad_()
    e_neg = torch.tensor(negative, dtype=torch.bool)[:, :, None, None, :]
    bags = torch.zeros(e_neg.shape, dtype=torch.int32) if bag_ids is None \
        else torch.tensor(bag_ids, dtype=torch.int32)[:, :, None, None, :]
    return logits, bags, bags > 0, e_neg


def test_the_negative_part_keeps_the_top_fraction_pooled_over_the_batch():
    # five eligible negatives over two items; ceil(0.4 · 5) = 2, and both of the highest lie in the second item
    logits, bags, e_pos, e_neg = _grid([[[0.0, -1.0, -2.0]], [[3.0, 1.0, 9.0]]],
                                       [[[True, True, True]], [[True, True, False]]])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, dict(SPEC, negative_topk_fraction=0.4))
    assert float(loss.detach()) == pytest.approx((_softplus(3.0) + _softplus(1.0)) / 2, abs=1e-6)
    loss.backward()
    grad = logits.grad.reshape(2, 3)
    assert not grad[0].any()                                    # the first item holds none of the top two
    assert float(grad[1, 0]) == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item() / 2, abs=1e-6)
    assert float(grad[1, 1]) == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item() / 2, abs=1e-6)
    assert grad[1, 2] == 0                                      # the highest logit of all is not an eligible negative


def test_the_negative_mask_decides_what_can_be_selected_not_the_bags():
    # the bag voxel (logit 8) and the ignored voxel (logit 7) are neither negatives; the complement of the bag is not used
    logits, bags, e_pos, e_neg = _grid([[[8.0, 7.0, 0.5, -3.0]]], [[[False, False, True, True]]], [[[1, 0, 0, 0]]])
    spec = dict(SPEC, topk_fraction=1.0, negative_topk_fraction=0.5)
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, spec)
    assert float(loss.detach()) == pytest.approx(_softplus(-8.0) + _softplus(0.5), abs=1e-6)
    loss.backward()
    grad = logits.grad.reshape(-1)
    assert grad[1] == 0 and grad[3] == 0 and grad[2] > 0 and grad[0] < 0


def test_classes_are_averaged_over_those_with_negative_support():
    # class 0: N = 4, k = 1 -> softplus(2); class 1: no eligible negative; class 2: N = 2, k = 1 -> softplus(-1)
    values = [[[2.0, 0.0, -1.0, -5.0], [9.0, 9.0, 9.0, 9.0], [-1.0, -4.0, 6.0, 6.0]]]
    negative = [[[True, True, True, True], [False, False, False, False], [True, True, False, False]]]
    logits, bags, e_pos, e_neg = _grid(values, negative)
    spec = dict(SPEC, negative_topk_fraction=0.25)
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, spec)
    assert float(loss.detach()) == pytest.approx((_softplus(2.0) + _softplus(-1.0)) / 2, abs=1e-6)
    loss.backward()
    assert not logits.grad[0, 1].any()                          # a class without support adds nothing


@pytest.mark.parametrize("count,fraction,kept", [(1, 0.01, 1), (99, 0.01, 1), (100, 0.01, 1), (101, 0.01, 2),
                                                 (700, 0.01, 7), (3, 1.0, 3), (3, 1, 3)])
def test_the_number_kept_is_the_ceiling_of_the_fraction_of_the_eligible_negatives(count, fraction, kept):
    values = torch.linspace(-3.0, 3.0, count).tolist()
    logits, bags, e_pos, e_neg = _grid([[values]], [[[True] * count]])
    terms = box_mil_terms_v5(logits, bags, e_pos, e_neg, dict(SPEC, negative_topk_fraction=fraction))
    assert (terms["classes"][0]["negative_eligible"], terms["classes"][0]["negative_selected"]) == (count, kept)
    top = sorted(values)[-kept:]
    assert float(terms["negative"].detach()) == pytest.approx(sum(_softplus(v) for v in top) / kept, abs=1e-5)
    terms["negative"].backward()
    assert int((logits.grad != 0).sum()) == kept                # gradient on the selected voxels only


def test_a_fraction_of_one_is_the_earlier_objective_bit_for_bit():
    generator = torch.Generator().manual_seed(7)
    shape = (2, 3, 2, 4, 5)
    bags = torch.zeros(shape, dtype=torch.int32)
    bags[0, 0, 0, :2, :3], bags[1, 2, 1, 1:, 2:] = 4, 6
    e_neg = (torch.rand(shape, generator=generator) > 0.3) & (bags == 0)
    values = torch.randn(shape, generator=generator)
    results = []
    for spec in (SPEC, dict(SPEC, negative_topk_fraction=1.0), dict(SPEC, negative_topk_fraction=1)):
        logits = values.clone().requires_grad_()
        loss = box_mil_objective_v5(logits, bags, bags > 0, e_neg, spec)
        loss.backward()
        results.append((loss.detach(), logits.grad))
    for loss, grad in results[1:]:
        assert torch.equal(loss, results[0][0]) and torch.equal(grad, results[0][1])


def test_the_terms_are_what_the_objective_adds_and_carry_the_counts_of_each_class():
    values = [[[2.0, 0.0, -1.0, -5.0], [1.0, -2.0, 0.5, 4.0]]]
    negative = [[[False, True, True, True], [True, True, True, False]]]
    logits, bags, e_pos, e_neg = _grid(values, negative, [[[3, 0, 0, 0], [0, 0, 0, 0]]])
    spec = dict(SPEC, topk_fraction=1.0, positive_weight=2.0, negative_weight=0.5, negative_topk_fraction=0.5)
    terms = box_mil_terms_v5(logits, bags, e_pos, e_neg, spec)
    record = box_terms_record_v5(terms)
    top = [(_softplus(0.0) + _softplus(-1.0)) / 2, (_softplus(1.0) + _softplus(0.5)) / 2]       # 2 of 3 in each class
    everything = [(_softplus(0.0) + _softplus(-1.0) + _softplus(-5.0)) / 3,
                  (_softplus(1.0) + _softplus(-2.0) + _softplus(0.5)) / 3]
    assert record["positive"] == pytest.approx(_softplus(-2.0), abs=1e-6) and record["bags"] == 1
    assert record["negative"] == pytest.approx(sum(top) / 2, abs=1e-6)
    assert record["negative_all"] == pytest.approx(sum(everything) / 2, abs=1e-6)
    assert [item["bags"] for item in record["classes"]] == [1, 0]
    assert [(item["negative_eligible"], item["negative_selected"]) for item in record["classes"]] == [(3, 2), (3, 2)]
    for index, item in enumerate(record["classes"]):
        assert item["negative"] == pytest.approx(top[index], abs=1e-6)
        assert item["negative_all"] == pytest.approx(everything[index], abs=1e-6)
    assert record["classes"][0]["positive"] == pytest.approx(_softplus(-2.0), abs=1e-6)
    assert record["classes"][1]["positive"] is None
    total = float(box_mil_objective_v5(logits, bags, e_pos, e_neg, spec).detach())
    assert total == pytest.approx(2.0 * record["positive"] + 0.5 * record["negative"], abs=1e-6)
    assert not terms["negative_all"].requires_grad               # the all-negative mean is recorded, never optimised


def test_without_the_key_every_eligible_negative_is_kept_and_both_negative_terms_agree():
    logits, bags, e_pos, e_neg = _grid([[[2.0, 0.0, -1.0]]], [[[True, True, True]]])
    record = box_terms_record_v5(box_mil_terms_v5(logits, bags, e_pos, e_neg, SPEC))
    assert record["negative"] == record["negative_all"]
    assert record["classes"][0]["negative_selected"] == record["classes"][0]["negative_eligible"] == 3
    assert record["positive"] is None and record["classes"][0]["positive"] is None


def test_terms_without_any_support_are_none():
    logits, bags, e_pos, e_neg = _grid([[[0.3, 1.0]]], [[[False, False]]])
    record = box_terms_record_v5(box_mil_terms_v5(logits, bags, e_pos, e_neg, dict(SPEC, negative_topk_fraction=0.01)))
    assert record == {"positive": None, "negative": None, "negative_all": None, "bags": 0,
                      "classes": [{"bags": 0, "positive": None, "negative": None, "negative_all": None,
                                   "negative_eligible": 0, "negative_selected": 0}]}


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, True, float("nan"), "0.01", None])
def test_an_invalid_negative_fraction_is_refused(fraction):
    logits, bags, e_pos, e_neg = _case([0.0], [0], [True])
    with pytest.raises(V5Error) as err:
        box_mil_objective_v5(logits, bags, e_pos, e_neg, dict(SPEC, negative_topk_fraction=fraction))
    assert (err.value.code, err.value.path) == ("INVALID_LOSS_WEIGHTS", "$.spec")
