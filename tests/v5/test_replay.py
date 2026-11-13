"""Replay of a member's patch sampling and the two parts of the box objective (diagnostics of 2026-11-11).
Hand-computed values."""
from __future__ import annotations

import math

import pytest
import torch

from anatobind.v41.engine.replay_v5 import batch_exposure_v5, box_objective_parts_v5
from anatobind.v41.losses import box_mil_objective_v5

SPEC = {"kind": "box_mil", "topk_fraction": 0.5, "positive_weight": 1.0, "negative_weight": 1.0,
        "box_policy_sha256": "p" * 64}


def _case(values, bag_ids, negative):
    logits = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)
    bags = torch.tensor(bag_ids, dtype=torch.int32).reshape(1, 1, 1, 1, -1)
    e_neg = torch.tensor(negative, dtype=torch.bool).reshape(1, 1, 1, 1, -1)
    return logits, bags, bags > 0, e_neg


def test_the_parts_are_the_two_means_the_objective_adds():
    case = _case([2.0, -1.0, 0.0, 3.0], [1, 1, 0, 0], [False, False, True, True])
    parts = box_objective_parts_v5(*case, SPEC)
    assert parts["positive"] == pytest.approx(math.log1p(math.exp(-2.0)), abs=1e-6)       # top 1 of the 2 bag voxels
    assert parts["negative"] == pytest.approx((math.log(2.0) + math.log1p(math.exp(3.0))) / 2, abs=1e-6)
    assert (parts["bags"], parts["negative_voxels"]) == (1, 2)
    weighted = dict(SPEC, positive_weight=2.0, negative_weight=0.5)
    total = float(box_mil_objective_v5(*case, weighted))
    assert 2.0 * parts["positive"] + 0.5 * parts["negative"] == pytest.approx(total, abs=1e-6)


def test_a_part_without_support_is_none():
    parts = box_objective_parts_v5(*_case([0.3, 1.0], [0, 0], [False, False]), SPEC)
    assert {key: parts[key] for key in ("positive", "negative", "negative_all", "bags", "negative_voxels")} == {
        "positive": None, "negative": None, "negative_all": None, "bags": 0, "negative_voxels": 0}


def test_the_parts_are_the_record_of_the_terms_training_logs():
    """One decomposition for training and replay: the parts are box_terms_record_v5 of box_mil_terms_v5."""
    from anatobind.v41.losses import box_mil_terms_v5, box_terms_record_v5

    case = _case([2.0, -1.0, 0.0, 3.0, -2.0], [1, 1, 0, 0, 0], [False, False, True, True, True])
    spec = dict(SPEC, negative_topk_fraction=0.5)                  # keeps 2 of the 3 eligible negatives
    parts = box_objective_parts_v5(*case, spec)
    assert parts == {**box_terms_record_v5(box_mil_terms_v5(*case, spec)), "negative_voxels": 3}
    assert parts["negative"] == pytest.approx((math.log(2.0) + math.log1p(math.exp(3.0))) / 2, abs=1e-6)
    assert parts["negative_all"] == pytest.approx(
        (math.log(2.0) + math.log1p(math.exp(3.0)) + math.log1p(math.exp(-2.0))) / 3, abs=1e-6)
    assert (parts["classes"][0]["negative_eligible"], parts["classes"][0]["negative_selected"]) == (3, 2)
    assert parts["positive"] + parts["negative"] == pytest.approx(float(box_mil_objective_v5(*case, spec)), abs=1e-6)
    without = box_objective_parts_v5(*case, SPEC)                   # an earlier plan: both negative terms are one
    assert without["negative"] == without["negative_all"] == pytest.approx(parts["negative_all"], abs=1e-6)


def test_the_exposure_of_a_batch_names_its_bags_and_the_logged_support():
    bags = torch.zeros((2, 2, 1, 1, 4), dtype=torch.int32)
    bags[0, 0, 0, 0, :2] = 5                                   # item 0, class 0: bag 5 on two voxels
    bags[1, 1, 0, 0, 1:] = 2                                   # item 1, class 1: bag 2 on three voxels
    e_pos = bags > 0
    e_pos[1, 1, 0, 0, 3] = False                               # one of them is outside the field of view
    e_neg = torch.tensor([[[True, True, True, True], [True, True, True, True]],
                          [[True, True, True, False], [True, False, False, False]]]).reshape(2, 2, 1, 1, 4)
    e_neg[0, 0, 0, 0, :2] = False                              # no negative inside the box
    anatomy = torch.zeros((2, 1, 1, 4), dtype=torch.int64)
    a_pos = torch.ones((2, 2, 1, 1, 4), dtype=torch.bool)
    a_pos[1, 0, 0, 0, 3] = False                               # one voxel where a class is not eligible
    exposure = batch_exposure_v5(["u0", "u1"], {"A": anatomy, "U": bags},
                                 {"A": {"e_pos": a_pos, "e_neg": torch.ones_like(a_pos)},
                                  "U": {"e_pos": e_pos, "e_neg": e_neg}})
    # U: 2 bags + (2 + 4 + 3 + 1) negative voxels, as train_unified_v5 logs it; A: 4 + 3 fully eligible voxels
    assert exposure == {"support": {"A": 7, "U": 12}, "bags": [("u0", 0, 5, 2), ("u1", 1, 2, 2)]}
