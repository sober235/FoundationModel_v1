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


# ---------------------------------------------------------------------------
# Reach of the hard-negative selection (pilot of 2026-11-12)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def test_the_reach_of_the_selection_counts_the_high_negatives_it_keeps():
    from anatobind.v41.engine.replay_v5 import negative_reach_v5

    # class 0: six eligible negatives, three at or above probability 0.5; class 1: five eligible (the 9.0 is not a
    # negative), four of them high; class 2: nothing eligible
    logits = torch.tensor([[[3.0, 2.0, 0.0, -1.0, -2.0, -3.0], [9.0, 4.0, 0.5, 0.2, 0.1, -5.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])[:, :, None, None, :]
    e_neg = torch.tensor([[[True] * 6, [False, True, True, True, True, True], [False] * 6]])[:, :, None, None, :]
    reach = negative_reach_v5(logits, e_neg, 0.5, [0.5, 0.5, 0.5])
    first, second, third = reach
    assert (first["negative_eligible"], first["negative_selected"], first["high"], first["high_selected"]) == (6, 3, 3, 3)
    assert first["cutoff_probability"] == pytest.approx(_sigmoid(0.0), abs=1e-6)
    assert first["selected_mean_probability"] == pytest.approx((_sigmoid(3) + _sigmoid(2) + _sigmoid(0)) / 3, abs=1e-6)
    assert first["eligible_mean_probability"] == pytest.approx(
        sum(_sigmoid(v) for v in (3, 2, 0, -1, -2, -3)) / 6, abs=1e-6)
    # ceil(0.5 · 5) = 3 of the four high negatives are kept: the selection does not reach all of them
    assert (second["negative_eligible"], second["negative_selected"], second["high"], second["high_selected"]) == (
        5, 3, 4, 3)
    assert second["cutoff_probability"] == pytest.approx(_sigmoid(0.2), abs=1e-6)
    assert third == {"negative_eligible": 0, "negative_selected": 0, "high": 0, "high_selected": 0,
                     "cutoff_probability": None, "selected_mean_probability": None, "eligible_mean_probability": None}
    # another threshold for the first class: only the two voxels above 0.85 are high, both are kept
    assert negative_reach_v5(logits, e_neg, 0.5, [0.85, 0.5, 0.5])[0]["high"] == 2
    # without a fraction every eligible negative is kept
    everything = negative_reach_v5(logits, e_neg, None, [0.5, 0.5, 0.5])
    assert (everything[1]["negative_selected"], everything[1]["high_selected"]) == (5, 4)
    assert everything[1]["cutoff_probability"] == pytest.approx(_sigmoid(-5.0), abs=1e-6)


def test_the_reach_agrees_with_the_counts_of_the_objective():
    from anatobind.v41.engine.replay_v5 import negative_reach_v5
    from anatobind.v41.losses import box_mil_terms_v5

    generator = torch.Generator().manual_seed(3)
    logits = torch.randn((2, 2, 3, 4, 5), generator=generator)
    e_neg = torch.rand((2, 2, 3, 4, 5), generator=generator) > 0.4
    bags = torch.zeros(logits.shape, dtype=torch.int32)
    spec = dict(SPEC, negative_topk_fraction=0.01)
    terms = box_mil_terms_v5(logits, bags, bags > 0, e_neg, spec)
    reach = negative_reach_v5(logits, e_neg, 0.01, [0.5, 0.5])
    for item, entry in zip(reach, terms["classes"]):
        assert (item["negative_eligible"], item["negative_selected"]) == (
            entry["negative_eligible"], entry["negative_selected"])


def test_the_counts_of_the_terms_need_no_weights():
    from anatobind.v41.engine.replay_v5 import box_counts_v5
    from anatobind.v41.losses import box_mil_terms_v5, box_terms_record_v5

    generator = torch.Generator().manual_seed(5)
    shape = (2, 2, 3, 4, 5)
    bags = torch.zeros(shape, dtype=torch.int32)
    bags[0, 0, 0, :2, :3], bags[0, 0, 2, 2:, 3:], bags[1, 1, 1, :, :2] = 1, 2, 1     # class 0: two bags; class 1: one
    e_pos = bags > 0
    e_pos[0, 0, 2, 2:, 3:] = False                      # the second bag of class 0 has no supervised voxel left
    e_neg = (torch.rand(shape, generator=generator) > 0.4) & (bags == 0)
    for spec in (SPEC, dict(SPEC, negative_topk_fraction=0.01), dict(SPEC, negative_topk_fraction=1)):
        counts = box_counts_v5(bags, e_pos, e_neg, spec)
        assert [sorted(entry) for entry in counts] == [["bags", "negative_eligible", "negative_selected"]] * 2
        assert [entry["bags"] for entry in counts] == [1, 1]
        for logits in (torch.randn(shape, generator=generator), torch.zeros(shape)):     # whatever the weights give
            record = box_terms_record_v5(box_mil_terms_v5(logits, bags, e_pos, e_neg, spec))
            assert counts == [{key: entry[key] for key in ("bags", "negative_eligible", "negative_selected")}
                              for entry in record["classes"]]
    kept = box_counts_v5(bags, e_pos, e_neg, dict(SPEC, negative_topk_fraction=0.01))
    assert [entry["negative_selected"] for entry in kept] == [
        math.ceil(entry["negative_eligible"] / 100) for entry in kept]
    assert box_counts_v5(bags, e_pos, torch.zeros(shape, dtype=torch.bool), SPEC)[0] == {
        "bags": 1, "negative_eligible": 0, "negative_selected": 0}
