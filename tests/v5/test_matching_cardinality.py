from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.evaluation.instances import match_instances, match_instances_v5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _interval(start, stop, uid, cls="target_a"):
    mask = np.zeros((680, 1, 1), bool)
    mask[start:stop] = True
    return {"instance_uid": uid, "class_name": cls, "mask": mask}


TRUTH = [_interval(100 * i, 100 * i + 100, f"t{i}") for i in range(6)]
PRED = [_interval(100 * i + 80, 100 * i + 180, f"p{i}") for i in range(6)]
POLICY = {"iou_match_threshold": 0.1, "class_aware": True}


def test_six_eligible_matches_beat_five_high_iou():
    pairs = match_instances_v5(PRED, TRUTH, POLICY)
    assert len(pairs) == 6
    assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == 6
    assert len(match_instances(PRED, TRUTH, POLICY)) == 5   # v41 历史行为保留


def _oracle(ious, threshold):
    best = (0, 0.0)
    n_pred, n_truth = ious.shape
    for k in range(min(n_pred, n_truth) + 1):
        for rows in itertools.combinations(range(n_pred), k):
            for cols in itertools.permutations(range(n_truth), k):
                if all(ious[r, c] >= threshold for r, c in zip(rows, cols)):
                    best = max(best, (k, sum(ious[r, c] for r, c in zip(rows, cols))))
    return best


@pytest.mark.parametrize("seed", range(40))

def test_matches_exhaustive_oracle(seed):
    rng = np.random.default_rng(seed)
    n_pred, n_truth = rng.integers(1, 5, size=2)
    pred = [{"instance_uid": f"p{i}", "class_name": "target_a", "mask": rng.random((6, 6)) < 0.4}
            for i in range(n_pred)]
    truth = [{"instance_uid": f"t{j}", "class_name": "target_a", "mask": rng.random((6, 6)) < 0.4}
             for j in range(n_truth)]
    ious = np.array([[iou(p["mask"], t["mask"]) for t in truth] for p in pred])
    pairs = match_instances_v5(pred, truth, {"iou_match_threshold": 0.1, "class_aware": True})
    k, total = _oracle(ious, 0.1)
    assert len(pairs) == k
    assert math.isclose(sum(ious[i, j] for i, j in pairs), total, abs_tol=1e-12)


def test_input_order_does_not_change_id_pairs():
    ids = {(PRED[i]["instance_uid"], TRUTH[j]["instance_uid"]) for i, j in match_instances_v5(PRED, TRUTH, POLICY)}
    pred_r, truth_r = PRED[::-1], TRUTH[::-1]
    ids_r = {(pred_r[i]["instance_uid"], truth_r[j]["instance_uid"])
             for i, j in match_instances_v5(pred_r, truth_r, POLICY)}
    assert ids == ids_r


def test_class_mismatch_is_not_matched():
    pred = [dict(PRED[0], class_name="target_b")]
    assert match_instances_v5(pred, TRUTH[:1], POLICY) == []


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, float("nan")])

def test_invalid_threshold_rejected_even_when_empty(threshold):
    with pytest.raises(V5Error) as error:
        match_instances_v5([], [], {"iou_match_threshold": threshold, "class_aware": True})
    assert error.value.code == "INVALID_IOU_THRESHOLD"


def test_duplicate_ids_rejected():
    with pytest.raises(V5Error) as error:
        match_instances_v5([PRED[0], PRED[0]], TRUTH, POLICY)
    assert error.value.code == "DUPLICATE_INSTANCE_ID"


def test_shape_mismatch_rejected():
    bad = {"instance_uid": "p9", "class_name": "target_a", "mask": np.zeros((10, 1, 1), bool)}
    with pytest.raises(V5Error) as error:
        match_instances_v5([bad], TRUTH, POLICY)
    assert error.value.code == "MASK_SHAPE_MISMATCH"
