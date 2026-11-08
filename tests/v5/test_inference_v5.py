"""v5.1 full-volume sliding-window inference and original-grid restore (plan_v5 T09c; Proposal §2.1, §4.3)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.infer_v5 import predict_volume_v5, restore_probabilities_v5
from anatobind.v41.geometry_v5 import pad_or_crop_v5
from anatobind.v41.models.interface_v5 import collate_model_inputs_v5
from tests.v5.factories import StubPerceptionModel, make_model_input_v5, make_module_scope, make_scope

SHAPE = (5, 7, 9)


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _constant(values):
    def logits(batch):
        size = (batch.image.shape[0], len(values), *batch.image.shape[2:])
        return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1, 1).expand(size).clone()
    return logits


def _u_scope():
    return validate_scope_v5(make_module_scope("U"))


def test_sliding_window_covers_volume_and_averages_probabilities():
    scope = validate_scope_v5(make_scope())
    model = StubPerceptionModel(scope, ("A", "U"), anatomy_fn=_constant([0.0, 1.0, 2.0]),
                                abnormality_fn=_constant([0.5, -1.0]))
    out = predict_volume_v5(model, make_model_input_v5(scope, SHAPE), (4, 4, 4), (2, 3, 4), "fp32")
    assert set(out) == {"anatomy", "abnormality"}
    assert out["anatomy"].shape == (3, *SHAPE) and out["abnormality"].shape == (2, *SHAPE)
    assert out["anatomy"].dtype == np.float32 and out["abnormality"].dtype == np.float32
    # A constant model's probabilities do not change when overlapping windows are averaged.
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0).numpy()
    np.testing.assert_allclose(out["anatomy"], np.broadcast_to(expected[:, None, None, None], out["anatomy"].shape),
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(out["abnormality"][0], _sigmoid(0.5), rtol=0, atol=1e-7)
    np.testing.assert_allclose(out["abnormality"][1], _sigmoid(-1.0), rtol=0, atol=1e-7)
    # Windows tile with the stride; the last one on each axis sits flush with the end of the volume.
    crops = [chain[-1] for chain in model.chains]
    assert all(crop["op"] == "crop" and crop["input_shape"] == list(SHAPE) for crop in crops)
    starts = [crop["start"] for crop in crops]
    assert [sorted({s[axis] for s in starts}) for axis in range(3)] == [[0, 1], [0, 3], [0, 4, 5]]
    assert len(starts) == 12
    covered = np.zeros(SHAPE, dtype=int)
    for s in starts:
        covered[s[0]:s[0] + 4, s[1]:s[1] + 4, s[2]:s[2] + 4] += 1
    assert (covered >= 1).all()


def test_probabilities_not_logits_are_averaged():
    scope = _u_scope()

    def by_window(batch):     # logit 4 in windows starting at x = 0, logit 0 in those starting at x = 1
        value = 4.0 if batch.to_original[0][-1]["start"][0] == 0 else 0.0
        return torch.full((1, 1, *batch.image.shape[2:]), value)

    model = StubPerceptionModel(scope, ("U",), abnormality_fn=by_window)
    out = predict_volume_v5(model, make_model_input_v5(scope, SHAPE), (4, 4, 4), (2, 3, 4), "fp32")
    assert set(out) == {"abnormality"}
    p = out["abnormality"][0]
    np.testing.assert_allclose(p[0], _sigmoid(4.0), rtol=1e-6)
    np.testing.assert_allclose(p[4], 0.5, rtol=1e-6)
    np.testing.assert_allclose(p[1:4], (_sigmoid(4.0) + 0.5) / 2, rtol=1e-6)   # averaging logits would give σ(2)


def test_volume_smaller_than_window_is_padded_then_cropped_back():
    scope = _u_scope()
    model = StubPerceptionModel(scope, ("U",), abnormality_fn=_constant([1.0]))
    out = predict_volume_v5(model, make_model_input_v5(scope, (3, 7, 9)), (4, 4, 4), (2, 3, 4), "fp32")
    assert out["abnormality"].shape == (1, 3, 7, 9)
    np.testing.assert_allclose(out["abnormality"], _sigmoid(1.0), rtol=1e-6)
    for chain in model.chains:
        assert chain[-2] == {"op": "pad_or_crop", "input_shape": [3, 7, 9]}
        assert chain[-1]["op"] == "crop" and chain[-1]["input_shape"] == [4, 7, 9]


def test_predict_volume_refuses_bad_arguments():
    scope = _u_scope()
    model = StubPerceptionModel(scope, ("U",), abnormality_fn=_constant([1.0]))
    batch = make_model_input_v5(scope, SHAPE)
    pair = collate_model_inputs_v5([batch, make_model_input_v5(scope, SHAPE)])
    _raises("BATCH_NOT_SINGLE", "$.image", predict_volume_v5, model, pair, (4, 4, 4), (2, 3, 4), "fp32")
    _raises("INVALID_WINDOW", "$.window", predict_volume_v5, model, batch, (4, 4), (2, 3), "fp32")
    _raises("INVALID_WINDOW", "$.window", predict_volume_v5, model, batch, (0, 4, 4), (1, 1, 1), "fp32")
    _raises("INVALID_WINDOW", "$.stride", predict_volume_v5, model, batch, (4, 4, 4), (5, 3, 4), "fp32")
    _raises("INVALID_PRECISION", "$.precision", predict_volume_v5, model, batch, (4, 4, 4), (2, 3, 4), "fp16")
    assert model.calls == 0


def test_restore_inverts_pad_crop_and_downsample():
    original = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    padded, _ = pad_or_crop_v5(original, (6, 4, 6))
    prob, represented = restore_probabilities_v5(padded[None], [{"op": "pad_or_crop", "input_shape": [4, 5, 6]}],
                                                 (4, 5, 6))
    assert prob.shape == (1, 4, 5, 6) and prob.dtype == np.float32 and represented.dtype == bool
    assert represented[:, :4].all() and not represented[:, 4].any()
    assert np.array_equal(prob[0][represented], original[represented]) and (prob[0][~represented] == 0).all()
    # Downsample: coarse sample i sits at original f·i; j takes the linear value at j/f, the edge value beyond.
    x, y, z = np.meshgrid(np.arange(8), np.arange(7), np.arange(5), indexing="ij")
    ramp = ((x + 2 * y + 3 * z) / 100.0).astype(np.float32)
    chain = [{"op": "downsample", "factor": [2, 2, 1], "input_shape": [8, 7, 5]}]
    prob, represented = restore_probabilities_v5(ramp[::2, ::2][None], chain, (8, 7, 5))
    assert prob.shape == (1, 8, 7, 5) and represented.all()
    np.testing.assert_allclose(prob[0, :7], ramp[:7], atol=1e-6)
    np.testing.assert_allclose(prob[0, 7], ramp[6], atol=1e-6)
    _raises("GRID_MISMATCH", "$.to_original[0]", restore_probabilities_v5, ramp[::2, ::3][None], chain, (8, 7, 5))
    # A crop goes back to its start; the rest of the volume is not represented.
    crop = [{"op": "crop", "start": [1, -1, 0], "input_shape": [5, 5, 5]}]
    prob, represented = restore_probabilities_v5(np.ones((1, 3, 3, 3), dtype=np.float32), crop, (5, 5, 5))
    expected = np.zeros((5, 5, 5), dtype=bool)
    expected[1:4, 0:2, 0:3] = True
    assert np.array_equal(represented, expected) and np.array_equal(prob[0] == 1, expected)
    # Operations are undone last first: here the padding added after a 6 -> 3 downsample is dropped first.
    chain = [{"op": "downsample", "factor": [2, 1, 1], "input_shape": [6, 4, 4]},
             {"op": "pad_or_crop", "input_shape": [3, 4, 4]}]
    coarse = np.full((1, 4, 4, 4), 0.5, dtype=np.float32)
    coarse[0, 3] = 0.9
    prob, represented = restore_probabilities_v5(coarse, chain, (6, 4, 4))
    assert represented.all() and np.allclose(prob, 0.5)
    _raises("GRID_MISMATCH", "$.original_shape", restore_probabilities_v5, coarse, chain, (6, 4, 5))
    _raises("TRANSFORM_UNSUPPORTED", "$.to_original[0].op", restore_probabilities_v5, coarse,
            [{"op": "rotate", "input_shape": [4, 4, 4]}], (4, 4, 4))


def test_discrete_maps_are_never_interpolated():
    chain = [{"op": "downsample", "factor": [2, 2, 1], "input_shape": [4, 4, 2]}]
    for discrete in (np.ones((1, 2, 2, 2), dtype=bool), np.ones((1, 2, 2, 2), dtype=np.int64)):
        _raises("DTYPE_MISMATCH", "$.prob", restore_probabilities_v5, discrete, chain, (4, 4, 2))
