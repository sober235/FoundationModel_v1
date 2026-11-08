"""Frozen v5.1 model input/output interface (plan_v5 §1.7, T09a)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import GEOMETRY_KEYS, V5Error, validate_scope_v5
from anatobind.v41.models.interface_v5 import (
    ModelInputV5, ModelOutputV5, PerceptionModelV5, build_model_input_v5, check_model_input_v5,
    check_model_output_v5, collate_model_inputs_v5)
from tests.v5.factories import make_module_scope

SHAPE = (4, 3, 5)
DOWNSAMPLE = {"op": "downsample", "factor": [1, 2, 2], "input_shape": [4, 6, 10]}


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _scope(task="A", **overrides):
    return validate_scope_v5(make_module_scope(task, **overrides))


def _au_scope():
    return _scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"])


def _grid(shape=SHAPE, spacing=(3.0, 0.5, 0.5)):
    return {"shape": list(shape), "axis_order": "zyx"[-len(shape):],
            "spacing_mm": None if spacing is None else list(spacing), "affine": None, "units_reliable": False,
            "transform_chain": []}


def _channels(scope, shape=SHAPE):
    return {name: np.full(shape, float(i + 1)) for i, name in enumerate(scope.input_modalities)}


def _input(scope, shape=SHAPE, **overrides):
    args = {"channels": _channels(scope, shape), "scope": scope, "valid_fov": np.ones(shape, dtype=bool),
            "grid": _grid(shape, spacing=(3.0, 0.5, 0.5)[-len(shape):]), "to_original": [DOWNSAMPLE]}
    args.update(overrides)
    return build_model_input_v5(**args)


class StubModel(PerceptionModelV5):
    """Returns zero logits with the channel counts it is told; counts calls to _forward_v5."""

    def __init__(self, scope, tasks, anatomy_channels=None, abnormality_channels=None):
        super().__init__(scope, tasks)
        self.calls = 0
        self.anatomy_channels = anatomy_channels
        self.abnormality_channels = abnormality_channels

    def _forward_v5(self, batch):
        self.calls += 1
        size = (batch.image.shape[0], *batch.image.shape[2:])

        def logits(channels):
            return None if channels is None else torch.zeros((size[0], channels, *size[1:]))

        return ModelOutputV5(anatomy_logits=logits(self.anatomy_channels),
                             abnormality_logits=logits(self.abnormality_channels))


def test_model_input_stacks_channels_in_scope_order():
    scope = _scope("A")
    assert scope.input_modalities == ["synthetic_e1", "synthetic_e2"]
    channels = {"synthetic_e2": np.full(SHAPE, 2.0), "synthetic_e1": np.full(SHAPE, 1.0)}
    batch = build_model_input_v5(channels, scope, np.ones(SHAPE, dtype=bool), _grid(), [DOWNSAMPLE])
    assert isinstance(batch, ModelInputV5)
    assert batch.image.dtype == torch.float32 and tuple(batch.image.shape) == (1, 2, *SHAPE)
    assert batch.image[0, 0].unique().tolist() == [1.0] and batch.image[0, 1].unique().tolist() == [2.0]
    assert batch.valid_fov.dtype == torch.bool and tuple(batch.valid_fov.shape) == (1, *SHAPE)
    assert batch.observation_mode == "3d"
    assert {key: batch.grids[0][key] for key in GEOMETRY_KEYS} == _grid() and batch.grids[0]["slice_axis"] == 0
    assert batch.to_original == ((DOWNSAMPLE,),)
    check_model_input_v5(batch, scope, ("3d",))


def test_missing_or_extra_modality_rejected():
    scope = _scope("A")
    channels = _channels(scope)
    del channels["synthetic_e2"]
    _raises("INPUT_MODALITY_MISSING", "$.input_modalities[1]", _input, scope, channels=channels)
    _raises("INPUT_MODALITY_UNEXPECTED", "$.channels", _input, scope,
            channels=dict(_channels(scope), synthetic_e3=np.zeros(SHAPE)))


def test_build_rejects_shape_and_grid_mismatch():
    scope = _scope("A")
    channels = _channels(scope)
    channels["synthetic_e2"] = np.zeros((4, 3, 4))
    _raises("GRID_MISMATCH", "$.channels", _input, scope, channels=channels)
    _raises("GRID_MISMATCH", "$.valid_fov", _input, scope, valid_fov=np.ones((4, 3, 4), dtype=bool))
    _raises("GRID_MISMATCH", "$.valid_fov", _input, scope, valid_fov=np.ones(SHAPE, dtype=np.uint8))
    _raises("GRID_MISMATCH", "$.grid.shape", _input, scope, grid=_grid((4, 3, 4)))
    _raises("GRID_MISMATCH", "$.grid", _input, scope, grid={k: v for k, v in _grid().items() if k != "affine"})
    _raises("GRID_MISMATCH", "$.grid.spacing_mm", _input, scope, grid=_grid(spacing=(3.0, 0.0, 0.5)))
    _raises("GRID_MISMATCH", "$.channels", _input, scope, channels=_channels(scope, shape=(3, 5)))


def test_valid_slices_follow_fov():
    scope = _scope("A")
    fov = np.ones(SHAPE, dtype=bool)
    fov[3] = False
    assert _input(scope, valid_fov=fov).valid_slices.tolist() == [[True, True, True, False]]
    assert _input(scope, grid=_grid(spacing=(0.5, 0.5, 3.0))).grids[0]["slice_axis"] == 2
    assert _input(scope, grid=_grid(spacing=(1.0, 1.0, 1.0))).grids[0]["slice_axis"] == 2
    assert _input(scope, grid=_grid(spacing=(3.0, 3.0, 0.5))).grids[0]["slice_axis"] == 1
    fov = np.ones(SHAPE, dtype=bool)
    fov[..., 4] = False
    unknown = _input(scope, valid_fov=fov, grid=_grid(spacing=None))
    assert unknown.grids[0]["slice_axis"] == 2 and unknown.valid_slices.tolist() == [[True] * 4 + [False]]
    flat = _scope("A", observation_mode="2d")
    single = _input(flat, shape=(3, 5), valid_fov=np.zeros((3, 5), dtype=bool))
    assert single.grids[0]["slice_axis"] is None and single.valid_slices.tolist() == [[False]]


def test_3d_only_model_rejects_2d_input():
    flat = _scope("A", observation_mode="2d")
    batch = _input(flat, shape=(3, 5))
    model = StubModel(_scope("A"), ("A",), anatomy_channels=3)
    _raises("MODE_NOT_SUPPORTED", "$.observation_mode", model, batch)
    assert model.calls == 0
    _raises("MODE_NOT_SUPPORTED", "$.scope.observation_mode", StubModel, flat, ("A",))

    class BothModes(StubModel):
        supported_modes = ("2d", "3d")

    both = BothModes(_scope("A"), ("A",), anatomy_channels=3)
    _raises("MODE_MISMATCH", "$.observation_mode", both, batch)
    assert both.calls == 0


def test_output_shapes_and_disabled_tasks_checked():
    scope = _au_scope()
    batch = _input(scope)
    output = StubModel(scope, ("A", "U"), anatomy_channels=3, abnormality_channels=1)(batch)
    assert tuple(output.anatomy_logits.shape) == (1, 3, *SHAPE) and tuple(output.abnormality_logits.shape) == (
        1, 1, *SHAPE)
    assert StubModel(scope, ("A",), anatomy_channels=3)(batch).abnormality_logits is None
    _raises("MODULE_TASKS_INVALID", "$.abnormality_logits", StubModel(scope, ("A",), 3, 1), batch)
    _raises("MODULE_TASKS_INVALID", "$.anatomy_logits", StubModel(scope, ("A", "U"), None, 1), batch)
    _raises("OUTPUT_SHAPE_MISMATCH", "$.anatomy_logits", StubModel(scope, ("A",), anatomy_channels=2), batch)
    _raises("OUTPUT_SHAPE_MISMATCH", "$.abnormality_logits", StubModel(scope, ("U",), abnormality_channels=2), batch)
    for tasks in ((), ("A", "A"), ("S",)):
        _raises("MODULE_TASKS_INVALID", "$.tasks", StubModel, scope, tasks)
    _raises("MODULE_TASKS_INVALID", "$.tasks", StubModel, _scope("A"), ("U",))
    wrong_batch = ModelOutputV5(anatomy_logits=torch.zeros((2, 3, *SHAPE)), abnormality_logits=None)
    _raises("OUTPUT_SHAPE_MISMATCH", "$.anatomy_logits", check_model_output_v5, wrong_batch, batch, scope, ("A",))


def test_nonfinite_input_rejected():
    scope = _scope("A")
    for bad in (math.nan, math.inf):
        channels = _channels(scope)
        channels["synthetic_e1"][0, 0, 0] = bad
        model = StubModel(scope, ("A",), anatomy_channels=3)
        _raises("NON_FINITE_INPUT", "$.image", model, _input(scope, channels=channels))
        assert model.calls == 0


def test_collate_rejects_mixed_grids():
    scope = _scope("A")
    first, second = _input(scope), _input(scope, to_original=[])
    batch = collate_model_inputs_v5([first, second])
    assert tuple(batch.image.shape) == (2, 2, *SHAPE) and tuple(batch.valid_slices.shape) == (2, 4)
    assert batch.to_original == ((DOWNSAMPLE,), ()) and len(batch.grids) == 2
    check_model_input_v5(batch, scope, ("3d",))
    other = _input(scope, shape=(4, 3, 6))
    _raises("GRID_MISMATCH", "$[1].image", collate_model_inputs_v5, [first, other])
    turned = _input(scope, grid=_grid(spacing=(0.5, 3.0, 0.5)))
    _raises("GRID_MISMATCH", "$[1].valid_slices", collate_model_inputs_v5, [first, turned])
    flat = _scope("A", observation_mode="2d")
    _raises("MODE_MISMATCH", "$[1].observation_mode", collate_model_inputs_v5, [_input(flat, shape=(3, 5)),
                                                                                 _input(scope)])
    _raises("INVALID_TYPE", "$", collate_model_inputs_v5, [])


def test_hand_built_batch_is_checked_not_trusted():
    scope = _scope("A")
    batch = _input(scope)
    skewed = dataclasses.replace(batch, grids=(dict(batch.grids[0], slice_axis=7),))
    _raises("GRID_MISMATCH", "$.grids[0]", check_model_input_v5, skewed, scope, ("3d",))
    _raises("CHANNEL_COUNT_MISMATCH", "$.image", check_model_input_v5,
            dataclasses.replace(batch, image=batch.image[:, :1]), scope, ("3d",))
    _raises("GRID_MISMATCH", "$.image", check_model_input_v5,
            dataclasses.replace(batch, image=batch.image.double()), scope, ("3d",))
    assert batch != _input(scope) and batch == batch
