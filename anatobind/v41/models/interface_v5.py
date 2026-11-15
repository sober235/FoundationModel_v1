"""Frozen v5.1 model input/output interface (plan_v5 §1.7, T09a).

A model sees image channels in the scope's ``input_modalities`` order, the valid field of view, the
observation mode, a valid-slice mask, the model-grid geometry and the transform back to the original grid.
Nothing else: labels, eligibility, review and visibility never enter a model (GC8). Logits stay on the model
grid; T09c restores them to the original grid. A model that does not support the batch's mode raises instead
of silently running another mode. Errors are V5Error(code, path) without input values.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..contracts_v5 import GEOMETRY_KEYS, SPATIAL_NDIM, ActiveScope, V5Error

_SPACING_TIE = 1e-6


@dataclass(frozen=True, eq=False)
class ModelInputV5:
    image: torch.Tensor          # float32 [B, C, *spatial], C in scope.input_modalities order
    valid_fov: torch.Tensor      # bool [B, *spatial]
    observation_mode: str
    valid_slices: torch.Tensor   # bool [B, n_slices]
    grids: tuple                 # tuple[dict, ...]: GEOMETRY_KEYS + slice_axis, one per item
    to_original: tuple           # tuple[tuple[dict, ...], ...]: transform chain to the original grid, one per item


@dataclass(frozen=True, eq=False)
class ModelOutputV5:
    anatomy_logits: torch.Tensor | None      # [B, 1+K, *spatial], channel 0 background, softmax
    abnormality_logits: torch.Tensor | None  # [B, C_U, *spatial], sigmoid


def _scope_tasks(scope: ActiveScope) -> tuple[str, ...]:
    return tuple(task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes))
                 if classes)


def slice_axis_v5(spacing_mm: list | None, mode: str) -> int | None:
    """Through-plane axis of a 3d grid: largest spacing; ties and unknown spacing take the last such axis.

    A 2d observation is a single slice, so it has no slice axis (None).
    """
    if mode != "3d":
        return None
    ndim = SPATIAL_NDIM[mode]
    if spacing_mm is None:
        return ndim - 1
    largest = max(spacing_mm)
    return max(i for i, s in enumerate(spacing_mm) if math.isclose(s, largest, rel_tol=_SPACING_TIE))


def _valid_slices(valid_fov: np.ndarray, slice_axis: int | None) -> np.ndarray:
    if slice_axis is None:
        return np.array([valid_fov.any()])
    others = tuple(axis for axis in range(valid_fov.ndim) if axis != slice_axis)
    return valid_fov.any(axis=others)


def _check_grid(grid: object, spatial: tuple) -> None:
    if not isinstance(grid, dict) or set(grid) != set(GEOMETRY_KEYS):
        raise V5Error("GRID_MISMATCH", "$.grid")
    if not isinstance(grid["shape"], (list, tuple)) or [int(s) for s in grid["shape"]] != list(spatial):
        raise V5Error("GRID_MISMATCH", "$.grid.shape")
    spacing = grid["spacing_mm"]
    if spacing is not None and not (
            isinstance(spacing, (list, tuple)) and len(spacing) == len(spatial)
            and all(isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and s > 0
                    for s in spacing)):
        raise V5Error("GRID_MISMATCH", "$.grid.spacing_mm")


def build_model_input_v5(channels: dict, scope: ActiveScope, valid_fov: np.ndarray, grid: dict,
                         to_original: list) -> ModelInputV5:
    """One observation (B=1) stacked in scope order; valid_slices is derived from valid_fov, not given."""
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not isinstance(channels, dict):
        raise V5Error("INVALID_TYPE", "$.channels")
    for index, name in enumerate(scope.input_modalities):
        if name not in channels:
            raise V5Error("INPUT_MODALITY_MISSING", f"$.input_modalities[{index}]")
    if set(channels) != set(scope.input_modalities):
        raise V5Error("INPUT_MODALITY_UNEXPECTED", "$.channels")
    mode = scope.observation_mode
    arrays = [channels[name] for name in scope.input_modalities]
    if any(not isinstance(a, np.ndarray) or a.ndim != SPATIAL_NDIM[mode] or a.shape != arrays[0].shape
           for a in arrays):
        raise V5Error("GRID_MISMATCH", "$.channels")
    spatial = arrays[0].shape
    if not isinstance(valid_fov, np.ndarray) or valid_fov.dtype != bool or valid_fov.shape != spatial:
        raise V5Error("GRID_MISMATCH", "$.valid_fov")
    _check_grid(grid, spatial)
    if not isinstance(to_original, list) or not all(isinstance(op, dict) for op in to_original):
        raise V5Error("INVALID_TYPE", "$.to_original")
    axis = slice_axis_v5(grid["spacing_mm"], mode)
    image = torch.from_numpy(np.stack([np.asarray(a, dtype=np.float32) for a in arrays])[None])
    return ModelInputV5(image=image, valid_fov=torch.from_numpy(valid_fov.copy())[None], observation_mode=mode,
                        valid_slices=torch.from_numpy(_valid_slices(valid_fov, axis))[None],
                        grids=(dict(copy.deepcopy(grid), slice_axis=axis),),
                        to_original=(tuple(copy.deepcopy(op) for op in to_original),))


def collate_model_inputs_v5(items: list) -> ModelInputV5:
    if not isinstance(items, list) or not items or not all(isinstance(item, ModelInputV5) for item in items):
        raise V5Error("INVALID_TYPE", "$")
    first = items[0]
    for index, item in enumerate(items):
        if item.observation_mode != first.observation_mode:
            raise V5Error("MODE_MISMATCH", f"$[{index}].observation_mode")
        if item.image.shape[1:] != first.image.shape[1:] or item.valid_fov.shape[1:] != first.valid_fov.shape[1:]:
            raise V5Error("GRID_MISMATCH", f"$[{index}].image")
        if item.valid_slices.shape[1:] != first.valid_slices.shape[1:]:
            raise V5Error("GRID_MISMATCH", f"$[{index}].valid_slices")
    return ModelInputV5(image=torch.cat([item.image for item in items]),
                        valid_fov=torch.cat([item.valid_fov for item in items]),
                        observation_mode=first.observation_mode,
                        valid_slices=torch.cat([item.valid_slices for item in items]),
                        grids=tuple(grid for item in items for grid in item.grids),
                        to_original=tuple(chain for item in items for chain in item.to_original))


def check_model_input_v5(batch: ModelInputV5, scope: ActiveScope, supported_modes: tuple) -> None:
    if not isinstance(batch, ModelInputV5):
        raise V5Error("INVALID_TYPE", "$")
    if batch.observation_mode not in supported_modes:
        raise V5Error("MODE_NOT_SUPPORTED", "$.observation_mode")
    if batch.observation_mode != scope.observation_mode:
        raise V5Error("MODE_MISMATCH", "$.observation_mode")
    image = batch.image
    if not isinstance(image, torch.Tensor) or image.dtype != torch.float32 \
            or image.ndim != 2 + SPATIAL_NDIM[batch.observation_mode]:
        raise V5Error("GRID_MISMATCH", "$.image")
    if image.shape[1] != len(scope.input_modalities):
        raise V5Error("CHANNEL_COUNT_MISMATCH", "$.image")
    size, spatial = image.shape[0], tuple(image.shape[2:])
    if not isinstance(batch.valid_fov, torch.Tensor) or batch.valid_fov.dtype != torch.bool \
            or tuple(batch.valid_fov.shape) != (size, *spatial):
        raise V5Error("GRID_MISMATCH", "$.valid_fov")
    if len(batch.grids) != size or len(batch.to_original) != size:
        raise V5Error("GRID_MISMATCH", "$.grids")
    for index, grid in enumerate(batch.grids):
        if not isinstance(grid, dict) or list(grid.get("shape", ())) != list(spatial):
            raise V5Error("GRID_MISMATCH", f"$.grids[{index}]")
        axis = grid.get("slice_axis")
        if axis is not None and not (isinstance(axis, int) and not isinstance(axis, bool) and 0 <= axis < len(spatial)):
            raise V5Error("GRID_MISMATCH", f"$.grids[{index}]")
        n_slices = 1 if axis is None else spatial[axis]
        if not isinstance(batch.valid_slices, torch.Tensor) or batch.valid_slices.dtype != torch.bool \
                or tuple(batch.valid_slices.shape) != (size, n_slices):
            raise V5Error("GRID_MISMATCH", "$.valid_slices")
    if not bool(torch.isfinite(image).all()):
        raise V5Error("NON_FINITE_INPUT", "$.image")


def _check_logits(logits: object, expected: tuple, enabled: bool, path: str) -> None:
    if not enabled:
        if logits is not None:
            raise V5Error("MODULE_TASKS_INVALID", path)
        return
    if logits is None:
        raise V5Error("MODULE_TASKS_INVALID", path)
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point() or tuple(logits.shape) != expected:
        raise V5Error("OUTPUT_SHAPE_MISMATCH", path)


def check_model_output_v5(output: ModelOutputV5, batch: ModelInputV5, scope: ActiveScope, tasks: tuple) -> None:
    if not isinstance(output, ModelOutputV5):
        raise V5Error("INVALID_TYPE", "$")
    size, spatial = batch.image.shape[0], tuple(batch.image.shape[2:])
    _check_logits(output.anatomy_logits, (size, 1 + len(scope.anatomy_classes), *spatial), "A" in tasks,
                  "$.anatomy_logits")
    _check_logits(output.abnormality_logits, (size, len(scope.abnormality_classes), *spatial), "U" in tasks,
                  "$.abnormality_logits")


class PerceptionModelV5(torch.nn.Module):
    """Base of every v5 model: forward checks the input, runs ``_forward_v5`` and checks the output."""

    supported_modes: tuple = ("3d",)

    def __init__(self, scope: ActiveScope, tasks: tuple):
        super().__init__()
        if not isinstance(scope, ActiveScope):
            raise V5Error("INVALID_TYPE", "$.scope")
        tasks = tuple(tasks)
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(_scope_tasks(scope)):
            raise V5Error("MODULE_TASKS_INVALID", "$.tasks")
        if scope.observation_mode not in self.supported_modes:
            raise V5Error("MODE_NOT_SUPPORTED", "$.scope.observation_mode")
        self.scope = scope
        self.tasks = tasks

    def forward(self, batch: ModelInputV5) -> ModelOutputV5:
        check_model_input_v5(batch, self.scope, self.supported_modes)
        output = self._forward_v5(batch)
        check_model_output_v5(output, batch, self.scope, self.tasks)
        return output

    def _forward_v5(self, batch: ModelInputV5) -> ModelOutputV5:
        raise NotImplementedError
