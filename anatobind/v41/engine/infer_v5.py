"""v5.1 full-volume inference and original-grid restore (plan_v5 T09c; Proposal §2.1, §4.3).

``predict_volume_v5`` tiles one observation (B=1) with windows on the model grid and averages softmax/sigmoid
probabilities by coverage, never logits. ``restore_probabilities_v5`` undoes the observation's transform chain
from last to first (chains list operations in the order they were applied): pad_or_crop with T07b
``restore_grid_v5``, crop by placing the window back, downsample by linear interpolation. Only float
probabilities are restored; discrete maps are never interpolated, and thresholds apply after restore.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import torch
from scipy import ndimage

from ..contracts_v5 import V5Error
from ..geometry_v5 import restore_grid_v5
from ..models.interface_v5 import ModelInputV5, PerceptionModelV5
from .train_v5 import PRECISIONS, autocast_v5

_REPRESENTED_TOLERANCE = 1e-6    # an interpolated support mask counts as represented only where it is ~1
_SLAB = 16                       # original-grid slices per map_coordinates call, to bound coordinate memory


# ---------------------------------------------------------------------------
# Sliding-window prediction


def _vector(value: object, ndim: int, path: str) -> tuple:
    if not (isinstance(value, (list, tuple)) and len(value) == ndim
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in value)):
        raise V5Error("INVALID_WINDOW", path)
    return tuple(int(v) for v in value)


def _starts(size: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] + window < size:
        starts.append(size - window)
    return starts


def _device(model: torch.nn.Module) -> torch.device:
    parameter = next(iter(model.parameters()), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _window_input(batch: ModelInputV5, image: torch.Tensor, fov: torch.Tensor, chain: list, start: tuple,
                  window: tuple, device: torch.device) -> ModelInputV5:
    region = tuple(slice(b, b + w) for b, w in zip(start, window))
    window_fov = fov[(slice(None), *region)]
    grid = dict(batch.grids[0], shape=list(window))
    if grid.get("affine") is not None:
        affine = np.asarray(grid["affine"], dtype=float)
        affine[:3, 3] += affine[:3, :3] @ np.asarray(start, dtype=float)
        grid["affine"] = affine.tolist()
    crop = {"op": "crop", "start": list(start), "input_shape": list(fov.shape[1:])}
    grid["transform_chain"] = [*chain, crop]
    axis = grid.get("slice_axis")
    if axis is None:
        slices = window_fov.reshape(1, -1).any(dim=1, keepdim=True)
    else:
        others = tuple(1 + d for d in range(len(window)) if d != axis)
        slices = window_fov.any(dim=others)
    return dataclasses.replace(batch, image=image[(slice(None), slice(None), *region)].to(device),
                               valid_fov=window_fov.to(device), valid_slices=slices.to(device), grids=(grid,),
                               to_original=(tuple(grid["transform_chain"]),))


def predict_volume_v5(model: PerceptionModelV5, batch: ModelInputV5, window: tuple, stride: tuple,
                      precision: str) -> dict:
    """Probabilities on the model grid: {"anatomy": [1+K, ...], "abnormality": [C_U, ...]} for enabled tasks."""
    if not isinstance(model, PerceptionModelV5):
        raise V5Error("INVALID_TYPE", "$.model")
    if not isinstance(batch, ModelInputV5):
        raise V5Error("INVALID_TYPE", "$.batch")
    if batch.image.ndim < 3 or batch.image.shape[0] != 1:
        raise V5Error("BATCH_NOT_SINGLE", "$.image")
    spatial = tuple(int(s) for s in batch.image.shape[2:])
    window = _vector(window, len(spatial), "$.window")
    stride = _vector(stride, len(spatial), "$.stride")
    if any(s > w for s, w in zip(stride, window)):
        raise V5Error("INVALID_WINDOW", "$.stride")
    if precision not in PRECISIONS:
        raise V5Error("INVALID_PRECISION", "$.precision")
    padded = tuple(max(s, w) for s, w in zip(spatial, window))
    image, fov, chain = batch.image, batch.valid_fov, list(batch.to_original[0])
    if padded != spatial:   # pad at the far end so every window fits; FOV is false in the padding
        inside = tuple(slice(0, s) for s in spatial)
        image = torch.zeros((1, image.shape[1], *padded), dtype=image.dtype, device=image.device)
        image[(slice(None), slice(None), *inside)] = batch.image
        fov = torch.zeros((1, *padded), dtype=torch.bool, device=fov.device)
        fov[(slice(None), *inside)] = batch.valid_fov
        chain.append({"op": "pad_or_crop", "input_shape": list(spatial)})
    device = _device(model)
    sums: dict[str, np.ndarray] = {}
    count = np.zeros(padded, dtype=np.int64)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in itertools.product(*(_starts(p, w, s) for p, w, s in zip(padded, window, stride))):
                region = tuple(slice(b, b + w) for b, w in zip(start, window))
                with autocast_v5(precision, device.type):
                    output = model(_window_input(batch, image, fov, chain, start, window, device))
                for key, logits in (("anatomy", output.anatomy_logits), ("abnormality", output.abnormality_logits)):
                    if logits is None:
                        continue
                    logits = logits.float()
                    prob = torch.softmax(logits, dim=1) if key == "anatomy" else torch.sigmoid(logits)
                    prob = prob[0].cpu().numpy().astype(np.float64)
                    if key not in sums:
                        sums[key] = np.zeros((prob.shape[0], *padded), dtype=np.float64)
                    sums[key][(slice(None), *region)] += prob
                count[region] += 1
    finally:
        model.train(was_training)
    keep = (slice(None), *(slice(0, s) for s in spatial))
    return {key: (total / count)[keep].astype(np.float32) for key, total in sums.items()}


# ---------------------------------------------------------------------------
# Restore to the original grid


def _shape(value: object, ndim: int, path: str) -> tuple:
    if not (isinstance(value, (list, tuple)) and len(value) == ndim
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in value)):
        raise V5Error("GRID_MISMATCH", path)
    return tuple(int(v) for v in value)


def _upsample_linear(array: np.ndarray, factor: tuple, shape: tuple) -> np.ndarray:
    """[C, *coarse] → [C, *shape]: original j reads the coarse coordinate j/f (order 1, edge values beyond)."""
    out = np.zeros((array.shape[0], *shape), dtype=np.float64)
    axes = [np.arange(n, dtype=np.float64) / f for n, f in zip(shape, factor)]
    for first in range(0, shape[0], _SLAB):
        part = [axes[0][first:first + _SLAB], *axes[1:]]
        coordinates = np.stack(np.meshgrid(*part, indexing="ij"))
        for c in range(array.shape[0]):
            out[c, first:first + _SLAB] = ndimage.map_coordinates(array[c].astype(np.float64), coordinates,
                                                                  order=1, mode="nearest")
    return out


def restore_probabilities_v5(prob: np.ndarray, to_original: list, original_shape: tuple) -> tuple:
    """Undo `to_original` last-first; returns (float probabilities [C, *original_shape], represented mask)."""
    if not isinstance(prob, np.ndarray) or not np.issubdtype(prob.dtype, np.floating):
        raise V5Error("DTYPE_MISMATCH", "$.prob")
    if prob.ndim < 2:
        raise V5Error("GRID_MISMATCH", "$.prob")
    if not isinstance(to_original, (list, tuple)):
        raise V5Error("INVALID_TYPE", "$.to_original")
    ndim = prob.ndim - 1
    current = prob.astype(np.float64)
    represented = np.ones(prob.shape[1:], dtype=bool)
    for index in reversed(range(len(to_original))):
        path = f"$.to_original[{index}]"
        op = to_original[index]
        if not isinstance(op, dict):
            raise V5Error("INVALID_TYPE", path)
        kind = op.get("op")
        if kind not in ("pad_or_crop", "crop", "downsample"):
            raise V5Error("TRANSFORM_UNSUPPORTED", f"{path}.op")
        target = _shape(op.get("input_shape"), ndim, path)
        size = current.shape[1:]
        if kind == "pad_or_crop":
            valid = np.zeros(size, dtype=bool)
            valid[tuple(slice(0, min(s, t)) for s, t in zip(size, target))] = True
            valid &= represented
            restored = [restore_grid_v5(channel, valid, target) for channel in current]
            current = np.stack([array for array, _ in restored])
            represented = restored[0][1]
        elif kind == "crop":
            start = op.get("start")
            if not (isinstance(start, (list, tuple)) and len(start) == ndim
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in start)):
                raise V5Error("GRID_MISMATCH", path)
            placed = np.zeros((current.shape[0], *target), dtype=np.float64)
            mask = np.zeros(target, dtype=bool)
            source, sink = [], []
            for b, n, t in zip(start, size, target):
                low, high = max(b, 0), min(b + n, t)
                source.append(slice(low - b, max(high, low) - b))
                sink.append(slice(low, max(high, low)))
            placed[(slice(None), *sink)] = current[(slice(None), *source)]
            mask[tuple(sink)] = represented[tuple(source)]
            current, represented = placed, mask
        else:
            factor = op.get("factor")
            if not (isinstance(factor, (list, tuple)) and len(factor) == ndim
                    and all(isinstance(f, int) and not isinstance(f, bool) and f >= 1 for f in factor)):
                raise V5Error("GRID_MISMATCH", path)
            if tuple(-(-t // f) for t, f in zip(target, factor)) != tuple(size):
                raise V5Error("GRID_MISMATCH", path)
            support = _upsample_linear(represented[None].astype(np.float64), factor, target)[0]
            current = _upsample_linear(current, factor, target)
            represented = support >= 1.0 - _REPRESENTED_TOLERANCE
        current = np.where(represented, current, 0.0)
    if tuple(current.shape[1:]) != tuple(original_shape):
        raise V5Error("GRID_MISMATCH", "$.original_shape")
    return current.astype(prob.dtype), represented
