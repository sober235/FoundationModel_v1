"""Geometry software checks for v5.1 (Proposal_v5 §3.2).

The 1e-4 voxel affine round trip is a software tolerance, not a registration
quality standard. Labels are resampled discretely with an explicit valid-FOV
mask, padding/cropping is exactly invertible with an explicit represented mask,
and downsampling reports positives that vanish so they are never used as negatives.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts_v5 import V5Error
from .geometry import inverse_transform_points, transform_points


def affine_roundtrip_error_v5(points_vox: np.ndarray, affine: np.ndarray) -> float:
    points = np.asarray(points_vox, dtype=float)
    back = inverse_transform_points(transform_points(points, affine), affine)
    return float(np.max(np.abs(back - points))) if len(points) else 0.0


def resample_labels_v5(labels: np.ndarray, src_affine: np.ndarray, dst_affine: np.ndarray,
                       dst_shape: tuple) -> tuple:
    """Nearest-neighbour label resampling; returns (labels on dst grid, valid_mask of in-FOV dst voxels)."""
    if not isinstance(labels, np.ndarray) or labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.integer):
        raise V5Error("DTYPE_MISMATCH", "$.labels")
    if labels.ndim != 3 or len(dst_shape) != 3:
        raise V5Error("GRID_MISMATCH", "$.labels")
    grid = np.indices(tuple(dst_shape)).reshape(3, -1).T
    source = inverse_transform_points(transform_points(grid, dst_affine), src_affine)
    nearest = np.rint(source)
    valid = np.all((nearest >= 0) & (nearest <= np.asarray(labels.shape) - 1), axis=1)
    out = np.zeros(len(grid), dtype=labels.dtype)
    idx = nearest[valid].astype(np.int64)
    out[valid] = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out.reshape(tuple(dst_shape)), valid.reshape(tuple(dst_shape))


def pad_or_crop_v5(array: np.ndarray, target_shape: tuple, fill=0) -> tuple:
    """Origin-anchored pad/crop per axis; returns (array in target shape, valid_mask of copied voxels)."""
    if len(target_shape) != array.ndim or any(int(t) < 1 for t in target_shape):
        raise V5Error("GRID_MISMATCH", "$.target_shape")
    out = np.full(tuple(target_shape), fill, dtype=array.dtype)
    valid = np.zeros(tuple(target_shape), dtype=bool)
    region = tuple(slice(0, min(s, t)) for s, t in zip(array.shape, target_shape))
    out[region] = array[region]
    valid[region] = True
    return out, valid


def restore_grid_v5(array: np.ndarray, valid_mask: np.ndarray, original_shape: tuple, fill=0) -> tuple:
    """Inverse of pad_or_crop_v5; returns (array on the original grid, represented_mask)."""
    if array.shape != valid_mask.shape or len(original_shape) != array.ndim:
        raise V5Error("GRID_MISMATCH", "$.original_shape")
    out = np.full(tuple(original_shape), fill, dtype=array.dtype)
    represented = np.zeros(tuple(original_shape), dtype=bool)
    region = tuple(slice(0, min(s, t)) for s, t in zip(array.shape, original_shape))
    keep = valid_mask[region]
    out[region] = np.where(keep, array[region], fill)
    represented[region] = keep
    return out, represented


def downsample_loss_report_v5(mask: np.ndarray, factor: tuple, class_names: list) -> dict:
    """Strided downsampling of bool [C, *spatial]; reports components (26-connected) that vanish."""
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim < 2:
        raise V5Error("DTYPE_MISMATCH", "$.mask")
    spatial = mask.shape[1:]
    if len(factor) != len(spatial) or not all(isinstance(f, (int, np.integer)) and f >= 1 for f in factor):
        raise V5Error("INVALID_FACTOR", "$.factor")
    if len(class_names) != mask.shape[0]:
        raise V5Error("CLASS_COUNT_MISMATCH", "$.class_names")
    steps = tuple(slice(None, None, int(f)) for f in factor)
    down_shape = tuple(-(-s // int(f)) for s, f in zip(spatial, factor))
    lattice = np.zeros(spatial, dtype=bool)
    lattice[steps] = True
    structure = ndimage.generate_binary_structure(len(spatial), len(spatial))
    lost = np.zeros((mask.shape[0], *down_shape), dtype=bool)
    classes = {}
    for c, name in enumerate(class_names):
        components, count = ndimage.label(mask[c], structure=structure)
        vanished = 0
        for label in range(1, count + 1):
            component = components == label
            if not (component & lattice).any():
                vanished += 1
                blocks = np.argwhere(component) // np.asarray(factor, dtype=np.int64)
                lost[(c, *blocks.T)] = True
        classes[name] = {"voxels_before": int(mask[c].sum()), "voxels_after": int(mask[c][steps].sum()),
                         "components_before": int(count), "components_vanished": vanished}
    return {"factor": [int(f) for f in factor], "classes": classes, "lost_positive_mask": lost}
