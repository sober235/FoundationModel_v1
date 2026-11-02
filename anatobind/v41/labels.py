"""Categorical label decoding with NIfTI proxy scaling preserved.

`discrete_labels` and `LABEL_ROUND_ATOL` are copied verbatim from
anatobind/aur/labels.py of sober235/Foundation_Model @4deeead
(blob 303c510eb507ca0ebf35520e3cdf7cf13eeded84). The tolerance repairs
floating-point representation only; probability maps are rejected.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from anatobind.v41.geometry import geometry_from_nifti

LABEL_ROUND_ATOL = 1e-5


def discrete_labels(values, dtype=np.int16):
    """Decode categorical maps, including NIfTI proxy scaling, without truncating labels.

    Rounding only repairs floating-point representation within LABEL_ROUND_ATOL;
    probability maps, nonfinite values and integer overflow are invalid inputs.
    Errors deliberately contain no private filenames or voxel values.
    """
    data = np.asarray(values)
    dtype = np.dtype(dtype)
    if data.dtype.kind not in "biuf" or dtype.kind not in "iu" or not np.isfinite(data).all():
        raise ValueError("discrete label map requires finite real categorical values")
    # Keep integer inputs exact, including int64/uint64 values beyond float64's
    # consecutive-integer range. Floating categorical maps still need rounding.
    rounded = data if data.dtype.kind in "biu" else np.rint(data)
    limits = np.iinfo(dtype)
    if ((data.dtype.kind == "f" and np.any(np.abs(data - rounded) > LABEL_ROUND_ATOL))
            or (rounded.size and (int(rounded.min()) < limits.min or int(rounded.max()) > limits.max))):
        raise ValueError("discrete label map requires near-integer values within the output range")
    return rounded.astype(dtype)


def label_value_report(labels: np.ndarray, allowed) -> dict:
    present = sorted(int(v) for v in np.unique(labels))
    allowed_set = {int(v) for v in allowed}
    unexpected = [v for v in present if v not in allowed_set]
    return {"values_present": present, "unexpected_values": unexpected, "pass": not unexpected}


def read_categorical_nifti(path: Path, dtype=np.int16) -> tuple[np.ndarray, dict]:
    image = nib.load(str(path))
    meta = geometry_from_nifti(image)
    slope, inter = image.header.get_slope_inter()
    meta["scl_slope"] = None if slope is None else float(slope)
    meta["scl_inter"] = None if inter is None else float(inter)
    meta["dtype_on_disk"] = str(image.get_data_dtype())
    labels = discrete_labels(np.asarray(image.dataobj), dtype)
    values = sorted(int(v) for v in np.unique(labels))
    meta["values_present"] = values
    if values == [0] or not values:
        meta["label_value_qc"] = "needs_review"
        meta["label_value_reason"] = "ALL_ZERO"
    else:
        meta["label_value_qc"] = "pass"
        meta["label_value_reason"] = None
    return labels, meta


def channel_stack_agreement(stack: np.ndarray, labels: np.ndarray) -> dict:
    """Match boolean channels against integer labels by Dice; used to verify source channel order."""
    if stack.shape[:-1] != labels.shape:
        raise ValueError("CHANNEL_STACK_SHAPE_MISMATCH")
    values = [int(v) for v in np.unique(labels) if v != 0]
    mapping: list[int | None] = []
    dice: list[float | None] = []
    for channel in range(stack.shape[-1]):
        mask = stack[..., channel]
        best, best_dice = None, 0.0
        if mask.any():
            for value in values:
                truth = labels == value
                score = 2.0 * np.logical_and(mask, truth).sum() / (mask.sum() + truth.sum())
                if score > best_dice:
                    best, best_dice = value, float(score)
        mapping.append(best)
        dice.append(None if best is None else round(best_dice, 6))
    assigned = [m for m in mapping if m is not None]
    consistent = len(assigned) == stack.shape[-1] and len(set(assigned)) == len(assigned) \
        and all(d is not None and d > 0.99 for d in dice)
    return {"channel_to_label": mapping, "dice": dice, "consistent": consistent, "labels_present": values}
