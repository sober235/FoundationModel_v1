"""Surface distances from the vendored Google DeepMind surface-distance at a pinned commit (Proposal_v5 §8.3, [R6]).

HD95 and NSD use that implementation's surfel-area weighting (boundary length in 2D); nothing here
re-implements the metric.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._vendor.surface_distance import metrics as _sd

SURFACE_SOURCE_COMMIT = "1f805cea446805e559badc80dd27f34175e4173d"


def surface_distances_v5(truth: np.ndarray, pred: np.ndarray, spacing_mm: tuple[float, ...]) -> dict:
    with warnings.catch_warnings():
        # The pinned source calls scipy.ndimage.filters/morphology, deprecated aliases in SciPy >= 1.8.
        warnings.simplefilter("ignore", DeprecationWarning)
        return _sd.compute_surface_distances(np.asarray(truth, dtype=bool), np.asarray(pred, dtype=bool),
                                             tuple(float(s) for s in spacing_mm))


def hd95_v5(distances: dict) -> float:
    return float(_sd.compute_robust_hausdorff(distances, 95))


def nsd_v5(distances: dict, tolerance_mm: float) -> float:
    return float(_sd.compute_surface_dice_at_tolerance(distances, tolerance_mm))
