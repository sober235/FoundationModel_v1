from __future__ import annotations

import numpy as np
import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.geometry_v5 import (
    affine_roundtrip_error_v5,
    downsample_loss_report_v5,
    pad_or_crop_v5,
    resample_labels_v5,
    restore_grid_v5,
)

# NumPy 2.0 with macOS Accelerate emits spurious divide-by-zero/overflow/invalid RuntimeWarnings from matmul for some shapes;
# the values are finite and match einsum. A real overflow still fails these tests' value assertions.
pytestmark = pytest.mark.filterwarnings("ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning")


def _anisotropic_rotated_affine() -> np.ndarray:
    theta = 0.3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.3125, 0.3125, 0.8])
    affine[:3, 3] = [-80.5, 12.25, 33.0]
    return affine


def test_affine_roundtrip_within_1e_4_voxel():
    points = np.random.default_rng(0).uniform(0, 512, size=(1000, 3))
    assert affine_roundtrip_error_v5(points, _anisotropic_rotated_affine()) < 1e-4


def test_label_resampling_is_discrete():
    labels = np.zeros((6, 6, 6), dtype=np.int16)
    labels[1:4, 2:5, 0:3] = 3
    labels[4:, :, 4:] = 5
    same, valid = resample_labels_v5(labels, np.eye(4), np.eye(4), labels.shape)
    assert np.array_equal(same, labels) and valid.all() and same.dtype == labels.dtype
    shifted = np.eye(4)
    shifted[:3, 3] = [2.0, 0.0, 0.0]
    moved, valid = resample_labels_v5(labels, np.eye(4), shifted, labels.shape)
    assert set(np.unique(moved[valid])) <= set(np.unique(labels))
    assert not valid[-2:].any() and valid[:-2].all()
    with pytest.raises(V5Error) as error:
        resample_labels_v5(labels.astype(np.float32), np.eye(4), np.eye(4), labels.shape)
    assert error.value.code == "DTYPE_MISMATCH"


def test_original_grid_write_back_is_exact():
    array = np.arange(4 * 5 * 3, dtype=np.int32).reshape(4, 5, 3)
    padded, valid = pad_or_crop_v5(array, (6, 5, 4), fill=0)
    assert padded.shape == (6, 5, 4) and valid.sum() == array.size and not valid[4:].any()
    restored, represented = restore_grid_v5(padded, valid, array.shape, fill=-1)
    assert np.array_equal(restored, array) and represented.all()


def test_cropped_voxels_are_not_reported_as_restored():
    array = np.ones((4, 4, 4), dtype=np.int16)
    cropped, valid = pad_or_crop_v5(array, (2, 4, 4), fill=0)
    restored, represented = restore_grid_v5(cropped, valid, array.shape, fill=-1)
    assert represented[:2].all() and not represented[2:].any() and (restored[2:] == -1).all()


def test_downsample_reports_vanished_small_lesion():
    mask = np.zeros((2, 16, 16, 16), dtype=bool)
    mask[0, 5, 5, 5] = True
    mask[1, 0:8, 0:8, 0:8] = True
    report = downsample_loss_report_v5(mask, (2, 2, 2), ["small", "large"])
    small, large = report["classes"]["small"], report["classes"]["large"]
    assert (small["components_before"], small["components_vanished"], small["voxels_after"]) == (1, 1, 0)
    assert (large["components_before"], large["components_vanished"]) == (1, 0) and large["voxels_after"] == 64
    lost = report["lost_positive_mask"]
    assert lost.shape == (2, 8, 8, 8) and lost[0, 2, 2, 2] and lost[0].sum() == 1 and not lost[1].any()


@pytest.mark.parametrize("mask,factor,names,code", [
    (np.zeros((1, 4, 4, 4), dtype=np.uint8), (2, 2, 2), ["a"], "DTYPE_MISMATCH"),
    (np.zeros((1, 4, 4, 4), dtype=bool), (2, 2), ["a"], "INVALID_FACTOR"),
    (np.zeros((1, 4, 4, 4), dtype=bool), (2, 0, 2), ["a"], "INVALID_FACTOR"),
    (np.zeros((1, 4, 4, 4), dtype=bool), (2, 2, 2), ["a", "b"], "CLASS_COUNT_MISMATCH"),
])
def test_downsample_rejects_non_bool_or_bad_factor(mask, factor, names, code):
    with pytest.raises(V5Error) as error:
        downsample_loss_report_v5(mask, factor, names)
    assert error.value.code == code


def test_downsample_reports_lost_blocks_of_surviving_thin_structure():
    mask = np.zeros((1, 8, 8, 8), dtype=bool)
    mask[0, 1, :, :] = True  # 1-voxel plate at odd x: never on the stride-2 lattice
    mask[0, 0, 0, 0] = True  # one lattice voxel keeps the component alive
    report = downsample_loss_report_v5(mask, (2, 2, 2), ["cartilage"])
    assert report["classes"]["cartilage"]["components_vanished"] == 0
    lost = report["lost_positive_mask"][0]
    assert lost.sum() == 15 and not lost[0, 0, 0] and not lost[1:].any()


def test_resample_half_voxel_tie_rounds_half_up():
    labels = np.arange(8, dtype=np.int16).reshape(8, 1, 1)
    dst = np.eye(4)
    dst[0, 3] = 0.5
    out, valid = resample_labels_v5(labels, np.eye(4), dst, (7, 1, 1))
    assert out.ravel().tolist() == [1, 2, 3, 4, 5, 6, 7] and valid.all()
