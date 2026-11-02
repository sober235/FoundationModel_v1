import nibabel as nib
import numpy as np
import pytest

from anatobind.v41.geometry import (
    audit_geometry,
    geometry_from_nifti,
    inverse_transform_points,
    pad_to_shape,
    resample_labels_nearest,
    transform_points,
)
from anatobind.v41.labels import (
    LABEL_ROUND_ATOL,
    channel_stack_agreement,
    discrete_labels,
    label_value_report,
    read_categorical_nifti,
)


def test_scaled_categorical_labels():
    assert discrete_labels(np.array([0.0, 0.999999]))[1] == 1
    with pytest.raises(ValueError):
        discrete_labels(np.array([0.25, 0.75]))


def test_discrete_labels_tolerance_and_overflow():
    assert LABEL_ROUND_ATOL == 1e-5
    assert discrete_labels(np.array([2.0 + 5e-6])).tolist() == [2]
    with pytest.raises(ValueError):
        discrete_labels(np.array([2.0 + 5e-5]))
    with pytest.raises(ValueError):
        discrete_labels(np.array([np.nan]))
    with pytest.raises(ValueError):
        discrete_labels(np.array([70000.0]), dtype=np.int16)
    assert discrete_labels(np.array([5], dtype=np.int64)).dtype == np.int16


def _save(path, data, affine, scl=None):
    image = nib.Nifti1Image(data, affine)
    if scl is not None:
        image.header.set_slope_inter(*scl)
        image.header.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def test_read_categorical_nifti_applies_proxy_scaling(tmp_path):
    raw = np.array([[[0, 1], [2, 3]]], dtype=np.int16)
    _save(tmp_path / "scaled.nii.gz", raw, np.diag([0.5, 0.5, 2.0, 1.0]), scl=(1.0, 0.0))
    labels, meta = read_categorical_nifti(tmp_path / "scaled.nii.gz")
    assert labels.dtype == np.int16 and labels.tolist() == raw.tolist()
    assert meta["spacing_mm"] == [0.5, 0.5, 2.0]
    assert meta["values_present"] == [0, 1, 2, 3]
    assert meta["label_value_qc"] == "pass"
    scaled = nib.Nifti1Image(np.array([[[0, 2], [4, 6]]], dtype=np.int16), np.eye(4))
    scaled.header.set_slope_inter(0.5, 0.0)
    nib.save(scaled, str(tmp_path / "half.nii.gz"))
    labels, meta = read_categorical_nifti(tmp_path / "half.nii.gz")
    assert labels.tolist() == [[[0, 1], [2, 3]]]


def test_read_categorical_nifti_rejects_probability_and_flags_all_zero(tmp_path):
    _save(tmp_path / "prob.nii.gz", np.array([[[0.25, 0.75]]], dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError):
        read_categorical_nifti(tmp_path / "prob.nii.gz")
    _save(tmp_path / "zero.nii.gz", np.zeros((2, 2, 2), dtype=np.int16), np.eye(4))
    labels, meta = read_categorical_nifti(tmp_path / "zero.nii.gz")
    assert meta["label_value_qc"] == "needs_review"
    assert meta["label_value_reason"] == "ALL_ZERO"


def test_label_value_report_flags_unexpected_values():
    report = label_value_report(np.array([0, 1, 6, 7]), allowed=range(0, 7))
    assert report["unexpected_values"] == [7]
    assert report["pass"] is False


def test_geometry_from_nifti_records_physical_space(tmp_path):
    affine = np.array([[0, -0.5, 0, 10], [0.5, 0, 0, -20], [0, 0, 2.0, 5], [0, 0, 0, 1]])
    _save(tmp_path / "g.nii.gz", np.zeros((4, 6, 8), dtype=np.int16), affine)
    geometry = geometry_from_nifti(nib.load(str(tmp_path / "g.nii.gz")))
    assert geometry["shape"] == [4, 6, 8]
    assert geometry["spacing_mm"] == [0.5, 0.5, 2.0]
    assert geometry["axis_codes"] == ["A", "L", "S"]
    assert geometry["physical_fov_mm"] == [2.0, 3.0, 16.0]
    assert geometry["units"] == "unknown" and geometry["units_declared"] == "unknown"
    assert "qform_code" in geometry and "sform_code" in geometry


def test_same_shape_wrong_affine_fails_audit():
    image = {"shape": [8, 8, 8], "spacing_mm": [1.0, 1.0, 1.0], "affine": np.eye(4).tolist(), "axis_codes": ["R", "A", "S"]}
    shifted = dict(image, affine=(np.eye(4) + np.array([[0, 0, 0, 3.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])).tolist())
    report = audit_geometry(image, shifted)
    assert report["same_shape"] is True
    assert report["same_affine"] is False
    assert report["status"] == "fail"
    assert "AFFINE_MISMATCH" in report["reasons"]
    assert audit_geometry(image, dict(image))["status"] == "pass"
    other_shape = dict(image, shape=[8, 8, 9])
    assert "SHAPE_MISMATCH" in audit_geometry(image, other_shape)["reasons"]


def test_affine_roundtrip_is_numerically_invertible():
    rng = np.random.default_rng(20261101)
    affine = np.eye(4)
    affine[:3, :3] = np.array([[0.3125, 0.01, 0], [-0.02, 0.3125, 0.05], [0, 0.03, 0.8]])
    affine[:3, 3] = [-80.5, 12.25, 33.0]
    points = rng.uniform(0, 512, size=(200, 3))
    world = transform_points(points, affine)
    back = inverse_transform_points(world, affine)
    assert np.allclose(back, points, atol=1e-4)
    assert world.shape == (200, 3)
    assert np.allclose(transform_points(np.zeros((1, 3)), affine)[0], affine[:3, 3])


def test_label_resampling_preserves_categories():
    labels = np.zeros((10, 10, 10), dtype=np.int16)
    labels[2:6, 2:6, 2:6] = 3
    labels[6:9, 6:9, 6:9] = 5
    src = np.diag([1.0, 1.0, 1.0, 1.0])
    dst = np.diag([0.5, 0.5, 0.5, 1.0])
    out = resample_labels_nearest(labels, src, dst, (20, 20, 20))
    assert out.dtype == labels.dtype
    assert set(np.unique(out).tolist()) <= {0, 3, 5}
    assert (out == 3).sum() == 8 * (labels == 3).sum()


def test_padding_is_separated_from_valid_fov():
    data = np.ones((4, 4, 4), dtype=np.int16)
    padded, valid = pad_to_shape(data, (6, 8, 4), fill=0)
    assert padded.shape == (6, 8, 4) and valid.shape == (6, 8, 4)
    assert valid.sum() == 64 and padded[valid].min() == 1
    assert padded[~valid].max() == 0
    with pytest.raises(ValueError, match="TARGET_SMALLER_THAN_INPUT"):
        pad_to_shape(data, (3, 4, 4))


def test_channel_stack_agreement_identifies_channel_order():
    labels = np.zeros((6, 6, 6), dtype=np.int16)
    labels[0] = 1
    labels[1] = 2
    labels[2] = 3
    stack = np.zeros((6, 6, 6, 3), dtype=bool)
    stack[..., 0] = labels == 2
    stack[..., 1] = labels == 1
    stack[..., 2] = labels == 3
    report = channel_stack_agreement(stack, labels)
    assert report["channel_to_label"] == [2, 1, 3]
    assert report["dice"] == [1.0, 1.0, 1.0]
    assert report["consistent"] is True
    stack[..., 2] = False
    report = channel_stack_agreement(stack, labels)
    assert report["channel_to_label"][2] is None
    assert report["consistent"] is False
