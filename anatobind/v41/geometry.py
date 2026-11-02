"""Physical-space description, comparison and explicit transforms.

Arrays keep NIfTI voxel order (x, y, z); world coordinates follow the header
affine. Nothing here edits a header to make two grids "match".
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_UNIT_NAMES = {"unknown": "unknown", "meter": "m", "mm": "mm", "micron": "um"}


def geometry_from_nifti(image) -> dict:
    header = image.header
    affine = np.asarray(image.affine, dtype=float)
    shape = [int(v) for v in image.shape[:3]]
    spacing = [float(round(v, 6)) for v in header.get_zooms()[:3]]
    try:
        xyz_unit = header.get_xyzt_units()[0]
    except Exception:  # noqa: BLE001 - header without unit fields
        xyz_unit = "unknown"
    return {
        "shape": shape,
        "spacing_mm": spacing,
        "affine": [[float(v) for v in row] for row in affine],
        "axis_codes": [str(c) for c in __import__("nibabel").aff2axcodes(affine)],
        "units": _UNIT_NAMES.get(xyz_unit, str(xyz_unit)),
        "units_declared": str(xyz_unit),
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
        "physical_fov_mm": [float(round(s * n, 6)) for s, n in zip(spacing, shape)],
    }


SCHEMA_GEOMETRY_KEYS = ("shape", "spacing_mm", "affine", "axis_codes", "units")


def to_schema_geometry(geometry: dict) -> dict:
    if geometry.get("units") != "mm":
        raise ValueError("UNITS_NOT_MM")
    return {key: geometry[key] for key in SCHEMA_GEOMETRY_KEYS}


def audit_geometry(image: dict, label: dict, atol_affine: float = 1e-3, atol_spacing: float = 1e-3) -> dict:
    reasons = []
    same_shape = list(image["shape"]) == list(label["shape"])
    if not same_shape:
        reasons.append("SHAPE_MISMATCH")
    same_spacing = np.allclose(image["spacing_mm"], label["spacing_mm"], atol=atol_spacing)
    if not same_spacing:
        reasons.append("SPACING_MISMATCH")
    same_affine = np.allclose(np.asarray(image["affine"], float), np.asarray(label["affine"], float), atol=atol_affine)
    if not same_affine:
        reasons.append("AFFINE_MISMATCH")
    same_axes = list(image.get("axis_codes", [])) == list(label.get("axis_codes", []))
    if not same_axes:
        reasons.append("AXIS_CODE_MISMATCH")
    return {
        "same_shape": same_shape,
        "same_spacing": bool(same_spacing),
        "same_affine": bool(same_affine),
        "same_axis_codes": same_axes,
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
    }


def transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("POINTS_SHAPE")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (np.asarray(affine, dtype=float) @ homogeneous.T).T[:, :3]


def inverse_transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return transform_points(points, np.linalg.inv(np.asarray(affine, dtype=float)))


def resample_labels_nearest(labels: np.ndarray, src_affine: np.ndarray, dst_affine: np.ndarray,
                            dst_shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(dst_shape).reshape(3, -1).T
    world = transform_points(grid, dst_affine)
    source_voxels = inverse_transform_points(world, src_affine)
    sampled = ndimage.map_coordinates(labels, source_voxels.T, order=0, mode="constant", cval=0)
    return sampled.reshape(dst_shape).astype(labels.dtype)


def pad_to_shape(array: np.ndarray, target_shape: tuple[int, ...], fill=0) -> tuple[np.ndarray, np.ndarray]:
    if len(target_shape) != array.ndim or any(t < s for t, s in zip(target_shape, array.shape)):
        raise ValueError("TARGET_SMALLER_THAN_INPUT")
    pads = [(0, t - s) for t, s in zip(target_shape, array.shape)]
    padded = np.pad(array, pads, mode="constant", constant_values=fill)
    valid = np.pad(np.ones(array.shape, dtype=bool), pads, mode="constant", constant_values=False)
    return padded, valid
