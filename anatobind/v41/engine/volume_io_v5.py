"""v5.1 volume readers: NIfTI, SKM-TEA H5 and MetaImage (plan_v5 T09b).

Every array keeps voxel order (x, y, z), as v41 ``geometry`` does; headers carry a RAS affine. A ref is a path
relative to the data root (H5 refs add ``#<dataset>``); absolute refs and ``..`` are refused. MetaImage is read
with numpy and zlib only. Errors are V5Error(code, path) and never echo file names or values.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import h5py
import nibabel as nib
import numpy as np

from ..contracts_v5 import V5Error
from ..geometry import geometry_from_nifti
from ..labels import discrete_labels, read_categorical_nifti

VOLUME_KINDS = ("image", "label")
HEADER_KEYS = ("shape", "spacing_mm", "affine", "axis_codes")
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
_MET_TYPES = {"MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_USHORT": "u2", "MET_SHORT": "i2", "MET_UINT": "u4",
              "MET_INT": "i4", "MET_FLOAT": "f4", "MET_DOUBLE": "f8"}


@dataclass(frozen=True, eq=False)
class VolumeV5:
    array: np.ndarray
    header: dict | None   # shape, spacing_mm, affine (RAS), axis_codes; None when the file has no geometry (H5)


def _split_ref(root: Path, ref: object) -> tuple[Path, str | None]:
    if not isinstance(ref, str):
        raise V5Error("INVALID_REF", "$.ref")
    location, mark, dataset = ref.partition("#")
    relative = PurePosixPath(location)
    if not location or relative.is_absolute() or ".." in relative.parts or "\\" in location:
        raise V5Error("INVALID_REF", "$.ref")
    return Path(root) / relative, (dataset if mark else None)


def _header_from_nifti(geometry: dict) -> dict:
    return {key: geometry[key] for key in HEADER_KEYS}


def _read_nifti(path: Path, kind: str) -> VolumeV5:
    if kind == "label":
        try:
            labels, meta = read_categorical_nifti(path)
        except ValueError:
            raise V5Error("LABEL_NOT_CATEGORICAL", "$.ref") from None
        if labels.ndim != 3:
            raise V5Error("VOLUME_NOT_3D", "$.ref")
        return VolumeV5(array=labels, header=_header_from_nifti(meta))
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise V5Error("VOLUME_NOT_3D", "$.ref")
    return VolumeV5(array=image.get_fdata(dtype=np.float32), header=_header_from_nifti(geometry_from_nifti(image)))


def _read_h5(path: Path, dataset: str | None, kind: str) -> VolumeV5:
    if not dataset:
        raise V5Error("H5_DATASET_REQUIRED", "$.ref")
    with h5py.File(path, "r") as handle:
        if dataset not in handle or not isinstance(handle[dataset], h5py.Dataset):
            raise V5Error("H5_DATASET_MISSING", "$.ref")
        data = handle[dataset][()]
    if kind == "image":
        if data.dtype.kind not in "biuf":
            raise V5Error("IMAGE_NOT_REAL", "$.ref")
        return VolumeV5(array=np.asarray(data, dtype=np.float32), header=None)
    if data.dtype.kind not in "biu" or not np.isin(data, (0, 1)).all():
        raise V5Error("LABEL_NOT_CATEGORICAL", "$.ref")
    return VolumeV5(array=data.astype(bool), header=None)


def _numbers(fields: dict, key: str, count: int) -> list[float]:
    try:
        values = [float(v) for v in fields[key].split()]
    except (KeyError, ValueError):
        raise V5Error("MHA_UNSUPPORTED", f"$.{key}") from None
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise V5Error("MHA_UNSUPPORTED", f"$.{key}")
    return values


def _flag(fields: dict, key: str, default: bool) -> bool:
    value = fields.get(key)
    if value is None:
        return default
    if value.lower() not in ("true", "false"):
        raise V5Error("MHA_UNSUPPORTED", f"$.{key}")
    return value.lower() == "true"


def read_mha_v5(path: Path) -> VolumeV5:
    """MetaImage (.mha, data LOCAL) as (x, y, z); TransformMatrix triple j is axis j's direction (LPS)."""
    blob = Path(path).read_bytes()
    fields: dict[str, str] = {}
    position = 0
    while True:
        end = blob.find(b"\n", position)
        if end < 0:
            raise V5Error("MHA_UNSUPPORTED", "$.ElementDataFile")
        line = blob[position:end].decode("ascii", errors="replace").strip()
        position = end + 1
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            fields[key] = value
        if key == "ElementDataFile":
            break
    if fields.get("ObjectType", "Image") != "Image":
        raise V5Error("MHA_UNSUPPORTED", "$.ObjectType")
    if fields.get("NDims") != "3":
        raise V5Error("MHA_UNSUPPORTED", "$.NDims")
    if fields["ElementDataFile"] != "LOCAL":
        raise V5Error("MHA_UNSUPPORTED", "$.ElementDataFile")
    if fields.get("ElementNumberOfChannels", "1") != "1":
        raise V5Error("MHA_UNSUPPORTED", "$.ElementNumberOfChannels")
    if not _flag(fields, "BinaryData", True):
        raise V5Error("MHA_UNSUPPORTED", "$.BinaryData")
    if fields.get("ElementType") not in _MET_TYPES:
        raise V5Error("MHA_UNSUPPORTED", "$.ElementType")
    msb = _flag(fields, "BinaryDataByteOrderMSB", _flag(fields, "ElementByteOrderMSB", False))
    compressed = _flag(fields, "CompressedData", False)
    dims = _numbers(fields, "DimSize", 3)
    if not all(d.is_integer() and d > 0 for d in dims):
        raise V5Error("MHA_UNSUPPORTED", "$.DimSize")
    shape = tuple(int(d) for d in dims)
    spacing = _numbers(fields, "ElementSpacing", 3) if "ElementSpacing" in fields else [1.0, 1.0, 1.0]
    offset = _numbers(fields, "Offset", 3) if "Offset" in fields else [0.0, 0.0, 0.0]
    direction = (_numbers(fields, "TransformMatrix", 9) if "TransformMatrix" in fields
                 else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    data = blob[position:]
    if compressed:
        try:
            data = zlib.decompress(data)
        except zlib.error:
            raise V5Error("MHA_DATA_SIZE_MISMATCH", "$.DimSize") from None
    dtype = np.dtype(_MET_TYPES[fields["ElementType"]]).newbyteorder(">" if msb else "<")
    if len(data) != int(np.prod(shape)) * dtype.itemsize:
        raise V5Error("MHA_DATA_SIZE_MISMATCH", "$.DimSize")
    array = np.frombuffer(data, dtype=dtype).reshape(shape[::-1]).transpose(2, 1, 0)
    array = np.ascontiguousarray(array).astype(dtype.newbyteorder("="))
    lps = np.eye(4)
    lps[:3, :3] = np.asarray(direction).reshape(3, 3).T @ np.diag(spacing)
    lps[:3, 3] = offset
    ras = _LPS_TO_RAS @ lps
    header = {"shape": list(shape), "spacing_mm": [float(s) for s in spacing],
              "affine": [[float(v) for v in row] for row in ras], "axis_codes": [str(c) for c in nib.aff2axcodes(ras)]}
    return VolumeV5(array=array, header=header)


def read_volume_v5(root: Path, ref: str, kind: str) -> VolumeV5:
    if kind not in VOLUME_KINDS:
        raise V5Error("INVALID_KIND", "$.kind")
    path, dataset = _split_ref(root, ref)
    name = path.name.lower()
    if not name.endswith((".h5", ".nii", ".nii.gz", ".mha")):
        raise V5Error("UNSUPPORTED_FORMAT", "$.ref")
    if dataset is not None and not name.endswith(".h5"):
        raise V5Error("INVALID_REF", "$.ref")
    if not path.is_file():
        raise V5Error("VOLUME_MISSING", "$.ref")
    if name.endswith(".h5"):
        return _read_h5(path, dataset, kind)
    if name.endswith(".mha"):
        volume = read_mha_v5(path)
        if kind == "image":
            return VolumeV5(array=volume.array.astype(np.float32), header=volume.header)
        try:
            return VolumeV5(array=discrete_labels(volume.array), header=volume.header)
        except ValueError:
            raise V5Error("LABEL_NOT_CATEGORICAL", "$.ref") from None
    return _read_nifti(path, kind)
