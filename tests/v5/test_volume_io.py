"""v5.1 volume readers: NIfTI, SKM H5 and MetaImage (plan_v5 T09b). Arrays are (x, y, z) voxel order."""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.engine.volume_io_v5 import VolumeV5, read_mha_v5, read_volume_v5
from tests.v5.factories import write_mha, write_nifti, write_skm_h5

AFFINE = np.array([[0.5, 0, 0, 10], [0, 0.6, 0, -5], [0, 0, 2.0, 3], [0, 0, 0, 1]])
# Row j is the direction of axis j (MetaIO TransformMatrix); a proper rotation that permutes the axes.
DIRECTION = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
SPACING = (0.5, 0.7, 3.3)
OFFSET = (10.0, 20.0, -5.0)


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def test_nifti_image_keeps_scaling_and_ras_header(tmp_path):
    raw = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    write_nifti(tmp_path / "img" / "a.nii.gz", raw, AFFINE, slope=0.5)
    volume = read_volume_v5(tmp_path, "img/a.nii.gz", "image")
    assert isinstance(volume, VolumeV5)
    assert volume.array.dtype == np.float32 and np.array_equal(volume.array, raw * np.float32(0.5))
    assert set(volume.header) == {"shape", "spacing_mm", "affine", "axis_codes"}
    assert volume.header["shape"] == [2, 3, 4] and volume.header["spacing_mm"] == pytest.approx([0.5, 0.6, 2.0])
    assert np.allclose(volume.header["affine"], AFFINE) and volume.header["axis_codes"] == ["R", "A", "S"]
    near = raw.astype(np.float32) + np.float32(1e-6)
    write_nifti(tmp_path / "lab.nii", near, AFFINE)
    label = read_volume_v5(tmp_path, "lab.nii", "label")
    assert np.array_equal(label.array, raw) and label.array.dtype.kind == "i"
    write_nifti(tmp_path / "prob.nii.gz", np.full((2, 2, 2), 0.5, dtype=np.float32), AFFINE)
    _raises("LABEL_NOT_CATEGORICAL", "$.ref", read_volume_v5, tmp_path, "prob.nii.gz", "label")
    write_nifti(tmp_path / "four.nii.gz", np.zeros((2, 2, 2, 2), dtype=np.float32), AFFINE)
    _raises("VOLUME_NOT_3D", "$.ref", read_volume_v5, tmp_path, "four.nii.gz", "image")


def test_skm_h5_reads_named_dataset_only(tmp_path):
    echo1 = np.full((2, 3, 4), 1.5, dtype=np.float32)
    echo2 = np.full((2, 3, 4), 2.5, dtype=np.float64)
    seg = np.zeros((2, 3, 4, 6), dtype=np.uint8)
    seg[0, 0, 0, 5] = 1
    write_skm_h5(tmp_path / "skm" / "case.h5", echo1, echo2, seg)
    first = read_volume_v5(tmp_path, "skm/case.h5#echo1", "image")
    assert first.header is None and first.array.dtype == np.float32 and np.array_equal(first.array, echo1)
    second = read_volume_v5(tmp_path, "skm/case.h5#echo2", "image")
    assert second.array.dtype == np.float32 and np.array_equal(second.array, echo2.astype(np.float32))
    labels = read_volume_v5(tmp_path, "skm/case.h5#seg", "label")
    assert labels.array.dtype == bool and labels.array.shape == (2, 3, 4, 6)
    assert labels.array[0, 0, 0, 5] and int(labels.array.sum()) == 1
    _raises("H5_DATASET_REQUIRED", "$.ref", read_volume_v5, tmp_path, "skm/case.h5", "image")
    _raises("H5_DATASET_REQUIRED", "$.ref", read_volume_v5, tmp_path, "skm/case.h5#", "image")
    _raises("H5_DATASET_MISSING", "$.ref", read_volume_v5, tmp_path, "skm/case.h5#echo3", "image")
    write_skm_h5(tmp_path / "skm" / "bad.h5", echo1.astype(np.complex64), echo2, seg * 2)
    _raises("IMAGE_NOT_REAL", "$.ref", read_volume_v5, tmp_path, "skm/bad.h5#echo1", "image")
    _raises("LABEL_NOT_CATEGORICAL", "$.ref", read_volume_v5, tmp_path, "skm/bad.h5#seg", "label")


def _expected_ras(spacing=SPACING, direction=DIRECTION, offset=OFFSET):
    lps = np.eye(4)
    lps[:3, :3] = np.asarray(direction).T @ np.diag(spacing)
    lps[:3, 3] = offset
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def test_mha_raw_and_zlib_agree_and_affine_is_ras(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    write_mha(tmp_path / "raw.mha", array, SPACING, DIRECTION, OFFSET, compressed=False)
    write_mha(tmp_path / "zip.mha", array, SPACING, DIRECTION, OFFSET, compressed=True)
    write_mha(tmp_path / "msb.mha", array, SPACING, DIRECTION, OFFSET, compressed=False, msb=True)
    raw, packed, msb = (read_mha_v5(tmp_path / name) for name in ("raw.mha", "zip.mha", "msb.mha"))
    for volume in (raw, packed, msb):
        assert volume.array.shape == (2, 3, 4) and np.array_equal(volume.array, array)
        assert volume.array.dtype == np.int16 and volume.array.dtype.byteorder in ("=", "|", "<")
    expected = _expected_ras()
    assert np.allclose(raw.header["affine"], expected) and np.allclose(packed.header["affine"], expected)
    assert raw.header["shape"] == [2, 3, 4] and raw.header["spacing_mm"] == list(SPACING)
    assert raw.header["axis_codes"] == [str(c) for c in nib.aff2axcodes(expected)]
    # Voxel (i, j, k) sits at Offset + sum_j index_j * spacing_j * direction_j in LPS (MetaIO), negated x/y in RAS.
    i, j, k = 1, 2, 3
    lps_point = np.asarray(OFFSET) + i * SPACING[0] * DIRECTION[0] + j * SPACING[1] * DIRECTION[1] \
        + k * SPACING[2] * DIRECTION[2]
    ras_point = np.asarray(raw.header["affine"]) @ np.array([i, j, k, 1.0])
    assert np.allclose(ras_point[:3], lps_point * np.array([-1.0, -1.0, 1.0]))
    image = read_volume_v5(tmp_path, "zip.mha", "image")
    assert image.array.dtype == np.float32 and np.array_equal(image.array, array.astype(np.float32))
    labels = read_volume_v5(tmp_path, "zip.mha", "label")
    assert labels.array.dtype.kind == "i" and np.array_equal(labels.array, array)


def test_mha_matches_simpleitk(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    array_zyx = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(SPACING)
    image.SetOrigin(OFFSET)
    image.SetDirection((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    sitk.WriteImage(image, str(tmp_path / "itk.mha"), True)
    volume = read_mha_v5(tmp_path / "itk.mha")
    assert np.array_equal(volume.array, sitk.GetArrayFromImage(image).transpose(2, 1, 0))
    lps = np.eye(4)
    lps[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    lps[:3, 3] = image.GetOrigin()
    assert np.allclose(volume.header["affine"], np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps)


@pytest.mark.parametrize("header, key", [
    ({"NDims": 2}, "NDims"),
    ({"ElementDataFile": "volume.raw"}, "ElementDataFile"),
    ({"ElementDataFile": None}, "ElementDataFile"),
    ({"ElementNumberOfChannels": 3}, "ElementNumberOfChannels"),
    ({"ElementType": "MET_LONG"}, "ElementType"),
    ({"BinaryData": "False"}, "BinaryData"),
    ({"ObjectType": "Transform"}, "ObjectType"),
])
def test_unsupported_mha_rejected(tmp_path, header, key):
    array = np.zeros((2, 2, 2), dtype=np.uint8)
    write_mha(tmp_path / "x.mha", array, SPACING, DIRECTION, OFFSET, compressed=False, **header)
    _raises("MHA_UNSUPPORTED", f"$.{key}", read_mha_v5, tmp_path / "x.mha")


def test_truncated_mha_rejected(tmp_path):
    write_mha(tmp_path / "x.mha", np.zeros((2, 2, 2), dtype=np.int16), SPACING, DIRECTION, OFFSET, compressed=False)
    data = (tmp_path / "x.mha").read_bytes()
    (tmp_path / "x.mha").write_bytes(data[:-1])
    _raises("MHA_DATA_SIZE_MISMATCH", "$.DimSize", read_mha_v5, tmp_path / "x.mha")


def test_ref_must_stay_inside_root(tmp_path):
    write_nifti(tmp_path / "img" / "a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), AFFINE)
    for ref in ("../a.nii.gz", "/img/a.nii.gz", "img/../../a.nii.gz", "img/./../../x.nii.gz", "", "#echo1"):
        _raises("INVALID_REF", "$.ref", read_volume_v5, tmp_path / "img", ref, "image")
    _raises("INVALID_REF", "$.ref", read_volume_v5, tmp_path, "img/a.nii.gz#echo1", "image")
    _raises("INVALID_KIND", "$.kind", read_volume_v5, tmp_path, "img/a.nii.gz", "mask")
    _raises("UNSUPPORTED_FORMAT", "$.ref", read_volume_v5, tmp_path, "img/a.mhd", "image")
    _raises("VOLUME_MISSING", "$.ref", read_volume_v5, tmp_path, "img/missing.nii.gz", "image")
    assert read_volume_v5(tmp_path, "img/a.nii.gz", "image").array.shape == (2, 2, 2)


def test_unreadable_volume_does_not_echo_file_name(tmp_path):
    """A corrupt file is VOLUME_UNREADABLE at $.ref; neither the message nor a chained exception carries its name."""
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "secret-name.nii").write_bytes(b"not a nifti file at all" * 20)
    write_nifti(tmp_path / "img" / "secret-name.nii.gz", np.zeros((8, 8, 8), dtype=np.float32), AFFINE)
    packed = (tmp_path / "img" / "secret-name.nii.gz").read_bytes()
    (tmp_path / "img" / "secret-name.nii.gz").write_bytes(packed[: len(packed) // 2])
    (tmp_path / "skm").mkdir()
    (tmp_path / "skm" / "secret-name.h5").write_bytes(b"not an hdf5 file" * 20)
    cases = [("img/secret-name.nii", "image"), ("img/secret-name.nii", "label"),
             ("img/secret-name.nii.gz", "image"), ("img/secret-name.nii.gz", "label"),
             ("skm/secret-name.h5#echo1", "image"), ("skm/secret-name.h5#seg", "label")]
    for ref, kind in cases:
        with pytest.raises(V5Error) as err:
            read_volume_v5(tmp_path, ref, kind)
        assert (err.value.code, err.value.path) == ("VOLUME_UNREADABLE", "$.ref"), (ref, kind)
        assert "secret-name" not in str(err.value) and "secret-name" not in repr(err.value)
        assert err.value.__cause__ is None and err.value.__suppress_context__


def _vec(values):
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def test_mha_geometry_synonyms_are_read_or_refused(tmp_path):
    """MetaIO also spells Offset as Origin/Position, TransformMatrix as Rotation/Orientation, and takes ElementSize
    as the spacing when ElementSpacing is absent; a synonym is never silently replaced by the default."""
    array = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    expected = _expected_ras()
    spellings = [{"Offset": None, "Position": _vec(OFFSET), "TransformMatrix": None, "Orientation": _vec(DIRECTION)},
                 {"Offset": None, "Origin": _vec(OFFSET), "TransformMatrix": None, "Rotation": _vec(DIRECTION)},
                 {"Origin": _vec(OFFSET), "Rotation": _vec(DIRECTION)},
                 {"ElementSpacing": None, "ElementSize": _vec(SPACING)},
                 {"ElementSize": _vec((9.0, 9.0, 9.0))}]
    for index, header in enumerate(spellings):
        write_mha(tmp_path / f"s{index}.mha", array, SPACING, DIRECTION, OFFSET, compressed=False, **header)
        volume = read_mha_v5(tmp_path / f"s{index}.mha")
        assert np.allclose(volume.header["affine"], expected), header
        assert volume.header["spacing_mm"] == list(SPACING), header
    write_mha(tmp_path / "o.mha", array, SPACING, DIRECTION, OFFSET, compressed=False, Origin=_vec((0, 0, 0)))
    _raises("MHA_UNSUPPORTED", "$.Offset", read_mha_v5, tmp_path / "o.mha")
    write_mha(tmp_path / "r.mha", array, SPACING, DIRECTION, OFFSET, compressed=False, Orientation=_vec(np.eye(3)))
    _raises("MHA_UNSUPPORTED", "$.TransformMatrix", read_mha_v5, tmp_path / "r.mha")
