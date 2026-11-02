import numpy as np
from PIL import Image

from anatobind.v41.overlays import pick_slices, render_overlay


def test_pick_slices_prefers_label_mass():
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[2, 3:6, 1] = 1
    labels[5, 2, 4:7] = 2
    picked = pick_slices(labels)
    assert picked == {"axis0": 2, "axis1": 2, "axis2": 1} or picked["axis0"] in (2, 5)
    assert all(0 <= v < 8 for v in picked.values())


def test_render_overlay_writes_png_with_label_colour(tmp_path):
    image = np.random.default_rng(0).normal(size=(16, 16, 6)).astype(np.float32)
    labels = np.zeros((16, 16, 6), dtype=np.int16)
    labels[4:8, 4:8, 3] = 1
    out = tmp_path / "case.png"
    info = render_overlay(image, labels, out, title="synthetic")
    assert out.is_file()
    assert info["slices"]["axis2"] == 3
    png = np.asarray(Image.open(out).convert("RGB"))
    assert png.shape[2] == 3 and png.shape[0] > 16
    assert png.reshape(-1, 3).max(axis=0)[0] == 255


def test_render_overlay_rejects_shape_mismatch(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        render_overlay(np.zeros((4, 4, 4)), np.zeros((4, 4, 5), dtype=np.int16), tmp_path / "x.png", title="t")
