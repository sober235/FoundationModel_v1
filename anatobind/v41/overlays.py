"""Private overlay renderings for human review of image/label alignment.

Overlays are evidence for a reviewer, not a QC verdict: a same-shape pair can
still be misaligned, which is why the reviewer sees the picked slices rather
than a pass flag.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from anatobind.v41.paths import PrivatePathError, write_private_json

_PALETTE = {
    1: (255, 0, 0), 2: (0, 200, 0), 3: (0, 120, 255), 4: (255, 200, 0), 5: (255, 0, 255), 6: (0, 255, 255),
}
_SCALE = 2


def pick_slices(labels: np.ndarray) -> dict[str, int]:
    picked = {}
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        mass = (labels != 0).sum(axis=other)
        picked[f"axis{axis}"] = int(mass.argmax()) if mass.max() > 0 else labels.shape[axis] // 2
    return picked


def _normalise(plane: np.ndarray) -> np.ndarray:
    plane = plane.astype(np.float64)
    lo, hi = np.percentile(plane, (1, 99))
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    return (np.clip((plane - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _panel(image_plane: np.ndarray, label_plane: np.ndarray) -> Image.Image:
    rgb = np.stack([_normalise(image_plane)] * 3, axis=-1)
    for value in np.unique(label_plane):
        if value == 0:
            continue
        colour = _PALETTE.get(int(value), (255, 255, 255))
        mask = label_plane == value
        rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array(colour)).astype(np.uint8)
    panel = Image.fromarray(np.ascontiguousarray(np.transpose(rgb, (1, 0, 2))))
    return panel.resize((panel.width * _SCALE, panel.height * _SCALE), Image.NEAREST)


def render_overlay(image: np.ndarray, labels: np.ndarray, out_png: Path, title: str) -> dict:
    image = np.asarray(image)
    labels = np.asarray(labels)
    if image.ndim != 3 or image.shape != labels.shape:
        raise ValueError("SHAPE_MISMATCH")
    slices = pick_slices(labels)
    panels = []
    for axis in range(3):
        index = slices[f"axis{axis}"]
        panels.append(_panel(np.take(image, index, axis=axis), np.take(labels, index, axis=axis)))
    header = 18
    width = sum(p.width for p in panels) + 4 * (len(panels) - 1)
    height = max(p.height for p in panels) + header
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 2), f"{title} slices={slices} values={sorted(int(v) for v in np.unique(labels))}", fill=(255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header))
        x += panel.width + 4
    out_png = Path(out_png)
    try:
        fd = os.open(out_png, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PrivatePathError("PRIVATE_FILE_EXISTS") from None
    with os.fdopen(fd, "wb") as handle:
        canvas.save(handle, format="PNG")
    return {"png": str(out_png), "slices": slices, "label_values": sorted(int(v) for v in np.unique(labels)),
            "shape": list(image.shape)}


def render_source_overlays(root: Path, assets: list[dict], out: Path, cases: int) -> dict:
    import h5py
    import nibabel as nib

    from anatobind.v41.adapters import isles, skm
    from anatobind.v41.labels import read_categorical_nifti

    root = Path(root)
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest = []
    for pair in [p for p in isles.pair_assets(assets) if p["label_relpath"]][:cases]:
        labels, meta = read_categorical_nifti(root / pair["label_relpath"])
        for role in ("dwi", "adc"):
            if role not in pair["images"]:
                continue
            image = np.asarray(nib.load(str(root / pair["images"][role])).dataobj)
            if image.shape != labels.shape:
                manifest.append({"source": pair["source"], "case_id": pair["case_id"], "role": role,
                                 "status": "skipped_shape_mismatch"})
                continue
            info = render_overlay(image, labels, out / f"{pair['source']}_{pair['case_id']}_{role}.png",
                                  title=f"{pair['source']} {pair['case_id']} {role} ({meta['label_value_qc']})")
            manifest.append({"source": pair["source"], "case_id": pair["case_id"], "role": role, "status": "rendered", **info})
    for pair in [p for p in skm.pair_assets(assets) if p["h5_relpath"] and p["dicom_track_relpath"]][:cases]:
        labels, meta = read_categorical_nifti(root / pair["dicom_track_relpath"])
        labels = skm.nifti_to_h5(labels)
        with h5py.File(root / pair["h5_relpath"], "r") as handle:
            image = np.asarray(handle["echo1"])
        if image.shape != labels.shape:
            manifest.append({"source": pair["source"], "case_id": pair["case_id"], "role": "echo1_vs_dicom_track",
                             "status": "skipped_shape_mismatch", "image_shape": list(image.shape),
                             "label_shape": list(labels.shape)})
            continue
        info = render_overlay(image, labels, out / f"{pair['source']}_{pair['case_id']}_echo1_dicom_track.png",
                              title=f"{pair['source']} {pair['case_id']} echo1/dicom-track {skm.NIFTI_TO_H5_TRANSFORM} ({meta['label_value_qc']})")
        manifest.append({"source": pair["source"], "case_id": pair["case_id"], "role": "echo1_vs_dicom_track",
                         "status": "rendered", **info})
    write_private_json(out / "overlay_manifest.json", manifest)
    return {"rendered": sum(1 for m in manifest if m["status"] == "rendered"),
            "skipped": sum(1 for m in manifest if m["status"] != "rendered")}


def _cli(args) -> int:
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.inventory import read_jsonl
    from anatobind.v41.paths import resolve_private_output

    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    assets = read_jsonl(Path(args.assets))
    if args.source:
        assets = [a for a in assets if a.get("source") in set(args.source)]
    print(json.dumps(render_source_overlays(Path(args.root), assets, out, args.cases)))
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("overlay-samples")
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--source", action="append", default=None)
    parser.set_defaults(func=_cli)
