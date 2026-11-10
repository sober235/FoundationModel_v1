"""SKM-TEA pairing and source channel-order verification (read-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SOURCE = "SKM-TEA_ltr"
_TRANSFORMS = {
    "identity": lambda a: a,
    "transpose_xy": lambda a: np.transpose(a, (1, 0, 2)),
    "transpose_xy_flip_z": lambda a: np.transpose(a, (1, 0, 2))[:, :, ::-1],
    "flip_z": lambda a: a[:, :, ::-1],
    "transpose_xy_flip_x": lambda a: np.transpose(a, (1, 0, 2))[::-1, :, :],
    "transpose_xy_flip_y": lambda a: np.transpose(a, (1, 0, 2))[:, ::-1, :],
}
# Verified by channel_check on 12/12 cases (dice 1.0 for all six classes): the
# dicom-track NIfTI equals the H5 seg stack only after swapping the first two axes.
NIFTI_TO_H5_TRANSFORM = "transpose_xy"


def nifti_to_h5(labels: np.ndarray) -> np.ndarray:
    return _TRANSFORMS[NIFTI_TO_H5_TRANSFORM](labels)


def nifti_to_h5_shape(shape) -> tuple[int, ...]:
    return tuple(nifti_to_h5(np.empty(tuple(shape), dtype=bool)).shape)


def h5_seg_shape(path: Path) -> tuple[int, ...]:
    import h5py

    with h5py.File(path, "r") as handle:
        return tuple(handle["seg"].shape[:3])


def pair_assets(assets: list[dict]) -> list[dict]:
    cases: dict[str, dict] = {}
    for asset in assets:
        if asset.get("source") != SOURCE or asset.get("kind") not in ("image", "label") or not asset.get("case_id"):
            continue
        entry = cases.setdefault(asset["case_id"], {"case_id": asset["case_id"], "source": SOURCE,
                                                     "h5_relpath": None, "dicom_track_relpath": None,
                                                     "raw_track_relpath": None})
        rel = asset["relpath"]
        if asset["kind"] == "image":
            entry["h5_relpath"] = rel
        elif "/dicom-track/" in rel:
            entry["dicom_track_relpath"] = rel
        elif "/raw-data-track/" in rel:
            entry["raw_track_relpath"] = rel
    return [cases[k] for k in sorted(cases)]


def pairing_report(assets: list[dict]) -> dict:
    """Counts only. One case is one scan: its echoes share one H5 file and its tracks are label versions of the
    same scan, so neither adds a scan or a patient."""
    seen: dict[tuple, int] = {}
    for asset in assets:
        if asset.get("source") != SOURCE or asset.get("kind") not in ("image", "label") or not asset.get("case_id"):
            continue
        rel = asset["relpath"]
        role = "h5" if asset["kind"] == "image" else "dicom_track" if "/dicom-track/" in rel else \
            "raw_track" if "/raw-data-track/" in rel else "other_label"
        seen[(asset["case_id"], role)] = seen.get((asset["case_id"], role), 0) + 1
    pairs = pair_assets(assets)
    return {"cases": len(pairs),
            "paired": sum(1 for p in pairs if p["h5_relpath"] and p["dicom_track_relpath"]),
            "missing_h5": sum(1 for p in pairs if not p["h5_relpath"]),
            "missing_dicom_track": sum(1 for p in pairs if not p["dicom_track_relpath"]),
            "with_raw_track": sum(1 for p in pairs if p["raw_track_relpath"]),
            "duplicate_role_assets": sum(n - 1 for n in seen.values() if n > 1),
            "cases_with_duplicate_assets": len({case for (case, _), n in seen.items() if n > 1}),
            "other_labels_not_registered": sum(n for (_, role), n in seen.items() if role == "other_label")}


def channel_check(root: Path, pair: dict) -> dict:
    import h5py

    from anatobind.v41.labels import channel_stack_agreement, read_categorical_nifti

    root = Path(root)
    with h5py.File(root / pair["h5_relpath"], "r") as handle:
        seg = handle["seg"][()].astype(bool)
    report = {"case_id": pair["case_id"], "h5_seg_shape": list(seg.shape), "tracks": {}}
    for track in ("dicom_track_relpath", "raw_track_relpath"):
        if not pair.get(track):
            report["tracks"][track] = None
            continue
        labels, meta = read_categorical_nifti(root / pair[track])
        best = None
        for name, fn in _TRANSFORMS.items():
            candidate = fn(labels)
            if candidate.shape != seg.shape[:3]:
                continue
            agreement = channel_stack_agreement(seg, candidate)
            score = sum(d or 0.0 for d in agreement["dice"])
            if best is None or score > best["score"]:
                best = {"transform": name, "score": score, **agreement}
        report["tracks"][track] = {
            "nifti_shape": meta["shape"], "spacing_mm": meta["spacing_mm"], "axis_codes": meta["axis_codes"],
            "values_present": meta["values_present"], "best": best,
        }
    return report
