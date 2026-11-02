"""`geometry-qc` command: per-pair physical-space checks and label value QC for P0 sources."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import nibabel as nib

from anatobind.v41.adapters import isles, skm
from anatobind.v41.geometry import audit_geometry, geometry_from_nifti
from anatobind.v41.inventory import read_jsonl
from anatobind.v41.labels import label_value_report, read_categorical_nifti
from anatobind.v41.paths import PrivatePathError, resolve_private_output, write_private_json

A6_ALLOWED = range(0, 7)
ISLES_ALLOWED = (0, 1)


def _geometry(path: Path) -> dict:
    return geometry_from_nifti(nib.load(str(path)))


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PrivatePathError("PRIVATE_FILE_EXISTS") from None
    with os.fdopen(fd, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: json.dumps(row.get(c)) if isinstance(row.get(c), (list, dict)) else row.get(c) for c in columns})


def run_geometry_qc(root: Path, assets: list[dict], out: Path, skm_channel_cases: int = 0,
                    label_values: bool = True, progress: bool = False) -> dict:
    root = Path(root)
    geometry_rows, label_rows, pairing_rows, channel_reports = [], [], [], []
    isles_pairs = isles.pair_assets(assets)
    for pair_index, pair in enumerate(isles_pairs):
        if progress and pair_index % 25 == 0:
            print(f"progress ISLES {pair_index}/{len(isles_pairs)}", flush=True)
        if not pair["label_relpath"]:
            pairing_rows.append({"source": pair["source"], "case_id": pair["case_id"], "status": "missing_label",
                                 "images": sorted(pair["images"])})
            continue
        label_geometry = _geometry(root / pair["label_relpath"])
        pairing_rows.append({"source": pair["source"], "case_id": pair["case_id"], "status": "paired",
                             "images": sorted(pair["images"]), "label_relpath": pair["label_relpath"]})
        for role, rel in sorted(pair["images"].items()):
            report = audit_geometry(_geometry(root / rel), label_geometry)
            geometry_rows.append({"source": pair["source"], "case_id": pair["case_id"], "image_role": role,
                                  "image_relpath": rel, "label_relpath": pair["label_relpath"], **report,
                                  "label_shape": label_geometry["shape"], "label_spacing_mm": label_geometry["spacing_mm"]})
        if label_values:
            labels, meta = read_categorical_nifti(root / pair["label_relpath"])
            values = label_value_report(labels, ISLES_ALLOWED)
            label_rows.append({"source": pair["source"], "case_id": pair["case_id"], "label_relpath": pair["label_relpath"],
                               "label_value_qc": meta["label_value_qc"], "reason": meta["label_value_reason"],
                               "values_present": values["values_present"], "unexpected_values": values["unexpected_values"],
                               "scl_slope": meta["scl_slope"], "dtype_on_disk": meta["dtype_on_disk"]})
    skm_pairs = skm.pair_assets(assets)
    for index, pair in enumerate(skm_pairs):
        if progress and index % 25 == 0:
            print(f"progress SKM {index}/{len(skm_pairs)}", flush=True)
        tracks = {k: pair[k] for k in ("dicom_track_relpath", "raw_track_relpath") if pair[k]}
        pairing_rows.append({"source": pair["source"], "case_id": pair["case_id"],
                             "status": "paired" if pair["h5_relpath"] and len(tracks) == 2 else "incomplete",
                             "images": ["h5"] if pair["h5_relpath"] else [], "label_relpath": json.dumps(tracks)})
        geometries = {k: _geometry(root / v) for k, v in tracks.items()}
        if len(geometries) == 2:
            report = audit_geometry(geometries["dicom_track_relpath"], geometries["raw_track_relpath"])
            geometry_rows.append({"source": pair["source"], "case_id": pair["case_id"], "image_role": "dicom_track_vs_raw_track",
                                  "image_relpath": tracks["dicom_track_relpath"], "label_relpath": tracks["raw_track_relpath"],
                                  **report, "label_shape": geometries["raw_track_relpath"]["shape"],
                                  "label_spacing_mm": geometries["raw_track_relpath"]["spacing_mm"]})
        if label_values:
            for track, rel in tracks.items():
                labels, meta = read_categorical_nifti(root / rel)
                values = label_value_report(labels, A6_ALLOWED)
                label_rows.append({"source": pair["source"], "case_id": pair["case_id"], "label_relpath": rel,
                                   "label_value_qc": meta["label_value_qc"], "reason": meta["label_value_reason"],
                                   "values_present": values["values_present"], "unexpected_values": values["unexpected_values"],
                                   "scl_slope": meta["scl_slope"], "dtype_on_disk": meta["dtype_on_disk"]})
        if index < skm_channel_cases and pair["h5_relpath"]:
            channel_reports.append(skm.channel_check(root, pair))
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    _csv(out / "geometry_qc.csv", geometry_rows, ["source", "case_id", "image_role", "image_relpath", "label_relpath",
                                                  "same_shape", "same_spacing", "same_affine", "same_axis_codes", "status",
                                                  "reasons", "label_shape", "label_spacing_mm"])
    _csv(out / "label_value_qc.csv", label_rows, ["source", "case_id", "label_relpath", "label_value_qc", "reason",
                                                  "values_present", "unexpected_values", "scl_slope", "dtype_on_disk"])
    _csv(out / "image_label_pairing.csv", pairing_rows, ["source", "case_id", "status", "images", "label_relpath"])
    if channel_reports:
        write_private_json(out / "skm_channel_check.json", channel_reports)
    summary = summarize_geometry_qc(geometry_rows, label_rows, pairing_rows, channel_reports)
    write_private_json(out / f"geometry_qc_summary_{time.time_ns()}.json", summary)
    return summary


def summarize_geometry_qc(geometry_rows, label_rows, pairing_rows, channel_reports) -> dict:
    def count(rows, key_fn):
        out: dict[str, int] = {}
        for row in rows:
            key = key_fn(row)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    channel_summary = {}
    for report in channel_reports:
        for track, info in report["tracks"].items():
            if not info or not info["best"]:
                continue
            key = f"{track}:{info['best']['transform']}:{info['best']['channel_to_label']}:{'consistent' if info['best']['consistent'] else 'inconsistent'}"
            channel_summary[key] = channel_summary.get(key, 0) + 1
    return {
        "geometry_by_source_role_status": count(geometry_rows, lambda r: f"{r['source']}|{r['image_role']}|{r['status']}|{','.join(r['reasons']) or '-'}"),
        "label_value_qc_by_source": count(label_rows, lambda r: f"{r['source']}|{r['label_value_qc']}|{r['reason'] or '-'}|unexpected={bool(r['unexpected_values'])}"),
        "pairing_by_source_status": count(pairing_rows, lambda r: f"{r['source']}|{r['status']}"),
        "skm_channel_check": channel_summary,
        "skm_channel_cases": len(channel_reports),
    }


def _cli(args) -> int:
    from anatobind.v41.cli import package_repo_root

    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    assets = read_jsonl(Path(args.assets))
    if args.source:
        wanted = set(args.source)
        assets = [a for a in assets if a.get("source") in wanted]
    summary = run_geometry_qc(Path(args.root), assets, out, skm_channel_cases=args.skm_channel_cases,
                              label_values=not args.skip_label_values, progress=True)
    print(json.dumps(summary))
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("geometry-qc")
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--skm-channel-cases", type=int, default=0)
    parser.add_argument("--skip-label-values", action="store_true")
    parser.add_argument("--source", action="append", default=None)
    parser.set_defaults(func=_cli)
