"""Real T07 preparation for v5.1 (plan_v5 §2B T07c real run; §2C T09b conventions).

Turns the v41 registration records of ISLES-2022 and SKM-TEA into records that
v5 migration accepts:

- geometry status and image geometry come from the v41 geometry QC rows; an
  SKM scan passes only when its per-case channel check finds one of the
  registered NIfTI-to-H5 transforms and the channel order of the v5 label
  map (real T07, 2026-11-08: 137/155 dicom tracks are the H5 seg with x/y
  swapped, 18/155 also have z reversed; the raw-data track is not registered
  and is left out of the counts);
- SKM images and anatomy labels are addressed as datasets inside the H5 file
  (``<path>#echo1``, ``#echo2``, ``#seg``), on the H5 grid;
- label map names move to their v5 names;
- ISLES lesion masks keep input visibility ``unknown``: the released-reference
  track does not pseudo-fill "visible" without an audit (Proposal_v5 §2.3);
- annotations of tasks the scope does not train are dropped, and released
  boxes are dropped for every scope (an skm_a or skm_joint_early record
  carries no boxes; T11 imports them from the release for the joint scope).

``run_prepare_v5`` is the driver: it runs the v41 geometry QC, prepares the
records and the downsampling loss reports, and writes everything under one new
private directory (700, files 600, never overwritten). Only the redacted
source-check markdown is meant to leave that directory.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .adapters import isles, skm
from .audits import build_registration_records
from .contracts_v5 import ActiveScope, V5Error
from .engine.data_v5 import decode_targets_v5, load_label_maps_v5
from .engine.volume_io_v5 import read_volume_v5
from .geometry import SCHEMA_GEOMETRY_KEYS, geometry_from_nifti
from .geometry_qc import run_geometry_qc
from .geometry_v5 import downsample_loss_report_v5
from .paths import write_private_json
from .schema import SampleRecord, validate_sample
from .source_check_v5 import render_source_check_md_v5, summarize_source_check_v5
from .source_policy_v5 import SourcePolicy

ISLES_DATASET, SKM_DATASET = isles.SOURCE, skm.SOURCE
SCOPE_SOURCE = {"isles_u": ISLES_DATASET, "skm_a": SKM_DATASET, "skm_joint_early": SKM_DATASET}
# v41 registration builds records per v41 scope name; the early joint scope reads the same SKM-TEA records
REGISTRATION_SCOPE = {"isles_u": "isles_u", "skm_a": "skm_a", "skm_joint_early": "skm_a"}
# v41 box annotation records are never carried into a v5 record, also when the scope has U classes: a released
# box is box-level and must not become a voxel annotation (Proposal_v5 §5.0). T11 imports the boxes from the
# release itself under the frozen box policy (real_t11_v5).
RELEASE_BOX_LABEL_MAPS = ("skmtea_boxes_v1",)
SKM_ECHOES = (("qdess_e1", "echo1"), ("qdess_e2", "echo2"))
SKM_SEG_DATASET = "seg"
# H5 seg channel c holds the dicom-track label value c + 1 (SKM-TEA DATASET.md order; label_maps_v5.yaml)
SKM_CHANNEL_TO_LABEL = [c + 1 for c in range(6)]
# v41 adapters.skm transforms that map the dicom-track NIfTI onto the H5 grid: h5[i, j, k] = nifti[j, i, k]
# or nifti[j, i, Z - 1 - k]
SKM_TRANSFORMS = ("transpose_xy", "transpose_xy_flip_z")
REGISTERED_TRACK = "dicom_track_relpath"
LABEL_MAP_V5 = {"isles2022_msk_v1": "isles_binary_v1", "skmtea_seg_dicom-track_v1": "skm_a6_h5_channels_v1"}
LABEL_MAPS_PATH = Path(__file__).resolve().parents[2] / "configs" / "v5" / "label_maps_v5.yaml"
_JSON_COLUMNS = ("reasons", "images", "label_shape", "label_spacing_mm")


def read_qc_csv_v5(path: Path) -> list[dict]:
    """Rows of a v41 geometry-qc CSV with their list-valued columns decoded."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            for key in _JSON_COLUMNS:
                if out.get(key):
                    out[key] = json.loads(out[key])
            rows.append(out)
    return rows


def skm_h5_geometry_v5(nifti_geometry: dict, units: str, transform: str = "transpose_xy") -> dict:
    """Schema geometry of the SKM H5 grid from the dicom-track NIfTI and the channel check's transform.

    ``units`` is the source-level unit decision (the NIfTI headers declare none); it is passed in, never guessed.
    """
    if transform not in SKM_TRANSFORMS:
        raise V5Error("SKM_TRANSFORM_UNSUPPORTED", "$.transform")
    shape, spacing = list(nifti_geometry["shape"]), list(nifti_geometry["spacing_mm"])
    to_nifti = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    if transform == "transpose_xy_flip_z":
        to_nifti[2] = [0, 0, -1, shape[2] - 1]
    affine = np.asarray(nifti_geometry["affine"], dtype=float) @ to_nifti
    return {"shape": [int(shape[1]), int(shape[0]), int(shape[2])],
            "spacing_mm": [float(spacing[1]), float(spacing[0]), float(spacing[2])],
            "affine": affine.tolist(), "axis_codes": [str(c) for c in nib.aff2axcodes(affine)], "units": units}


def skm_transform_v5(channel_report: dict | None) -> str | bool | None:
    """The verified transform name; False when the check found another transform or order; None when unchecked."""
    track = ((channel_report or {}).get("tracks") or {}).get(REGISTERED_TRACK)
    best = track.get("best") if isinstance(track, dict) else None
    if not isinstance(best, dict):
        return None
    verified = (best.get("transform") in SKM_TRANSFORMS and best.get("consistent") is True
                and best.get("channel_to_label") == SKM_CHANNEL_TO_LABEL)
    return best["transform"] if verified else False


def _geometry_status(record: dict, rows: list, channel_report: dict | None) -> str:
    files = {ref.split("#")[0] for ref in record["image_paths"]}
    own = [r for r in rows if r.get("source") == record["source_dataset"] and r.get("image_relpath") in files]
    if any(r.get("status") != "pass" for r in own):
        return "fail"
    if not own or {r["image_relpath"] for r in own} != files:
        return "not_checked"
    if record["source_dataset"] == SKM_DATASET:
        verified = skm_transform_v5(channel_report)
        if verified is None:
            return "not_checked"
        if not verified:
            return "fail"
    return "pass"


def _scope_tasks(scope: ActiveScope) -> set:
    return {task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes)) if classes}


def prepare_record_v5(record: dict, scope: ActiveScope, geometry_rows: list, image_geometry: dict | None,
                      channel_report: dict | None = None) -> SampleRecord:
    body = copy.deepcopy(record)
    source = body.get("source_dataset")
    if source not in (ISLES_DATASET, SKM_DATASET):
        raise V5Error("SOURCE_NOT_SUPPORTED", "$.source_dataset")
    if SCOPE_SOURCE.get(scope.scope_id) != source:
        raise V5Error("SCOPE_MISMATCH", "$.source_dataset")
    status = _geometry_status(body, geometry_rows, channel_report)
    if status != "not_checked" and image_geometry is None:
        raise V5Error("GEOMETRY_MISSING", "$.image_geometry")
    body["geometry_status"] = status
    body["image_geometry"] = copy.deepcopy(image_geometry)
    tasks = _scope_tasks(scope)
    annotations = []
    for index, annotation in enumerate(body["annotation_records"]):
        if annotation["task"] not in tasks or annotation["label_map_version"] in RELEASE_BOX_LABEL_MAPS:
            continue
        name = LABEL_MAP_V5.get(annotation["label_map_version"])
        if name is None:
            raise V5Error("LABEL_MAP_UNKNOWN", f"$.annotation_records[{index}].label_map_version")
        annotation["label_map_version"] = name
        annotations.append(annotation)
    if source == SKM_DATASET:
        if len(body["image_paths"]) != 1:
            raise V5Error("H5_LAYOUT_MISMATCH", "$.image_paths")
        h5 = body["image_paths"][0]
        body.update(image_paths=[f"{h5}#{dataset}" for _, dataset in SKM_ECHOES],
                    modality_names=[name for name, _ in SKM_ECHOES],
                    modality_is_synthetic=[False] * len(SKM_ECHOES), available_modality_mask=[True] * len(SKM_ECHOES))
        for annotation in annotations:
            annotation["path"] = f"{h5}#{SKM_SEG_DATASET}"
    else:
        for annotation in annotations:
            annotation["input_visibility"] = "unknown"
            annotation["visibility_evidence"] = (f"{annotation['visibility_evidence']}; input visibility not audited, "
                                                 "released_reference track (Proposal_v5 §2.3)")
    body["annotation_records"] = annotations
    return validate_sample(body)


# ---------------------------------------------------------------------------
# Driver


def _write_private_text(path: Path, text: str) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def _image_geometry(raw: Path, record: dict, pair: dict | None, skm_units: str | None,
                    channel: dict | None) -> dict | None:
    if record["source_dataset"] == ISLES_DATASET:
        geometry = geometry_from_nifti(nib.load(str(raw / record["image_paths"][0])))
        return {key: geometry[key] for key in SCHEMA_GEOMETRY_KEYS} if geometry["units"] == "mm" else None
    if pair is None or not pair.get("dicom_track_relpath") or skm_units is None:
        return None
    transform = skm_transform_v5(channel) or skm.NIFTI_TO_H5_TRANSFORM   # unverified scans are marked "fail"
    return skm_h5_geometry_v5(geometry_from_nifti(nib.load(str(raw / pair["dicom_track_relpath"]))), skm_units,
                              transform)


def _loss_report(raw: Path, prepared: SampleRecord, scope: ActiveScope, label_maps: dict, factor: tuple) -> dict:
    (annotation,) = prepared.annotation_records
    classes = list(scope.anatomy_classes if annotation.task == "A" else scope.abnormality_classes)
    volume = read_volume_v5(raw, annotation.path, "label")
    targets = decode_targets_v5(volume, annotation, classes, label_maps)
    report = downsample_loss_report_v5(targets, tuple(factor), classes)
    return {"factor": report["factor"], "classes": report["classes"]}


def run_prepare_v5(raw: Path, assets: list, scope: ActiveScope, policy: SourcePolicy, out: Path, *,
                   factor: tuple, skm_annotations: dict | None = None, skm_units: str | None = None,
                   reuse_qc: Path | None = None) -> dict:
    """``reuse_qc``: an earlier run's private geometry_qc directory for the same assets, read instead of rerun."""
    raw, out = Path(raw), Path(out)
    if policy.scope_id != scope.scope_id or SCOPE_SOURCE.get(scope.scope_id) != policy.source_dataset:
        raise V5Error("SCOPE_MISMATCH", "$.policy.scope_id")
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    assets = [a for a in assets if a.get("source") == policy.source_dataset]
    records = build_registration_records(REGISTRATION_SCOPE[scope.scope_id], assets, skm_annotations)
    pairs = skm.pair_assets(assets) if policy.source_dataset == SKM_DATASET else []
    pair_by_h5 = {p["h5_relpath"]: p for p in pairs if p["h5_relpath"]}
    qc_dir = Path(reuse_qc) if reuse_qc is not None else out / "geometry_qc"
    if reuse_qc is None:
        run_geometry_qc(raw, assets, qc_dir, skm_channel_cases=len(pairs))
    geometry_rows = read_qc_csv_v5(qc_dir / "geometry_qc.csv")
    pairing_rows = read_qc_csv_v5(qc_dir / "image_label_pairing.csv")
    channel_path = qc_dir / "skm_channel_check.json"
    checked = json.loads(channel_path.read_text()) if channel_path.exists() else []
    channel_reports = [{"case_id": r["case_id"],
                        "tracks": {REGISTERED_TRACK: (r.get("tracks") or {}).get(REGISTERED_TRACK)}} for r in checked]
    channel_by_case = {r["case_id"]: r for r in channel_reports}
    label_maps = load_label_maps_v5(LABEL_MAPS_PATH)
    prepared, failures, losses = [], [], []
    for index, record in enumerate(records):
        try:
            pair = pair_by_h5.get(record["image_paths"][0])
            channel = channel_by_case.get(pair["case_id"]) if pair else None
            sample = prepare_record_v5(record, scope, geometry_rows,
                                       _image_geometry(raw, record, pair, skm_units, channel), channel)
            if sample.geometry_status == "pass" and len(sample.annotation_records) == 1:
                losses.append(_loss_report(raw, sample, scope, label_maps, factor))
            prepared.append(sample)
        except V5Error as error:
            failures.append({"index": index, "code": error.code, "path": error.path})
    summary = summarize_source_check_v5(geometry_rows, pairing_rows, channel_reports, losses, policy)
    markdown = render_source_check_md_v5(summary)
    lines = "".join(json.dumps(_asdict(s), sort_keys=True) + "\n" for s in prepared)
    _write_private_text(out / "records_prepared.jsonl", lines)
    _write_private_text(out / "source_check.md", markdown)
    write_private_json(out / "loss_reports.json", losses)
    report = {"records": len(prepared), "failures": failures, "qc_reused": reuse_qc is not None,
              "geometry_status": dict(sorted(Counter(s.geometry_status for s in prepared).items())),
              "summary": summary, "source_check_sha256": hashlib.sha256(markdown.encode()).hexdigest()}
    write_private_json(out / "prepare_report.json", report)
    return report


def _asdict(sample: SampleRecord) -> dict:
    import dataclasses
    return dataclasses.asdict(sample)


# ---------------------------------------------------------------------------
# Registration report (import and registration only: nothing is prepared, migrated or frozen)


def registration_report_v5(assets: list, scope: ActiveScope, annotations_dir: Path | None = None) -> tuple:
    """(report, details) for one scope's source. `report` holds counts only; `details` names cases and is private.

    SKM-TEA: all three official split files are checked and read, and each scan keeps its official split as
    `provenance.release_split`. The formal `split` of every registered scan stays `unassigned`: it is given
    only by the freeze step, after grouping and the exposure check. ISLES: DWI, ADC and mask pairing.
    """
    from .audits import registration_summary, scan_skm_release

    source = SCOPE_SOURCE.get(scope.scope_id)
    if source is None:
        raise V5Error("SOURCE_NOT_SUPPORTED", "$.scope.scope_id")
    assets = [a for a in assets if a.get("source") == source]
    release = details = annotations = None
    if source == SKM_DATASET:
        if annotations_dir is None:
            raise V5Error("RELEASE_REQUIRED", "$.annotations_dir")
        annotations, release, details = scan_skm_release(Path(annotations_dir))
        pairing = skm.pairing_report(assets)
        registered = {p["case_id"] for p in skm.pair_assets(assets) if p["h5_relpath"]}
        release["release_scans_without_assets"] = len(set(annotations) - registered)
        release["registered_scans_without_release_entry"] = len(registered - set(annotations))
        details["release_scans_without_assets"] = sorted(set(annotations) - registered)
        details["registered_scans_without_release_entry"] = sorted(registered - set(annotations))
    else:
        pairing = isles.pairing_report(assets)
    records = build_registration_records(REGISTRATION_SCOPE[scope.scope_id], assets, annotations)
    report = {"scope_id": scope.scope_id, "source_dataset": source, "pairing": pairing, "release": release,
              "registration": registration_summary(records)}
    return report, details


def render_registration_md_v5(report: dict) -> str:
    registration, pairing, release = report["registration"], report["pairing"], report["release"]
    lines = [f"# 导入与登记汇总：{report['source_dataset']} → {report['scope_id']}", "",
             "只含计数；不含病例 ID、路径或文件名。本汇总只到导入与登记：没有冻结正式划分，没有训练，没有解封测试。", "",
             "## 扫描与患者", "",
             f"- 登记扫描 {registration['scans']}；可确认的独立患者 {registration['patients_confirmed']}；"
             f"患者身份不能确认的扫描 {registration['scans_patient_unconfirmed']}",
             f"- 泄漏组 {registration['leakage_groups']}；跨多个泄漏组的患者 "
             f"{registration['patients_in_multiple_leakage_groups']}；重复的样本号 "
             f"{registration['duplicate_sample_uids']}",
             "- 回波、标签轨道和其他派生版本不增加扫描数，也不增加患者数。", "",
             "## 配对", "", "| 项 | 计数 |", "| --- | --- |"]
    lines += [f"| {key} | {value} |" for key, value in pairing.items()]
    if release is not None:
        lines += ["", "## 官方划分文件（来源划分，不是正式划分）", "",
                  "| 文件 | 存在 | 可读 | 扫描 | 可确认患者 | 无患者号的扫描 | 框 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for split, entry in release["files"].items():
            lines.append(f"| {split} | {'是' if entry['present'] else '否'} | {'是' if entry['readable'] else '否'} | "
                         f"{entry['scans']} | {entry['patients']} | {entry['scans_without_subject']} | "
                         f"{entry['boxes']} |")
        lines += ["", f"- 缺失文件：{'、'.join(release['missing_files']) or '无'}；不可读文件："
                  f"{'、'.join(release['unreadable_files']) or '无'}",
                  f"- 三个文件合计：扫描 {release['scans']}，可确认患者 {release['patients']}",
                  f"- 同一文件内重复的病例 {release['duplicate_cases_within_file']}；出现在多个划分的病例 "
                  f"{release['cases_in_multiple_splits']}；出现在多个划分的患者 "
                  f"{release['patients_in_multiple_splits']}",
                  f"- 发布文件有而本地无影像的扫描 {release['release_scans_without_assets']}；本地有而发布文件无条目的"
                  f"扫描 {release['registered_scans_without_release_entry']}；找不到扫描的标注 "
                  f"{release['orphan_annotations']}",
                  "", "### 登记记录按来源划分", "", "| 来源划分 | 扫描 | 可确认患者 | 患者不能确认的扫描 |",
                  "| --- | --- | --- | --- |"]
        lines += [f"| {key} | {row['scans']} | {row['patients']} | {row['scans_patient_unconfirmed']} |"
                  for key, row in registration["by_release_split"].items()]
    lines += ["", "## 正式划分与曝光", "",
              "- 正式 split：" + "、".join(f"{k} {v}" for k, v in registration["formal_split"].items()),
              "- 曝光登记（按角色，条目数）：" + ("、".join(f"{k} {v}" for k, v in registration["exposure_roles"].items())
                                       or "无") + f"；没有曝光条目的扫描 {registration['scans_without_exposure_entry']}",
              f"- 没有标签的扫描 {registration['scans_without_label']}",
              "- 官方 test 的名称不构成盲测资格。曝光为 unknown 的病例保持 unknown，冻结时不能进入 locked_test。"]
    return "\n".join(lines) + "\n"


def run_registration_report_v5(assets: list, scope: ActiveScope, out: Path,
                               annotations_dir: Path | None = None) -> dict:
    report, details = registration_report_v5(assets, scope, annotations_dir)
    out = Path(out)
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    markdown = render_registration_md_v5(report)
    report["registration_md_sha256"] = hashlib.sha256(markdown.encode()).hexdigest()
    _write_private_text(out / "registration.md", markdown)
    write_private_json(out / "registration_report.json", report)
    if details is not None:
        write_private_json(out / "registration_details.json", details)
    return report


# ---------------------------------------------------------------------------
# Command line (run on the data server; prints counts only)


def run_migrate_v5(prepared_path: Path, policy: SourcePolicy, scope: ActiveScope, out: Path) -> dict:
    """Migrate prepared records (v5 migration, plan_v5 T07a) into a new private JSONL; failures carry no IDs."""
    import dataclasses

    from .migrate_v5 import migrate_v41_records_v5

    records = [validate_sample(json.loads(line)) for line in Path(prepared_path).read_text().splitlines() if line]
    migrated, failures = migrate_v41_records_v5(records, policy, scope)
    _write_private_text(Path(out), "".join(json.dumps(dataclasses.asdict(s), sort_keys=True) + "\n"
                                           for s in migrated))
    return {"records": len(migrated), "failures": failures,
            "index_alignment": dict(sorted(Counter(s.index_alignment_status for s in migrated).items())),
            "physical_geometry": dict(sorted(Counter(s.physical_geometry_status for s in migrated).items()))}


def main(argv: list | None = None) -> int:
    import argparse

    import yaml

    from .contracts_v5 import validate_scope_v5
    from .inventory import read_jsonl
    from .audits import load_skm_annotations
    from .source_policy_v5 import load_source_policies_v5

    parser = argparse.ArgumentParser(prog="anatobind.v41.real_t07_v5")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--raw", required=True)
    prepare.add_argument("--assets", required=True)
    prepare.add_argument("--scope", required=True)
    prepare.add_argument("--policies", required=True)
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--factor", type=int, nargs=3, required=True)
    prepare.add_argument("--skm-annotations-dir")
    prepare.add_argument("--skm-units")
    prepare.add_argument("--reuse-qc")
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--prepared", required=True)
    migrate.add_argument("--scope", required=True)
    migrate.add_argument("--policies", required=True)
    migrate.add_argument("--source-id", required=True)
    migrate.add_argument("--out", required=True)
    register = sub.add_parser("registration-report")
    register.add_argument("--assets", required=True)
    register.add_argument("--scope", required=True)
    register.add_argument("--out", required=True)
    register.add_argument("--skm-annotations-dir")
    args = parser.parse_args(argv)
    scope = validate_scope_v5(yaml.safe_load(Path(args.scope).read_text()))
    if args.command == "registration-report":
        report = run_registration_report_v5(
            read_jsonl(Path(args.assets)), scope, Path(args.out),
            Path(args.skm_annotations_dir) if args.skm_annotations_dir else None)
        print(render_registration_md_v5(report), end="")
        return 0
    policy = load_source_policies_v5(Path(args.policies))[args.source_id]
    if args.command == "prepare":
        annotations = load_skm_annotations(Path(args.skm_annotations_dir)) if args.skm_annotations_dir else None
        report = run_prepare_v5(Path(args.raw), read_jsonl(Path(args.assets)), scope, policy, Path(args.out),
                                factor=tuple(args.factor), skm_annotations=annotations, skm_units=args.skm_units,
                                reuse_qc=Path(args.reuse_qc) if args.reuse_qc else None)
        report = {k: report[k] for k in ("records", "failures", "geometry_status", "source_check_sha256")}
    else:
        report = run_migrate_v5(Path(args.prepared), policy, scope, Path(args.out))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
