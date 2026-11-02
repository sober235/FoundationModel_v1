"""Scope audit bundle builders: provenance, visibility and eligibility reports
plus minimal registration records for P0 sources.

Records carry identity and supervision metadata only; they never contain
voxel data. Historical exposure to legacy experiments is recorded as
unverified so no case can later be renamed an unexposed blind test.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

from anatobind.v41.adapters import isles, skm
from anatobind.v41.eligibility import annotation_eligibility
from anatobind.v41.paths import PrivatePathError

LEGACY_EXPOSURE = {"experiment_id": "legacy_foundation_model_unverified",
                   "role": "unknown", "recorded_at": "2026-11-02T00:00:00Z"}
SCOPES = ("isles_u", "skm_a")
_SCOPE_SOURCE = {"isles_u": "ISLES_ltr", "skm_a": "SKM-TEA_ltr"}
A6_CLASSES = ["patellar_cartilage", "femoral_cartilage", "tibial_cartilage_medial",
              "tibial_cartilage_lateral", "meniscus_medial", "meniscus_lateral"]


def _exclusive_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PrivatePathError("PRIVATE_FILE_EXISTS") from None
    with os.fdopen(fd, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: json.dumps(row.get(c)) if isinstance(row.get(c), (list, dict)) else row.get(c)
                             for c in columns})


def load_skm_annotations(annotations_dir: Path) -> dict:
    """Per-case box and subject metadata from SKM-TEA annotations v1.0.0 (read-only)."""
    annotations_dir = Path(annotations_dir)
    per_case: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = annotations_dir / f"{split}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        categories = {c["id"]: c["supercategory"] for c in doc["categories"]}
        images = {im["id"]: im for im in doc["images"]}
        for image in doc["images"]:
            case = Path(image["file_name"]).name.removesuffix(".h5")
            entry = per_case.setdefault(case, {"subject_id": image.get("subject_id"),
                                               "scan_id": image.get("scan_id"),
                                               "timepoint": image.get("timepoint"),
                                               "inspected": image.get("inspected"),
                                               "split_release": split, "boxes": []})
        for ann in doc["annotations"]:
            image = images[ann["image_id"]]
            case = Path(image["file_name"]).name.removesuffix(".h5")
            per_case.setdefault(case, {"subject_id": image.get("subject_id"), "scan_id": image.get("scan_id"),
                                       "timepoint": image.get("timepoint"), "inspected": image.get("inspected"),
                                       "split_release": split, "boxes": []})
            per_case[case]["boxes"].append({
                "supercategory": categories[ann["category_id"]],
                "category_id": ann["category_id"],
                "tissue_id": ann.get("tissue_id"),
                "confidence": ann.get("confidence"),
            })
    return per_case


def build_registration_records(scope: str, assets: list[dict], skm_annotations: dict | None = None) -> list[dict]:
    if scope not in SCOPES:
        raise ValueError("UNKNOWN_SCOPE")
    records = []
    if scope == "isles_u":
        for pair in isles.pair_assets(assets):
            case = pair["case_id"]
            records.append({
                "schema_version": "v41.0",
                "sample_uid": f"isles2022:{case}:ses-0001",
                "source_dataset": "ISLES_ltr",
                "source_release": "ISLES-2022",
                "cohort": "isles2022",
                "patient_uid": f"isles2022:{case}",
                "study_uid": f"isles2022:{case}:ses-0001",
                "session_uid": f"isles2022:{case}:ses-0001",
                "parent_scan_uid": None,
                "derivative_uid": None,
                "leakage_group_uid": f"isles2022:{case}",
                "exposure_history": [dict(LEGACY_EXPOSURE)],
                "split": "unassigned",
                "native_acquisition_dim": 3,
                "observation_mode": "3d",
                "image_paths": [pair["images"][role] for role in ("dwi", "adc", "flair") if role in pair["images"]],
                "modality_names": [role for role in ("dwi", "adc", "flair") if role in pair["images"]],
                "modality_is_synthetic": [False] * len(pair["images"]),
                "available_modality_mask": [role in pair["images"] for role in ("dwi", "adc", "flair")],
                "image_geometry": None,
                "original_geometry": None,
                "transform_chain": [],
                "geometry_status": "not_checked",
                "label_value_qc": "not_checked",
                "anatomy_ontology_version": "none_brain_a_pending",
                "abnormality_ontology_version": "u_v1",
                "relation_semantics_version": "r_v1",
                "annotation_records": [{
                    "annotation_uid": f"isles2022:{case}:msk",
                    "task": "U",
                    "class_scope": ["ischemic_infarct"],
                    "representation": "voxel",
                    "path": pair["label_relpath"],
                    "label_map_version": "isles2022_msk_v1",
                    "provenance_type": "manual",
                    "creator_or_model": "ISLES-2022 release",
                    "creator_version": "2022",
                    "parent_annotation_uid": None,
                    "completeness": "unknown",
                    "review_region_path": None,
                    "input_visibility": "unknown",
                    "quality_status": "accepted",
                    "intended_role": "train",
                    "reference_role": "expert_voxel",
                    "adjudication_status": "not_applicable",
                    "reader_ids": ["isles2022_release"],
                    "protocol_hash": None,
                    "visibility_evidence": "release protocol verification pending",
                }] if pair["label_relpath"] else [],
                "review_regions": [],
                "visibility_records": [],
                "quality_records": [],
                "provenance": {"ingest_commit": "v41_t07", "ingest_time": "2026-11-02T00:00:00Z"},
            })
    else:
        skm_annotations = skm_annotations or {}
        for pair in skm.pair_assets(assets):
            case = pair["case_id"]
            meta = skm_annotations.get(case, {})
            annotations = []
            for track, track_name in (("dicom_track_relpath", "dicom-track"), ("raw_track_relpath", "raw-data-track")):
                if pair.get(track):
                    annotations.append({
                        "annotation_uid": f"skmtea:{case}:seg:{track_name}",
                        "task": "A",
                        "class_scope": list(A6_CLASSES),
                        "representation": "voxel",
                        "path": pair[track],
                        "label_map_version": f"skmtea_seg_{track_name}_v1",
                        "provenance_type": "manual",
                        "creator_or_model": "SKM-TEA release",
                        "creator_version": "v1.0.0",
                        "parent_annotation_uid": None,
                        "completeness": "target_complete",
                        "review_region_path": None,
                        "input_visibility": "visible",
                        "quality_status": "accepted",
                        "intended_role": "train",
                        "reference_role": "expert_voxel",
                        "adjudication_status": "not_applicable",
                        "reader_ids": ["skmtea_release"],
                        "protocol_hash": None,
                        "visibility_evidence": "SKM-TEA segmentation track",
                    })
            for index, box in enumerate(meta.get("boxes", [])):
                annotations.append({
                    "annotation_uid": f"skmtea:{case}:box:{index}",
                    "task": "U",
                    "class_scope": [box["supercategory"].lower().replace(" ", "_")],
                    "representation": "bbox",
                    "path": f"annotations/v1.0.0/{meta.get('split_release', 'unknown')}.json",
                    "label_map_version": "skmtea_boxes_v1",
                    "provenance_type": "manual",
                    "creator_or_model": "SKM-TEA release",
                    "creator_version": "v1.0.0",
                    "parent_annotation_uid": None,
                    "completeness": "positive_only",
                    "review_region_path": None,
                    "input_visibility": "visible",
                    "quality_status": "accepted" if (box.get("confidence") or 0) > 0 else "rejected",
                    "intended_role": "train",
                    "reference_role": "weak_localization",
                    "adjudication_status": "not_applicable",
                    "reader_ids": ["skmtea_release"],
                    "protocol_hash": None,
                    "visibility_evidence": f"tissue_id={box.get('tissue_id')} confidence={box.get('confidence')}",
                })
            subject = meta.get("subject_id")
            records.append({
                "schema_version": "v41.0",
                "sample_uid": f"skmtea:{case}",
                "source_dataset": "SKM-TEA_ltr",
                "source_release": "SKM-TEA v1.0.0",
                "cohort": "skmtea",
                "patient_uid": f"skmtea:subject:{subject}" if subject is not None else f"skmtea:{case}",
                "study_uid": f"skmtea:{case}",
                "session_uid": f"skmtea:{case}",
                "parent_scan_uid": None,
                "derivative_uid": None,
                "leakage_group_uid": f"skmtea:subject:{subject}" if subject is not None else f"skmtea:{case}",
                "exposure_history": [dict(LEGACY_EXPOSURE)],
                "split": "unassigned",
                "native_acquisition_dim": 3,
                "observation_mode": "3d",
                "image_paths": [pair["h5_relpath"]] if pair["h5_relpath"] else [],
                "modality_names": ["qdess_dual_echo"],
                "modality_is_synthetic": [False],
                "available_modality_mask": [True],
                "image_geometry": None,
                "original_geometry": None,
                "transform_chain": [],
                "geometry_status": "not_checked",
                "label_value_qc": "not_checked",
                "anatomy_ontology_version": "skm_a6_v1",
                "abnormality_ontology_version": "u_v1",
                "relation_semantics_version": "r_v1",
                "annotation_records": annotations,
                "review_regions": [],
                "visibility_records": [],
                "quality_records": [],
                "provenance": {"ingest_commit": "v41_t07", "ingest_time": "2026-11-02T00:00:00Z",
                               "inspected_release_flag": meta.get("inspected"),
                               "release_split": meta.get("split_release")},
            })
    return records


def provenance_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        for ann in record["annotation_records"]:
            rows.append({
                "sample_uid": record["sample_uid"],
                "source_dataset": record["source_dataset"],
                "annotation_uid": ann["annotation_uid"],
                "task": ann["task"],
                "representation": ann["representation"],
                "provenance_type": ann["provenance_type"],
                "reference_role": ann["reference_role"],
                "completeness": ann["completeness"],
                "quality_status": ann["quality_status"],
                "intended_role": ann["intended_role"],
                "creator_or_model": ann["creator_or_model"],
            })
    return rows


def visibility_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            "sample_uid": record["sample_uid"],
            "source_dataset": record["source_dataset"],
            "modalities": record["modality_names"],
            "available_modality_mask": record["available_modality_mask"],
            "image_count": len(record["image_paths"]),
            "annotation_input_visibility": sorted({a["input_visibility"] for a in record["annotation_records"]}),
        })
    return rows


def eligibility_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        for ann in record["annotation_records"]:
            decision = annotation_eligibility(ann, ann["task"], ann["class_scope"])
            rows.append({
                "sample_uid": record["sample_uid"],
                "source_dataset": record["source_dataset"],
                "annotation_uid": ann["annotation_uid"],
                "task": ann["task"],
                "role": decision["role"],
                "localization_only": decision["localization_only"],
                "reason": decision["reason"],
                "positive_classes": [k for k, v in decision["classes"].items() if v["positive"]],
                "negative_classes": [k for k, v in decision["classes"].items() if v["negative"]],
            })
    return rows


def summarize(rows: list[dict], *keys: str) -> dict:
    counter = Counter(tuple(str(row.get(k)) for k in keys) for row in rows)
    return {", ".join(parts): count for parts, count in sorted(counter.items())}


def write_scope_audit(scope: str, assets: list[dict], out: Path, annotations_dir: Path | None = None) -> dict:
    if scope not in SCOPES:
        raise ValueError("UNKNOWN_SCOPE")
    source = _SCOPE_SOURCE[scope]
    scoped = [a for a in assets if a.get("source") == source]
    skm_annotations = load_skm_annotations(annotations_dir) if scope == "skm_a" and annotations_dir else None
    records = build_registration_records(scope, scoped, skm_annotations)
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    _exclusive_csv(out / "annotation_provenance_report.csv", provenance_rows(records),
                   ["sample_uid", "source_dataset", "annotation_uid", "task", "representation",
                    "provenance_type", "reference_role", "completeness", "quality_status",
                    "intended_role", "creator_or_model"])
    _exclusive_csv(out / "input_visibility_report.csv", visibility_rows(records),
                   ["sample_uid", "source_dataset", "modalities", "available_modality_mask",
                    "image_count", "annotation_input_visibility"])
    _exclusive_csv(out / "eligibility_coverage.csv", eligibility_rows(records),
                   ["sample_uid", "source_dataset", "annotation_uid", "task", "role",
                    "localization_only", "reason", "positive_classes", "negative_classes"])
    try:
        fd = os.open(out / "registration_records.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PrivatePathError("PRIVATE_FILE_EXISTS") from None
    with os.fdopen(fd, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    prov = provenance_rows(records)
    elig = eligibility_rows(records)
    summary = {
        "scope": scope,
        "records": len(records),
        "provenance_by_task_role": summarize(prov, "task", "provenance_type", "reference_role", "completeness"),
        "eligibility_by_role": summarize(elig, "task", "role", "localization_only"),
        "patients": len({r["patient_uid"] for r in records}),
        "exposure_note": "all records carry an unverified legacy exposure entry; none may be called an unexposed blind test",
    }
    if skm_annotations is not None:
        summary["skm_subjects"] = len({m["subject_id"] for m in skm_annotations.values() if m.get("subject_id") is not None})
        summary["skm_cases_with_boxes"] = sum(1 for m in skm_annotations.values() if m["boxes"])
        summary["skm_release_splits"] = dict(Counter(m["split_release"] for m in skm_annotations.values()))
    return summary
