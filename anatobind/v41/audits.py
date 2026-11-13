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
ISLES_PROTOCOL_EVIDENCE = ("ISLES 2022 protocol (doi:10.1038/s41597-022-01875-5): hybrid U-Net pre-segmentation edited by trained raters, "
                           "approved by an attending neuroradiologist; acute/sub-acute infarct only; raters also read FLAIR, "
                           "which is not a model input")
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


SKM_RELEASE_SPLITS = ("train", "val", "test")


def scan_skm_release(annotations_dir: Path) -> tuple[dict, dict, dict]:
    """Read every official split file of SKM-TEA annotations v1.0.0 (read-only).

    Returns (per_case, report, details). All three official files are checked and read; nothing is skipped
    silently. `report` holds counts only: per file whether it is present and readable, its scans, the patients
    that can be confirmed (distinct release subject_id) and its boxes, plus missing files, duplicate cases and
    split conflicts. `details` names the cases and subjects behind each issue and is private.

    A case listed in more than one official split keeps every split in `release_splits` and gets
    `split_release` None, so it cannot be given a formal split by the official rule until the conflict is
    resolved. Its boxes are kept from the first file only, so a conflict never doubles them.
    """
    annotations_dir = Path(annotations_dir)
    per_case: dict[str, dict] = {}
    files: dict[str, dict] = {}
    details = {"duplicate_cases_within_file": [], "cases_in_multiple_splits": [], "patients_in_multiple_splits": [],
               "scans_without_subject": []}
    subject_splits: dict = {}
    orphans = 0
    for split in SKM_RELEASE_SPLITS:
        path = annotations_dir / f"{split}.json"
        entry = files[split] = {"present": path.is_file(), "readable": False, "image_entries": 0, "scans": 0,
                                "patients": 0, "scans_without_subject": 0, "boxes": 0}
        if not entry["present"]:
            continue
        try:
            doc = json.loads(path.read_text())
            categories = {c["id"]: c["supercategory"] for c in doc["categories"]}
            images, annotations = list(doc["images"]), list(doc["annotations"])
        except (ValueError, KeyError, TypeError):
            continue
        entry["readable"] = True
        entry["image_entries"] = len(images)
        case_of: dict = {}
        cases_here: set = set()
        subjects_here: set = set()
        for image in images:
            case = Path(image["file_name"]).name.removesuffix(".h5")
            case_of[image["id"]] = case
            if case in cases_here:
                details["duplicate_cases_within_file"].append(case)
                per_case[case]["duplicate_entries"] += 1
                continue
            cases_here.add(case)
            subject = image.get("subject_id")
            if subject is None:
                entry["scans_without_subject"] += 1
                details["scans_without_subject"].append(case)
            else:
                subjects_here.add(subject)
                subject_splits.setdefault(subject, set()).add(split)
            if case in per_case:
                per_case[case]["release_splits"].append(split)
                per_case[case]["split_release"] = None
                continue
            per_case[case] = {"subject_id": subject, "scan_id": image.get("scan_id"),
                              "timepoint": image.get("timepoint"), "inspected": image.get("inspected"),
                              "split_release": split, "release_splits": [split], "duplicate_entries": 0,
                              "boxes": []}
        entry["scans"], entry["patients"] = len(cases_here), len(subjects_here)
        counted: set = set()
        for ann in annotations:
            case = case_of.get(ann.get("image_id"))
            if case is None or ann.get("category_id") not in categories:
                orphans += 1
                continue
            entry["boxes"] += 1
            if per_case[case]["release_splits"][0] != split:
                continue                                  # a conflicting case keeps the boxes of its first file
            if (case, ann.get("id")) in counted and ann.get("id") is not None:
                continue
            counted.add((case, ann.get("id")))
            per_case[case]["boxes"].append({
                "supercategory": categories[ann["category_id"]],
                "category_id": ann["category_id"],
                "tissue_id": ann.get("tissue_id"),
                "confidence": ann.get("confidence"),
            })
    details["cases_in_multiple_splits"] = sorted(c for c, m in per_case.items() if len(m["release_splits"]) > 1)
    details["patients_in_multiple_splits"] = sorted(str(s) for s, v in subject_splits.items() if len(v) > 1)
    for key in ("duplicate_cases_within_file", "scans_without_subject"):
        details[key] = sorted(set(details[key]))
    report = {"files": files,
              "missing_files": [s for s in SKM_RELEASE_SPLITS if not files[s]["present"]],
              "unreadable_files": [s for s in SKM_RELEASE_SPLITS if files[s]["present"] and not files[s]["readable"]],
              "scans": len(per_case), "patients": len(subject_splits),
              "scans_without_subject": len(details["scans_without_subject"]),
              "duplicate_cases_within_file": len(details["duplicate_cases_within_file"]),
              "cases_in_multiple_splits": len(details["cases_in_multiple_splits"]),
              "patients_in_multiple_splits": len(details["patients_in_multiple_splits"]),
              "orphan_annotations": orphans}
    return per_case, report, details


def load_skm_annotations(annotations_dir: Path) -> dict:
    """Per-case box and subject metadata from SKM-TEA annotations v1.0.0 (read-only); see scan_skm_release."""
    return scan_skm_release(annotations_dir)[0]


def build_registration_records(scope: str, assets: list[dict], skm_annotations: dict | None = None) -> list[dict]:
    if scope not in SCOPES:
        raise ValueError("UNKNOWN_SCOPE")
    records = []
    if scope == "isles_u":
        for pair in isles.pair_assets(assets):
            case = pair["case_id"]
            present = [role for role in isles.ROLES if role in pair["images"]]
            if not present:
                continue
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
                "image_paths": [pair["images"][role] for role in present],
                "modality_names": list(present),
                "modality_is_synthetic": [False] * len(present),
                "available_modality_mask": [True] * len(present),
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
                    "completeness": "target_complete",
                    "review_region_path": None,
                    "input_visibility": "visible",
                    "quality_status": "accepted",
                    "intended_role": "train",
                    "reference_role": "expert_voxel",
                    "adjudication_status": "not_applicable",
                    "reader_ids": ["isles2022_release"],
                    "protocol_hash": None,
                    "visibility_evidence": ISLES_PROTOCOL_EVIDENCE,
                }] if pair["label_relpath"] else [],
                "review_regions": [],
                "visibility_records": [],
                "quality_records": [],
                "provenance": {"ingest_commit": "v41_t07", "ingest_time": "2026-11-02T00:00:00Z"},
            })
    else:
        release_given = skm_annotations is not None
        skm_annotations = skm_annotations or {}
        for pair in skm.pair_assets(assets):
            if not pair["h5_relpath"]:
                continue
            case = pair["case_id"]
            meta = skm_annotations.get(case, {})
            annotations = []
            for track, track_name in (("dicom_track_relpath", "dicom-track"),):
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
                    "path": f"annotations/v1.0.0/{meta.get('split_release') or 'unknown'}.json",
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
                "image_paths": [pair["h5_relpath"]],
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
                               "release_split": meta.get("split_release"),
                               **(_release_flags(meta) if release_given else {})},
            })
    return records


def _release_flags(meta: dict) -> dict:
    """Provenance flags written only when something is wrong, so a clean record keeps its earlier bytes."""
    flags = {}
    if len(meta.get("release_splits") or []) > 1:
        flags["release_split_conflict"] = list(meta["release_splits"])
    if meta.get("duplicate_entries"):
        flags["release_duplicate_entries"] = meta["duplicate_entries"]
    if not meta:
        flags["release_entry"] = "missing"
    elif meta.get("subject_id") is None:
        flags["patient_identity"] = "unconfirmed_case_fallback"
    return flags


def registration_summary(records: list[dict]) -> dict:
    """Counts only. Scans and patients are counted separately: a patient is counted once however many scans,
    echoes, tracks or follow-ups it has, and only when its identity is confirmed by the source."""
    def confirmed(record: dict) -> bool:
        flags = record.get("provenance") or {}
        return flags.get("release_entry") != "missing" and "patient_identity" not in flags

    uids = [r["sample_uid"] for r in records]
    by_release: dict = {}
    for record in records:
        provenance = record.get("provenance") or {}
        if "release_split" not in provenance:
            continue
        key = provenance["release_split"] or ("conflict" if "release_split_conflict" in provenance else "missing")
        row = by_release.setdefault(key, {"scans": 0, "patients": set(), "scans_patient_unconfirmed": 0})
        row["scans"] += 1
        if confirmed(record):
            row["patients"].add(record["patient_uid"])
        else:
            row["scans_patient_unconfirmed"] += 1
    group_of: dict = {}
    for record in records:
        group_of.setdefault(record["patient_uid"], set()).add(record["leakage_group_uid"])
    return {
        "scans": len(records),
        "duplicate_sample_uids": len(uids) - len(set(uids)),
        "patients_confirmed": len({r["patient_uid"] for r in records if confirmed(r)}),
        "scans_patient_unconfirmed": sum(1 for r in records if not confirmed(r)),
        "leakage_groups": len({r["leakage_group_uid"] for r in records}),
        "patients_in_multiple_leakage_groups": sum(1 for groups in group_of.values() if len(groups) > 1),
        "by_release_split": {k: {**v, "patients": len(v["patients"])} for k, v in sorted(by_release.items())},
        "formal_split": dict(sorted(Counter(r["split"] for r in records).items())),
        "exposure_roles": dict(sorted(Counter(e["role"] for r in records for e in r["exposure_history"]).items())),
        "scans_without_exposure_entry": sum(1 for r in records if not r["exposure_history"]),
        "scans_without_label": sum(1 for r in records if not r["annotation_records"]),
    }


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
