"""Per-task/class/region supervision eligibility.

Eligibility answers "may this voxel/class produce a gradient for this task?",
separately for the positive and the negative sign, and separately for
qualified (human-reviewed) and pseudo supervision. It never converts a
missing review into a negative.
"""
from __future__ import annotations

import numpy as np

QUALIFIED_PROVENANCE = ("manual", "expert_corrected")
PSEUDO_PROVENANCE = ("automated",)
NEVER_SUPERVISE = ("box_fill", "geometry", "unknown")
LOCALIZATION_ONLY = ("bbox", "instance", "study")
COMPLETE = ("target_complete",)
REGION = ("region_complete",)
POSITIVE_ONLY = ("positive_only",)


def annotation_eligibility(annotation: dict, task: str, class_scope: list[str]) -> dict:
    classes = {name: {"positive": False, "negative": False} for name in class_scope}
    decision = {"role": "none", "reason": None, "localization_only": False, "classes": classes}
    if annotation.get("task") != task or annotation.get("quality_status") != "accepted":
        decision["reason"] = "TASK_OR_QUALITY"
        return decision
    provenance = annotation.get("provenance_type")
    if provenance in NEVER_SUPERVISE:
        decision["reason"] = "PROVENANCE_NEVER_SUPERVISES"
        return decision
    role = "qualified" if provenance in QUALIFIED_PROVENANCE else "pseudo"
    if annotation.get("reference_role") == "unverified":
        role = "pseudo"
    if annotation.get("representation") in LOCALIZATION_ONLY:
        decision.update(role=role, reason="LOCALIZATION_ONLY", localization_only=True)
        return decision
    completeness = annotation.get("completeness")
    if completeness not in COMPLETE + REGION + POSITIVE_ONLY:
        decision.update(role=role, reason="COMPLETENESS_UNREVIEWED")
        return decision
    covered = set(annotation.get("class_scope") or [])
    for name in class_scope:
        if name not in covered:
            continue
        classes[name]["positive"] = True
        classes[name]["negative"] = completeness in COMPLETE + REGION
    decision.update(role=role, reason=None)
    return decision


def build_eligibility(record: dict, task: str, class_scope: list[str], shape: tuple[int, ...],
                      review_regions: dict[str, np.ndarray] | None = None,
                      targets: dict[str, np.ndarray] | None = None) -> np.ndarray:
    review_regions = review_regions or {}
    targets = targets or {}
    eligible = np.zeros((len(class_scope), *shape), dtype=bool)
    for annotation in record.get("annotation_records", []):
        decision = annotation_eligibility(annotation, task, class_scope)
        if decision["role"] == "none" or decision["localization_only"]:
            continue
        completeness = annotation.get("completeness")
        for index, name in enumerate(class_scope):
            grants = decision["classes"][name]
            if not (grants["positive"] or grants["negative"]):
                continue
            if completeness in COMPLETE:
                eligible[index] |= True
            elif completeness in REGION:
                region = review_regions.get(annotation.get("review_region_path"))
                if region is None:
                    continue
                region = np.asarray(region, dtype=bool)
                if region.shape != shape:
                    raise ValueError("REVIEW_REGION_SHAPE")
                eligible[index] |= region
            elif completeness in POSITIVE_ONLY:
                target = targets.get(annotation.get("annotation_uid"))
                if target is None:
                    continue
                eligible[index] |= np.asarray(target) > 0
    return eligible


def supervision_counts(record: dict, task_tensors: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    roles = {}
    for annotation in record.get("annotation_records", []):
        decision = annotation_eligibility(annotation, annotation.get("task", ""), annotation.get("class_scope") or [])
        if decision["role"] != "none":
            roles.setdefault(annotation.get("task"), decision["role"])
    rows = []
    for task in sorted(task_tensors):
        eligible, target = task_tensors[task]
        eligible = np.asarray(eligible, dtype=bool)
        target = np.asarray(target)
        positive = int(np.logical_and(eligible, target > 0).sum())
        negative = int(np.logical_and(eligible, target == 0).sum())
        role = roles.get(task, "none")
        rows.append({
            "sample_uid": record.get("sample_uid"),
            "source_dataset": record.get("source_dataset"),
            "task": task,
            "role": role,
            "positive_voxels": positive if role == "qualified" else 0,
            "negative_voxels": negative if role == "qualified" else 0,
            "pseudo_voxels": positive + negative if role == "pseudo" else 0,
            "eligible_voxels": int(eligible.sum()),
        })
    return rows
