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
EXCLUDED_INTENDED_ROLES = ("excluded", "pseudo_only")


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
    if completeness == "unreviewed":
        decision.update(role="none", reason="COMPLETENESS_UNREVIEWED")
        return decision
    if completeness not in COMPLETE + REGION + POSITIVE_ONLY:
        decision.update(role="none", reason="COMPLETENESS_UNKNOWN")
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
                      targets: dict[str, np.ndarray] | None = None,
                      roles: tuple[str, ...] = ("qualified",)) -> np.ndarray:
    review_regions = review_regions or {}
    targets = targets or {}
    eligible = np.zeros((len(class_scope), *shape), dtype=bool)
    for annotation in record.get("annotation_records", []):
        if annotation.get("intended_role") in EXCLUDED_INTENDED_ROLES:
            continue
        decision = annotation_eligibility(annotation, task, class_scope)
        if decision["role"] not in roles or decision["localization_only"]:
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


# ---------------------------------------------------------------------------
# v5.1 signed eligibility (Proposal_v5 §2.3, §4.2). v41 functions above are unchanged.

from .contracts_v5 import V5Error  # noqa: E402  (v5.1 section is appended; v41 code above untouched)

TRACKS_V5 = ("observable_target", "released_reference")
CHANNELS_V5 = ("expert", "pseudo")


def annotation_channel_v5(annotation, task: str) -> str | None:
    """'expert', 'pseudo' or None (never supervises this task)."""
    if annotation.task != task or annotation.quality_status != "accepted":
        return None
    if annotation.provenance_type in NEVER_SUPERVISE or annotation.representation in LOCALIZATION_ONLY:
        return None
    if annotation.intended_role == "excluded":
        return None
    if annotation.provenance_type not in QUALIFIED_PROVENANCE + PSEUDO_PROVENANCE:
        return None
    if (annotation.provenance_type in QUALIFIED_PROVENANCE and annotation.reference_role != "unverified"
            and annotation.intended_role != "pseudo_only"):
        return "expert"
    return "pseudo"


def region_mask_v5(regions: dict, key, shape: tuple, code: str, path: str) -> np.ndarray:
    if key is None or key not in regions:
        raise V5Error(code, path)
    region = np.asarray(regions[key], dtype=bool)
    if region.shape != shape:
        raise V5Error("GRID_MISMATCH", path)
    return region


def visibility_support_v5(annotation, record, visibility_regions: dict, shape: tuple, policy: dict,
                   path: str) -> np.ndarray:
    visibility = annotation.input_visibility
    if policy["track"] == "observable_target":
        if visibility == "visible":
            return np.ones(shape, dtype=bool)
        if visibility == "limited":
            return region_mask_v5(visibility_regions, annotation.visibility_region_path, shape,
                           "VISIBILITY_REGION_MISSING", f"{path}.visibility_region_path")
        return np.zeros(shape, dtype=bool)
    if visibility in ("visible", "limited"):
        return np.ones(shape, dtype=bool)
    if visibility == "unknown" and policy.get("allow_unknown_visibility") is True \
            and policy.get("reference_policy_id") == record.reference_policy_id:
        return np.ones(shape, dtype=bool)
    return np.zeros(shape, dtype=bool)


def build_eligibility_v5(record, task: str, classes: list[str], targets: dict, review_regions: dict,
                         visibility_regions: dict, valid_fov: np.ndarray, policy: dict,
                         channel: str) -> tuple[np.ndarray, np.ndarray]:
    if policy.get("track") not in TRACKS_V5:
        raise V5Error("UNKNOWN_TRACK", "$.policy.track")
    if policy["track"] == "observable_target" and policy.get("allow_unknown_visibility"):
        raise V5Error("UNKNOWN_VISIBILITY_NOT_ALLOWED", "$.policy.allow_unknown_visibility")
    if channel not in CHANNELS_V5:
        raise V5Error("UNKNOWN_CHANNEL", "$.channel")
    fov = np.asarray(valid_fov, dtype=bool)
    shape = fov.shape
    e_pos = np.zeros((len(classes), *shape), dtype=bool)
    e_neg = np.zeros((len(classes), *shape), dtype=bool)
    for i, annotation in enumerate(record.annotation_records):
        path = f"$.annotation_records[{i}]"
        if annotation_channel_v5(annotation, task) != channel:
            continue
        indices = [k for k, name in enumerate(classes) if name in annotation.class_scope]
        if not indices:
            continue
        support = visibility_support_v5(annotation, record, visibility_regions, shape, policy, path) & fov
        completeness = annotation.completeness
        if completeness == "target_complete":
            for k in indices:
                e_pos[k] |= support
                e_neg[k] |= support
        elif completeness == "region_complete":
            reviewed = support & region_mask_v5(review_regions, annotation.review_region_path, shape,
                                         "REVIEW_REGION_MISSING", f"{path}.review_region_path")
            for k in indices:
                e_pos[k] |= reviewed
                e_neg[k] |= reviewed
        elif completeness == "positive_only":
            class_values = policy.get("class_values")
            if not isinstance(class_values, dict) or any(classes[k] not in class_values for k in indices):
                raise V5Error("POSITIVE_ONLY_CLASS_MAP_REQUIRED", "$.policy.class_values")
            if annotation.annotation_uid not in targets:
                raise V5Error("TARGET_MISSING", f"{path}.annotation_uid")
            target = np.asarray(targets[annotation.annotation_uid])
            if target.shape != shape:
                raise V5Error("GRID_MISMATCH", f"{path}.annotation_uid")
            for k in indices:
                e_pos[k] |= support & (target == class_values[classes[k]])
    return e_pos, e_neg
