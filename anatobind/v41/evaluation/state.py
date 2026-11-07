"""StateReference and direct tissue identity-state evaluation (Proposal_v5 §4.1, §8.5).

Evaluation-role data only: deployment code (state_rule_v5, inference) must
never import this module. Tissue grouping and the denominator come from the
reference A and reference eligibility, never from the predicted A; rejected
or out-of-scope predictions stay in the fixed denominator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts_v5 import ActiveScope, PerceptionOutput, SampleV5, V5Error, array_sha256, canonical_sha256
from ..eligibility import annotation_channel_v5, region_mask_v5, visibility_support_v5
from ..state_rule_v5 import NOT_ASSESSABLE, REASON_BITS, TARGET_NEGATIVE, TARGET_POSITIVE
from .statistics import patient_bootstrap_v5

STATE_ELIGIBILITY_BITS = {
    "a_unqualified": 1,
    "u_incomplete": 2,
    "not_reviewed": 4,
    "not_visible": 8,
    "out_of_fov": 16,
    "not_state_tissue": 32,
    "single_slice_unreadable": 64,
}
STATE_NAMES = (("target_positive", TARGET_POSITIVE), ("target_negative", TARGET_NEGATIVE))
_SPATIAL_NDIM = {"2d": 2, "3d": 3}
_U_COMPLETE = ("target_complete", "region_complete")


@dataclass(frozen=True)
class StateReference:
    observation_uid: str
    mode: str
    scope_hash: str
    anatomy_truth: np.ndarray
    state_truth: np.ndarray
    state_eligible: np.ndarray
    eligibility_reason: np.ndarray
    reference_set_hash: str
    geometry: dict


def _reference_array(arrays: dict, name: str, shape: tuple, kind: str) -> np.ndarray:
    if name not in arrays or not isinstance(arrays[name], np.ndarray):
        raise V5Error("ARRAY_REQUIRED", f"$.reference_arrays.{name}")
    array = arrays[name]
    if kind == "bool" and array.dtype != np.bool_:
        raise V5Error("DTYPE_MISMATCH", f"$.reference_arrays.{name}")
    if kind == "int" and (array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer)):
        raise V5Error("DTYPE_MISMATCH", f"$.reference_arrays.{name}")
    if array.shape != shape:
        raise V5Error("GRID_MISMATCH", f"$.reference_arrays.{name}")
    return array


_OBSERVABLE_TRACK = {"track": "observable_target"}


def _reference_annotations(record: SampleV5, task: str) -> list:
    """Expert-source voxel references only: stricter than the expert training channel."""
    return [(i, a) for i, a in enumerate(record.annotation_records)
            if annotation_channel_v5(a, task) == "expert" and a.reference_role == "expert_voxel"]


def build_state_reference_v5(record: SampleV5, reference_arrays: dict, scope: ActiveScope,
                             observation_uid: str) -> StateReference:
    """Freeze the per-voxel state reference before any prediction is scored.

    Review and visibility come from the record's qualifying annotations (regions looked up by their
    path in reference_arrays["review_regions"] / ["visibility_regions"]) under the observable_target
    rules of build_eligibility_v5, so unknown or unobservable visibility never yields evaluable normal.
    """
    if scope.scope_kind != "joint":
        raise V5Error("SCOPE_MISMATCH", "$.scope")
    anatomy = reference_arrays.get("anatomy")
    if not isinstance(anatomy, np.ndarray) or anatomy.ndim != _SPATIAL_NDIM[scope.observation_mode]:
        raise V5Error("GRID_MISMATCH", "$.reference_arrays.anatomy")
    shape = anatomy.shape
    anatomy = _reference_array(reference_arrays, "anatomy", shape, "int")
    abnormality = _reference_array(reference_arrays, "abnormality", (len(scope.abnormality_classes), *shape), "bool")
    valid_fov = _reference_array(reference_arrays, "valid_fov", shape, "bool")
    readable = reference_arrays.get("single_slice_readable")
    if readable is not None and not isinstance(readable, bool):
        raise V5Error("INVALID_TYPE", "$.reference_arrays.single_slice_readable")
    review_regions = reference_arrays.get("review_regions", {})
    visibility_regions = reference_arrays.get("visibility_regions", {})

    a_covered = set()
    for _, a in _reference_annotations(record, "A"):
        if a.completeness == "target_complete":
            a_covered |= set(a.class_scope)
    reviewed = np.ones(shape, dtype=bool)
    supported = np.ones(shape, dtype=bool)
    u_annotations = [(i, a) for i, a in _reference_annotations(record, "U") if a.completeness in _U_COMPLETE]
    for name in scope.abnormality_classes:
        class_reviewed = np.zeros(shape, dtype=bool)
        class_supported = np.zeros(shape, dtype=bool)
        for i, a in u_annotations:
            if name not in a.class_scope:
                continue
            path = f"$.annotation_records[{i}]"
            if a.completeness == "target_complete":
                region = np.ones(shape, dtype=bool)
            else:
                region = region_mask_v5(review_regions, a.review_region_path, shape, "REVIEW_REGION_MISSING",
                                        f"{path}.review_region_path")
            visible = visibility_support_v5(a, record, visibility_regions, shape, _OBSERVABLE_TRACK, path)
            class_reviewed |= region
            class_supported |= region & visible
        reviewed &= class_reviewed
        supported &= class_supported
    u_covered = {name for _, a in u_annotations for name in a.class_scope}

    bits = STATE_ELIGIBILITY_BITS
    reason = np.zeros(shape, dtype=np.uint8)
    if not set(scope.state_anatomy_classes) <= a_covered:
        reason |= bits["a_unqualified"]
    if not set(scope.abnormality_classes) <= u_covered:
        reason |= bits["u_incomplete"]
    if scope.observation_mode == "2d" and readable is not True:
        reason |= bits["single_slice_unreadable"]
    reason[~reviewed] |= bits["not_reviewed"]
    reason[reviewed & ~supported] |= bits["not_visible"]
    reason[~valid_fov] |= bits["out_of_fov"]
    reason[~np.isin(anatomy, np.asarray(scope.state_tissue_ids))] |= bits["not_state_tissue"]

    stored = {"anatomy_truth": anatomy.copy(), "state_truth": np.any(abnormality, axis=0).astype(np.uint8),
              "eligibility_reason": reason, "state_eligible": reason == 0}
    for array in stored.values():
        array.setflags(write=False)
    geometry = {"shape": list(shape), "physical_geometry_status": record.physical_geometry_status}
    return StateReference(observation_uid=observation_uid, mode=scope.observation_mode, scope_hash=scope.scope_hash,
                          reference_set_hash=_state_reference_hash(observation_uid, scope.observation_mode,
                                                                   scope.scope_hash, geometry, stored),
                          geometry=geometry, **stored)


def _state_reference_hash(observation_uid: str, mode: str, scope_hash: str, geometry: dict, stored: dict) -> str:
    return canonical_sha256({
        "observation_uid": observation_uid, "mode": mode, "scope_hash": scope_hash, "geometry": geometry,
        "arrays": {name: array_sha256(stored[name]) for name in ("anatomy_truth", "state_truth", "eligibility_reason")},
    })


def verify_state_reference_v5(reference: StateReference) -> None:
    stored = {"anatomy_truth": reference.anatomy_truth, "state_truth": reference.state_truth,
              "eligibility_reason": reference.eligibility_reason}
    expected = _state_reference_hash(reference.observation_uid, reference.mode, reference.scope_hash,
                                     reference.geometry, stored)
    if expected != reference.reference_set_hash \
            or not np.array_equal(reference.state_eligible, reference.eligibility_reason == 0):
        raise V5Error("REFERENCE_HASH_MISMATCH", "$.reference_set_hash")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _overlap_metrics(g: np.ndarray, p: np.ndarray) -> dict:
    tp, n_g, n_p = int((g & p).sum()), int(g.sum()), int(p.sum())
    return {"tp": tp, "fp": n_p - tp, "fn": n_g - tp, "support": n_g, "dice": _ratio(2 * tp, n_g + n_p),
            "precision": _ratio(tp, n_p), "recall": _ratio(tp, n_g)}


def tissue_state_metrics_v5(pred: PerceptionOutput, reference: StateReference, scope: ActiveScope) -> dict:
    if (pred.observation_uid != reference.observation_uid or pred.mode != reference.mode
            or not pred.scope_hash == reference.scope_hash == scope.scope_hash
            or pred.tissue_state_map.shape != reference.state_eligible.shape):
        raise V5Error("IDENTITY_MISMATCH", "$.reference")
    verify_state_reference_v5(reference)
    e = reference.state_eligible
    a_ref, s_ref = reference.anatomy_truth, reference.state_truth
    a_pred, s_pred = pred.anatomy_map, pred.tissue_state_map
    accepted_s = s_pred != NOT_ASSESSABLE

    per_tissue_state = {}
    for name in scope.state_anatomy_classes:
        k = scope.tissue_id(name)
        per_tissue_state[name] = {
            state_name: _overlap_metrics(e & (a_ref == k) & (s_ref == z), e & (a_pred == k) & (s_pred == z) & accepted_s)
            for state_name, z in STATE_NAMES
        }
    u_pred = np.any(pred.abnormality_map & pred.acceptance["abnormality"], axis=0) & e
    u_ref = (s_ref == TARGET_POSITIVE) & e
    fp_voxels = int((e & (s_ref == TARGET_NEGATIVE) & (s_pred == TARGET_POSITIVE)).sum())
    rejected = e & (s_pred == NOT_ASSESSABLE)
    rejection_reasons = {}
    for name, bit in REASON_BITS.items():
        count = int((rejected & ((pred.reason["state"] & bit) != 0)).sum())
        if count:
            rejection_reasons[name] = count
    return {
        "reference_count": int(e.sum()),
        "per_tissue_state": per_tissue_state,
        "u_dice_in_reference_tissue": _ratio(2 * int((u_pred & u_ref).sum()), int(u_pred.sum() + u_ref.sum())),
        "false_normal_on_accepted": _ratio(int((e & (s_pred == TARGET_NEGATIVE) & (s_ref == TARGET_POSITIVE)).sum()),
                                           int((e & (s_pred == TARGET_NEGATIVE)).sum())),
        "accepted_positive_fnr": _ratio(int((e & (s_ref == TARGET_POSITIVE) & (s_pred == TARGET_NEGATIVE)).sum()),
                                        int((e & (s_ref == TARGET_POSITIVE) & accepted_s).sum())),
        "normal_fp_fraction": _ratio(fp_voxels, int((e & (s_ref == TARGET_NEGATIVE)).sum())),
        "fp_voxels": fp_voxels,
        "coverage": _ratio(int((e & accepted_s).sum()), int(e.sum())),
        "rejection_reasons": rejection_reasons,
        "reference_set_hash": reference.reference_set_hash,
    }


def summarize_state_by_patient_v5(results: list[tuple[str, dict]], metric: tuple[str, str, str], seed: int,
                                  n_boot: int, ci_level: float) -> dict:
    """Per patient: mean of the metric's non-None values over observations; then patient-equal bootstrap."""
    tissue, state_name, metric_name = metric
    values: dict[str, list[float]] = {}
    patients = set()
    for patient, result in results:
        patients.add(patient)
        value = result["per_tissue_state"][tissue][state_name][metric_name]
        if value is not None:
            values.setdefault(patient, []).append(float(value))
    summary = patient_bootstrap_v5({p: sum(v) / len(v) for p, v in values.items()}, seed=seed, n_boot=n_boot,
                                   ci_level=ci_level)
    summary["n_patients_without_support"] = len(patients - set(values))
    return summary
