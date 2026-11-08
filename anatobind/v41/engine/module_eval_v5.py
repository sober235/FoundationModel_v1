"""v5.1 module baseline evaluation on development_val (plan_v5 T10; Proposal §1.1, §6.1, §8.3).

References are frozen before any prediction: each array is copied read-only and hashed with the patient, split
and geometry, and the set carries reference_set_hash; evaluation recomputes both and refuses a mismatch. Only
development_val references can be frozen. A is scored per tissue with T06a segmentation_metrics_v5 on
pred = (anatomy_map == k): a rejected voxel is no prediction and stays in the denominator; U the same way on
accepted positives, plus T06c lesion detection with instances from extract_instances_v5 on both sides under the
acceptance policy's connectivity (the reference keeps every component). Millimetre metrics are None unless the
physical geometry is verified. The summary needs one row per frozen observation (no partial baseline), averages
within each patient, then weights patients equally; it
describes development_val only (no intervals; scientific_conclusion NOT_EVALUATED) and never adds module
results together. Reports carry no observation or patient identifiers.
"""
from __future__ import annotations

import copy
import re

import numpy as np

from ..contracts_v5 import (
    GEOMETRY_KEYS, UNKNOWN_KEY, ActiveScope, ModuleOutput, V5Error, array_sha256, canonical_sha256)
from ..evaluation.instances import lesion_detection_metrics_v5, summarize_lesion_detection_v5
from ..evaluation.policy import validate_metric_policy_v5
from ..evaluation.segmentation import segmentation_metrics_v5
from ..perception import extract_instances_v5, validate_acceptance_policy_v5
from .smoke_v5 import _count, _number, _text

SPLIT = "development_val"
CONCLUSION = "NOT_EVALUATED"
_REFERENCE_KEYS = ("patient_uid", "split", "geometry", "tasks")
_FROZEN_KEYS = ("scope", "scope_hash", "entries", "entry_sha256", "reference_set_hash")
_ROW_SHARED = ("split", "scope_id", "scope_hash", "reference_set_hash", "reference_entries", "checkpoint_sha256",
               "acceptance_policy_id", "metric_policy_sha256", "metrics")
_SEGMENTATION_FIELDS = (("dice", "dice"), ("hd95", "hd95_mm"), ("nsd", "nsd"))
_DETECTION_FIELDS = ("pooled_sensitivity", "fp_per_scan")
_HEX64 = re.compile(r"[0-9a-f]{64}")


# ---------------------------------------------------------------------------
# Frozen development references


def _task_sizes(scope: ActiveScope) -> dict:
    sizes = {}
    if scope.anatomy_classes:
        sizes["A"] = len(scope.anatomy_classes)
    if scope.abnormality_classes:
        sizes["U"] = len(scope.abnormality_classes)
    return sizes


def _check_reference(reference: object, scope: ActiveScope, path: str) -> None:
    """truth: A integer tissue ids [*shape] in 0..K, U bool [C_U, *shape]; eligible: bool [classes, *shape]."""
    if not isinstance(reference, dict):
        raise V5Error("INVALID_TYPE", path)
    if any(key not in _REFERENCE_KEYS for key in reference):
        raise V5Error("UNKNOWN_FIELD", f"{path}.{UNKNOWN_KEY}")
    for key in _REFERENCE_KEYS:
        if key not in reference:
            raise V5Error("MISSING_FIELD", f"{path}.{key}")
    if not isinstance(reference["patient_uid"], str) or not reference["patient_uid"]:
        raise V5Error("INVALID_REFERENCE", f"{path}.patient_uid")
    if reference["split"] != SPLIT:
        raise V5Error("SPLIT_NOT_ALLOWED", f"{path}.split")
    geometry = reference["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != set(GEOMETRY_KEYS) \
            or not isinstance(geometry["shape"], (list, tuple)) \
            or not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in geometry["shape"]):
        raise V5Error("GRID_MISMATCH", f"{path}.geometry")
    canonical_sha256(geometry)  # JSON only, so it can be hashed and compared with an output's geometry
    shape, sizes, tasks = tuple(geometry["shape"]), _task_sizes(scope), reference["tasks"]
    if not isinstance(tasks, dict) or not tasks or any(task not in sizes for task in tasks):
        raise V5Error("INVALID_REFERENCE", f"{path}.tasks")
    for task, block in tasks.items():
        task_path, n = f"{path}.tasks.{task}", sizes[task]
        if not isinstance(block, dict) or set(block) != {"truth", "eligible"}:
            raise V5Error("INVALID_REFERENCE", task_path)
        truth, eligible = block["truth"], block["eligible"]
        if not isinstance(eligible, np.ndarray) or eligible.dtype != bool or eligible.shape != (n, *shape):
            raise V5Error("GRID_MISMATCH", f"{task_path}.eligible")
        if task == "A":
            valid = isinstance(truth, np.ndarray) and np.issubdtype(truth.dtype, np.integer) and truth.shape == shape \
                and (truth.size == 0 or (int(truth.min()) >= 0 and int(truth.max()) <= n))
        else:
            valid = isinstance(truth, np.ndarray) and truth.dtype == bool and truth.shape == (n, *shape)
        if not valid:
            raise V5Error("GRID_MISMATCH", f"{task_path}.truth")


def _read_only(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, copy=True)
    frozen.setflags(write=False)
    return frozen


def _entry_sha256(entry: dict) -> str:
    return canonical_sha256({"patient_uid": entry["patient_uid"], "split": entry["split"],
                             "geometry": entry["geometry"],
                             "tasks": {task: {key: array_sha256(block[key]) for key in ("truth", "eligible")}
                                       for task, block in entry["tasks"].items()}})


def _set_sha256(scope_hash: str, entry_hashes: dict) -> str:
    return canonical_sha256({"scope_hash": scope_hash, "entries": entry_hashes})


def freeze_module_reference_v5(references: dict, scope: ActiveScope) -> dict:
    """Freeze {observation_uid: {patient_uid, split, geometry, tasks: {A|U: {truth, eligible}}}} before prediction.

    Returns {scope, scope_hash, entries (read-only copies), entry_sha256, reference_set_hash}. Error paths use the
    position in sorted observation order, never the identifier.
    """
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    if not isinstance(references, dict) or not references:
        raise V5Error("REFERENCES_REQUIRED", "$.references")
    if not all(isinstance(uid, str) and uid for uid in references):
        raise V5Error("INVALID_TYPE", "$.references")
    entries, hashes = {}, {}
    for index, uid in enumerate(sorted(references)):
        reference = references[uid]
        _check_reference(reference, scope, f"$.references[{index}]")
        entries[uid] = {"patient_uid": reference["patient_uid"], "split": SPLIT,
                        "geometry": copy.deepcopy(reference["geometry"]),
                        "tasks": {task: {key: _read_only(block[key]) for key in ("truth", "eligible")}
                                  for task, block in reference["tasks"].items()}}
        hashes[uid] = _entry_sha256(entries[uid])
    return {"scope": scope, "scope_hash": scope.scope_hash, "entries": entries, "entry_sha256": hashes,
            "reference_set_hash": _set_sha256(scope.scope_hash, hashes)}


def _verified_entry(reference: dict, uid: str) -> dict:
    entry = reference["entries"].get(uid)
    if entry is None:
        raise V5Error("REFERENCE_MISSING", "$.observation_uid")
    try:
        intact = _entry_sha256(entry) == reference["entry_sha256"].get(uid)
    except (KeyError, TypeError, AttributeError, V5Error):
        intact = False
    if not intact:
        raise V5Error("REFERENCE_HASH_MISMATCH", "$.reference")
    if set(reference["entry_sha256"]) != set(reference["entries"]) \
            or _set_sha256(reference["scope_hash"], reference["entry_sha256"]) != reference["reference_set_hash"]:
        raise V5Error("REFERENCE_HASH_MISMATCH", "$.reference_set_hash")
    return entry


# ---------------------------------------------------------------------------
# Per-observation metrics


def _lesion_detection(output: ModuleOutput, block: dict, scope: ActiveScope, metric_policy: dict,
                      acceptance_policy: dict, spacing) -> dict:
    """Both sides cut to the evaluable region; the reference keeps every component (min_voxels 1)."""
    eligible = block["eligible"]
    predicted = extract_instances_v5(output.abnormality_map & eligible, output.acceptance["abnormality"], scope,
                                     acceptance_policy)
    truth_rule = copy.deepcopy(acceptance_policy)
    truth_rule["instances"]["min_voxels"] = 1
    truth = extract_instances_v5(block["truth"] & eligible, np.ones(eligible.shape, dtype=bool), scope, truth_rule)
    voxel_volume = None if spacing is None else float(np.prod(spacing))
    return lesion_detection_metrics_v5(predicted, truth, metric_policy, voxel_volume)


def module_metrics_v5(output: ModuleOutput, reference: dict, metric_policy: dict, acceptance_policy: dict) -> dict:
    """Metrics of one development_val observation against the frozen reference set.

    The acceptance policy must be the engineering policy the output was produced under (B2); its connectivity
    builds the instances. metric_policy is the registered development policy; lesion detection also needs its
    class_aware and size_strata_voxels.
    """
    if not isinstance(output, ModuleOutput):
        raise V5Error("INVALID_TYPE", "$.output")
    if not isinstance(reference, dict) or set(reference) != set(_FROZEN_KEYS) \
            or not isinstance(reference["scope"], ActiveScope):
        raise V5Error("INVALID_TYPE", "$.reference")
    scope = reference["scope"]
    if reference["scope_hash"] != scope.scope_hash or output.scope_hash != scope.scope_hash:
        raise V5Error("SCOPE_MISMATCH", "$.scope_hash")
    entry = _verified_entry(reference, output.observation_uid)
    validate_acceptance_policy_v5(acceptance_policy, scope)
    if acceptance_policy["policy_role"] != "engineering":
        raise V5Error("POLICY_ROLE_NOT_ALLOWED", "$.acceptance_policy.policy_role")
    if acceptance_policy["policy_id"] != output.provenance.get("output_policy_version"):
        raise V5Error("POLICY_MISMATCH", "$.acceptance_policy.policy_id")
    if canonical_sha256(output.geometry) != canonical_sha256(entry["geometry"]):
        raise V5Error("GRID_MISMATCH", "$.geometry")
    for task in output.enabled_tasks:
        if task not in entry["tasks"]:
            raise V5Error("REFERENCE_TASK_MISSING", f"$.reference.tasks.{task}")
    if not isinstance(metric_policy, dict) or metric_policy.get("metrics") is None:
        raise V5Error("METRIC_POLICY_NULL", "$.metrics")
    checked = validate_metric_policy_v5(metric_policy, "development")

    geometry = entry["geometry"]
    verified = geometry["units_reliable"] is True
    physical = {"physical_geometry_status": "verified" if verified else "unverified",
                "spacing_mm": geometry["spacing_mm"]}
    segmentation, detection, detection_reasons = {}, None, []
    if "A" in output.enabled_tasks:
        block, accepted = entry["tasks"]["A"], output.acceptance["anatomy"]
        segmentation["A"] = {name: segmentation_metrics_v5((output.anatomy_map == k) & accepted, block["truth"] == k,
                                                           block["eligible"][k - 1], physical, metric_policy)
                             for k, name in enumerate(scope.anatomy_classes, start=1)}
    if "U" in output.enabled_tasks:
        block = entry["tasks"]["U"]
        predicted = output.abnormality_map & output.acceptance["abnormality"]
        segmentation["U"] = {name: segmentation_metrics_v5(predicted[c], block["truth"][c], block["eligible"][c],
                                                           physical, metric_policy)
                             for c, name in enumerate(scope.abnormality_classes)}
        if "lesion_detection" in checked["ok_metrics"]:
            detection = _lesion_detection(output, block, scope, metric_policy, acceptance_policy,
                                          geometry["spacing_mm"] if verified else None)
        else:
            detection_reasons.append("POLICY_VALUE_UNSET" if "lesion_detection" in checked["blocked_metrics"]
                                     else "NOT_REQUESTED")
    return {"observation_uid": output.observation_uid, "patient_uid": entry["patient_uid"], "split": SPLIT,
            "scope_id": scope.scope_id, "scope_hash": scope.scope_hash,
            "reference_set_hash": reference["reference_set_hash"], "reference_entries": len(reference["entries"]),
            "checkpoint_sha256": output.checkpoint_sha256, "acceptance_policy_id": acceptance_policy["policy_id"],
            "metric_policy_sha256": canonical_sha256(metric_policy), "metrics": list(metric_policy["metrics"]),
            "segmentation": segmentation, "detection": detection, "detection_reasons": detection_reasons}


# ---------------------------------------------------------------------------
# Patient-weighted summary


def _stat(patient_values: list, n_scans: int, n_none: int) -> dict:
    present = [value for value in patient_values if value is not None]
    return {"mean": sum(present) / len(present) if present else None, "n_patients": len(present),
            "n_scans": n_scans, "n_none": n_none}


def _segmentation_stat(groups: list) -> dict:
    """groups: per patient, the scans' values (float or None); mean within the patient, then patients equally."""
    patient_values, n_scans, n_none = [], 0, 0
    for values in groups:
        present = [value for value in values if value is not None]
        n_scans += len(present)
        n_none += len(values) - len(present)
        patient_values.append(sum(present) / len(present) if present else None)
    return _stat(patient_values, n_scans, n_none)


def _detection_stats(groups: list) -> dict:
    """groups: per patient, the scans' detection results (dict or None); each patient pooled by T06c first."""
    values = {key: [] for key in _DETECTION_FIELDS}
    counts = {key: [0, 0] for key in _DETECTION_FIELDS}
    for results in groups:
        found = [result for result in results if result is not None]
        with_truth = sum(1 for result in found if result["tp"] + result["fn"] > 0)
        counts["pooled_sensitivity"][0] += with_truth
        counts["pooled_sensitivity"][1] += len(results) - with_truth
        counts["fp_per_scan"][0] += len(found)
        counts["fp_per_scan"][1] += len(results) - len(found)
        pooled = summarize_lesion_detection_v5(found) if found else {key: None for key in _DETECTION_FIELDS}
        for key in _DETECTION_FIELDS:
            values[key].append(pooled[key])
    return {key: _stat(values[key], *counts[key]) for key in _DETECTION_FIELDS}


def summarize_module_metrics_v5(rows: list) -> dict:
    """Patient-weighted development_val description of one model, scope and frozen reference set; no intervals."""
    if not isinstance(rows, list) or not rows:
        raise V5Error("ROWS_REQUIRED", "$.rows")
    first, seen = rows[0], set()
    for index, row in enumerate(rows):
        path = f"$.rows[{index}]"
        if not isinstance(row, dict):
            raise V5Error("INVALID_TYPE", path)
        for key in _ROW_SHARED:
            if row.get(key) != first.get(key):
                raise V5Error("MIXED_ROWS", f"{path}.{key}")
        if not isinstance(row.get("segmentation"), dict) or set(row["segmentation"]) != set(first["segmentation"]):
            raise V5Error("MIXED_ROWS", f"{path}.segmentation")
        if row.get("observation_uid") in seen:
            raise V5Error("DUPLICATE_OBSERVATION", path)
        seen.add(row["observation_uid"])
    if len(rows) != first["reference_entries"]:   # every frozen observation is scored: no partial baseline
        raise V5Error("MISSING_OBSERVATION", "$.rows")
    if first["split"] != SPLIT:
        raise V5Error("SPLIT_NOT_ALLOWED", "$.rows[0].split")
    patients = {}
    for row in rows:
        patients.setdefault(row["patient_uid"], []).append(row)
    groups = [patients[uid] for uid in sorted(patients)]
    fields = [field for name, field in _SEGMENTATION_FIELDS if name in first["metrics"]]
    segmentation = {task: {name: {field: _segmentation_stat([[row["segmentation"][task][name][field] for row in group]
                                                             for group in groups])
                                  for field in fields}
                           for name in first["segmentation"][task]}
                    for task in first["segmentation"]}
    detection = None
    if "U" in first["segmentation"] and "lesion_detection" in first["metrics"]:
        detection = _detection_stats([[row["detection"] for row in group] for group in groups])
    return {"scope_id": first["scope_id"], "scope_hash": first["scope_hash"], "split": SPLIT,
            "scientific_conclusion": CONCLUSION, "aggregation": "patient_equal_weight",
            "reference_set_hash": first["reference_set_hash"], "checkpoint_sha256": first["checkpoint_sha256"],
            "acceptance_policy_id": first["acceptance_policy_id"],
            "metric_policy_sha256": first["metric_policy_sha256"], "n_patients": len(groups), "n_scans": len(rows),
            "segmentation": segmentation, "detection": detection}


# ---------------------------------------------------------------------------
# Redacted report


def _stat_cells(stat: object, path: str) -> str:
    if not isinstance(stat, dict):
        raise V5Error("INVALID_SUMMARY", path)
    return " | ".join([_number(stat.get("mean"), f"{path}.mean", 3), _count(stat.get("n_patients"), f"{path}.n_patients"),
                       _count(stat.get("n_scans"), f"{path}.n_scans"), _count(stat.get("n_none"), f"{path}.n_none")])


def render_module_baseline_md_v5(summary: dict) -> str:
    """Markdown of a summarize_module_metrics_v5 summary; reads only the fields it shows (no identifiers)."""
    if not isinstance(summary, dict):
        raise V5Error("INVALID_SUMMARY", "$")
    if summary.get("split") != SPLIT:
        raise V5Error("INVALID_SUMMARY", "$.split")
    if summary.get("scientific_conclusion") != CONCLUSION:
        raise V5Error("INVALID_SUMMARY", "$.scientific_conclusion")
    for key in ("checkpoint_sha256", "reference_set_hash", "metric_policy_sha256"):
        if not isinstance(summary.get(key), str) or not _HEX64.fullmatch(summary[key]):
            raise V5Error("INVALID_SUMMARY", f"$.{key}")
    segmentation = summary.get("segmentation")
    if not isinstance(segmentation, dict) or not segmentation or any(task not in ("A", "U") for task in segmentation):
        raise V5Error("INVALID_SUMMARY", "$.segmentation")
    lines = [f"# 模块基线：{_text(summary.get('scope_id'), '$.scope_id')}", "",
             f"- split: {SPLIT}",
             f"- scientific_conclusion: {CONCLUSION}",
             "- 只作开发集描述：逐例指标先在患者内平均，再患者等权；不计算区间（区间属于 T14 的 H1 比较）。",
             "- 各模块结果分开报告，不相加为同例联合成功（Proposal §1.1）。",
             f"- 患者数：{_count(summary.get('n_patients'), '$.n_patients')}；"
             f"扫描数：{_count(summary.get('n_scans'), '$.n_scans')}",
             f"- acceptance_policy_id: {_text(summary.get('acceptance_policy_id'), '$.acceptance_policy_id')}",
             f"- checkpoint_sha256: {summary['checkpoint_sha256']}",
             f"- reference_set_hash: {summary['reference_set_hash']}",
             f"- metric_policy_sha256: {summary['metric_policy_sha256']}"]
    titles = {"A": "A（解剖）分割", "U": "U（异常）分割"}
    for task in sorted(segmentation):
        path = f"$.segmentation.{task}"
        if not isinstance(segmentation[task], dict):
            raise V5Error("INVALID_SUMMARY", path)
        lines += ["", f"## {titles[task]}", "", "| 类别 | 指标 | 患者等权均值 | 有值患者数 | 有值扫描数 | None 扫描数 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, metrics in segmentation[task].items():
            if not isinstance(metrics, dict) or any(field not in dict(_SEGMENTATION_FIELDS).values() for field in metrics):
                raise V5Error("INVALID_SUMMARY", path)
            lines += [f"| {_text(name, path)} | {field} | {_stat_cells(stat, path)} |" for field, stat in metrics.items()]
    detection = summary.get("detection")
    if detection is not None:
        if not isinstance(detection, dict) or set(detection) != set(_DETECTION_FIELDS):
            raise V5Error("INVALID_SUMMARY", "$.detection")
        lines += ["", "## U 病灶检测（患者内按 T06c 汇总，再患者等权）", "",
                  "| 指标 | 患者等权均值 | 有值患者数 | 有值扫描数 | None 扫描数 |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {key} | {_stat_cells(detection[key], f'$.detection.{key}')} |" for key in _DETECTION_FIELDS]
    return "\n".join(lines) + "\n"
