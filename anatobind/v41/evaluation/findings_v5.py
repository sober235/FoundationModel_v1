"""Findings scored against the released boxes (Proposal_v5 §0.6, §5.0; adopted criterion S7; design §5.5).

The reference is the readers' primary boxes. A box is hit when a finding of its class has its anchor inside the
box; the host is correct when the hit finding (the largest candidate) names the box's tissue group. Every
denominator is the full set of primary boxes: a missed box counts against sensitivity, host accuracy and IoBB.

Findings that match no box of their class are counted, never called false positives: the release has no confirmed
negatives, so an unmarked place is not a normal place. Boxes the readers drew outside the primary analysis are
ignore regions. Rows carry a patient id for patient-equal summaries; summaries and reports carry counts and metrics
only.

The three counts of a row are decided by the anchor in this order: `in_reference` (a primary box of the finding's
class), else `in_ignored` (any ignore region), else `outside`. `outside` is therefore "the rest": it also holds a
finding whose anchor lies in a primary box of ANOTHER class, and is not "geometrically outside every box".
finding_location_v5 splits that rest in two without changing the three counts.
"""
from __future__ import annotations

import numpy as np

CONCLUSION = "NOT_EVALUATED"
OUTSIDE_NOTE = "描述性计数，不是误报"


def _is_primary(box: dict) -> bool:
    return box.get("status") == "included" and box.get("analysis") == "primary"


def _inside(point: list, box: dict) -> bool:
    return all(start <= p < start + size for p, start, size in zip(point, box["start"], box["size"]))


def _iobb(finding_box: dict, reference_box: dict) -> float:
    """|finding box ∩ reference box| / |finding box|."""
    intersection, volume = 1.0, 1.0
    for a0, da, b0, db in zip(finding_box["start"], finding_box["size"], reference_box["start"], reference_box["size"]):
        intersection *= max(0.0, min(a0 + da, b0 + db) - max(a0, b0))
        volume *= da
    return intersection / volume if volume > 0 else 0.0


def score_findings_v5(findings: list, reference: dict, patient_uid: str, classes: list,
                      iobb_thresholds: list) -> dict:
    """One scan's row: each primary box with hit / host_correct / iobb, where the findings fall, tissue positives."""
    primary = [box for box in reference["boxes"] if _is_primary(box)]
    ignored = [box for box in reference["boxes"] if not _is_primary(box)]
    boxes = []
    for box in primary:
        candidates = [f for f in findings if f["abnormality_class"] == box["abnormality_class"]
                      and _inside(f["anchor"], box["box"])]
        chosen = min(candidates, key=lambda f: (-f["voxels"], f["finding_uid"])) if candidates else None
        group = box["host"]["tissue_group"]
        boxes.append({"abnormality_class": box["abnormality_class"], "tissue_group": group, "hit": chosen is not None,
                      "host_correct": chosen is not None and chosen["host_group"] == group,
                      "iobb": max(_iobb(f["box"], box["box"]) for f in candidates) if candidates else None})
    counts = {"total": len(findings), "in_reference": 0, "in_ignored": 0, "outside": 0}
    for finding in findings:
        if any(box["abnormality_class"] == finding["abnormality_class"] and _inside(finding["anchor"], box["box"])
               for box in primary):
            counts["in_reference"] += 1
        elif any(_inside(finding["anchor"], box["box"]) for box in ignored):
            counts["in_ignored"] += 1
        else:
            counts["outside"] += 1
    pairs = sorted({(box["abnormality_class"], box["tissue_group"]) for box in boxes},
                   key=lambda pair: (classes.index(pair[0]), pair[1]))
    tissue_positive = [{"abnormality_class": cls, "tissue_group": group,
                        "predicted": any(f["abnormality_class"] == cls and f["host_group"] == group for f in findings)}
                       for cls, group in pairs]
    return {"patient_uid": patient_uid, "classes": list(classes), "iobb_thresholds": [float(t) for t in iobb_thresholds],
            "boxes": boxes, "findings": counts, "tissue_positive": tissue_positive}


LOCATIONS = ("same_class_primary", "ignored", "other_class_primary", "outside_all")


def finding_location_v5(finding: dict, reference: dict) -> dict:
    """Where the anchor of one finding lies: exactly one of LOCATIONS, decided in that order.

    same_class_primary = score_findings_v5's in_reference; ignored = its in_ignored; other_class_primary and
    outside_all together are its `outside`. An anchor inside both an ignore region and another class's primary box
    is `ignored`, as in the original count. The two flags tell the memberships whatever the precedence."""
    primary = [box for box in reference["boxes"] if _is_primary(box)]
    inside = [box for box in primary if _inside(finding["anchor"], box["box"])]
    same = any(box["abnormality_class"] == finding["abnormality_class"] for box in inside)
    other = any(box["abnormality_class"] != finding["abnormality_class"] for box in inside)
    ignored = any(_inside(finding["anchor"], box["box"]) for box in reference["boxes"] if not _is_primary(box))
    location = "same_class_primary" if same else "ignored" if ignored else "other_class_primary" if other \
        else "outside_all"
    return {"location": location, "in_other_class_primary": other, "in_ignored": ignored}


def _ratio(numerator: float, denominator: float):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _box_stats(rows: list, cls: str | None, thresholds: list) -> dict:
    by_patient: dict = {}
    for row in rows:
        selected = [box for box in row["boxes"] if cls is None or box["abnormality_class"] == cls]
        by_patient.setdefault(row["patient_uid"], []).extend(selected)
    everything = [box for boxes in by_patient.values() for box in boxes]
    with_boxes = [boxes for boxes in by_patient.values() if boxes]

    def rate(key: str) -> dict:
        per_patient = [sum(box[key] for box in boxes) / len(boxes) for boxes in with_boxes]
        return {"patient_mean": float(np.mean(per_patient)) if per_patient else None,
                "pooled": _ratio(sum(box[key] for box in everything), len(everything))}

    tissue = [item for row in rows for item in row["tissue_positive"] if cls is None or item["abnormality_class"] == cls]
    return {"n_patients": len(by_patient), "n_patients_with_boxes": len(with_boxes), "n_boxes": len(everything),
            "box_sensitivity": rate("hit"), "host_accuracy": rate("host_correct"),
            "iobb_at": {format(t, "g"): _ratio(sum(1 for box in everything if box["iobb"] is not None
                                                   and box["iobb"] >= t), len(everything)) for t in thresholds},
            "tissue_positive_recall": _ratio(sum(item["predicted"] for item in tissue), len(tissue))}


def summarize_findings_v5(rows: list) -> dict:
    """Patient-equal and pooled rates over all rows and per class; a rate without support is None."""
    classes = list(rows[0]["classes"]) if rows else []
    thresholds = list(rows[0]["iobb_thresholds"]) if rows else []
    overall = _box_stats(rows, None, thresholds)
    overall["n_scans"] = len(rows)
    overall["findings"] = {key: sum(row["findings"][key] for row in rows)
                           for key in ("total", "in_reference", "in_ignored", "outside")}
    overall["findings_outside_reference_per_scan"] = _ratio(overall["findings"]["outside"], len(rows))
    return {"scientific_conclusion": CONCLUSION, "overall": overall,
            "by_class": {cls: _box_stats(rows, cls, thresholds) for cls in classes}}


def _cell(value) -> str:
    return "NA" if value is None else f"{value:.3f}"


def render_findings_eval_md_v5(summary: dict) -> str:
    """Counts and metrics only: no patient, sample or finding identifiers."""
    overall = summary["overall"]
    lines = ["# 所见评价（对照医生的主分析框）", "",
             f"- scientific_conclusion: {summary['scientific_conclusion']}",
             f"- 患者 {overall['n_patients']}，扫描 {overall['n_scans']}，有主分析框的患者 "
             f"{overall['n_patients_with_boxes']}，主分析框 {overall['n_boxes']}",
             "- 分母固定为医生的全部主分析框；未命中的框计入每一项的分母。", "",
             "| 范围 | 框数 | 框命中率（患者等权） | 框命中率（合并） | 归属正确率（患者等权） | 归属正确率（合并） | 组织级阳性召回 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, stats in [("合计", overall), *summary["by_class"].items()]:
        lines.append(f"| {name} | {stats['n_boxes']} | {_cell(stats['box_sensitivity']['patient_mean'])} | "
                     f"{_cell(stats['box_sensitivity']['pooled'])} | {_cell(stats['host_accuracy']['patient_mean'])} | "
                     f"{_cell(stats['host_accuracy']['pooled'])} | {_cell(stats['tissue_positive_recall'])} |")
    lines += ["", "| IoBB 阈值 | 达标的主分析框比例 |", "| --- | --- |"]
    lines += [f"| {threshold} | {_cell(value)} |" for threshold, value in overall["iobb_at"].items()]
    counts = overall["findings"]
    lines += ["", f"- 所见总数 {counts['total']}：定位点在同类主分析框内 {counts['in_reference']}，"
                  f"在非主分析框（忽略区）内 {counts['in_ignored']}，"
                  f"其余 {counts['outside']}（含定位点落在异类主分析框内的所见）",
              f"- 其余所见，每次扫描平均 {_cell(overall['findings_outside_reference_per_scan'])}"
              f"（{OUTSIDE_NOTE}：医生没有标注不等于正常）"]
    return "\n".join(lines) + "\n"


def anatomy_dice_v5(anatomy_map: np.ndarray, truth: np.ndarray, eligible: np.ndarray, classes: list) -> dict:
    """Per-class Dice of a predicted label map inside `eligible` (segmentation_metrics_v5's rule: None when there
    is nothing evaluable or both sides are empty)."""
    eligible = np.asarray(eligible, dtype=bool)
    result = {}
    for k, name in enumerate(classes, start=1):
        pred, reference = (np.asarray(anatomy_map) == k) & eligible, (np.asarray(truth) == k) & eligible
        total = int(pred.sum()) + int(reference.sum())
        result[name] = None if total == 0 else 2.0 * float((pred & reference).sum()) / float(total)
    return result
