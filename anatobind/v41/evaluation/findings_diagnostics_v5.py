"""Diagnostics of the findings chain (2026-11-11): where each released primary box is lost on the way from the raw
abnormality response to the final finding. Diagnostic only — nothing here is an official metric.

One row per scan. For every primary box of the readers, in the order of the chain:

1. response    the restored probability of the box's class inside the box (the training window of box_data_v5) and
               on its reference host tissue (the bag the box objective trained on), and whether any voxel reaches
               the acceptance threshold;
2. components  the connected components of accepted positives of that class that touch the box;
3. size        whether one of them survives the acceptance policy's min_voxels;
4. anchor      whether a finding of the class has its anchor inside the box (the official hit);
5. host        whether the hit finding names the box's tissue group — with the PREDICTED anatomy (the official
               reading) and, for diagnosis only, with the REFERENCE anatomy on the same predicted instance.

`stage` names the first step a box fails. Every finding also gets its place (finding_location_v5), both host
readings and the share of its voxels that lie on a tissue its class may sit on. Components are counted before and
after the size filter. Official results keep using the predicted anatomy; the reference-anatomy reading only tells
whether a lost host is lost by the anatomy prediction or by where the abnormality response lies.

Rows carry a patient id; summaries and the report carry counts and distributions only.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..contracts_v5 import ActiveScope, V5Error
from ..engine.box_data_v5 import _is_primary, _window
from ..findings_v5 import _group_ids, host_of_relation_v5, host_relation_fn_v5
from ..perception import CONNECTIVITY
from .findings_v5 import LOCATIONS, _inside, finding_location_v5

STAGES = ("no_response", "below_min_voxels", "anchor_outside", "hit_host_wrong", "hit_host_correct")
SIZE_BINS = ((1, "1"), (2, "2-9"), (10, "10-49"), (50, "50-99"), (100, "100-999"), (1000, "1000-9999"),
             (10000, ">=10000"))
RESPONSE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
HOST_PAIRS = ("bound_bound", "bound_undetermined", "undetermined_bound", "undetermined_undetermined")
_FUNNEL_KEYS = ("responded", "kept_touching", "hit", "host_correct", "host_correct_reference_anatomy")


def _histogram(sizes: np.ndarray) -> dict:
    edges = np.array([edge for edge, _ in SIZE_BINS])
    counts = np.bincount(np.searchsorted(edges, sizes, side="right") - 1, minlength=len(SIZE_BINS)) \
        if len(sizes) else np.zeros(len(SIZE_BINS), dtype=np.int64)
    return {label: int(counts[index]) for index, (_, label) in enumerate(SIZE_BINS)}


def _anchor_distance(anchor: list, box: dict) -> float:
    """Euclidean distance in voxels from the anchor to the nearest voxel of the box; 0 inside."""
    gaps = [max(start - p, p - (start + size - 1), 0.0) for p, start, size in zip(anchor, box["start"], box["size"])]
    return float(math.sqrt(sum(gap * gap for gap in gaps)))


def diagnose_scan_v5(*, findings: list, instances: list, reference: dict, scope: ActiveScope,
                     probability: np.ndarray, positive: np.ndarray, accepted: np.ndarray,
                     anatomy_predicted: np.ndarray, anatomy_accepted: np.ndarray, anatomy_reference: np.ndarray,
                     anatomy_eligible: np.ndarray, groups: dict, groups_by_class: dict, acceptance_policy: dict,
                     topk_fraction: float, patient_uid: str) -> dict:
    """One scan's diagnostic row; `findings` and `instances` are the scan's official findings and their instances."""
    classes = list(scope.abnormality_classes)
    shape = tuple(int(n) for n in np.shape(anatomy_reference))
    positive, accepted, probability = np.asarray(positive), np.asarray(accepted), np.asarray(probability)
    if positive.dtype != bool or accepted.dtype != bool or positive.shape != (len(classes), *shape) \
            or accepted.shape != positive.shape or probability.shape != positive.shape \
            or any(tuple(np.shape(a)) != shape for a in (anatomy_predicted, anatomy_accepted, anatomy_eligible)):
        raise V5Error("GRID_MISMATCH", "$.positive")
    if not (isinstance(topk_fraction, (int, float)) and 0 < topk_fraction <= 1):
        raise V5Error("INVALID_LOSS_WEIGHTS", "$.topk_fraction")
    min_voxels = acceptance_policy["instances"]["min_voxels"]
    structure = ndimage.generate_binary_structure(3, CONNECTIVITY[acceptance_policy["instances"]["connectivity"]])
    ids = _group_ids(scope, groups)
    labels, sizes = [], []
    for c in range(len(classes)):
        labelled, count = ndimage.label(positive[c] & accepted[c], structure=structure)
        labels.append(labelled)
        sizes.append(np.bincount(labelled.ravel(), minlength=count + 1))

    # every official finding is one kept component: its anchor is a voxel of that component
    by_uid = {instance["instance_uid"]: instance for instance in instances}
    component = {}
    for finding in findings:
        c = classes.index(finding["abnormality_class"])
        label = int(labels[c][tuple(finding["anchor"])])
        if finding["finding_uid"] not in by_uid or label == 0 or int(sizes[c][label]) != finding["voxels"]:
            raise V5Error("DIAGNOSTIC_INSTANCE_MISMATCH", "$.findings")
        component[finding["finding_uid"]] = label
    for c, name in enumerate(classes):
        kept = int((sizes[c][1:] >= min_voxels).sum())
        if kept != sum(1 for finding in findings if finding["abnormality_class"] == name):
            raise V5Error("DIAGNOSTIC_INSTANCE_MISMATCH", "$.instances")

    reference_relations, _ = host_relation_fn_v5(groups, groups_by_class)(
        instances, np.asarray(anatomy_reference), np.asarray(anatomy_eligible, dtype=bool), scope)
    reference_host = {relation["instance_uid"]: host_of_relation_v5(relation, ids) for relation in reference_relations}

    finding_rows = []
    for finding in findings:
        c = classes.index(finding["abnormality_class"])
        instance = by_uid[finding["finding_uid"]]
        ndim = len(instance["bbox"]) // 2
        region = tuple(slice(a, b) for a, b in zip(instance["bbox"][:ndim], instance["bbox"][ndim:]))
        mask = np.asarray(instance["mask"], dtype=bool)[region]
        allowed = sorted({i for group in groups_by_class.get(finding["abnormality_class"], []) for i in ids[group]})
        on_predicted = np.isin(np.asarray(anatomy_predicted)[region], allowed) & np.asarray(anatomy_accepted)[region]
        on_reference = np.isin(np.asarray(anatomy_reference)[region], allowed) & np.asarray(anatomy_eligible)[region]
        values = probability[c][region][mask]
        group, status = reference_host[finding["finding_uid"]]
        finding_rows.append({
            "abnormality_class": finding["abnormality_class"], "voxels": finding["voxels"],
            **finding_location_v5(finding, reference),
            "host_group": finding["host_group"], "host_status": finding["host_status"],
            "host_group_reference_anatomy": group, "host_status_reference_anatomy": status,
            "on_tissue_fraction_predicted": float((on_predicted & mask).sum()) / finding["voxels"],
            "on_tissue_fraction_reference": float((on_reference & mask).sum()) / finding["voxels"],
            "probability_mean": float(values.mean()), "probability_max": float(values.max())})

    box_rows = []
    for box in reference["boxes"]:
        if not _is_primary(box):
            continue
        name, group = box["abnormality_class"], box["host"]["tissue_group"]
        c = classes.index(name)
        window = _window(box["box"], shape)
        entry = {"abnormality_class": name, "tissue_group": group, "box_voxels": 0, "bag_voxels": 0,
                 "probability": {"max_box": None, "max_bag": None, "topk_mean_bag": None,
                                 "positive_fraction_box": None, "positive_fraction_bag": None},
                 "positive_voxels_box": 0, "components_touching": 0, "largest_touching_voxels": 0,
                 "kept_touching": 0, "hit": False, "host_correct": False, "host_correct_reference_anatomy": False,
                 "chosen": None, "largest_kept_touching": None, "stage": "no_response"}
        touching = np.zeros(0, dtype=np.int64)
        if window is not None:
            values, inside = probability[c][window], labels[c][window]
            bag = np.isin(np.asarray(anatomy_reference)[window], ids.get(group, []))
            marked = inside > 0
            touching = np.unique(inside[marked])
            entry.update(box_voxels=int(values.size), bag_voxels=int(bag.sum()),
                         positive_voxels_box=int(marked.sum()), components_touching=int(len(touching)),
                         largest_touching_voxels=int(sizes[c][touching].max()) if len(touching) else 0,
                         kept_touching=int((sizes[c][touching] >= min_voxels).sum()))
            entry["probability"].update(max_box=float(values.max()),
                                        positive_fraction_box=float(marked.sum()) / values.size)
            if bag.any():
                in_bag = np.sort(values[bag])
                top = in_bag[-max(1, math.ceil(topk_fraction * in_bag.size)):]
                entry["probability"].update(max_bag=float(in_bag[-1]), topk_mean_bag=float(top.mean()),
                                            positive_fraction_bag=float((marked & bag).sum()) / in_bag.size)
        candidates = [finding for finding in findings
                      if finding["abnormality_class"] == name and _inside(finding["anchor"], box["box"])]
        if candidates:                                        # score_findings_v5's choice: the largest candidate
            chosen = min(candidates, key=lambda finding: (-finding["voxels"], finding["finding_uid"]))
            inside_voxels = int((labels[c][window] == component[chosen["finding_uid"]]).sum())
            reference_group = reference_host[chosen["finding_uid"]][0]
            entry.update(hit=True, host_correct=chosen["host_group"] == group,
                         host_correct_reference_anatomy=reference_group == group,
                         chosen={"voxels": chosen["voxels"], "inside_fraction": inside_voxels / chosen["voxels"],
                                 "host_group": chosen["host_group"],
                                 "host_group_reference_anatomy": reference_group})
            entry["stage"] = "hit_host_correct" if entry["host_correct"] else "hit_host_wrong"
        elif entry["kept_touching"]:
            kept = touching[sizes[c][touching] >= min_voxels]
            largest = int(kept[np.argmax(sizes[c][kept])])
            anchor = next(finding["anchor"] for finding in findings
                          if finding["abnormality_class"] == name and component[finding["finding_uid"]] == largest)
            entry["largest_kept_touching"] = {
                "voxels": int(sizes[c][largest]),
                "inside_fraction": float((labels[c][window] == largest).sum()) / float(sizes[c][largest]),
                "anchor_distance_voxels": _anchor_distance(anchor, box["box"])}
            entry["stage"] = "anchor_outside"
        elif entry["positive_voxels_box"]:
            entry["stage"] = "below_min_voxels"
        box_rows.append(entry)

    components = {name: {"positive_voxels": int(sizes[c][1:].sum()), "count": int(len(sizes[c]) - 1),
                         "kept": int((sizes[c][1:] >= min_voxels).sum()), "size_histogram": _histogram(sizes[c][1:])}
                  for c, name in enumerate(classes)}
    return {"patient_uid": patient_uid, "classes": classes, "boxes": box_rows, "findings": finding_rows,
            "components": components}


# ---------------------------------------------------------------------------
# Summaries


def _distribution(values: list) -> dict:
    """Count, mean and quantiles of the values that are present; all None without any."""
    present = np.array([float(value) for value in values if value is not None], dtype=np.float64)
    keys = ("min", "p10", "p25", "p50", "p75", "p90", "max")
    if not len(present):
        return {"n": 0, "mean": None, **{key: None for key in keys}}
    quantiles = np.quantile(present, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"n": int(len(present)), "mean": float(present.mean()),
            **{key: float(value) for key, value in zip(keys, quantiles)}}


def _per_class(rows: list, key: str, build) -> dict:
    classes = list(rows[0]["classes"]) if rows else []
    items = [item for row in rows for item in row[key]]
    return {"overall": build(items),
            "by_class": {name: build([item for item in items if item["abnormality_class"] == name])
                         for name in classes}}


def _funnel(boxes: list) -> dict:
    flags = {"responded": lambda box: box["positive_voxels_box"] > 0,
             "kept_touching": lambda box: box["kept_touching"] > 0}
    return {"n_boxes": len(boxes),
            **{key: sum(1 for box in boxes if (flags[key](box) if key in flags else box[key])) for key in _FUNNEL_KEYS},
            "stages": {stage: sum(1 for box in boxes if box["stage"] == stage) for stage in STAGES}}


def _response(boxes: list) -> dict:
    missed = [box for box in boxes if not box["hit"]]
    return {**{key: _distribution([box["probability"][key] for box in boxes])
               for key in ("max_box", "max_bag", "topk_mean_bag", "positive_fraction_bag")},
            "max_box_of_missed": _distribution([box["probability"]["max_box"] for box in missed]),
            "boxes_with_max_at_least": {format(level, "g"): sum(
                1 for box in boxes if box["probability"]["max_box"] is not None
                and box["probability"]["max_box"] >= level) for level in RESPONSE_LEVELS}}


def _supervision(boxes: list) -> dict:
    return {key: _distribution([box[key] for box in boxes]) for key in ("box_voxels", "bag_voxels")}


def _chain(boxes: list) -> dict:
    """What the instances that reach a box look like: the hit one, or the largest kept one whose anchor missed."""
    chosen = [box["chosen"] for box in boxes if box["chosen"] is not None]
    missed = [box["largest_kept_touching"] for box in boxes if box["largest_kept_touching"] is not None]
    return {"hit": {key: _distribution([item[key] for item in chosen]) for key in ("voxels", "inside_fraction")},
            "anchor_outside": {key: _distribution([item[key] for item in missed])
                               for key in ("voxels", "inside_fraction", "anchor_distance_voxels")},
            "largest_touching_voxels_below_min": _distribution(
                [box["largest_touching_voxels"] for box in boxes if box["stage"] == "below_min_voxels"])}


def _locations(findings: list) -> dict:
    return {"total": len(findings),
            **{place: sum(1 for finding in findings if finding["location"] == place) for place in LOCATIONS},
            **{flag: sum(1 for finding in findings if finding[flag]) for flag in ("in_other_class_primary", "in_ignored")}}


def _hosts(findings: list) -> dict:
    def pair(finding: dict) -> str:
        return f"{finding['host_status']}_{finding['host_status_reference_anatomy']}"

    both = [finding for finding in findings if pair(finding) == "bound_bound"]
    return {"status": {key: sum(1 for finding in findings if pair(finding) == key) for key in HOST_PAIRS},
            "same_group_when_both_bound": sum(
                1 for finding in both if finding["host_group"] == finding["host_group_reference_anatomy"]),
            **{key: _distribution([finding[key] for finding in findings])
               for key in ("on_tissue_fraction_predicted", "on_tissue_fraction_reference")}}


def _instance_sizes(findings: list) -> dict:
    return {"voxels": _distribution([finding["voxels"] for finding in findings]),
            **{key: _distribution([finding[key] for finding in findings])
               for key in ("probability_mean", "probability_max")}}


def summarize_diagnostics_v5(rows: list) -> dict:
    """Counts and distributions over a split's rows; no patient, sample or finding identifier."""
    classes = list(rows[0]["classes"]) if rows else []
    components = {}
    for name in classes:
        items = [row["components"][name] for row in rows]
        components[name] = {
            **{key: sum(item[key] for item in items) for key in ("positive_voxels", "count", "kept")},
            "kept_per_scan": _distribution([item["kept"] for item in items]),
            "size_histogram": {label: sum(item["size_histogram"][label] for item in items) for _, label in SIZE_BINS}}
    return {"n_scans": len(rows), "classes": classes,
            "funnel": _per_class(rows, "boxes", _funnel), "response": _per_class(rows, "boxes", _response),
            "supervision_sizes": _per_class(rows, "boxes", _supervision), "chain": _per_class(rows, "boxes", _chain),
            "locations": _per_class(rows, "findings", _locations), "hosts": _per_class(rows, "findings", _hosts),
            "instance_sizes": _per_class(rows, "findings", _instance_sizes), "components": components}


# ---------------------------------------------------------------------------
# Report


def _cell(value) -> str:
    if value is None:
        return "NA"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def render_diagnostics_md_v5(by_role: dict) -> str:
    """Counts and distributions of each split side by side: no patient, sample or finding identifiers."""
    roles = list(by_role)
    classes = list(next(iter(by_role.values()))["classes"]) if roles else []

    def table(header: str, rows: list) -> list:
        lines = [f"| {header} | " + " | ".join(roles) + " |", "| --- |" + " --- |" * len(roles)]
        for label, getter in rows:
            lines.append(f"| {label} | " + " | ".join(_cell(getter(by_role[role])) for role in roles) + " |")
        return lines

    def scopes():
        yield "合计", "overall", None
        for name in classes:
            yield name, "by_class", name

    def pick(summary: dict, section: str, where: str, name: str | None) -> dict:
        return summary[section]["overall"] if name is None else summary[section][where][name]

    lines = ["# 所见链路诊断（对照医生的主分析框）", "",
             "本报告只含计数与分布；不含样本或患者 ID、路径或文件名。全部是对当前检查点的诊断，不是正式指标；"
             "train 上的数值只用于诊断，不能代替泛化评价。", "",
             "- 每个主分析框按链路依次判断：框内响应 → 连通块 → 体素数过滤 → 定位点 → 所在组织。",
             "- “参照解剖”一行只作诊断：预测的异常实例不变，只把判断所在组织时用的解剖图换成参照解剖。"
             "正式结果仍然用预测解剖。",
             "- 所见位置按定位点归入四类，顺序固定：同类主分析框 → 忽略区 → 异类主分析框 → 所有参照框之外。"
             "未匹配的所见不是误报：医生没有标注不等于正常。", ""]
    for title, where, name in scopes():
        lines += [f"## 链路漏斗：{title}", ""]
        lines += table("项", [
            ("主分析框", lambda s: pick(s, "funnel", where, name)["n_boxes"]),
            ("框内有达到阈值的响应", lambda s: pick(s, "funnel", where, name)["responded"]),
            ("有通过体素数过滤的连通块触及框", lambda s: pick(s, "funnel", where, name)["kept_touching"]),
            ("定位点落在框内（命中）", lambda s: pick(s, "funnel", where, name)["hit"]),
            ("命中且所在组织正确（预测解剖，正式口径）", lambda s: pick(s, "funnel", where, name)["host_correct"]),
            ("命中且所在组织正确（参照解剖，仅诊断）",
             lambda s: pick(s, "funnel", where, name)["host_correct_reference_anatomy"])])
        lines += ["", "失败点（每个框只计一次，取链路上最先失败的一步）：", ""]
        labels = {"no_response": "框内没有达到阈值的响应", "below_min_voxels": "有响应，但连通块都小于体素数下限",
                  "anchor_outside": "有保留的连通块触及框，但定位点在框外", "hit_host_wrong": "命中，所在组织不对",
                  "hit_host_correct": "命中，所在组织正确"}
        lines += table("失败点", [(labels[stage], lambda s, stage=stage: pick(s, "funnel", where, name)["stages"][stage])
                                 for stage in STAGES])
        lines += ["", "框内响应（该类别的概率，恢复到原始网格后）：", ""]
        lines += table("项", [
            *[(f"框内最大概率 ≥ {format(level, 'g')} 的框",
               lambda s, level=level: pick(s, "response", where, name)["boxes_with_max_at_least"][format(level, "g")])
              for level in RESPONSE_LEVELS],
            ("框内最大概率：中位数", lambda s: pick(s, "response", where, name)["max_box"]["p50"]),
            ("宿主组织上最大概率：中位数", lambda s: pick(s, "response", where, name)["max_bag"]["p50"]),
            ("宿主组织上前 k 比例体素的平均概率：中位数",
             lambda s: pick(s, "response", where, name)["topk_mean_bag"]["p50"]),
            ("宿主组织上达到阈值的体素比例：中位数",
             lambda s: pick(s, "response", where, name)["positive_fraction_bag"]["p50"]),
            ("未命中的框的框内最大概率：中位数", lambda s: pick(s, "response", where, name)["max_box_of_missed"]["p50"])])
        lines += ["", "监督区域与预测实例的大小（体素）：", ""]
        lines += table("项", [
            *[(f"主分析框体素数：{label}", lambda s, key=key: pick(s, "supervision_sizes", where, name)["box_voxels"][key])
              for key, label in (("min", "最小"), ("p50", "中位数"), ("max", "最大"))],
            *[(f"框内宿主组织体素数：{label}",
               lambda s, key=key: pick(s, "supervision_sizes", where, name)["bag_voxels"][key])
              for key, label in (("min", "最小"), ("p50", "中位数"), ("max", "最大"))],
            ("所见条数", lambda s: pick(s, "instance_sizes", where, name)["voxels"]["n"]),
            *[(f"所见体素数：{label}", lambda s, key=key: pick(s, "instance_sizes", where, name)["voxels"][key])
              for key, label in (("min", "最小"), ("p50", "中位数"), ("p90", "第 90 百分位"), ("max", "最大"))],
            ("命中的所见落在框内的体素比例：中位数",
             lambda s: pick(s, "chain", where, name)["hit"]["inside_fraction"]["p50"]),
            ("定位点在框外的最大连通块：体素数中位数",
             lambda s: pick(s, "chain", where, name)["anchor_outside"]["voxels"]["p50"]),
            ("定位点在框外的最大连通块：落在框内的体素比例中位数",
             lambda s: pick(s, "chain", where, name)["anchor_outside"]["inside_fraction"]["p50"]),
            ("定位点在框外的最大连通块：定位点到框的距离中位数（体素）",
             lambda s: pick(s, "chain", where, name)["anchor_outside"]["anchor_distance_voxels"]["p50"])])
        lines += ["", "所见的位置（按定位点，四类互斥）：", ""]
        lines += table("项", [
            ("所见总数", lambda s: pick(s, "locations", where, name)["total"]),
            ("定位点在同类主分析框内", lambda s: pick(s, "locations", where, name)["same_class_primary"]),
            ("定位点在忽略区内", lambda s: pick(s, "locations", where, name)["ignored"]),
            ("定位点在异类主分析框内", lambda s: pick(s, "locations", where, name)["other_class_primary"]),
            ("定位点在所有参照框之外", lambda s: pick(s, "locations", where, name)["outside_all"]),
            ("（不分先后）定位点落在某个异类主分析框内",
             lambda s: pick(s, "locations", where, name)["in_other_class_primary"]),
            ("（不分先后）定位点落在某个忽略区内", lambda s: pick(s, "locations", where, name)["in_ignored"])])
        lines += ["", "所在组织（全部所见；预测解剖为正式口径，参照解剖仅诊断）：", ""]
        lines += table("项", [
            ("两种解剖下都能确定", lambda s: pick(s, "hosts", where, name)["status"]["bound_bound"]),
            ("其中两种解剖给出同一组织组", lambda s: pick(s, "hosts", where, name)["same_group_when_both_bound"]),
            ("预测解剖能确定，参照解剖不能", lambda s: pick(s, "hosts", where, name)["status"]["bound_undetermined"]),
            ("预测解剖不能确定，参照解剖能", lambda s: pick(s, "hosts", where, name)["status"]["undetermined_bound"]),
            ("两种解剖下都不能确定", lambda s: pick(s, "hosts", where, name)["status"]["undetermined_undetermined"]),
            ("所见体素落在可归属组织上的比例（预测解剖）：中位数",
             lambda s: pick(s, "hosts", where, name)["on_tissue_fraction_predicted"]["p50"]),
            ("所见体素落在可归属组织上的比例（参照解剖）：中位数",
             lambda s: pick(s, "hosts", where, name)["on_tissue_fraction_reference"]["p50"])])
        lines.append("")
    for name in classes:
        lines += [f"## 连通块（体素数过滤之前与之后）：{name}", ""]
        lines += table("项", [
            ("扫描数", lambda s: s["n_scans"]),
            ("达到阈值的体素总数", lambda s: s["components"][name]["positive_voxels"]),
            ("连通块总数", lambda s: s["components"][name]["count"]),
            ("保留的连通块（即所见）", lambda s: s["components"][name]["kept"]),
            ("每次扫描保留的连通块：中位数", lambda s: s["components"][name]["kept_per_scan"]["p50"]),
            *[(f"体素数 {label} 的连通块", lambda s, label=label: s["components"][name]["size_histogram"][label])
              for _, label in SIZE_BINS]])
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
