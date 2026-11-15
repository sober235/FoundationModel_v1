"""Aggregates over the private rows of the evaluate-only run. Prints counts and distributions only:
no patient, sample or finding identifier, no path, no time. Usage: rows_aggregate.py <rows dir> <member>"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

ROWS, MEMBER = sys.argv[1], sys.argv[2]
BINS = [(50, "50-99"), (100, "100-999"), (1000, "1000-9999"), (10000, ">=10000")]
PLACES = ("same_class_primary", "ignored", "other_class_primary", "outside_all")


def size_bin(v):
    label = None
    for edge, name in BINS:
        if v >= edge:
            label = name
    return label


def dist(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    out = {"n": int(len(a)), "mean": float(a.mean())}
    out.update({k: float(v) for k, v in zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), q)})
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


result = {}
for role in ("train", "development_val"):
    rows = [json.loads(line) for line in open(f"{ROWS}/{role}.{MEMBER}.jsonl") if line.strip()]
    classes = rows[0]["diagnostic"]["classes"]
    out = {"n_scans": len(rows)}
    for name in classes:
        findings = [f for r in rows for f in r["diagnostic"]["findings"] if f["abnormality_class"] == name]
        boxes = [b for r in rows for b in r["diagnostic"]["boxes"] if b["abnormality_class"] == name]
        total_positive = sum(r["diagnostic"]["components"][name]["positive_voxels"] for r in rows)
        kept_voxels = sum(f["voxels"] for f in findings)
        block = {"positive_voxels_total": total_positive, "kept_finding_voxels_total": kept_voxels,
                 "kept_share_of_positive_voxels": round(kept_voxels / total_positive, 4) if total_positive else None,
                 "findings": len(findings), "boxes": len(boxes)}
        # A. findings by size bin
        by_bin = {}
        for _, label in BINS:
            items = [f for f in findings if size_bin(f["voxels"]) == label]
            vox = sum(f["voxels"] for f in items)
            by_bin[label] = {
                "count": len(items), "voxels": vox,
                "share_of_kept_voxels": round(vox / kept_voxels, 4) if kept_voxels else None,
                "places": {p: sum(1 for f in items if f["location"] == p) for p in PLACES},
                "host_predicted_bound": sum(1 for f in items if f["host_status"] == "bound"),
                "host_reference_bound": sum(1 for f in items if f["host_status_reference_anatomy"] == "bound"),
                "host_groups_predicted": dict(Counter(str(f["host_group"]) for f in items)),
                "on_tissue_predicted": dist([f["on_tissue_fraction_predicted"] for f in items]),
                "on_tissue_reference": dist([f["on_tissue_fraction_reference"] for f in items]),
                "on_tissue_predicted_zero": sum(1 for f in items if f["on_tissue_fraction_predicted"] == 0),
                "on_tissue_reference_zero": sum(1 for f in items if f["on_tissue_fraction_reference"] == 0),
                "voxel_weighted_on_tissue_predicted": round(
                    sum(f["voxels"] * f["on_tissue_fraction_predicted"] for f in items) / vox, 4) if vox else None,
                "voxel_weighted_on_tissue_reference": round(
                    sum(f["voxels"] * f["on_tissue_fraction_reference"] for f in items) / vox, 4) if vox else None,
                "probability_mean": dist([f["probability_mean"] for f in items])}
        block["findings_by_size"] = by_bin
        # B. voxel-weighted places and tissue share of all kept findings
        block["kept_voxels_by_place_of_anchor"] = {
            p: sum(f["voxels"] for f in findings if f["location"] == p) for p in PLACES}
        block["kept_voxels_on_tissue_predicted"] = round(
            sum(f["voxels"] * f["on_tissue_fraction_predicted"] for f in findings) / kept_voxels, 4) if kept_voxels else None
        block["kept_voxels_on_tissue_reference"] = round(
            sum(f["voxels"] * f["on_tissue_fraction_reference"] for f in findings) / kept_voxels, 4) if kept_voxels else None
        block["findings_by_place_on_tissue_reference_zero"] = {
            p: [sum(1 for f in findings if f["location"] == p and f["on_tissue_fraction_reference"] == 0),
                sum(1 for f in findings if f["location"] == p)] for p in PLACES}
        # C. boxes
        block["boxes_positive_voxels_sum"] = sum(b["positive_voxels_box"] for b in boxes)
        block["boxes_voxels_sum"] = sum(b["box_voxels"] for b in boxes)
        block["boxes_bag_voxels_sum"] = sum(b["bag_voxels"] for b in boxes)
        block["positive_fraction_box"] = dist([b["probability"]["positive_fraction_box"] for b in boxes])
        block["positive_fraction_bag"] = dist([b["probability"]["positive_fraction_bag"] for b in boxes])
        block["positive_in_box_on_bag_share"] = dist([
            (b["probability"]["positive_fraction_bag"] * b["bag_voxels"]) / b["positive_voxels_box"]
            for b in boxes if b["positive_voxels_box"] and b["probability"]["positive_fraction_bag"] is not None])
        block["components_touching"] = dist([b["components_touching"] for b in boxes])
        block["kept_touching"] = dist([b["kept_touching"] for b in boxes])
        block["largest_touching_voxels"] = dist([b["largest_touching_voxels"] for b in boxes])
        block["boxes_whose_largest_touching_is_at_least_10000"] = sum(1 for b in boxes if b["largest_touching_voxels"] >= 10000)
        block["boxes_whose_largest_touching_is_at_least_1000"] = sum(1 for b in boxes if b["largest_touching_voxels"] >= 1000)
        stages = Counter(b["stage"] for b in boxes)
        block["stages"] = dict(stages)
        if boxes:
            edges = np.quantile([b["box_voxels"] for b in boxes], [1 / 3, 2 / 3])
            tertile = lambda b: int(np.searchsorted(edges, b["box_voxels"], side="right"))
            block["hit_by_box_size_tertile"] = {
                f"t{t + 1}": [sum(1 for b in boxes if tertile(b) == t and b["hit"]), sum(1 for b in boxes if tertile(b) == t)]
                for t in range(3)}
            block["box_size_tertile_edges"] = [float(e) for e in edges]
            block["hit_chosen_voxels"] = dist([b["chosen"]["voxels"] for b in boxes if b["chosen"]])
            block["hit_chosen_at_least_10000"] = sum(1 for b in boxes if b["chosen"] and b["chosen"]["voxels"] >= 10000)
            block["hit_host_groups"] = dict(Counter(
                f'{b["tissue_group"]}->{b["chosen"]["host_group"]}' for b in boxes if b["chosen"]))
            block["box_tissue_groups"] = dict(Counter(b["tissue_group"] for b in boxes))
            block["hit_by_tissue_group"] = {g: [sum(1 for b in boxes if b["tissue_group"] == g and b["hit"]),
                                               sum(1 for b in boxes if b["tissue_group"] == g)]
                                           for g in sorted({b["tissue_group"] for b in boxes})}
        # D. scans with and without a primary box of the class
        with_box = [r for r in rows if any(b["abnormality_class"] == name for b in r["diagnostic"]["boxes"])]
        without = [r for r in rows if not any(b["abnormality_class"] == name for b in r["diagnostic"]["boxes"])]
        for label, group in (("scans_with_primary_box", with_box), ("scans_without_primary_box", without)):
            block[label] = {
                "n": len(group),
                "positive_voxels_per_scan": dist([r["diagnostic"]["components"][name]["positive_voxels"] for r in group]),
                "kept_per_scan": dist([r["diagnostic"]["components"][name]["kept"] for r in group]),
                "largest_finding_voxels_per_scan": dist([
                    max([f["voxels"] for f in r["diagnostic"]["findings"] if f["abnormality_class"] == name] or [0])
                    for r in group]),
                "scans_with_a_finding_of_at_least_10000": sum(
                    1 for r in group if any(f["abnormality_class"] == name and f["voxels"] >= 10000
                                            for f in r["diagnostic"]["findings"])),
                "scans_with_a_finding_of_at_least_1000": sum(
                    1 for r in group if any(f["abnormality_class"] == name and f["voxels"] >= 1000
                                            for f in r["diagnostic"]["findings"]))}
        out[name] = block
    # scans without any primary box at all
    none = [r for r in rows if not r["diagnostic"]["boxes"]]
    out["scans_without_any_primary_box"] = {
        "n": len(none), "findings_per_scan": dist([len(r["diagnostic"]["findings"]) for r in none])}
    out["scans_with_a_primary_box"] = {
        "n": len(rows) - len(none),
        "findings_per_scan": dist([len(r["diagnostic"]["findings"]) for r in rows if r["diagnostic"]["boxes"]])}
    out["reused_identical"] = [sum(1 for r in rows if r["reused"]), sum(1 for r in rows if r["identical"])]
    result[role] = out
json.dump(result, sys.stdout, ensure_ascii=False, indent=1, sort_keys=True)
