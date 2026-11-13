"""Join the private rows of the evaluate-only run with the frozen box references, by the order the driver wrote
them (sorted sample uid inside a role). Prints counts and distributions only: no identifier, no path, no time.
Entries of locked_test are never used. Usage: rows_by_boxes.py <rows dir> <member> <split registry> <box references>"""
import json
import math
import sys
from collections import Counter

import numpy as np

ROWS, MEMBER, REGISTRY, REFERENCES = sys.argv[1:5]
ROLES = ("train", "development_val")


def dist(values):
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    out = {"n": int(len(a)), "mean": round(float(a.mean()), 1)}
    out.update({k: round(float(v), 1) for k, v in zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), q)})
    return out


def primary(box):
    return box.get("status") == "included" and box.get("analysis") == "primary"


def union_voxels(boxes):
    if not boxes:
        return 0
    lows = [[max(int(math.floor(s)), 0) for s in b["box"]["start"]] for b in boxes]
    highs = [[int(math.ceil(s + n)) for s, n in zip(b["box"]["start"], b["box"]["size"])] for b in boxes]
    shape = [max(h[d] for h in highs) for d in range(3)]
    if min(shape) <= 0:
        return 0
    mask = np.zeros(shape, dtype=bool)
    for low, high in zip(lows, highs):
        mask[tuple(slice(a, b) for a, b in zip(low, high))] = True
    return int(mask.sum())


assignment = json.load(open(REGISTRY))["assignment"]
wanted = {uid for uid, role in assignment.items() if role in ROLES}
references = {}
for line in open(REFERENCES, encoding="utf-8"):
    if line.strip():
        entry = json.loads(line)
        if entry.get("sample_uid") in wanted:            # everything else, locked_test included, is dropped
            references[entry["sample_uid"]] = entry

result = {}
for role in ROLES:
    rows = [json.loads(line)["diagnostic"] for line in open(f"{ROWS}/{role}.{MEMBER}.jsonl") if line.strip()]
    uids = sorted(uid for uid, r in assignment.items() if r == role)
    assert len(uids) == len(rows), "row count differs from the split"
    classes = rows[0]["classes"]
    scans = []
    for uid, row in zip(uids, rows):
        boxes = references.get(uid, {"boxes": []})["boxes"]
        mine = [b for b in boxes if primary(b)]
        assert len(mine) == len(row["boxes"]), "primary boxes differ: the join by order is wrong"
        assert Counter(b["abnormality_class"] for b in mine) == Counter(b["abnormality_class"] for b in row["boxes"])
        scans.append({"row": row, "primary": mine, "ignored": [b for b in boxes if not primary(b)]})
    out = {"n_scans": len(scans), "join_checked": True,
           "non_primary_boxes_total": sum(len(s["ignored"]) for s in scans),
           "non_primary_boxes_per_scan": dist(len(s["ignored"]) for s in scans),
           "non_primary_by_status_analysis": dict(Counter(
               f'{b.get("status")}/{b.get("analysis")}' for s in scans for b in s["ignored"])),
           "non_primary_by_class": dict(Counter(str(b.get("abnormality_class")) for s in scans for b in s["ignored"])),
           "ignore_union_voxels_per_scan": dist(union_voxels(s["ignored"]) for s in scans),
           "ignore_union_voxels_total": sum(union_voxels(s["ignored"]) for s in scans),
           "primary_union_voxels_total": {name: sum(union_voxels([b for b in s["primary"] if b["abnormality_class"] == name])
                                                    for s in scans) for name in classes}}

    def describe(group):
        block = {"n": len(group), "findings_per_scan": dist(len(s["row"]["findings"]) for s in group)}
        for name, giant in zip(classes, (10000, 1000)):
            largest = [max([f["voxels"] for f in s["row"]["findings"] if f["abnormality_class"] == name] or [0])
                       for s in group]
            block[name] = {
                "positive_voxels_per_scan": dist(s["row"]["components"][name]["positive_voxels"] for s in group),
                "largest_finding_voxels_per_scan": dist(largest),
                f"scans_with_a_finding_of_at_least_{giant}": sum(1 for v in largest if v >= giant)}
        return block

    out["scans_without_any_box"] = describe([s for s in scans if not s["primary"] and not s["ignored"]])
    out["scans_with_only_non_primary_boxes"] = describe([s for s in scans if not s["primary"] and s["ignored"]])
    out["scans_with_a_primary_box"] = describe([s for s in scans if s["primary"]])
    for name in classes:
        none = [s for s in scans if not s["ignored"] and not any(b["abnormality_class"] == name for b in s["primary"])]
        out[f"scans_without_ignore_and_without_primary_{name}"] = describe(none)
    result[role] = out
json.dump(result, sys.stdout, ensure_ascii=False, indent=1, sort_keys=True)
