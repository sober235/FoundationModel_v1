"""Distributions over the per-box rows of the within-scan relocation control (run on the server).

Reads the private rows of one evaluate-only run (one JSON line per scan; field "relocation" is the list of that
scan's primary boxes) and prints, per role and class, how many placements each box was compared with, the share of
them that the scan's findings hit, and the host-tissue voxel count of the box. Counts and distributions only: no
identifier, no path, no time. Usage: relocation_rows.py <rows dir> <member>
"""
import json
import sys

import numpy as np

ROWS, MEMBER = sys.argv[1:3]
ROLES = ("train", "development_val")


def dist(values, digits):
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    out = {"n": int(len(a)), "mean": round(float(a.mean()), digits)}
    out.update({k: round(float(v), digits) for k, v in zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), q)})
    return out


def block(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    out = {"n_boxes": len(rows), "n_usable": len(ok),
           "excluded": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows} - {"ok"})},
           "comparable_placements": dist((r["comparable_placements"] for r in ok), 1),
           "boxes_with_fewer_than_100_comparable_placements": sum(r["comparable_placements"] < 100 for r in ok),
           "boxes_with_fewer_than_1000_comparable_placements": sum(r["comparable_placements"] < 1000 for r in ok),
           "host_voxels": dist((r["host_voxels"] for r in ok), 1)}
    for key in ("hit", "host_correct"):
        p = [r[f"p_{key}"] for r in ok]
        out[key] = {"observed": sum(bool(r[key]) for r in ok), "expected": round(float(sum(p)), 3),
                    "p_per_box": dist(p, 4),
                    "boxes_with_p_at_least_0_9": sum(v >= 0.9 for v in p),
                    "boxes_with_p_at_most_0_1": sum(v <= 0.1 for v in p),
                    "p_per_box_of_observed": dist((r[f"p_{key}"] for r in ok if r[key]), 4),
                    "p_per_box_of_not_observed": dist((r[f"p_{key}"] for r in ok if not r[key]), 4)}
    return out


result = {}
for role in ROLES:
    scans = [json.loads(line) for line in open(f"{ROWS}/{role}.{MEMBER}.jsonl", encoding="utf-8") if line.strip()]
    rows = [r for scan in scans for r in scan["relocation"]]
    entry = {"n_scans": len(scans), "scans_with_a_primary_box": sum(bool(scan["relocation"]) for scan in scans),
             "overall": block(rows), "by_class": {}}
    for name in sorted({r["abnormality_class"] for r in rows}):
        entry["by_class"][name] = block([r for r in rows if r["abnormality_class"] == name])
    result[role] = entry
json.dump(result, sys.stdout, ensure_ascii=False, indent=1, sort_keys=True)
