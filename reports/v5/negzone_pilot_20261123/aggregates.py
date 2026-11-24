"""Aggregates of the zone-balanced negative-term control experiment (plan unified_knee_brain_negzone_pilot_v1).

Reads, on the server, the private per-scan rows of three evaluate-only runs at step 6000 (the zone run, the
original run, the 5 % pilot) and the zone run's training log, and prints one JSON of counts, medians and pooled
fractions: no sample or patient identifier, path or time. Scans are paired by patient inside this script only.

    python aggregates.py <zone dev rows> <zone partial train rows> <original eval dir> <5 % eval dir> <zone metrics.jsonl>

Groups follow reports/v5/negzone_pilot_prereg_20261120.md §4: group 1 (box hits and the within-scan relocation
expectation), group 2 (findings per scan, scans with a cartilage-lesion finding of at least 10000 voxels or a
meniscal-tear finding of at least 1000 voxels, by whether the scan has a primary box), group 3 (share of the host
tissue at or above threshold: same-class primary boxes, ignore regions, outside all boxes), group 6 (training log
by 500-step segments). The thresholds are the frozen ones; nothing here changes an evaluation rule.
"""
import json
import statistics as st
import sys

CLASSES = ("cartilage_lesion", "meniscal_tear")
LARGE = {"cartilage_lesion": 10000, "meniscal_tear": 1000}
ZONES = ("same_class_primary", "ignored", "outside")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return {row["official"]["patient_uid"]: row for row in map(json.loads, handle)}


def groups(rows):
    totals = [row["official"]["findings"]["total"] for row in rows]
    out = {"scans": len(rows), "findings_per_scan": {"median": st.median(totals), "min": min(totals),
                                                     "max": max(totals)}}
    for name in CLASSES:
        with_box, without_box = [], []
        for row in rows:
            sizes = [f["voxels"] for f in row["diagnostic"]["findings"] if f["abnormality_class"] == name]
            (with_box if row["official"]["boxes"] else without_box).append(any(v >= LARGE[name] for v in sizes))
        sizes_all = [f["voxels"] for row in rows for f in row["diagnostic"]["findings"] if f["abnormality_class"] == name]
        largest = [max([f["voxels"] for f in row["diagnostic"]["findings"] if f["abnormality_class"] == name] or [0])
                   for row in rows]
        boxes = [b for row in rows for b in row["official"]["boxes"] if b["abnormality_class"] == name]
        moved = [x for row in rows for x in row["relocation"] if x["abnormality_class"] == name and x["p_hit"] is not None]
        host = {}
        for zone in ZONES:
            on = sum(row["diagnostic"]["zones"][name][zone]["positive_on_host"] for row in rows)
            voxels = sum(row["diagnostic"]["zones"][name][zone]["host_voxels"] for row in rows)
            host[zone] = round(on / voxels, 4) if voxels else None
        out[name] = {
            "large_finding_scans": {"with_box": [sum(with_box), len(with_box)],
                                    "without_box": [sum(without_box), len(without_box)]},
            "findings": len(sizes_all), "finding_voxels_median": st.median(sizes_all) if sizes_all else None,
            "largest_finding_per_scan_median": st.median(largest),
            "boxes": len(boxes), "hits": sum(bool(b["hit"]) for b in boxes),
            "hits_host_correct": sum(bool(b["host_correct"]) for b in boxes),
            "relocation_expected_hits": round(sum(x["p_hit"] for x in moved), 2),
            "host_positive_share": host}
    return out


def side_by_side(runs, keys):
    return {name: groups([rows[k] for k in keys]) for name, rows in runs.items()}


def log_segments(path, size=500):
    steps = [r for r in map(json.loads, open(path, encoding="utf-8")) if r.get("event") == "step"]

    def mean(values):
        values = [v for v in values if v is not None]
        return round(st.mean(values), 4) if values else None
    segments = []
    for start in range(0, len(steps), size):
        block = steps[start:start + size]
        entry = {"steps": [block[0]["step"], block[-1]["step"]]}
        for index, name in enumerate(CLASSES):
            terms = [r["members"]["knee_qdess"]["tasks"]["U"]["box"]["classes"][index] for r in block]
            zones = [t["negative_zones"] for t in terms]
            entry[name] = {"positive": mean([t["positive"] for t in terms]),
                           "negative": mean([t["negative"] for t in terms]),
                           "host_mean": mean([z["host"]["mean"] for z in zones if z["host"]["eligible"]]),
                           "rest_mean": mean([z["rest"]["mean"] for z in zones if z["rest"]["eligible"]]),
                           "host_eligible_mean": round(st.mean([z["host"]["eligible"] for z in zones])),
                           "rest_eligible_mean": round(st.mean([z["rest"]["eligible"] for z in zones])),
                           "steps_without_host": sum(z["host"]["eligible"] == 0 for z in zones)}
        segments.append(entry)
    return segments


def main(zone_dev, zone_train, original, pilot5, metrics):
    zone_d, zone_t = load(zone_dev), load(zone_train)
    other = {"original_6000": original, "pilot5_6000": pilot5}
    dev = {"zone_6000": zone_d, **{k: load(f"{v}/rows/development_val.knee_qdess.jsonl") for k, v in other.items()}}
    train = {"zone_6000": zone_t, **{k: load(f"{v}/rows/train.knee_qdess.jsonl") for k, v in other.items()}}
    dev_keys, train_keys = sorted(zone_d), sorted(zone_t)
    for runs, keys in ((dev, dev_keys), (train, train_keys)):
        if any(k not in rows for rows in runs.values() for k in keys):
            raise SystemExit("SCANS_DIFFER")
    print(json.dumps({"development_val": side_by_side(dev, dev_keys),
                      "train_partial": side_by_side(train, train_keys),
                      "log_segments_500": log_segments(metrics)}, ensure_ascii=False, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:6])
