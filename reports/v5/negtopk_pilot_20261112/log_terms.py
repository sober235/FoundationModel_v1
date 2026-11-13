"""Window means of the box-objective terms that the pilot run logged at every step (run on the server).

Reads one training log (metrics.jsonl) and prints, for each window of steps, the mean of the logged knee U terms:
the positive term (steps whose batch holds a bag), the optimised negative term (the top fraction of the negatives)
and the mean over every eligible negative, in total and per class. Only the fields "event", "step" and
members.knee_qdess.tasks.U.box are read; the output holds counts and means only.

Usage: python3 log_terms.py <metrics.jsonl> <window> > log_terms.json
"""
import json
import statistics
import sys

CLASSES = ("cartilage_lesion", "meniscal_tear")


def _mean(values):
    return None if not values else round(statistics.fmean(values), 6)


def main(path: str, window: int) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            box = record["members"]["knee_qdess"]["tasks"]["U"]["box"]
            rows[record["step"]] = box
    steps = sorted(rows)
    if steps != list(range(1, len(steps) + 1)):
        raise SystemExit("the log does not hold every step once")
    windows = []
    for start in range(1, len(steps) + 1, window):
        chunk = [rows[s] for s in range(start, min(start + window, len(steps) + 1))]
        entry = {"steps": [start, start + len(chunk) - 1], "steps_with_a_bag": sum(b["bags"] > 0 for b in chunk),
                 "positive": _mean([b["positive"] for b in chunk if b["positive"] is not None]),
                 "negative": _mean([b["negative"] for b in chunk if b["negative"] is not None]),
                 "negative_all": _mean([b["negative_all"] for b in chunk if b["negative_all"] is not None]),
                 "classes": {}}
        for c, name in enumerate(CLASSES):
            entry["classes"][name] = {
                "steps_with_a_bag": sum(b["classes"][c]["bags"] > 0 for b in chunk),
                "positive": _mean([b["classes"][c]["positive"] for b in chunk if b["classes"][c]["positive"] is not None]),
                "negative": _mean([b["classes"][c]["negative"] for b in chunk if b["classes"][c]["negative"] is not None]),
                "negative_all": _mean([b["classes"][c]["negative_all"] for b in chunk
                                       if b["classes"][c]["negative_all"] is not None]),
                "selected_share": _mean([b["classes"][c]["negative_selected"] / b["classes"][c]["negative_eligible"]
                                         for b in chunk if b["classes"][c]["negative_eligible"]])}
        windows.append(entry)
    return {"steps": len(steps), "window": window, "windows": windows}


if __name__ == "__main__":
    json.dump(main(sys.argv[1], int(sys.argv[2])), sys.stdout, ensure_ascii=False, indent=1, sort_keys=True)
