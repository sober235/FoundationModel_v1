"""T11 early phase: SKM-TEA released boxes and tissue_id as reference candidates (plan_v5 T11; Proposal_v5 §5.0).

Only the original FM_data labels are imported; no reader, no new annotation. The rules are the frozen
`configs/v5/skm_box_policy_v1.yaml` (S1–S7, adopted 2026-11-07):

- U is box-level: a box is kept as a box (`reference_granularity: box`). It never becomes voxel truth, and no
  eligibility or mask is produced here.
- R is tissue-level: the host tissue group comes from the box's `tissue_id`, looked up by name in the release's
  own `tissues` table (never by integer id), and is stored as an explicit assignment.
- S is tissue-level and positive-only: a tissue group is `positive` for a class when an included primary box of
  that class names it; otherwise it is `annotation_unknown`. Nothing is ever recorded negative, and `inspected`
  does not create negatives.
- A box whose category does not fit its tissue, whose confidence is 0, whose size is zero, whose coordinates are
  not integers, which leaves the grid, or which holds no host tissue of reference A is excluded with its reason
  and counted; it stays unknown. Grade-1 cartilage lesions are a secondary analysis; extrusion, ligament tears
  and effusion are out of scope.

Pure functions over the release JSON, the scan's grid shape and its reference A channels; no file is read here.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from .contracts_v5 import V5Error, canonical_sha256

POLICY_KEYS = ("policy_id", "source", "box_format", "targets", "out_of_scope_categories", "supercategory_hosts",
               "no_host_tissue_id", "host_tissues", "out_of_scope_tissues", "tissue_groups_by_class",
               "min_confidence", "sides")
ANALYSES = ("primary", "secondary")
NO_HOST = "no_host"
ADOPTED_MIN_CONFIDENCE = 1                    # S4 (adopted 2026-11-07): only confidence 0 is excluded
EXCLUSION_CASCADE = ("category_tissue_mismatch", "confidence_zero", "degenerate_box", "non_integer_box",
                     "out_of_bounds", "no_host_tissue_in_box")
_STAGES = (("confidence_nonzero", "confidence_zero"), ("nonzero_size", "degenerate_box"),
           ("integer_coordinates", "non_integer_box"), ("in_bounds", "out_of_bounds"),
           ("host_tissue_in_box", "no_host_tissue_in_box"))


def _bad(path: str) -> V5Error:
    return V5Error("INVALID_SKM_BOX_POLICY", path)


def _names(value: object, path: str) -> list:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise _bad(path)
    return value


def load_skm_box_policy_v5(policy: dict | str | Path) -> dict:
    """Validate the box policy and add its `policy_sha256` (over everything else)."""
    if not isinstance(policy, dict):
        import yaml

        policy = yaml.safe_load(Path(policy).read_text())
    if not isinstance(policy, dict) or set(policy) != set(POLICY_KEYS):
        raise _bad("$")
    policy = copy.deepcopy(policy)
    source = policy["source"]
    if not (isinstance(source, dict) and set(source) == {"source_id", "source_version"}):
        raise _bad("$.source")
    classes = policy["tissue_groups_by_class"]
    if not isinstance(classes, dict) or not classes:
        raise _bad("$.tissue_groups_by_class")
    for name, target in policy["targets"].items():
        if not (isinstance(target, dict) and set(target) == {"abnormality", "analysis"}
                and target["abnormality"] in classes and target["analysis"] in ANALYSES):
            raise _bad(f"$.targets.{name}")
    if set(policy["targets"]) & set(_names(policy["out_of_scope_categories"], "$.out_of_scope_categories")):
        raise _bad("$.out_of_scope_categories")
    groups = set()
    for name, host in policy["host_tissues"].items():
        if not (isinstance(host, dict) and set(host) == {"group", "anatomy"} and isinstance(host["group"], str)):
            raise _bad(f"$.host_tissues.{name}")
        _names(host["anatomy"], f"$.host_tissues.{name}.anatomy")
        groups.add(host["group"])
    _names(policy["out_of_scope_tissues"], "$.out_of_scope_tissues")
    known_tissues = set(policy["host_tissues"]) | set(policy["out_of_scope_tissues"]) | {NO_HOST}
    for name, hosts in policy["supercategory_hosts"].items():
        if not set(_names(hosts, f"$.supercategory_hosts.{name}")) <= known_tissues:
            raise _bad(f"$.supercategory_hosts.{name}")
    for name, members in classes.items():
        if not set(_names(members, f"$.tissue_groups_by_class.{name}")) <= groups:
            raise _bad(f"$.tissue_groups_by_class.{name}")
    for group, sides in policy["sides"].items():
        if group not in groups or not (isinstance(sides, dict) and set(sides) == {"medial", "lateral"}):
            raise _bad(f"$.sides.{group}")
    if not isinstance(policy["no_host_tissue_id"], int):
        raise _bad("$.no_host_tissue_id")
    if policy["min_confidence"] != ADOPTED_MIN_CONFIDENCE:
        raise V5Error("SKM_POLICY_NOT_ADOPTED", "$.min_confidence")
    policy["policy_sha256"] = canonical_sha256(policy)
    return policy


def skm_release_tables_v5(release: dict, policy: dict) -> dict:
    """The release's category and tissue tables, by id; every name must be one the policy decides."""
    categories, tissues = {}, {}
    known = set(policy["targets"]) | set(policy["out_of_scope_categories"])
    for item in release.get("categories", []):
        if item.get("name") not in known or item.get("supercategory") not in policy["supercategory_hosts"]:
            raise V5Error("SKM_CATEGORY_UNKNOWN", "$.categories")
        categories[item["id"]] = {"name": item["name"], "supercategory": item["supercategory"]}
    for item in release.get("tissues", []):
        if item.get("name") not in set(policy["host_tissues"]) | set(policy["out_of_scope_tissues"]):
            raise V5Error("SKM_TISSUE_UNKNOWN", "$.tissues")
        tissues[item["id"]] = item["name"]
    if not categories or not tissues:
        raise V5Error("SKM_RELEASE_TABLES_MISSING", "$")
    return {"categories": categories, "tissues": tissues}


def _box_values(box: dict, index: int) -> tuple[list, list, float]:
    bbox, confidence = box.get("bbox"), box.get("confidence")
    if not (isinstance(bbox, list) and len(bbox) == 6
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)):
        raise V5Error("INVALID_BOX", f"$.annotations[{index}].bbox")
    if not (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)):
        raise V5Error("INVALID_BOX", f"$.annotations[{index}].confidence")
    return list(bbox[:3]), list(bbox[3:]), float(confidence)


def _exclusion(start, size, confidence, shape, policy) -> str | None:
    if confidence < policy["min_confidence"]:
        return "confidence_zero"
    if any(s <= 0 for s in size):
        return "degenerate_box"
    if not all(float(v).is_integer() for v in start + size):
        return "non_integer_box"
    if any(a < 0 or a + s > n for a, s, n in zip(start, size, shape)):
        return "out_of_bounds"
    return None


def import_skm_boxes_v5(*, sample_uid: str, shape: tuple, annotations: list, tables: dict, reference_a: dict,
                        policy: dict) -> dict:
    """One scan's boxes as reference candidates. `reference_a` maps each A6 class to a boolean array on the
    scan grid (the boxes' index space); it is read only to check the host tissue inside a box and its side."""
    shape = tuple(int(n) for n in shape)
    needed = {name for host in policy["host_tissues"].values() for name in host["anatomy"]}
    for name in needed:
        array = reference_a.get(name)
        if array is None:
            raise V5Error("REFERENCE_A_MISSING", f"$.reference_a.{name}")
        if tuple(np.shape(array)) != shape:
            raise V5Error("GRID_MISMATCH", f"$.reference_a.{name}")
    target_supercategories = {tables["categories"][cid]["supercategory"] for cid in tables["categories"]
                              if tables["categories"][cid]["name"] in policy["targets"]}
    boxes = []
    for index, ann in enumerate(annotations):
        category = tables["categories"].get(ann.get("category_id"))
        if category is None:
            raise V5Error("SKM_CATEGORY_UNKNOWN", f"$.annotations[{index}].category_id")
        tissue_id = ann.get("tissue_id")
        tissue = NO_HOST if tissue_id == policy["no_host_tissue_id"] else tables["tissues"].get(tissue_id)
        if tissue is None:
            raise V5Error("SKM_TISSUE_UNKNOWN", f"$.annotations[{index}].tissue_id")
        start, size, confidence = _box_values(ann, index)
        name, supercategory = category["name"], category["supercategory"]
        record = {"annotation_id": ann.get("id"), "reference_granularity": "box",
                  "origin": {**policy["source"], "labeler": ann.get("labeler"), "category": name,
                             "tissue": None if tissue == NO_HOST else tissue},
                  "target_supercategory": supercategory in target_supercategories,
                  "abnormality_class": None, "analysis": None, "status": None, "reason": None,
                  "confidence": confidence, "box": {"start": start, "size": size}, "host_voxels_in_box": None,
                  "host": None}
        boxes.append(record)
        if tissue not in policy["supercategory_hosts"][supercategory]:
            record.update(status="excluded", reason="category_tissue_mismatch")
            continue
        if not record["target_supercategory"]:                      # S3: ligament tears and effusion
            relation = NO_HOST if tissue == NO_HOST else "out_of_scope"
            record.update(status="out_of_scope", reason="out_of_scope_category",
                          host={"tissue_group": None, "relation": relation, "reference_granularity": "tissue",
                                "assignment": "released_tissue_id", "side": None})
            continue
        reason = _exclusion(start, size, confidence, shape, policy)
        host = policy["host_tissues"][tissue]
        if reason is None:
            window = tuple(slice(int(a), int(a + s)) for a, s in zip(start, size))
            record["host_voxels_in_box"] = int(sum(int(np.count_nonzero(np.asarray(reference_a[c])[window]))
                                                   for c in host["anatomy"]))
            if record["host_voxels_in_box"] == 0:
                reason = "no_host_tissue_in_box"
        if reason is not None:
            record.update(status="excluded", reason=reason)
            continue
        side = None
        if host["group"] in policy["sides"]:
            counts = {s: int(np.count_nonzero(np.asarray(reference_a[c])[window]))
                      for s, c in policy["sides"][host["group"]].items()}
            present = [s for s in ("medial", "lateral") if counts[s] > 0]
            side = present[0] if len(present) == 1 else "both"
        record["host"] = {"tissue_group": host["group"], "relation": "bound", "reference_granularity": "tissue",
                          "assignment": "released_tissue_id", "side": side}
        if name in policy["out_of_scope_categories"]:                   # S3: meniscal extrusion
            record.update(status="out_of_scope", reason="out_of_scope_category")
            continue
        target = policy["targets"][name]
        record.update(status="included", abnormality_class=target["abnormality"], analysis=target["analysis"])
    states = {analysis: {cls: {group: "annotation_unknown" for group in groups}
                         for cls, groups in policy["tissue_groups_by_class"].items()} for analysis in ANALYSES}
    for record in boxes:
        if record["status"] == "included":
            states[record["analysis"]][record["abnormality_class"]][record["host"]["tissue_group"]] = "positive"
    return {"sample_uid": sample_uid, "policy_sha256": policy["policy_sha256"], "source": dict(policy["source"]),
            "boxes": boxes, "tissue_state": {"reference_granularity": "tissue", "states": states["primary"],
                                             "secondary": states["secondary"]}}


def summarize_skm_import_v5(results: list, splits: list | None = None) -> dict:
    """Counts only (no identifiers): the exclusion cascade in the research report's order, exclusions and
    out-of-scope boxes by name, and positive scans per split."""
    splits = splits or ["all"] * len(results)
    if len(splits) != len(results):
        raise V5Error("INVALID_TYPE", "$.splits")
    cascade = {"target_supercategory_consistent": 0, **{stage: 0 for stage, _ in _STAGES},
               "primary": 0, "secondary": 0, "out_of_scope_extrusion": 0}
    excluded, out_of_scope, positives, without = {}, {}, {}, {}
    for result, split in zip(results, splits):
        positives.setdefault(split, {})
        for box in result["boxes"]:
            if box["status"] == "excluded":
                excluded[box["reason"]] = excluded.get(box["reason"], 0) + 1
            if box["status"] == "out_of_scope":
                category = box["origin"]["category"]
                out_of_scope[category] = out_of_scope.get(category, 0) + 1
            if not box["target_supercategory"] or box["reason"] == "category_tissue_mismatch":
                continue
            cascade["target_supercategory_consistent"] += 1
            for position, (stage, _) in enumerate(_STAGES):
                if box["reason"] not in {reason for _, reason in _STAGES[:position + 1]}:
                    cascade[stage] += 1
            if box["status"] == "included":
                cascade[box["analysis"]] += 1
            elif box["status"] == "out_of_scope":
                cascade["out_of_scope_extrusion"] += 1
        has_primary = False
        for cls, groups in result["tissue_state"]["states"].items():
            for group, state in groups.items():
                if state == "positive":
                    has_primary = True
                    key = f"{cls}:{group}"
                    positives[split][key] = positives[split].get(key, 0) + 1
        if not has_primary:
            without[split] = without.get(split, 0) + 1
    return {"scans": len(results), "boxes": sum(len(r["boxes"]) for r in results), "cascade": cascade,
            "excluded": excluded, "out_of_scope": dict(sorted(out_of_scope.items())),
            "positive_scans_by_split": positives, "scans_without_primary_box": without}
