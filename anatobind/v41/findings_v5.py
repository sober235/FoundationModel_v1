"""Protocol table and findings of the unified model (Proposal_v5 §0.6; design 2026-11-10 §5.1, §5.4).

A finding is what a reader would mark: a box around one predicted abnormality instance, its class, and (where the
protocol has tissue labels) the tissue group it sits on, with one fixed sentence. Findings are derived from the
existing output chain's products (accepted binary maps, instances, relations); nothing here reads a reference.

The anchor of a finding is the instance voxel nearest to the instance's centroid. The adopted criterion S7 names
the probability peak, but the output chain keeps binary maps, not probabilities (design §5.4, a recorded deviation).

"未见所列异常" is never "normal": a protocol without confirmed negatives (the knee boxes) only says that none of
the listed abnormalities was marked.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from .annotation_v5 import load_skm_box_policy_v5
from .contracts_v5 import ActiveScope, V5Error, validate_scope_v5
from .perception import extract_instances_v5

PROTOCOL_KEYS = ("scope", "source_id", "description", "box_policy", "display")
HOST_STATUSES = ("bound", "undetermined", "not_reported")
_PROTOCOL_ID = re.compile(r"[a-z][a-z0-9_]*")


def load_protocols_v5(path: Path, config_dir: Path) -> dict:
    """Protocol id -> {scope, source_id, description, box_policy, groups, groups_by_class, display}."""
    table = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(table, dict) or not table:
        raise V5Error("PROTOCOL_INVALID", "$")
    protocols = {}
    for pid, entry in table.items():
        if not isinstance(pid, str) or not _PROTOCOL_ID.fullmatch(pid) \
                or not isinstance(entry, dict) or set(entry) != set(PROTOCOL_KEYS) \
                or not isinstance(entry["display"], dict):
            raise V5Error("PROTOCOL_INVALID", f"$.{pid}" if isinstance(pid, str) else "$")
        scope = validate_scope_v5(yaml.safe_load((Path(config_dir) / entry["scope"]).read_text(encoding="utf-8")))
        policy = groups = groups_by_class = None
        if entry["box_policy"] is not None:
            policy = load_skm_box_policy_v5(Path(config_dir) / entry["box_policy"])
            groups = {host["group"]: list(host["anatomy"]) for host in policy["host_tissues"].values()}
            groups_by_class = {name: list(members) for name, members in policy["tissue_groups_by_class"].items()}
        protocols[pid] = {"scope": scope, "source_id": entry["source_id"], "description": entry["description"],
                          "box_policy": policy, "groups": groups, "groups_by_class": groups_by_class,
                          "display": dict(entry["display"])}
    return protocols


def _group_ids(scope: ActiveScope, groups: dict) -> dict:
    return {group: [scope.anatomy_classes.index(name) + 1 for name in names] for group, names in groups.items()}


def host_relation_fn_v5(groups: dict, groups_by_class: dict):
    """A relation_fn for run_perception_v5: each instance is bound to the allowed tissue group that holds most of
    its accepted predicted-anatomy voxels (ties: the first group name); none of them gives status unknown."""

    def relation_fn(instances: list, anatomy_map: np.ndarray, anatomy_accepted: np.ndarray, scope: ActiveScope):
        ids = _group_ids(scope, groups)
        contract = scope.relation_contract
        relations, reasons = [], {}
        for instance in instances:
            voxels = np.asarray(anatomy_map)[instance["mask"] & np.asarray(anatomy_accepted, dtype=bool)]
            best, best_count = None, 0
            for group in sorted(groups_by_class.get(instance["class_name"], [])):
                count = int(np.isin(voxels, ids[group]).sum())
                if count > best_count:
                    best, best_count = group, count
            tissue_ids = [] if best is None else sorted(int(t) for t in np.unique(voxels) if int(t) in ids[best])
            relations.append({"instance_uid": instance["instance_uid"], "predicate": contract["predicate"],
                              "tissue_ids": tissue_ids, "status": "bound" if tissue_ids else "unknown",
                              "accepted": True, "completion_role": contract["completion_role"]})
            reasons[instance["instance_uid"]] = 0
        return relations, reasons

    return relation_fn


def host_of_relation_v5(relation: dict | None, ids: dict) -> tuple:
    """(tissue group, status) of one instance's relation: bound when its tissues all belong to one group."""
    if relation is not None and relation["status"] == "bound":
        owners = [group for group, members in sorted(ids.items()) if set(relation["tissue_ids"]) <= set(members)]
        if owners:
            return owners[0], "bound"
    return None, "undetermined"


def _anchor(mask: np.ndarray) -> list:
    coordinates = np.argwhere(mask)                       # lexicographic order, so the first minimum is the lowest
    distance = ((coordinates - coordinates.mean(axis=0)) ** 2).sum(axis=1)
    return [int(v) for v in coordinates[int(np.argmin(distance))]]


def findings_from_output_v5(output, scope: ActiveScope, acceptance_policy: dict, groups: dict | None) -> list:
    """One finding per accepted abnormality instance of a joint (PerceptionOutput) or module (ModuleOutput) output."""
    if output.output_kind == "joint":
        instances = list(output.abnormal_instances)
        relations = {relation["instance_uid"]: relation for relation in output.tissue_relations}
    else:
        instances = extract_instances_v5(output.abnormality_map, output.acceptance["abnormality"], scope,
                                         acceptance_policy)
        relations = {}
    ids = _group_ids(scope, groups) if groups is not None else None
    findings = []
    for instance in sorted(instances, key=lambda item: item["instance_uid"]):
        ndim = len(instance["bbox"]) // 2
        start, stop = instance["bbox"][:ndim], instance["bbox"][ndim:]
        host_group, host_status = None, "not_reported"
        if ids is not None:
            host_group, host_status = host_of_relation_v5(relations.get(instance["instance_uid"]), ids)
        findings.append({"finding_uid": instance["instance_uid"], "abnormality_class": instance["class_name"],
                         "box": {"start": [int(v) for v in start], "size": [int(b - a) for a, b in zip(start, stop)]},
                         "anchor": _anchor(np.asarray(instance["mask"], dtype=bool)),
                         "voxels": int(np.count_nonzero(instance["mask"])), "host_group": host_group,
                         "host_status": host_status})
    return findings


def render_findings_zh_v5(findings: list, scope: ActiveScope, display: dict) -> list:
    """The fixed wording of each finding; without findings, one sentence that never says "normal"."""

    def name(key: str) -> str:
        if not isinstance(display, dict) or not isinstance(display.get(key), str):
            raise V5Error("PROTOCOL_DISPLAY_MISSING", "$.display")
        return display[key]

    if not findings:
        listed = "、".join(name(cls) for cls in scope.abnormality_classes)
        return [f"未见所列异常（仅限：{listed}）。医生没有标注不等于正常。"]
    sentences = []
    for finding in findings:
        abnormality = name(finding["abnormality_class"])
        if finding["host_status"] == "bound":
            sentences.append(f"在{name(finding['host_group'])}上，疑似{abnormality}")
        elif finding["host_status"] == "undetermined":
            sentences.append(f"疑似{abnormality}，所在结构未能确定")
        else:
            sentences.append(f"疑似{abnormality}")
    return sentences
