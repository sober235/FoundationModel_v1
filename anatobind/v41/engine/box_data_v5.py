"""Box supervision targets for the unified model (Proposal_v5 §0.6, §5.0; design 2026-11-10 §5.2).

A released box is never filled into voxel truth. A primary box becomes a *bag*: the reference host tissue inside
the box, where the model must mark at least a little of the class. Outside every box the class is negative (the
model imitates what the readers marked; this is a training rule, never a statement that the tissue is normal).
A box the readers drew but that is not in the primary analysis (grade 1, excluded, out of scope) is an ignore
region for every class: neither positive nor negative.

Box references are read through the frozen split by role (train / development_val); locked_test is refused, so the
released test boxes in the import output cannot reach training or model selection by this path.
Errors are V5Error(code, path) without identifiers or values.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..contracts_v5 import ActiveScope, V5Error
from ..splits_v5 import verify_split_registry_v5
from .data_v5 import TrainingSampleV5, load_training_sample_v5
from .registration_v5 import READABLE_ROLES

BAG_DTYPE = np.int32


def _window(box: dict, shape: tuple) -> tuple | None:
    """The box [start, start + size) clipped to the volume as slices; None when nothing is left."""
    slices = []
    for start, size, limit in zip(box["start"], box["size"], shape):
        low, high = max(int(np.floor(start)), 0), min(int(np.ceil(start + size)), int(limit))
        if high <= low:
            return None
        slices.append(slice(low, high))
    return tuple(slices)


def _is_primary(box: dict) -> bool:
    return box.get("status") == "included" and box.get("analysis") == "primary"


def box_bags_v5(reference: dict, shape: tuple, scope: ActiveScope, anatomy: dict, box_policy: dict) -> dict:
    """Bags, positive and negative support of one scan's boxes on the scan grid; see the module docstring."""
    shape = tuple(int(n) for n in shape)
    classes = list(scope.abnormality_classes)
    group_anatomy = {host["group"]: list(host["anatomy"]) for host in box_policy["host_tissues"].values()}
    needed = sorted({name for names in group_anatomy.values() for name in names})
    if not isinstance(anatomy, dict) or any(
            name not in anatomy or tuple(np.shape(anatomy[name])) != shape for name in needed):
        raise V5Error("GRID_MISMATCH", "$.anatomy")
    bags = np.zeros((len(classes), *shape), dtype=BAG_DTYPE)
    e_neg = np.ones((len(classes), *shape), dtype=bool)
    regions = []                                  # (voxels, bag id, class index, window, region inside the window)
    for index, box in enumerate(reference["boxes"]):
        window = _window(box["box"], shape)
        if not _is_primary(box):
            if window is not None:
                e_neg[(slice(None), *window)] = False
            continue
        if box["abnormality_class"] not in classes:
            raise V5Error("BOX_CLASS_UNKNOWN", f"$.boxes[{index}].abnormality_class")
        names = group_anatomy.get((box.get("host") or {}).get("tissue_group"))
        region = None
        if window is not None and names:
            region = np.zeros(tuple(s.stop - s.start for s in window), dtype=bool)
            for name in names:
                region |= np.asarray(anatomy[name], dtype=bool)[window]
        if region is None or not region.any():
            raise V5Error("BOX_HOST_EMPTY", f"$.boxes[{index}]")
        c = classes.index(box["abnormality_class"])
        e_neg[(c, *window)] = False
        regions.append((int(region.sum()), index + 1, c, window, region))
    # Larger bags first, so where two bags of one class overlap the smaller one (then the lower id) keeps the voxel.
    for _, bag_id, c, window, region in sorted(regions, key=lambda item: (-item[0], -item[1])):
        bags[(c, *window)][region] = bag_id
    return {"bags": bags, "e_pos": bags > 0, "e_neg": e_neg,
            "n_bags": {name: int(sum(1 for item in regions if item[2] == c)) for c, name in enumerate(classes)}}


def load_box_references_v5(path: Path, registry: dict, roles: list[str]) -> dict[str, dict]:
    """Box references of the samples the frozen split assigns to `roles`; other lines are dropped unread."""
    verify_split_registry_v5(registry)
    if not isinstance(roles, list) or not roles:
        raise V5Error("INVALID_ROLE", "$.roles")
    for index, role in enumerate(roles):
        if role == "locked_test":
            raise V5Error("LOCKED_TEST_NOT_READABLE", f"$.roles[{index}]")
        if role not in READABLE_ROLES or role in roles[:index]:
            raise V5Error("INVALID_ROLE", f"$.roles[{index}]")
    assignment = registry["assignment"]
    seen, selected = set(), {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        uid = entry.get("sample_uid") if isinstance(entry, dict) else None
        if not isinstance(uid, str) or uid in seen:
            raise V5Error("DUPLICATE_SAMPLE" if uid in seen else "INVALID_TYPE", "$.references")
        seen.add(uid)
        if assignment.get(uid) in roles:
            selected[uid] = entry
    return selected


def box_positive_centers_v5(sample: TrainingSampleV5) -> np.ndarray:
    """Patch centres of a box-supervised sample: A foreground (as positive_centers_v5) or any voxel of a bag.

    positive_centers_v5 cannot be used: its `target & e_pos` is a bitwise and, which drops even bag ids."""
    mask = np.zeros(sample.observation.valid_fov.shape, dtype=bool)
    if "A" in sample.targets:
        mask |= (sample.targets["A"] > 0) & sample.e_pos["A"].all(axis=0) & sample.e_neg["A"].all(axis=0)
    if "U" in sample.targets:
        mask |= ((sample.targets["U"] > 0) & sample.e_pos["U"]).any(axis=0)
    return np.argwhere(mask)


def load_box_training_sample_v5(root: Path, record, scope: ActiveScope, supervision: dict, recipe: dict,
                                label_maps: dict, regions: dict, reference: dict,
                                box_policy: dict) -> TrainingSampleV5:
    """A from the released segmentation (as every training sample), U from the scan's box reference."""
    if [int(f) for f in recipe["downsample_factor"]] != [1, 1, 1]:
        raise V5Error("BOX_DOWNSAMPLE_NOT_SUPPORTED", "$.recipe.downsample_factor")
    if not isinstance(reference, dict) or reference.get("sample_uid") != record.sample_uid:
        raise V5Error("BOX_REFERENCE_MISMATCH", "$.reference.sample_uid")
    if reference.get("policy_sha256") != box_policy["policy_sha256"]:
        raise V5Error("BOX_POLICY_MISMATCH", "$.reference.policy_sha256")
    anatomy_only = dict(copy.deepcopy(supervision), tasks={"A": copy.deepcopy(supervision["tasks"]["A"])})
    sample = load_training_sample_v5(root, record, scope, anatomy_only, recipe, label_maps, regions)
    labels = sample.targets["A"]
    anatomy = {name: labels == k + 1 for k, name in enumerate(scope.anatomy_classes)}
    boxes = box_bags_v5(reference, labels.shape, scope, anatomy, box_policy)
    fov = sample.observation.valid_fov
    return TrainingSampleV5(observation=sample.observation,
                            targets={**sample.targets, "U": boxes["bags"]},
                            e_pos={**sample.e_pos, "U": boxes["e_pos"] & fov},
                            e_neg={**sample.e_neg, "U": boxes["e_neg"] & fov})
