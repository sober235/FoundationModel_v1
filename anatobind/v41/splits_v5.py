"""v5.1 group-level splits (Proposal_v5 §3.3; plan_v5 T08b).

Splits are assigned per group, so all records of a group share one split; a
group mapping that puts one patient or leakage group into two groups is
rejected. Official source splits come first; otherwise groups are ordered within
each stratum by a seeded hash and cut by the policy fractions. Every value
comes from the policy (no defaults). A group whose exposure ledger blocks the
locked test of the lineage or of any ancestor lineage (L6) is moved out of
``locked_test`` to the policy's ``exposed_groups`` split. Pure function:
nothing is written. The returned registry carries its lineage, ancestors and
policy, so ``verify_split_registry_v5`` can recompute its hash (L4).
"""
from __future__ import annotations

import copy
import hashlib
import math

from .contracts_v5 import SampleV5, V5Error, canonical_sha256
from .exposure_v5 import lineage_blockers_v5

SPLITS_V5 = ("train", "development_val", "locked_test", "unassigned", "excluded")
OFFICIAL_MAP = {"train": "train", "val": "development_val", "test": "locked_test"}
FRACTION_KEYS = ("train", "development_val", "locked_test")
EXPOSED_TARGETS = ("train", "excluded")
SMALL_STRATUM_RULES = ("merge",)
MERGED_STRATUM = "__merged__"
_FRACTION_TOLERANCE = 1e-9


def _bad(key: str) -> V5Error:
    return V5Error("INVALID_SPLIT_POLICY", f"$.policy.{key}")


def _require(policy: dict, key: str):
    if key not in policy:
        raise _bad(key)
    return policy[key]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _record_groups(records: list, groups: dict) -> dict[str, list[SampleV5]]:
    if not isinstance(records, list):
        raise V5Error("INVALID_TYPE", "$.records")
    if not isinstance(groups, dict):
        raise V5Error("INVALID_TYPE", "$.groups")
    seen: set[str] = set()
    members: dict[str, list[SampleV5]] = {}
    owner: dict[tuple[str, str], str] = {}
    for index, record in enumerate(records):
        path = f"$.records[{index}]"
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", path)
        if record.sample_uid in seen:
            raise V5Error("DUPLICATE_SAMPLE", path)
        seen.add(record.sample_uid)
        group = groups.get(record.sample_uid)
        if group is None:
            raise V5Error("MISSING_GROUP", path)
        if not isinstance(group, str) or not group.strip():
            raise V5Error("INVALID_GROUP", path)
        for field in ("leakage_group_uid", "patient_uid"):
            if owner.setdefault((field, getattr(record, field)), group) != group:
                raise V5Error("LEAKAGE_GROUP_SPLIT", f"{path}.{field}")
        members.setdefault(group, []).append(record)
    return members


def _official(policy: dict, group_names: list[str]) -> dict[str, str]:
    mapping = _require(policy, "official_split")
    if not isinstance(mapping, dict) or any(value not in OFFICIAL_MAP for value in mapping.values()):
        raise _bad("official_split")
    return {g: OFFICIAL_MAP[mapping[g]] if g in mapping else "unassigned" for g in group_names}


def _fractions(policy: dict) -> tuple[float, float, float]:
    fractions = _require(policy, "fractions")
    if not isinstance(fractions, dict) or set(fractions) != set(FRACTION_KEYS):
        raise _bad("fractions")
    values = tuple(fractions[key] for key in FRACTION_KEYS)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise _bad("fractions")
    if abs(sum(values) - 1.0) > _FRACTION_TOLERANCE:
        raise _bad("fractions")
    return values


def _stratified(policy: dict, group_names: list[str]) -> dict[str, str]:
    seed = _require(policy, "seed")
    if not _is_int(seed):
        raise _bad("seed")
    fractions = _fractions(policy)
    strata = _require(policy, "strata")
    if not isinstance(strata, dict) or any(
            not isinstance(strata.get(g), str) or not strata[g].strip() or strata[g] == MERGED_STRATUM
            for g in group_names):
        raise _bad("strata")
    minimum = _require(policy, "min_stratum_groups")
    if not _is_int(minimum) or minimum < 1:
        raise _bad("min_stratum_groups")
    if _require(policy, "small_stratum_rule") not in SMALL_STRATUM_RULES:
        raise _bad("small_stratum_rule")
    sizes: dict[str, int] = {}
    for g in group_names:
        sizes[strata[g]] = sizes.get(strata[g], 0) + 1
    by_stratum: dict[str, list[str]] = {}
    for g in group_names:
        stratum = strata[g] if sizes[strata[g]] >= minimum else MERGED_STRATUM
        by_stratum.setdefault(stratum, []).append(g)
    out: dict[str, str] = {}
    for stratum, names in sorted(by_stratum.items()):
        ordered = sorted(names, key=lambda g: hashlib.sha256(f"{seed}:{stratum}:{g}".encode("utf-8")).hexdigest())
        n = len(ordered)
        cut_train = math.floor(n * fractions[0] + 0.5)
        cut_val = max(cut_train, min(n, math.floor(n * (fractions[0] + fractions[1]) + 0.5)))
        for position, g in enumerate(ordered):
            out[g] = "train" if position < cut_train else "development_val" if position < cut_val else "locked_test"
    return out


def split_registry_sha256_v5(assignment: dict, lineage_id: str, ancestor_lineages: list[str], policy: dict) -> str:
    return canonical_sha256({"assignment": assignment, "lineage_id": lineage_id,
                             "ancestor_lineages": ancestor_lineages, "policy": policy})


def freeze_splits_v5(records: list[SampleV5], groups: dict[str, str], policy: dict, lineage_id: str, *,
                     ancestor_lineages: list[str]) -> dict:
    lineage_blockers_v5([], lineage_id, ancestor_lineages)
    ancestors = sorted(ancestor_lineages)
    if not isinstance(policy, dict):
        raise V5Error("INVALID_SPLIT_POLICY", "$.policy")
    members = _record_groups(records, groups)
    group_names = sorted(members)
    kind = policy.get("kind")
    if kind not in ("official", "stratified"):
        raise _bad("kind")
    exposed_target = _require(policy, "exposed_groups")
    if exposed_target not in EXPOSED_TARGETS:
        raise _bad("exposed_groups")
    by_group = _official(policy, group_names) if kind == "official" else _stratified(policy, group_names)
    exposed = 0
    for g in group_names:
        if by_group[g] == "locked_test" and any(
                lineage_blockers_v5(list(r.exposure_history), lineage_id, ancestors) for r in members[g]):
            by_group[g] = exposed_target
            exposed += 1
    assignment = {r.sample_uid: by_group[g] for g in group_names for r in members[g]}
    assignment = dict(sorted(assignment.items()))
    counts = {split: 0 for split in sorted(SPLITS_V5)}
    for split in by_group.values():
        counts[split] += 1
    frozen_policy = copy.deepcopy(policy)
    return {"assignment": assignment, "counts": counts, "exposed_groups": exposed,
            "unassigned_groups": counts["unassigned"], "lineage_id": lineage_id, "ancestor_lineages": ancestors,
            "policy": frozen_policy,
            "split_sha256": split_registry_sha256_v5(assignment, lineage_id, ancestors, frozen_policy)}


_REGISTRY_KEYS = ("assignment", "lineage_id", "ancestor_lineages", "policy", "split_sha256")


def verify_split_registry_v5(registry: dict) -> dict:
    """Recompute a frozen registry's hash; only a registry that proves itself may be read (L4)."""
    if not isinstance(registry, dict):
        raise V5Error("INVALID_SPLIT_REGISTRY", "$")
    for key in _REGISTRY_KEYS:
        if key not in registry:
            raise V5Error("INVALID_SPLIT_REGISTRY", f"$.{key}")
    assignment = registry["assignment"]
    if not isinstance(assignment, dict) or any(
            not isinstance(uid, str) or split not in SPLITS_V5 for uid, split in assignment.items()):
        raise V5Error("INVALID_SPLIT_REGISTRY", "$.assignment")
    if not isinstance(registry["policy"], dict):
        raise V5Error("INVALID_SPLIT_REGISTRY", "$.policy")
    lineage_blockers_v5([], registry["lineage_id"], registry["ancestor_lineages"])
    expected = split_registry_sha256_v5(assignment, registry["lineage_id"], registry["ancestor_lineages"],
                                        registry["policy"])
    if registry["split_sha256"] != expected:
        raise V5Error("SPLIT_HASH_MISMATCH", "$.split_sha256")
    return registry
