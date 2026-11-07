from __future__ import annotations

import hashlib

import pytest

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5
from anatobind.v41.splits_v5 import freeze_splits_v5
from tests.v5.factories import make_sample_v5

SPLITS = ("development_val", "excluded", "locked_test", "train", "unassigned")


def _exposure(role, lineage_id="lin-a"):
    return {"experiment_id": "exp-1", "role": role, "recorded_at": "2026-11-06T00:00:00Z",
            "lineage_id": lineage_id, "evidence": None}


def _records(n_groups, per_group=2, exposed=()):
    records, groups = [], {}
    for g in range(n_groups):
        for k in range(per_group):
            uid = f"s-{g}-{k}"
            history = [_exposure("training")] if g in exposed and k == 0 else []
            records.append(validate_sample_v5(make_sample_v5(
                sample_uid=uid, exposure_history=history,
                patient_uid=f"synthetic:p{g:04d}", leakage_group_uid=f"synthetic:p{g:04d}")))
            groups[uid] = f"g{g}"
    return records, groups


def _by_group(result, groups):
    out = {}
    for uid, split in result["assignment"].items():
        out.setdefault(groups[uid], set()).add(split)
    assert all(len(s) == 1 for s in out.values())
    return {g: s.pop() for g, s in out.items()}


def _stratified(**overrides):
    policy = {"kind": "stratified", "seed": 7, "fractions": {"train": 0.6, "development_val": 0.2, "locked_test": 0.2},
              "strata": {}, "min_stratum_groups": 3, "small_stratum_rule": "merge", "exposed_groups": "excluded"}
    policy.update(overrides)
    return policy


def test_official_split_is_mapped():
    records, groups = _records(3)
    policy = {"kind": "official", "official_split": {"g0": "train", "g1": "val", "g2": "test"},
              "exposed_groups": "excluded"}
    result = freeze_splits_v5(records, groups, policy, "lin-a")
    assert _by_group(result, groups) == {"g0": "train", "g1": "development_val", "g2": "locked_test"}
    assert result["counts"] == {"development_val": 1, "excluded": 0, "locked_test": 1, "train": 1, "unassigned": 0}
    assert (result["exposed_groups"], result["unassigned_groups"]) == (0, 0)
    assert len(result["split_sha256"]) == 64


def test_missing_official_group_stays_unassigned():
    records, groups = _records(3)
    policy = {"kind": "official", "official_split": {"g0": "train"}, "exposed_groups": "excluded"}
    result = freeze_splits_v5(records, groups, policy, "lin-a")
    assert _by_group(result, groups) == {"g0": "train", "g1": "unassigned", "g2": "unassigned"}
    assert result["unassigned_groups"] == 2 and result["counts"]["unassigned"] == 2


def test_official_exposed_test_group_moves_out_of_locked_test():
    records, groups = _records(2, exposed=(1,))
    policy = {"kind": "official", "official_split": {"g0": "test", "g1": "test"}, "exposed_groups": "train"}
    result = freeze_splits_v5(records, groups, policy, "lin-a")
    assert _by_group(result, groups) == {"g0": "locked_test", "g1": "train"}
    assert result["exposed_groups"] == 1
    other_lineage = freeze_splits_v5(records, groups, policy, "lin-b")
    assert _by_group(other_lineage, groups) == {"g0": "locked_test", "g1": "locked_test"}


def test_stratified_split_is_deterministic_and_group_consistent():
    records, groups = _records(20)
    strata = {f"g{g}": ("s1" if g < 10 else "s2") for g in range(20)}
    first = freeze_splits_v5(records, groups, _stratified(strata=strata), "lin-a")
    again = freeze_splits_v5(list(reversed(records)), dict(reversed(list(groups.items()))),
                             _stratified(strata=dict(reversed(list(strata.items())))), "lin-a")
    assert first == again
    by_group = _by_group(first, groups)
    assert first["counts"] == {"development_val": 4, "excluded": 0, "locked_test": 4, "train": 12, "unassigned": 0}
    for stratum in ("s1", "s2"):
        members = [g for g, s in strata.items() if s == stratum]
        assert sorted(by_group[g] for g in members).count("train") == 6
    reseeded = freeze_splits_v5(records, groups, _stratified(strata=strata, seed=8), "lin-a")
    assert reseeded["split_sha256"] != first["split_sha256"]


def test_small_strata_are_merged_by_rule():
    records, groups = _records(12)
    strata = {f"g{g}": "big" for g in range(10)} | {"g10": "tiny_a", "g11": "tiny_b"}
    fractions = {"train": 0.5, "development_val": 0.0, "locked_test": 0.5}
    result = freeze_splits_v5(records, groups, _stratified(strata=strata, fractions=fractions), "lin-a")
    by_group = _by_group(result, groups)
    order = sorted(["g10", "g11"], key=lambda g: hashlib.sha256(f"7:__merged__:{g}".encode()).hexdigest())
    assert (by_group[order[0]], by_group[order[1]]) == ("train", "locked_test")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exposure_survives_reseeding(seed):
    exposed = range(8)
    records, groups = _records(10, exposed=exposed)
    strata = {f"g{g}": "s" for g in range(10)}
    fractions = {"train": 0.4, "development_val": 0.2, "locked_test": 0.4}
    result = freeze_splits_v5(records, groups, _stratified(seed=seed, strata=strata, fractions=fractions), "lin-a")
    by_group = _by_group(result, groups)
    assert all(by_group[f"g{g}"] != "locked_test" for g in exposed)
    assert result["exposed_groups"] >= 2 and result["counts"]["excluded"] == result["exposed_groups"]
    assert result["counts"]["locked_test"] == 4 - result["exposed_groups"]


def test_policy_without_seed_or_rule_raises():
    records, groups = _records(3)
    strata = {f"g{g}": "s" for g in range(3)}
    for key in ("seed", "fractions", "strata", "min_stratum_groups", "small_stratum_rule", "exposed_groups"):
        policy = _stratified(strata=strata)
        del policy[key]
        with pytest.raises(V5Error) as err:
            freeze_splits_v5(records, groups, policy, "lin-a")
        assert (err.value.code, err.value.path) == ("INVALID_SPLIT_POLICY", f"$.policy.{key}")
    bad = [("small_stratum_rule", "drop"), ("seed", True), ("seed", 1.5), ("min_stratum_groups", 0),
           ("fractions", {"train": 0.5, "development_val": 0.2, "locked_test": 0.2}),
           ("fractions", {"train": 1.0}), ("exposed_groups", "locked_test"),
           ("strata", {"g0": "s"})]
    for key, value in bad:
        with pytest.raises(V5Error) as err:
            freeze_splits_v5(records, groups, _stratified(**{"strata": strata, key: value}), "lin-a")
        assert (err.value.code, err.value.path) == ("INVALID_SPLIT_POLICY", f"$.policy.{key}")
    for policy in ({"kind": "random"}, {"official_split": {}, "exposed_groups": "train"}):
        with pytest.raises(V5Error) as err:
            freeze_splits_v5(records, groups, policy, "lin-a")
        assert (err.value.code, err.value.path) == ("INVALID_SPLIT_POLICY", "$.policy.kind")
    with pytest.raises(V5Error) as err:
        freeze_splits_v5(records, groups, {"kind": "official", "official_split": {"g0": "holdout"},
                                           "exposed_groups": "train"}, "lin-a")
    assert (err.value.code, err.value.path) == ("INVALID_SPLIT_POLICY", "$.policy.official_split")


def test_records_need_groups_and_unique_uids():
    records, groups = _records(2)
    del groups["s-1-1"]
    policy = {"kind": "official", "official_split": {}, "exposed_groups": "train"}
    with pytest.raises(V5Error) as err:
        freeze_splits_v5(records, groups, policy, "lin-a")
    assert (err.value.code, err.value.path) == ("MISSING_GROUP", "$.records[3]")
    records, groups = _records(1)
    with pytest.raises(V5Error) as err:
        freeze_splits_v5(records + records[:1], groups, policy, "lin-a")
    assert (err.value.code, err.value.path) == ("DUPLICATE_SAMPLE", "$.records[2]")


def test_group_mapping_cannot_split_a_patient_or_leakage_group():
    policy = {"kind": "official", "official_split": {"g0": "train", "g1": "test"}, "exposed_groups": "train"}
    records, groups = _records(2)
    groups["s-0-1"] = "g1"
    with pytest.raises(V5Error) as err:
        freeze_splits_v5(records, groups, policy, "lin-a")
    assert (err.value.code, err.value.path) == ("LEAKAGE_GROUP_SPLIT", "$.records[1].leakage_group_uid")
    assert "p0000" not in str(err.value)
    records, groups = _records(2)
    records[1] = validate_sample_v5(make_sample_v5(sample_uid="s-0-1", patient_uid="synthetic:p0000",
                                                   leakage_group_uid="synthetic:other"))
    groups["s-0-1"] = "g1"
    with pytest.raises(V5Error) as err:
        freeze_splits_v5(records, groups, policy, "lin-a")
    assert (err.value.code, err.value.path) == ("LEAKAGE_GROUP_SPLIT", "$.records[1].patient_uid")
