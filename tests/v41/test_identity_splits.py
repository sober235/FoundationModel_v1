import json

import pytest

from anatobind.v41.identity import AliasError, alias_report_rows, resolve_groups
from anatobind.v41.schema import validate_sample
from anatobind.v41.splits import freeze_splits, validate_splits


def _records(make_sample, split_of_followup="train"):
    rows = [
        make_sample(sample_uid="s1-e1", patient_uid="src:p1", parent_scan_uid="src:p1:scan1", split="train"),
        make_sample(sample_uid="s1-e2", patient_uid="src:p1", parent_scan_uid="src:p1:scan1", split="train"),
        make_sample(sample_uid="s1-track-b", patient_uid="src:p1", parent_scan_uid="src:p1:scan1-rawtrack", split="train"),
        make_sample(sample_uid="s1-crop", patient_uid="src:p1", parent_scan_uid="src:p1:scan1", derivative_uid="crop-v1", split="train"),
        make_sample(sample_uid="s1-fu", patient_uid="src:p1-FU90", parent_scan_uid="src:p1-FU90:scan1", split=split_of_followup),
        make_sample(sample_uid="s9-other", patient_uid="other:q7", parent_scan_uid="other:q7:scan1", split="development_val"),
        make_sample(sample_uid="s2", patient_uid="src:p2", parent_scan_uid="src:p2:scan1", split="locked_test"),
    ]
    return [validate_sample(r) for r in rows]


@pytest.fixture
def related_records(make_sample):
    return _records(make_sample, split_of_followup="locked_test")


@pytest.fixture
def alias_edges():
    return [
        {"a": "src:p1", "b": "src:p1-FU90", "kind": "followup", "evidence": "official FU mapping table v1"},
        {"a": "src:p1", "b": "other:q7", "kind": "cross_source", "evidence": "curated cross-source mapping sheet"},
    ]


def test_lineage_split_lock(related_records, alias_edges):
    groups = resolve_groups(related_records, alias_edges)
    report = validate_splits(related_records, groups)
    assert report["cross_split_groups"] == 1
    assert report["pass"] is False


def test_related_records_share_one_group(related_records, alias_edges):
    groups = resolve_groups(related_records, alias_edges)
    assert len({groups[u] for u in ("s1-e1", "s1-e2", "s1-track-b", "s1-crop", "s1-fu", "s9-other")}) == 1
    assert groups["s2"] != groups["s1-e1"]
    assert groups["s1-e1"] == "group:other:q7"


def test_clean_split_passes(make_sample, alias_edges):
    records = _records(make_sample, split_of_followup="train")
    records = [r for r in records if r.sample_uid != "s9-other"]
    groups = resolve_groups(records, alias_edges)
    report = validate_splits(records, groups)
    assert report["cross_split_groups"] == 0
    assert report["pass"] is True
    assert report["groups"] == 2


def test_alias_without_evidence_is_rejected(related_records):
    with pytest.raises(AliasError, match="ALIAS_WITHOUT_EVIDENCE"):
        resolve_groups(related_records, [{"a": "src:p1", "b": "src:p2", "kind": "followup", "evidence": ""}])
    with pytest.raises(AliasError, match="ALIAS_UNKNOWN_KIND"):
        resolve_groups(related_records, [{"a": "src:p1", "b": "src:p2", "kind": "looks_similar", "evidence": "x"}])


def test_sibbms_cohorts_with_same_number_are_not_merged(make_sample):
    records = [
        validate_sample(make_sample(sample_uid="ms-1", patient_uid="sibbms:MS:sub-001", split="train")),
        validate_sample(make_sample(sample_uid="norm-1", patient_uid="sibbms:Norm:sub-001", split="locked_test")),
    ]
    groups = resolve_groups(records, [])
    assert groups["ms-1"] != groups["norm-1"]
    assert validate_splits(records, groups)["cross_split_groups"] == 0


def test_unresolved_alias_is_conservatively_merged_and_disclosed(make_sample):
    records = [
        validate_sample(make_sample(sample_uid="a", patient_uid="src:a", split="train")),
        validate_sample(make_sample(sample_uid="b", patient_uid="src:b", split="train")),
    ]
    edges = [{"a": "src:a", "b": "src:b", "kind": "cross_source", "evidence": "name collision only", "status": "unresolved"}]
    groups = resolve_groups(records, edges)
    assert groups["a"] == groups["b"]
    rows = alias_report_rows(records, edges, groups)
    assert any(r["status"] == "unresolved" for r in rows)


def test_exposed_record_cannot_be_called_unexposed_locked_test(make_sample):
    exposed = make_sample(
        sample_uid="x", patient_uid="src:x", split="locked_test",
        exposure_history=[{"experiment_id": "legacy_run_v5", "role": "test", "recorded_at": "2026-10-01T00:00:00Z"}],
    )
    records = [validate_sample(exposed)]
    report = validate_splits(records, resolve_groups(records, []))
    assert report["exposed_groups_in_locked_test"] == 1
    assert report["pass"] is False


def test_freeze_splits_is_deterministic_group_level_and_exposure_aware(make_sample, tmp_path):
    rows = []
    for i in range(30):
        for echo in (1, 2):
            rows.append(make_sample(sample_uid=f"p{i}-e{echo}", patient_uid=f"src:p{i}", parent_scan_uid=f"src:p{i}:scan"))
    rows[0]["exposure_history"] = [{"experiment_id": "legacy", "role": "train", "recorded_at": "2026-01-01T00:00:00Z"}]
    records = [validate_sample(r) for r in rows]
    policy = {"seed": 20261101, "fractions": {"train": 0.6, "development_val": 0.2, "locked_test": 0.2},
              "exposed_groups": "historical_only"}
    out = tmp_path / "splits_v1"
    result = freeze_splits(records, policy, out)
    again = freeze_splits(records, policy, tmp_path / "splits_v1_again")
    assert result["split_sha256"] == again["split_sha256"]
    frozen = json.loads((out / "splits.json").read_text())
    assignment = frozen["assignment"]
    assert set(assignment) == {r.sample_uid for r in records}
    for i in range(30):
        assert assignment[f"p{i}-e1"] == assignment[f"p{i}-e2"]
    assert assignment["p0-e1"] != "locked_test"
    assert frozen["historical_groups"] == ["group:src:p0"]
    assert result["counts"]["locked_test"] > 0
    assert result["leakage"]["pass"] is True
    with pytest.raises(ValueError, match="PRIVATE_FILE_EXISTS"):
        freeze_splits(records, policy, out)
    ledger = (out / "exposure_ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 1 and json.loads(ledger[0])["experiment_id"] == "legacy"
