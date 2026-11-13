from __future__ import annotations

import pytest

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5, validate_scope_v5
from anatobind.v41.exposure_v5 import EXPOSURE_ROLES_V5, V41_ROLE_MAP, lineage_blockers_v5, locked_test_blockers_v5
from anatobind.v41.migrate_v5 import migrate_v41_record_v5
from anatobind.v41.schema import validate_sample
from tests.v41.conftest import _sample
from tests.v5.factories import make_module_scope, make_sample_v5, make_source_policy


def _entry(role, lineage_id=None, experiment_id="exp-1"):
    return {"experiment_id": experiment_id, "role": role, "recorded_at": "2026-11-06T00:00:00Z",
            "lineage_id": lineage_id, "evidence": None}


def test_v5_roles_validate_and_v41_roles_rejected():
    history = [_entry(role, "lin-a") for role in EXPOSURE_ROLES_V5]
    sample = validate_sample_v5(make_sample_v5(exposure_history=history))
    assert [e["role"] for e in sample.exposure_history] == list(EXPOSURE_ROLES_V5)
    for v41_role in ("train", "val", "test", "ssl", "pseudo_label"):
        with pytest.raises(V5Error) as err:
            validate_sample_v5(make_sample_v5(exposure_history=[_entry(v41_role)]))
        assert err.value.path == "$.exposure_history[0].role"
    legacy = {"experiment_id": "exp-1", "role": "unknown", "recorded_at": "2026-11-06T00:00:00Z"}
    with pytest.raises(V5Error):
        validate_sample_v5(make_sample_v5(exposure_history=[legacy]))


def test_v41_roles_are_mapped_on_migration():
    v41_roles = ["train", "val", "test", "ssl", "pseudo_label", "unknown"]
    body = _sample()
    body["exposure_history"] = [{"experiment_id": f"exp-{i}", "role": role, "recorded_at": "2026-11-02T00:00:00Z"}
                                for i, role in enumerate(v41_roles)]
    migrated = migrate_v41_record_v5(validate_sample(body), make_source_policy(),
                                     validate_scope_v5(make_module_scope("U", input_modalities=["qdess_e1"])))
    assert [e["role"] for e in migrated.exposure_history] == [V41_ROLE_MAP[r] for r in v41_roles]
    assert [e["role"] for e in migrated.exposure_history] == [
        "training", "model_selection", "test_results_seen", "unlabeled_pretraining", "training", "unknown"]
    for i, entry in enumerate(migrated.exposure_history):
        assert entry == {"experiment_id": f"exp-{i}", "role": entry["role"], "recorded_at": "2026-11-02T00:00:00Z",
                         "lineage_id": None, "evidence": "migrated_from_v41"}


def test_administrative_qc_never_blocks():
    history = [_entry("administrative_qc", "lin-a"), _entry("administrative_qc", None)]
    assert locked_test_blockers_v5(history, "lin-a") == []
    assert locked_test_blockers_v5(history, "lin-b") == []


def test_unknown_and_seen_results_block_every_lineage():
    history = [_entry("unknown", "lin-other"), _entry("test_results_seen", "lin-other"), _entry("unknown", None)]
    assert locked_test_blockers_v5(history, "lin-a") == ["test_results_seen", "unknown"]
    assert locked_test_blockers_v5(history, "lin-other") == ["test_results_seen", "unknown"]


def test_training_blocks_only_its_lineage_or_null():
    other = [_entry(role, "lin-other") for role in ("training", "unlabeled_pretraining", "model_selection")]
    assert locked_test_blockers_v5(other, "lin-a") == []
    assert locked_test_blockers_v5(other, "lin-other") == ["model_selection", "training", "unlabeled_pretraining"]
    assert locked_test_blockers_v5([_entry("model_selection", None)], "lin-a") == ["model_selection"]


def test_blockers_reject_bad_inputs():
    with pytest.raises(V5Error) as err:
        locked_test_blockers_v5([_entry("train", "lin-a")], "lin-a")
    assert (err.value.code, err.value.path) == ("INVALID_EXPOSURE_ROLE", "$.exposure_history[0].role")
    for lineage in ("", None, 3):
        with pytest.raises(V5Error) as err:
            locked_test_blockers_v5([], lineage)
        assert (err.value.code, err.value.path) == ("INVALID_LINEAGE", "$.lineage_id")


def test_blank_lineage_entry_blocks_like_null():
    for blank in ("", "   "):
        assert locked_test_blockers_v5([_entry("training", blank)], "lin-a") == ["training"]
    assert locked_test_blockers_v5([_entry("training", " lin-a ")], "lin-a") == ["training"]
    with pytest.raises(V5Error) as err:
        locked_test_blockers_v5([_entry("training", 3)], "lin-a")
    assert (err.value.code, err.value.path) == ("INVALID_LINEAGE", "$.exposure_history[0].lineage_id")


def test_renamed_lineage_still_blocked_by_ancestor_exposure():
    history = [_entry("training", "lin-a")]
    assert locked_test_blockers_v5(history, "lin-b") == []
    assert lineage_blockers_v5(history, "lin-b", ["lin-a"]) == ["training"]
    assert lineage_blockers_v5(history, "lin-b", []) == []
    assert lineage_blockers_v5([_entry("test_results_seen", "lin-z")], "lin-b", ["lin-a"]) == ["test_results_seen"]
    for bad in ([""], [" "], ["lin-a", "lin-a"], ["lin-a", " lin-a"], ["lin-b"], [None], "lin-a"):
        with pytest.raises(V5Error) as err:
            lineage_blockers_v5(history, "lin-b", bad)
        assert (err.value.code, err.value.path) == ("INVALID_LINEAGE", "$.ancestor_lineages")
    with pytest.raises(V5Error) as err:
        lineage_blockers_v5(history, " ", ["lin-a"])
    assert (err.value.code, err.value.path) == ("INVALID_LINEAGE", "$.lineage_id")
