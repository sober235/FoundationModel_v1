"""v5.1 exposure ledger roles and locked-test blockers (Proposal_v5 §3.3).

Roles say how a sample has been seen. A lineage is the line of models one
locked test is meant to judge. Seeing test results (or an unknown exposure)
is a research-selection exposure and blocks every lineage; training-type
exposure blocks only its own lineage, or every lineage when the entry does
not name one (a blank lineage counts as not named); administrative QC never blocks.
"""
from __future__ import annotations

from .contracts_v5 import V5Error

EXPOSURE_ROLES_V5 = ("training", "unlabeled_pretraining", "model_selection", "test_results_seen",
                     "administrative_qc", "unknown")
V41_ROLE_MAP = {"train": "training", "val": "model_selection", "test": "test_results_seen",
                "ssl": "unlabeled_pretraining", "pseudo_label": "training", "unknown": "unknown"}
GLOBAL_BLOCKING_ROLES = ("test_results_seen", "unknown")
LINEAGE_BLOCKING_ROLES = ("training", "unlabeled_pretraining", "model_selection")
MIGRATION_EVIDENCE = "migrated_from_v41"


def migrate_exposure_v41(entry: dict, path: str) -> dict:
    role = entry.get("role")
    if role not in V41_ROLE_MAP:
        raise V5Error("INVALID_EXPOSURE_ROLE", f"{path}.role")
    return {"experiment_id": entry["experiment_id"], "role": V41_ROLE_MAP[role],
            "recorded_at": entry["recorded_at"], "lineage_id": None, "evidence": MIGRATION_EVIDENCE}


def locked_test_blockers_v5(exposure_history: list[dict], lineage_id: str) -> list[str]:
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise V5Error("INVALID_LINEAGE", "$.lineage_id")
    if not isinstance(exposure_history, list):
        raise V5Error("INVALID_TYPE", "$.exposure_history")
    blockers = set()
    for index, entry in enumerate(exposure_history):
        path = f"$.exposure_history[{index}]"
        if not isinstance(entry, dict):
            raise V5Error("INVALID_TYPE", path)
        role = entry.get("role")
        if role not in EXPOSURE_ROLES_V5:
            raise V5Error("INVALID_EXPOSURE_ROLE", f"{path}.role")
        entry_lineage = entry.get("lineage_id")
        if entry_lineage is not None and not isinstance(entry_lineage, str):
            raise V5Error("INVALID_LINEAGE", f"{path}.lineage_id")
        if role in GLOBAL_BLOCKING_ROLES:
            blockers.add(role)
        elif role in LINEAGE_BLOCKING_ROLES and (not entry_lineage or not entry_lineage.strip()
                                                 or entry_lineage.strip() == lineage_id.strip()):
            blockers.add(role)
    return sorted(blockers)


def lineage_blockers_v5(exposure_history: list[dict], lineage_id: str, ancestor_lineages: list[str]) -> list[str]:
    """Locked-test blockers for a lineage and every lineage its weights descend from (Proposal_v5 §3.3).

    A model initialised from a checkpoint inherits that checkpoint's exposure; a new lineage ID does not clear it.
    """
    locked_test_blockers_v5([], lineage_id)
    if not isinstance(ancestor_lineages, list) or any(
            not isinstance(name, str) or not name.strip() for name in ancestor_lineages):
        raise V5Error("INVALID_LINEAGE", "$.ancestor_lineages")
    names = [name.strip() for name in ancestor_lineages]
    if len(set(names)) != len(names) or lineage_id.strip() in names:
        raise V5Error("INVALID_LINEAGE", "$.ancestor_lineages")
    blockers = set(locked_test_blockers_v5(exposure_history, lineage_id))
    for name in names:
        blockers.update(locked_test_blockers_v5(exposure_history, name))
    return sorted(blockers)
