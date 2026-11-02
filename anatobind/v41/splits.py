"""Frozen patient-group splits with exposure history.

Assignment is per leakage group, deterministic in (seed, group id), and a
group with any recorded exposure can never become locked_test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from anatobind.v41.identity import alias_report_rows, group_members
from anatobind.v41.paths import PrivatePathError, write_private_json
from anatobind.v41.schema import SampleRecord

ASSIGNABLE = ("train", "development_val", "locked_test")


def _exposed(records: list[SampleRecord]) -> bool:
    return any(record.exposure_history for record in records)


def validate_splits(records: list[SampleRecord], groups: dict[str, str]) -> dict:
    members = group_members(records, groups)
    cross = []
    exposed_locked = []
    for group, rows in members.items():
        splits = {r.split for r in rows} - {"unassigned", "excluded"}
        if len(splits) > 1:
            cross.append(group)
        if _exposed(rows) and "locked_test" in {r.split for r in rows}:
            exposed_locked.append(group)
    unassigned = sum(1 for r in records if r.split == "unassigned")
    return {
        "records": len(records),
        "groups": len(members),
        "cross_split_groups": len(cross),
        "cross_split_group_ids": sorted(cross),
        "exposed_groups_in_locked_test": len(exposed_locked),
        "exposed_group_ids_in_locked_test": sorted(exposed_locked),
        "unassigned_records": unassigned,
        "pass": not cross and not exposed_locked,
    }


def _bucket(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def freeze_splits(records: list[SampleRecord], policy: dict, out: Path, groups: dict[str, str] | None = None,
                  aliases: list[dict] | None = None) -> dict:
    from anatobind.v41.identity import resolve_groups

    aliases = aliases or []
    groups = groups or resolve_groups(records, aliases)
    seed = int(policy["seed"])
    fractions = policy["fractions"]
    if set(fractions) != set(ASSIGNABLE) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("SPLIT_POLICY_INVALID")
    members = group_members(records, groups)
    historical = sorted(g for g, rows in members.items() if _exposed(rows))
    preassigned = {}
    for group, rows in members.items():
        fixed = {r.split for r in rows} - {"unassigned", "excluded"}
        if len(fixed) > 1:
            raise ValueError("SPLIT_CONFLICT_EXISTING")
        if fixed:
            preassigned[group] = fixed.pop()
    bounds = []
    acc = 0.0
    for name in ASSIGNABLE:
        acc += fractions[name]
        bounds.append((acc, name))
    assignment_by_group = {}
    for group in sorted(members):
        if group in preassigned:
            assignment_by_group[group] = preassigned[group]
            continue
        value = _bucket(group, seed)
        name = next(n for upper, n in bounds if value <= upper or n == ASSIGNABLE[-1])
        if group in historical and name == "locked_test":
            name = "train" if policy.get("exposed_groups", "historical_only") == "historical_only" else "excluded"
        assignment_by_group[group] = name
    assignment = {}
    for group, rows in members.items():
        for record in rows:
            assignment[record.sample_uid] = "excluded" if record.split == "excluded" else assignment_by_group[group]
    body = {
        "policy": policy, "seed": seed, "groups": groups, "assignment": assignment,
        "historical_groups": historical, "preassigned_groups": preassigned,
    }
    body["split_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    frozen = [
        SampleRecord(**{**r.__dict__, "split": assignment[r.sample_uid]}) for r in records
    ]
    leakage = validate_splits(frozen, groups)
    body["leakage"] = leakage
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_private_json(out / "splits.json", body)
    write_private_json(out / "split_leakage_report.json", leakage)
    _write_csv(out / "patient_alias_report.csv", alias_report_rows(records, aliases, groups),
               ["a", "b", "kind", "status", "evidence", "group", "a_in_records", "b_in_records"])
    _write_jsonl(out / "exposure_ledger.jsonl", [
        {"sample_uid": r.sample_uid, "group": groups[r.sample_uid], **entry}
        for r in records for entry in r.exposure_history
    ])
    counts = {name: 0 for name in (*ASSIGNABLE, "excluded")}
    for group, name in assignment_by_group.items():
        counts[name] += 1
    return {"split_sha256": body["split_sha256"], "counts": counts, "historical_groups": len(historical),
            "leakage": leakage, "out": str(out.name)}


def _exclusive(path: Path):
    try:
        return os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w", newline="")
    except FileExistsError:
        raise PrivatePathError("PRIVATE_FILE_EXISTS") from None


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with _exclusive(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with _exclusive(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_records(path: Path) -> list[SampleRecord]:
    from anatobind.v41.schema import validate_sample

    with open(path) as handle:
        return [validate_sample(json.loads(line)) for line in handle if line.strip()]


def _cli_resolve(args) -> int:
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.identity import resolve_groups
    from anatobind.v41.paths import resolve_private_output

    records = _load_records(Path(args.records))
    aliases = json.loads(Path(args.aliases).read_text()) if args.aliases else []
    groups = resolve_groups(records, aliases)
    report = validate_splits(records, groups)
    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_private_json(out / "groups.json", groups)
    write_private_json(out / "split_leakage_report.json", report)
    _write_csv(out / "patient_alias_report.csv", alias_report_rows(records, aliases, groups),
               ["a", "b", "kind", "status", "evidence", "group", "a_in_records", "b_in_records"])
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_ids")}))
    return 0


def _cli_freeze(args) -> int:
    from anatobind.v41.cli import package_repo_root
    from anatobind.v41.paths import resolve_private_output

    records = _load_records(Path(args.records))
    aliases = json.loads(Path(args.aliases).read_text()) if args.aliases else []
    policy = json.loads(Path(args.policy).read_text())
    out = resolve_private_output(Path(args.out), Path(args.private_root), package_repo_root())
    result = freeze_splits(records, policy, out, aliases=aliases)
    print(json.dumps({"split_sha256": result["split_sha256"], "counts": result["counts"],
                      "pass": result["leakage"]["pass"]}))
    return 0


def register_cli(subparsers) -> None:
    resolve = subparsers.add_parser("resolve-identities")
    resolve.add_argument("--records", required=True)
    resolve.add_argument("--aliases")
    resolve.add_argument("--out", required=True)
    resolve.add_argument("--private-root", required=True)
    resolve.set_defaults(func=_cli_resolve)
    freeze = subparsers.add_parser("freeze-splits")
    freeze.add_argument("--records", required=True)
    freeze.add_argument("--aliases")
    freeze.add_argument("--policy", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--private-root", required=True)
    freeze.set_defaults(func=_cli_freeze)
