"""G-DATA evidence from the real T07/T08 outputs (Proposal_v5 §9; plan_v5 §2C T10 real-run prerequisite 2).

Each of the six G-DATA evidence items is judged by an explicit, conservative
rule over counts the real runs produced (no IDs):

- readability: every record was prepared (no failures);
- semantics: every released label passed value QC or is an all-zero mask
  (reported, kept as a complete negative), with no unexpected values; for
  SKM-TEA every scan's channel check confirmed the registered transform and
  channel order;
- identity: the frozen split covers every migrated record, grouped by patient;
- pairing: every record has its label;
- task_geometry: every record's geometry passed and its index alignment is
  verified after migration;
- provenance: migration (bound to the source release, L1) kept every record.

All six items carry the sha256 of one redacted markdown that states these
counts and the hashes of the source-check and split summaries. The data-use
approval is not produced here: the researcher signs it.
"""
from __future__ import annotations

import hashlib

from .adapters import skm
from .contracts_v5 import ActiveScope
from .gates_v5 import GATE_REQUIREMENTS, gate_task_v5
from .source_policy_v5 import SourcePolicy, source_binding_v5

_ITEMS = GATE_REQUIREMENTS["G-DATA"]["evidence"]


def _label_qc(label_qc: dict, source: str, records: int) -> tuple[bool, dict]:
    own = {k: v for k, v in label_qc.items() if k.split("|")[0] == source}
    allowed = all(k.endswith("|unexpected=False") and k.split("|")[1] in ("pass", "needs_review")
                  and (k.split("|")[1] == "pass" or k.split("|")[2] == "ALL_ZERO") for k in own)
    return allowed and sum(own.values()) == records, own


def gdata_evidence_v5(*, scope: ActiveScope, policy: SourcePolicy, prepare: dict, label_qc: dict, migrate: dict,
                      split: dict) -> tuple[dict, str]:
    records = prepare["records"]
    summary = prepare["summary"]
    pairing, channel = summary["pairing"], summary["channel"]
    label_ok, label_counts = _label_qc(label_qc, policy.source_dataset, records)
    if policy.source_dataset == skm.SOURCE or channel["cases_checked"]:
        channel_ok = channel["cases_checked"] == records and channel["consistent"] == records
    else:
        channel_ok = True
    passed = {
        "readability": not prepare["failures"] and records > 0,
        "semantics": label_ok and channel_ok,
        "identity": split["samples"] == migrate["records"] and split["groups"] > 0,
        "pairing": pairing.get("paired") == records and sum(pairing.values()) == records,
        "task_geometry": prepare["geometry_status"] == {"pass": records}
        and migrate["index_alignment"] == {"verified": migrate["records"]},
        "provenance": not migrate["failures"] and migrate["records"] == records,
    }
    binding = source_binding_v5(policy)
    lines = [f"# G-DATA 证据：{binding['source_id']}（{binding['source_version']}）→ {scope.scope_id}", "",
             "只含计数与哈希；不含病例 ID、路径或文件名。数据使用批准由研究者签署，不在本文件中。", "",
             f"- scope_hash：`{scope.scope_hash}`；任务：{gate_task_v5(scope)}",
             f"- 来源核验摘要 sha256：`{prepare['source_check_sha256']}`",
             f"- 冻结划分 split_sha256：`{split['split_sha256']}`", "",
             "| 证据项 | 判定 | 依据（计数） |", "| --- | --- | --- |",
             f"| readability | {'PASS' if passed['readability'] else 'FAIL'} | 准备记录 {records}，失败 "
             f"{len(prepare['failures'])} |",
             f"| semantics | {'PASS' if passed['semantics'] else 'FAIL'} | 标签取值 QC："
             + "；".join(f"{'/'.join(k.split('|')[1:])} {v}" for k, v in sorted(label_counts.items()))
             + f"；通道核验 {channel['consistent']}/{channel['cases_checked']} |",
             f"| identity | {'PASS' if passed['identity'] else 'FAIL'} | 划分覆盖样本 {split['samples']}/"
             f"{migrate['records']}，患者组 {split['groups']} |",
             f"| pairing | {'PASS' if passed['pairing'] else 'FAIL'} | "
             + "；".join(f"{k} {v}" for k, v in sorted(pairing.items())) + " |",
             f"| task_geometry | {'PASS' if passed['task_geometry'] else 'FAIL'} | 几何 "
             + "；".join(f"{k} {v}" for k, v in sorted(prepare["geometry_status"].items()))
             + "；索引对齐 " + "；".join(f"{k} {v}" for k, v in sorted(migrate["index_alignment"].items()))
             + "；物理几何 " + "；".join(f"{k} {v}" for k, v in sorted(migrate["physical_geometry"].items())) + " |",
             f"| provenance | {'PASS' if passed['provenance'] else 'FAIL'} | 迁移 {migrate['records']}/{records}，"
             f"失败 {len(migrate['failures'])}；release 绑定 `{policy.source_release}` |"]
    markdown = "\n".join(lines) + "\n"
    digest = hashlib.sha256(markdown.encode()).hexdigest()
    evidence = {key: {"status": "PASS" if passed[key] else "FAIL", "sha256": digest, "scope_hash": scope.scope_hash,
                      "task": gate_task_v5(scope), **binding} for key in _ITEMS}
    return evidence, markdown


def main(argv: list | None = None) -> int:
    import argparse
    import json
    import os
    from pathlib import Path

    import yaml

    from .contracts_v5 import validate_scope_v5
    from .paths import write_private_json
    from .source_policy_v5 import load_source_policies_v5

    parser = argparse.ArgumentParser(prog="anatobind.v41.real_gdata_v5")
    for name in ("--prepare-report", "--qc-summary", "--migrate-report", "--split-summary", "--scope", "--policies",
                 "--source-id", "--out"):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text())                 # noqa: E731
    evidence, markdown = gdata_evidence_v5(
        scope=validate_scope_v5(yaml.safe_load(Path(args.scope).read_text())),
        policy=load_source_policies_v5(Path(args.policies))[args.source_id],
        prepare=load(args.prepare_report), label_qc=load(args.qc_summary)["label_value_qc_by_source"],
        migrate=load(args.migrate_report), split=load(args.split_summary))
    out = Path(args.out)
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    write_private_json(out / "gdata_evidence.json", evidence)
    fd = os.open(out / "gdata_evidence.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
