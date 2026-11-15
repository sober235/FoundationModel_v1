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

Before any item is judged, the inputs must be one data chain (2026-11-09):
the policy is the scope's own (scope, reference policy, input modalities,
task); the prepare run's source-check summary is the one the policy
registered; the frozen registry is the split summary's and covers exactly the
migrated records; and every migrated record carries this scope and this
source release. A mismatch raises instead of judging, so counts from one
source can never become another scope's evidence.

The policy must register the annotation modalities of every task the scope has
(A, U, or both for a joint scope). A joint scope's U comes from released boxes
(Proposal_v5 §5.0), which no migrated record carries, so its evidence also
needs the real box import (real_t11_v5): the import must be of this scope,
source and box policy, cover exactly the migrated records (all official
splits) with every official release file present, and have no failure. The
official split is where a scan came from, not its formal split; the evidence
opens data use, and each later use is still limited by the frozen split's role.
"""
from __future__ import annotations

import hashlib

from .adapters import skm
from .contracts_v5 import ActiveScope, SampleV5, V5Error
from .gates_v5 import GATE_REQUIREMENTS, gate_task_v5
from .source_policy_v5 import SourcePolicy, source_binding_v5
from .splits_v5 import verify_split_registry_v5

_ITEMS = GATE_REQUIREMENTS["G-DATA"]["evidence"]


def _check_chain(scope: ActiveScope, policy: SourcePolicy, prepare: dict, migrate: dict, split: dict,
                 records: list, registry: dict) -> None:
    for field, expected, actual in (("scope_id", scope.scope_id, policy.scope_id),
                                    ("reference_policy_id", scope.reference_policy_id, policy.reference_policy_id),
                                    ("model_modalities", list(scope.input_modalities), list(policy.model_modalities))):
        if actual != expected:
            raise V5Error("SCOPE_POLICY_MISMATCH", f"$.policy.{field}")
    for task in _scope_tasks(scope):
        if not policy.annotation_modalities.get(task):
            raise V5Error("SCOPE_POLICY_MISMATCH", f"$.policy.annotation_modalities.{task}")
    if not policy.index_evidence or str(prepare.get("source_check_sha256")) not in policy.index_evidence:
        raise V5Error("SOURCE_CHECK_NOT_REGISTERED", "$.prepare.source_check_sha256")
    verify_split_registry_v5(registry)
    if registry["split_sha256"] != split.get("split_sha256"):
        raise V5Error("SPLIT_HASH_MISMATCH", "$.split.split_sha256")
    if not isinstance(records, list) or len(records) != migrate.get("records"):
        raise V5Error("RECORDS_MISMATCH", "$.records")
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$.records[{index}]")
        for field, expected in (("active_scope_id", scope.scope_id),
                                ("reference_policy_id", scope.reference_policy_id),
                                ("model_modalities", list(scope.input_modalities))):
            if getattr(record, field) != expected:
                raise V5Error("RECORD_SCOPE_MISMATCH", f"$.records[{index}].{field}")
        for field, expected in (("source_dataset", policy.source_dataset), ("source_release", policy.source_release)):
            if getattr(record, field) != expected:
                raise V5Error("RECORD_SOURCE_MISMATCH", f"$.records[{index}].{field}")
    if set(registry["assignment"]) != {record.sample_uid for record in records}:
        raise V5Error("SPLIT_RECORDS_MISMATCH", "$.registry.assignment")


def _scope_tasks(scope: ActiveScope) -> list:
    return [task for task, classes in (("A", scope.anatomy_classes), ("U", scope.abnormality_classes)) if classes]


def _check_box_import(scope: ActiveScope, policy: SourcePolicy, records: list, box_import: dict | None) -> None:
    """A joint scope's box import must belong to this chain; a module scope must not bring one."""
    if scope.scope_kind != "joint":
        if box_import is not None:
            raise V5Error("BOX_IMPORT_NOT_EXPECTED", "$.box_import")
        return
    if not isinstance(box_import, dict):
        raise V5Error("BOX_IMPORT_MISSING", "$.box_import")
    from .real_t11_v5 import records_sha256_v5

    for field, expected in (("scope_hash", scope.scope_hash), ("source", source_binding_v5(policy)),
                            ("policy_id", scope.reference_policy_id), ("records", len(records)),
                            ("records_sha256", records_sha256_v5(records))):
        if box_import.get(field) != expected:
            raise V5Error("BOX_IMPORT_MISMATCH", f"$.box_import.{field}")
    for field in ("policy_sha256", "box_import_sha256"):
        value = box_import.get(field)
        if not (isinstance(value, str) and len(value) == 64):
            raise V5Error("BOX_IMPORT_MISMATCH", f"$.box_import.{field}")
    if not isinstance(box_import.get("release_files"), dict) or not isinstance(box_import.get("by_release_split"), dict):
        raise V5Error("BOX_IMPORT_MISMATCH", "$.box_import.release_files")


def _label_qc(label_qc: dict, source: str, records: int) -> tuple[bool, dict]:
    own = {k: v for k, v in label_qc.items() if k.split("|")[0] == source}
    allowed = all(k.endswith("|unexpected=False") and k.split("|")[1] in ("pass", "needs_review")
                  and (k.split("|")[1] == "pass" or k.split("|")[2] == "ALL_ZERO") for k in own)
    return allowed and sum(own.values()) == records, own


def gdata_evidence_v5(*, scope: ActiveScope, policy: SourcePolicy, prepare: dict, label_qc: dict, migrate: dict,
                      split: dict, records: list, registry: dict, box_import: dict | None = None) -> tuple[dict, str]:
    """`records` are the migrated SampleV5 records and `registry` the frozen split registry (both private); they
    only bind the chain and never enter the markdown, which keeps its counts-only form. `box_import` is the
    real_t11_v5 import report, required for a joint scope and refused for a module scope."""
    _check_chain(scope, policy, prepare, migrate, split, records, registry)
    _check_box_import(scope, policy, records, box_import)
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
    if box_import is not None:
        imported = box_import["summary"]["scans"]
        files_ok = bool(box_import["release_files"]) and all(box_import["release_files"].values())
        passed["semantics"] = passed["semantics"] and not box_import["failures"] and imported > 0
        passed["pairing"] = passed["pairing"] and imported == box_import["records"] and files_ok
        passed["provenance"] = passed["provenance"] and files_ok and not (
            set(box_import["by_release_split"]) & {"conflict", "missing"})
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
    if box_import is not None:
        by_release = "、".join(f"{k} {v}" for k, v in box_import["by_release_split"].items())
        files = "、".join(f"{k} {'存在' if v else '缺失'}" for k, v in box_import["release_files"].items())
        lines += ["", "## 框导入（U 为框级，R 为组织级；并入 semantics、pairing 与 provenance 的判定）", "",
                  f"- 框导入摘要 sha256：`{box_import['box_import_sha256']}`",
                  f"- 框策略 `{box_import['policy_id']}`，policy_sha256：`{box_import['policy_sha256']}`",
                  f"- 官方划分文件：{files}；登记扫描按来源划分：{by_release}",
                  f"- 登记扫描 {box_import['records']}，导入 {box_import['summary']['scans']}，失败 "
                  f"{len(box_import['failures'])}",
                  "- 来源划分不是正式划分。本证据放行数据使用；训练、模型选择和评价仍按冻结划分的角色读取。"]
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

    from .contracts_v5 import validate_sample_v5, validate_scope_v5
    from .paths import write_private_json
    from .source_policy_v5 import load_source_policies_v5

    parser = argparse.ArgumentParser(prog="anatobind.v41.real_gdata_v5")
    for name in ("--prepare-report", "--qc-summary", "--migrate-report", "--split-summary", "--records",
                 "--split-registry", "--scope", "--policies", "--source-id", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--box-import", help="real_t11_v5 import_report.json; required for a joint scope")
    args = parser.parse_args(argv)
    load = lambda path: json.loads(Path(path).read_text())                 # noqa: E731
    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    evidence, markdown = gdata_evidence_v5(
        scope=validate_scope_v5(yaml.safe_load(Path(args.scope).read_text())),
        policy=load_source_policies_v5(Path(args.policies))[args.source_id],
        prepare=load(args.prepare_report), label_qc=load(args.qc_summary)["label_value_qc_by_source"],
        migrate=load(args.migrate_report), split=load(args.split_summary), records=records,
        registry=load(args.split_registry), box_import=load(args.box_import) if args.box_import else None)
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
