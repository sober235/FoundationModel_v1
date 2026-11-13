"""Real T11 import of the SKM-TEA released boxes for the early joint scope (plan_v5 T11; Proposal_v5 §5.0).

The driver joins two things that already exist: the joint scope's migrated records and the release annotation
files. For every registered scan it reads the scan's reference A (the H5 `seg`, decoded as training decodes it)
and imports its boxes with `import_skm_boxes_v5` under the frozen box policy.

Import covers all three official splits (researcher decision 2026-11-10, replacing the same day's earlier
"train and development_val only"):

- every registered scan is imported, whatever its official split; each result keeps the official split
  (`release_split`, from `provenance.release_split`) as where the data came from;
- the official split is not the formal split. Import assigns no formal split and needs no frozen registry:
  a record's `split` stays what registration gave it (`unassigned`). Training, model selection and evaluation
  later read these references only through the frozen split of their run, by role;
- all three release files are checked first. A missing file, a scan whose official split is unresolved
  (conflict or absent) and a scan missing from its file are recorded as failures; none becomes "no box".

Import is contact, not exposure. It builds reference candidates under a policy frozen before the run, trains
nothing and scores nothing. The contact of every scan is written to a separate ledger with the role
`administrative_qc` (Proposal_v5 §3.3), apart from the exposure a training or evaluation run registers. The
counts of the official test split become known to whoever reads the summary: changing the box policy or any
rule after reading them is no longer administrative contact and must be recorded as a research exposure.

Everything is written under one new private directory (700, files 600, never overwritten):
`box_references.jsonl` (per scan, with identifiers), `contact_ledger.jsonl`, `import_report.json` and
`box_import.md`. Only the markdown, which holds counts and hashes, is meant to leave that directory. A scan
that cannot be imported is recorded as a failure (index, code, field path; no identifiers) and the run
continues, so one report shows every failure; G-DATA accepts the import only without failures.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .annotation_v5 import import_skm_boxes_v5, load_skm_box_policy_v5, skm_release_tables_v5, summarize_skm_import_v5
from .contracts_v5 import ActiveScope, SampleV5, V5Error, canonical_sha256
from .engine.data_v5 import decode_targets_v5, load_label_maps_v5
from .engine.volume_io_v5 import read_volume_v5
from .paths import write_private_json
from .source_policy_v5 import SourcePolicy, source_binding_v5

RELEASE_SPLITS = ("train", "val", "test")
CONTACT_ROLE = "administrative_qc"
CONTACT_EXPERIMENT = "t11_skm_box_import"
CONTACT_EVIDENCE = ("reference import: released boxes and reference A read to build reference candidates under "
                    "the frozen box policy; no model, no training, no scoring")
SAMPLE_PREFIX = "skmtea:"
LABEL_MAPS_PATH = Path(__file__).resolve().parents[2] / "configs" / "v5" / "label_maps_v5.yaml"


def records_sha256_v5(records: list) -> str:
    """Digest of the set of registered scans an import covered (sorted sample uids)."""
    return canonical_sha256(sorted(record.sample_uid for record in records))


def check_import_chain_v5(scope: ActiveScope, source_policy: SourcePolicy, box_policy: dict, records: list) -> None:
    """Scope, source binding, box policy and records must be one chain before any file is read."""
    if scope.scope_kind != "joint":
        raise V5Error("SCOPE_TASK_MISMATCH", "$.scope.scope_kind")
    for field, expected, actual in (
            ("scope_id", scope.scope_id, source_policy.scope_id),
            ("reference_policy_id", scope.reference_policy_id, source_policy.reference_policy_id),
            ("model_modalities", list(scope.input_modalities), list(source_policy.model_modalities))):
        if actual != expected:
            raise V5Error("SCOPE_POLICY_MISMATCH", f"$.policy.{field}")
    if box_policy["policy_id"] != scope.reference_policy_id:
        raise V5Error("SCOPE_POLICY_MISMATCH", "$.box_policy.policy_id")
    if box_policy["source"]["source_version"] != source_policy.source_version:
        raise V5Error("SCOPE_POLICY_MISMATCH", "$.box_policy.source.source_version")
    if set(box_policy["tissue_groups_by_class"]) != set(scope.abnormality_classes):
        raise V5Error("SCOPE_POLICY_MISMATCH", "$.box_policy.tissue_groups_by_class")
    if not isinstance(records, list) or not records:
        raise V5Error("INVALID_TYPE", "$.records")
    seen: set = set()
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$.records[{index}]")
        if record.sample_uid in seen:
            raise V5Error("DUPLICATE_SAMPLE", f"$.records[{index}]")
        seen.add(record.sample_uid)
        if record.active_scope_id != scope.scope_id or record.reference_policy_id != scope.reference_policy_id:
            raise V5Error("RECORD_SCOPE_MISMATCH", f"$.records[{index}]")
        if (record.source_dataset, record.source_release) != (source_policy.source_dataset,
                                                              source_policy.source_release):
            raise V5Error("RECORD_SOURCE_MISMATCH", f"$.records[{index}]")


class _Release:
    """The three official release annotation files; each is checked first and opened when a scan needs it."""

    def __init__(self, directory: Path, box_policy: dict):
        self.directory, self.box_policy, self.loaded = Path(directory), box_policy, {}
        self.present = {split: (self.directory / f"{split}.json").is_file() for split in RELEASE_SPLITS}

    def scan(self, release_split: str, case: str) -> tuple[dict, list]:
        if release_split not in self.loaded:
            path = self.directory / f"{release_split}.json"
            if not path.is_file():
                raise V5Error("RELEASE_FILE_MISSING", "$.provenance.release_split")
            try:
                doc = json.loads(path.read_text())
            except ValueError:
                raise V5Error("RELEASE_FILE_UNREADABLE", "$.provenance.release_split") from None
            tables = skm_release_tables_v5(doc, self.box_policy)
            images = {Path(str(image.get("file_name"))).name[:-len(".h5")]: image.get("id")
                      for image in doc.get("images", []) if str(image.get("file_name")).endswith(".h5")}
            boxes: dict = {}
            for annotation in doc.get("annotations", []):
                boxes.setdefault(annotation.get("image_id"), []).append(annotation)
            self.loaded[release_split] = (tables, images, boxes)
        tables, images, boxes = self.loaded[release_split]
        if case not in images:
            raise V5Error("RELEASE_ENTRY_MISSING", "$.sample_uid")
        return tables, boxes.get(images[case], [])


def _reference_a(raw: Path, record: SampleV5, scope: ActiveScope, label_maps: dict) -> dict:
    anatomy = [a for a in record.annotation_records if a.task == "A"]
    if len(anatomy) != 1:
        raise V5Error("REFERENCE_A_MISSING", "$.annotation_records")
    classes = list(scope.anatomy_classes)
    targets = decode_targets_v5(read_volume_v5(raw, anatomy[0].path, "label"), anatomy[0], classes, label_maps)
    return {name: targets[k] for k, name in enumerate(classes)}


def render_box_import_md_v5(report: dict) -> str:
    summary, cascade = report["summary"], report["summary"]["cascade"]
    lines = [f"# 框导入摘要：{report['source']['source_id']}（{report['source']['source_version']}）→ "
             f"{report['scope_id']}", "",
             "只含计数与哈希；不含病例 ID、路径或文件名。框保持框级，不转成体素真值；无框记 annotation_unknown，"
             "不作阴性。本摘要不是评价结果，`scientific_conclusion: NOT_EVALUATED`。", "",
             f"- scope_hash：`{report['scope_hash']}`",
             f"- 框策略 `{report['policy_id']}`，policy_sha256：`{report['policy_sha256']}`",
             f"- 登记记录集 records_sha256：`{report['records_sha256']}`",
             "- 官方划分文件：" + "、".join(f"{split} {'存在' if ok else '缺失'}"
                                      for split, ok in report["release_files"].items()),
             "- 登记扫描按来源划分：" + "、".join(f"{split} {n}" for split, n in report["by_release_split"].items()),
             f"- 登记扫描 {report['records']}，导入 {summary['scans']}，失败 {len(report['failures'])}；"
             f"框总数 {summary['boxes']}",
             "- 来源划分只记录数据出处，不是正式划分；导入不分配正式划分。训练、模型选择和评价只能按冻结划分的角色"
             "读取这些参照。",
             f"- 每个导入扫描的接触另记为 `{report['contact_role']}`（{summary['scans']} 条），与训练、模型选择和"
             "查看测试结果的曝光分开。官方 test 的计数已在本摘要中可见：此后据此修改框策略或规则属于研究性曝光，"
             "须另行登记。", "",
             "## 目标框的逐级计数", "", "| 阶段 | 框数 |", "| --- | --- |"]
    lines += [f"| {stage} | {count} |" for stage, count in cascade.items()]
    lines += ["", "## 排除与范围外", "", "| 原因或类别 | 框数 |", "| --- | --- |"]
    lines += [f"| 排除：{reason} | {count} |" for reason, count in sorted(summary["excluded"].items())]
    lines += [f"| 范围外：{name} | {count} |" for name, count in summary["out_of_scope"].items()]
    lines += ["", "## 主分析阳性扫描数（按来源划分）", "", "| 来源划分 | 异常类别:组织组 | 扫描数 |",
              "| --- | --- | --- |"]
    for split, groups in sorted(summary["positive_scans_by_split"].items()):
        lines += [f"| {split} | {key} | {count} |" for key, count in sorted(groups.items())]
    lines += ["", "无主分析框的扫描数（annotation_unknown，不是阴性）："
              + ("、".join(f"{split} {n}" for split, n in sorted(summary["scans_without_primary_box"].items()))
                 or "0")]
    if report["failures"]:
        lines += ["", "## 失败（按错误码）", "", "| 错误码 | 扫描数 |", "| --- | --- |"]
        lines += [f"| {code} | {count} |" for code, count in sorted(Counter(f["code"] for f in report["failures"]).items())]
    return "\n".join(lines) + "\n"


def run_skm_import_v5(raw: Path, records: list, scope: ActiveScope, source_policy: SourcePolicy, box_policy: dict,
                      annotations_dir: Path, out: Path, *, recorded_at: str, label_maps: dict | None = None) -> dict:
    check_import_chain_v5(scope, source_policy, box_policy, records)
    if not isinstance(recorded_at, str) or not recorded_at.strip():
        raise V5Error("INVALID_TYPE", "$.recorded_at")
    out = Path(out)
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    label_maps = label_maps if label_maps is not None else load_label_maps_v5(LABEL_MAPS_PATH)
    ordered = sorted(records, key=lambda r: r.sample_uid)
    release = _Release(Path(annotations_dir), box_policy)
    results, result_splits, failures = [], [], []
    by_release: Counter = Counter()
    for index, record in enumerate(ordered):
        provenance = record.provenance or {}
        release_split = provenance.get("release_split")
        by_release[release_split if release_split in RELEASE_SPLITS else
                   "conflict" if "release_split_conflict" in provenance else "missing"] += 1
        try:
            if release_split not in RELEASE_SPLITS or not record.sample_uid.startswith(SAMPLE_PREFIX):
                raise V5Error("RELEASE_SPLIT_UNRESOLVED", "$.provenance.release_split")
            tables, annotations = release.scan(release_split, record.sample_uid[len(SAMPLE_PREFIX):])
            reference_a = _reference_a(Path(raw), record, scope, label_maps)
            shape = next(iter(reference_a.values())).shape
            results.append(import_skm_boxes_v5(sample_uid=record.sample_uid, shape=shape, annotations=annotations,
                                               tables=tables, reference_a=reference_a, policy=box_policy))
            result_splits.append(release_split)
        except V5Error as error:
            failures.append({"index": index, "code": error.code, "path": error.path})
    report = {"scope_id": scope.scope_id, "scope_hash": scope.scope_hash, "source": source_binding_v5(source_policy),
              "policy_id": box_policy["policy_id"], "policy_sha256": box_policy["policy_sha256"],
              "records": len(ordered), "records_sha256": records_sha256_v5(ordered),
              "release_files": dict(release.present),
              "by_release_split": {key: by_release[key] for key in (*RELEASE_SPLITS, "conflict", "missing")
                                   if by_release[key] or key in RELEASE_SPLITS},
              "formal_split": dict(sorted(Counter(r.split for r in ordered).items())),
              "contact_role": CONTACT_ROLE, "failures": failures,
              "summary": summarize_skm_import_v5(results, result_splits)}
    markdown = render_box_import_md_v5(report)
    report["box_import_sha256"] = hashlib.sha256(markdown.encode()).hexdigest()
    by_uid = {record.sample_uid: record for record in ordered}
    fd = os.open(out / "box_references.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        for result, split in zip(results, result_splits):
            handle.write(json.dumps({**result, "release_split": split, "split": by_uid[result["sample_uid"]].split},
                                    sort_keys=True) + "\n")
    fd = os.open(out / "contact_ledger.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        for result in results:
            handle.write(json.dumps({"sample_uid": result["sample_uid"], "experiment_id": CONTACT_EXPERIMENT,
                                     "role": CONTACT_ROLE, "recorded_at": recorded_at, "lineage_id": None,
                                     "evidence": CONTACT_EVIDENCE}, sort_keys=True) + "\n")
    fd = os.open(out / "box_import.md", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(markdown)
    write_private_json(out / "import_report.json", report)
    return report


def main(argv: list | None = None) -> int:
    """Run on the data server; prints the redacted markdown only."""
    import argparse

    import yaml

    from .contracts_v5 import validate_sample_v5, validate_scope_v5
    from .source_policy_v5 import load_source_policies_v5

    parser = argparse.ArgumentParser(prog="anatobind.v41.real_t11_v5")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("import-skm")
    for name in ("--raw", "--records", "--scope", "--policies", "--source-id", "--box-policy", "--annotations-dir",
                 "--recorded-at", "--out"):
        command.add_argument(name, required=True)
    args = parser.parse_args(argv)
    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    report = run_skm_import_v5(
        Path(args.raw), records, validate_scope_v5(yaml.safe_load(Path(args.scope).read_text())),
        load_source_policies_v5(Path(args.policies))[args.source_id], load_skm_box_policy_v5(Path(args.box_policy)),
        Path(args.annotations_dir), Path(args.out), recorded_at=args.recorded_at)
    print(render_box_import_md_v5(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
