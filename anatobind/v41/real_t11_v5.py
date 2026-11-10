"""Real T11 import of the SKM-TEA released boxes for the early joint scope (plan_v5 T11; Proposal_v5 §5.0).

The driver joins three things that already exist: the joint scope's migrated records, its frozen split
registry, and the release annotation files. For every scan in a readable split it reads the scan's reference A
(the H5 `seg`, decoded as training decodes it) and imports its boxes with `import_skm_boxes_v5` under the frozen
box policy.

What it reads is decided by the frozen split, not by the caller:

- only `train` and `development_val` may be imported (researcher decision 2026-11-10). A scan in any other
  split is not read: neither its reference A nor its boxes. The release file of a split with no selected scan
  is never opened, so the official test annotations stay sealed until G-LOCKED;
- the registry must cover exactly the given records, and every record must belong to this scope and source.

The import builds reference candidates; it trains nothing and scores nothing, so it registers no exposure.

Everything is written under one new private directory (700, files 600, never overwritten):
`box_references.jsonl` (per scan, with identifiers), `import_report.json` and `box_import.md`. Only the
markdown, which holds counts and hashes, is meant to leave that directory. A scan that cannot be imported is
recorded as a failure (index, code, field path; no identifiers) and the run continues, so one report shows
every failure; G-DATA accepts the import only without failures.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .annotation_v5 import import_skm_boxes_v5, load_skm_box_policy_v5, skm_release_tables_v5, summarize_skm_import_v5
from .contracts_v5 import ActiveScope, SampleV5, V5Error
from .engine.data_v5 import decode_targets_v5, load_label_maps_v5
from .engine.volume_io_v5 import read_volume_v5
from .paths import write_private_json
from .source_policy_v5 import SourcePolicy, source_binding_v5
from .splits_v5 import verify_split_registry_v5

READABLE_SPLITS = ("train", "development_val")
SAMPLE_PREFIX = "skmtea:"
LABEL_MAPS_PATH = Path(__file__).resolve().parents[2] / "configs" / "v5" / "label_maps_v5.yaml"


def check_import_chain_v5(scope: ActiveScope, source_policy: SourcePolicy, box_policy: dict, records: list,
                          registry: dict, splits: tuple) -> None:
    """Scope, source binding, box policy, records and frozen split must be one chain before any file is read."""
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
    if not splits or not set(splits) <= set(READABLE_SPLITS):
        raise V5Error("SPLIT_SEALED", "$.splits")
    verify_split_registry_v5(registry)
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$.records[{index}]")
        if record.active_scope_id != scope.scope_id or record.reference_policy_id != scope.reference_policy_id:
            raise V5Error("RECORD_SCOPE_MISMATCH", f"$.records[{index}]")
        if (record.source_dataset, record.source_release) != (source_policy.source_dataset,
                                                              source_policy.source_release):
            raise V5Error("RECORD_SOURCE_MISMATCH", f"$.records[{index}]")
    if set(registry["assignment"]) != {record.sample_uid for record in records}:
        raise V5Error("SPLIT_RECORDS_MISMATCH", "$.registry.assignment")


class _Release:
    """The release annotation files, opened one split at a time and only when a selected scan needs them."""

    def __init__(self, directory: Path, box_policy: dict):
        self.directory, self.box_policy, self.loaded = Path(directory), box_policy, {}

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
             f"- 冻结划分 split_sha256：`{report['split_sha256']}`",
             "- 读取的划分：" + "、".join(f"{split} {n}" for split, n in report["selected"].items())
             + "；未读取：" + ("、".join(f"{split} {n}" for split, n in report["not_read"].items()) or "无"),
             f"- 导入扫描 {summary['scans']}，失败 {len(report['failures'])}；框总数 {summary['boxes']}", "",
             "## 目标框的逐级计数", "", "| 阶段 | 框数 |", "| --- | --- |"]
    lines += [f"| {stage} | {count} |" for stage, count in cascade.items()]
    lines += ["", "## 排除与范围外", "", "| 原因或类别 | 框数 |", "| --- | --- |"]
    lines += [f"| 排除：{reason} | {count} |" for reason, count in sorted(summary["excluded"].items())]
    lines += [f"| 范围外：{name} | {count} |" for name, count in summary["out_of_scope"].items()]
    lines += ["", "## 主分析阳性扫描数（按划分）", "", "| 划分 | 异常类别:组织组 | 扫描数 |", "| --- | --- | --- |"]
    for split, groups in sorted(summary["positive_scans_by_split"].items()):
        lines += [f"| {split} | {key} | {count} |" for key, count in sorted(groups.items())]
    lines += ["", "无主分析框的扫描数（annotation_unknown，不是阴性）："
              + ("、".join(f"{split} {n}" for split, n in sorted(summary["scans_without_primary_box"].items()))
                 or "0")]
    if report["failures"]:
        lines += ["", "## 失败（按错误码）", "", "| 错误码 | 扫描数 |", "| --- | --- |"]
        lines += [f"| {code} | {count} |" for code, count in sorted(Counter(f["code"] for f in report["failures"]).items())]
    return "\n".join(lines) + "\n"


def run_skm_import_v5(raw: Path, records: list, registry: dict, scope: ActiveScope, source_policy: SourcePolicy,
                      box_policy: dict, annotations_dir: Path, out: Path, *,
                      splits: tuple = READABLE_SPLITS, label_maps: dict | None = None) -> dict:
    check_import_chain_v5(scope, source_policy, box_policy, records, registry, tuple(splits))
    out = Path(out)
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    label_maps = label_maps if label_maps is not None else load_label_maps_v5(LABEL_MAPS_PATH)
    assignment = registry["assignment"]
    selected = sorted((r for r in records if assignment[r.sample_uid] in splits), key=lambda r: r.sample_uid)
    release = _Release(Path(annotations_dir), box_policy)
    results, result_splits, failures = [], [], []
    for index, record in enumerate(selected):
        try:
            release_split = (record.provenance or {}).get("release_split")
            if not isinstance(release_split, str) or not record.sample_uid.startswith(SAMPLE_PREFIX):
                raise V5Error("RELEASE_ENTRY_MISSING", "$.provenance.release_split")
            tables, annotations = release.scan(release_split, record.sample_uid[len(SAMPLE_PREFIX):])
            reference_a = _reference_a(Path(raw), record, scope, label_maps)
            shape = next(iter(reference_a.values())).shape
            results.append(import_skm_boxes_v5(sample_uid=record.sample_uid, shape=shape, annotations=annotations,
                                               tables=tables, reference_a=reference_a, policy=box_policy))
            result_splits.append(assignment[record.sample_uid])
        except V5Error as error:
            failures.append({"index": index, "code": error.code, "path": error.path})
    counts = Counter(assignment.values())
    report = {"scope_id": scope.scope_id, "scope_hash": scope.scope_hash, "source": source_binding_v5(source_policy),
              "policy_id": box_policy["policy_id"], "policy_sha256": box_policy["policy_sha256"],
              "split_sha256": registry["split_sha256"], "splits": list(splits),
              "selected": {split: counts.get(split, 0) for split in splits},
              "not_read": {split: n for split, n in sorted(counts.items()) if split not in splits},
              "failures": failures, "summary": summarize_skm_import_v5(results, result_splits)}
    markdown = render_box_import_md_v5(report)
    report["box_import_sha256"] = hashlib.sha256(markdown.encode()).hexdigest()
    fd = os.open(out / "box_references.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        for result, split in zip(results, result_splits):
            handle.write(json.dumps({**result, "split": split}, sort_keys=True) + "\n")
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
    for name in ("--raw", "--records", "--split-registry", "--scope", "--policies", "--source-id", "--box-policy",
                 "--annotations-dir", "--out"):
        command.add_argument(name, required=True)
    args = parser.parse_args(argv)
    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    report = run_skm_import_v5(
        Path(args.raw), records, json.loads(Path(args.split_registry).read_text()),
        validate_scope_v5(yaml.safe_load(Path(args.scope).read_text())),
        load_source_policies_v5(Path(args.policies))[args.source_id], load_skm_box_policy_v5(Path(args.box_policy)),
        Path(args.annotations_dir), Path(args.out))
    print(render_box_import_md_v5(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
