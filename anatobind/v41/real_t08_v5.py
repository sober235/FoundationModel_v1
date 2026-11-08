"""Real T08 split freezing for v5.1 (plan_v5 §2B T08b; §1.6 L8; Proposal_v5 §3.3).

The split rule was registered by the researcher before this run (2026-11-08):
SKM-TEA keeps its official patient split (release train/val/test); ISLES-2022
is stratified 70/15/15 with a fixed seed, the strata being tertiles of the
patient lesion volume measured from the released masks, and strata with fewer
groups than the policy minimum are merged and reported. Groups are patients
(leakage groups). A group whose exposure ledger blocks the locked test
(e.g. the legacy ``unknown`` exposure) is moved to the policy's
``exposed_groups`` split by ``freeze_splits_v5``.

The frozen registry names samples and groups, so it stays in one new private
directory (700, files 600, never overwritten); only ``render_split_md_v5``'s
counts, strata sizes, thresholds and hash may leave it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .contracts_v5 import SampleV5, V5Error
from .paths import write_private_json
from .splits_v5 import MERGED_STRATUM, freeze_splits_v5

RELEASE_SPLITS = ("train", "val", "test")
TERTILES = ("volume_t1", "volume_t2", "volume_t3")


def official_split_from_release_v5(records: list) -> dict[str, str]:
    """{group: release split} from each record's ``provenance.release_split``; one split per group."""
    out: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, SampleV5):
            raise V5Error("INVALID_TYPE", f"$.records[{index}]")
        split = (record.provenance or {}).get("release_split")
        if split is None:
            continue
        path = f"$.records[{index}].provenance.release_split"
        if split not in RELEASE_SPLITS:
            raise V5Error("INVALID_RELEASE_SPLIT", path)
        if out.setdefault(record.leakage_group_uid, split) != split:
            raise V5Error("OFFICIAL_SPLIT_CONFLICT", path)
    return dict(sorted(out.items()))


def volume_tertile_strata_v5(volumes: dict[str, float]) -> tuple[dict[str, str], list[float]]:
    """Stratum per group from the tertiles of the group volumes; returns (strata, [q1/3, q2/3])."""
    values = list(volumes.values())
    if not values or any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in values):
        raise V5Error("INVALID_VOLUME", "$.volumes")
    q1, q2 = (float(q) for q in np.quantile(np.asarray(values, dtype=float), [1 / 3, 2 / 3]))
    strata = {g: TERTILES[0] if v <= q1 else TERTILES[1] if v <= q2 else TERTILES[2]
              for g, v in sorted(volumes.items())}
    return strata, [q1, q2]


def lesion_volumes_ml_v5(raw: Path, records: list, scope, label_maps: dict) -> dict[str, float]:
    """Patient lesion volume (mL): voxels of the scope's U classes in the released mask times voxel volume."""
    from .engine.data_v5 import decode_targets_v5
    from .engine.volume_io_v5 import read_volume_v5

    classes = list(scope.abnormality_classes)
    out: dict[str, float] = {}
    for index, record in enumerate(records):
        spacing = (record.image_geometry or {}).get("spacing_mm")
        if not spacing:
            raise V5Error("SPACING_UNKNOWN", f"$.records[{index}].image_geometry")
        voxels = 0
        for annotation in record.annotation_records:
            if annotation.task == "U":
                volume = read_volume_v5(raw, annotation.path, "label")
                voxels += int(decode_targets_v5(volume, annotation, classes, label_maps).any(axis=0).sum())
        out[record.leakage_group_uid] = out.get(record.leakage_group_uid, 0.0) + \
            voxels * float(np.prod(spacing)) / 1000.0
    return dict(sorted(out.items()))


def run_freeze_v5(records: list, policy: dict, lineage_id: str, out: Path, *,
                  volumes: dict[str, float] | None = None) -> dict:
    """Freeze one scope's split (no ancestors: B2 baselines start from random initialisation)."""
    out = Path(out)
    groups = {r.sample_uid: r.leakage_group_uid for r in records if isinstance(r, SampleV5)}
    policy = dict(policy)
    thresholds = strata_sizes = None
    if policy.get("kind") == "official":
        policy["official_split"] = official_split_from_release_v5(records)
    elif policy.get("kind") == "stratified":
        if volumes is None or set(volumes) != set(groups.values()):
            raise V5Error("STRATUM_VOLUMES_MISSING", "$.volumes")
        policy["strata"], thresholds = volume_tertile_strata_v5(volumes)
    registry = freeze_splits_v5(records, groups, policy, lineage_id, ancestor_lineages=[])
    if policy.get("kind") == "stratified":
        sizes: dict[str, int] = {}
        for stratum in policy["strata"].values():
            sizes[stratum] = sizes.get(stratum, 0) + 1
        minimum = policy["min_stratum_groups"]
        strata_sizes = {s: n for s, n in sorted(sizes.items()) if n >= minimum}
        merged = sum(n for n in sizes.values() if n < minimum)
        if merged:
            strata_sizes[MERGED_STRATUM] = merged
    out.mkdir(mode=0o700)                                                # FileExistsError: never overwritten
    os.chmod(out, 0o700)
    write_private_json(out / "split_registry.json", registry)
    summary = {"lineage_id": lineage_id, "kind": policy["kind"], "groups": len(set(groups.values())),
               "samples": len(groups), "counts": registry["counts"], "exposed_groups": registry["exposed_groups"],
               "exposed_target": policy["exposed_groups"], "strata": strata_sizes,
               "volume_thresholds_ml": thresholds, "split_sha256": registry["split_sha256"],
               "rule": {k: policy[k] for k in ("seed", "fractions", "min_stratum_groups", "small_stratum_rule")
                        if k in policy}}
    write_private_json(out / "split_summary.json", summary)
    return summary


def render_split_md_v5(summary: dict) -> str:
    lines = [f"# 冻结划分：{summary['lineage_id']}", "",
             "只含计数、规则与哈希；不含样本或患者 ID。", "",
             f"- 规则：{summary['kind']}；组（患者）数 {summary['groups']}，样本数 {summary['samples']}",
             f"- 被曝光阻断而移出 locked_test 的组：{summary['exposed_groups']}（移到 `{summary['exposed_target']}`）",
             f"- `split_sha256`：`{summary['split_sha256']}`", ""]
    for key, value in summary["rule"].items():
        lines.append(f"- {key}：`{value}`")
    lines += ["", "## 各划分组数", "", "| 划分 | 组数 |", "| --- | --- |"]
    lines += [f"| {split} | {n} |" for split, n in summary["counts"].items()]
    if summary["strata"] is not None:
        q = summary["volume_thresholds_ml"]
        lines += ["", "## 分层", "", f"病灶体积三分位阈值（mL）：{q[0]:.3f}、{q[1]:.3f}", "",
                  "| 层 | 组数 |", "| --- | --- |"]
        lines += [f"| {name} | {n} |" for name, n in summary["strata"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list | None = None) -> int:
    """Freeze one scope's split on the data server; prints the redacted summary only."""
    import argparse
    import json

    import yaml

    from .contracts_v5 import validate_sample_v5, validate_scope_v5
    from .engine.data_v5 import load_label_maps_v5

    parser = argparse.ArgumentParser(prog="anatobind.v41.real_t08_v5")
    parser.add_argument("--records", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--policy", required=True, help="YAML split rule without group lists")
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw")
    parser.add_argument("--label-maps", default=str(Path(__file__).resolve().parents[2] / "configs" / "v5"
                                                     / "label_maps_v5.yaml"))
    args = parser.parse_args(argv)
    scope = validate_scope_v5(yaml.safe_load(Path(args.scope).read_text()))
    records = [validate_sample_v5(json.loads(line)) for line in Path(args.records).read_text().splitlines() if line]
    policy = yaml.safe_load(Path(args.policy).read_text())
    volumes = None
    if policy.get("kind") == "stratified":
        volumes = lesion_volumes_ml_v5(Path(args.raw), records, scope, load_label_maps_v5(Path(args.label_maps)))
    summary = run_freeze_v5(records, policy, args.lineage_id, Path(args.out), volumes=volumes)
    print(render_split_md_v5(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
