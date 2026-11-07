"""Redacted source-check summaries for v5.1 (plan_v5 T07c).

Takes the rows that v41 ``geometry_qc.run_geometry_qc`` produces, plus
``downsample_loss_report_v5`` reports, and reduces them to counts for one
source. Case IDs, paths, file names and track names never reach the output;
unrecognised reason strings are counted as ``OTHER`` rather than echoed.
"""
from __future__ import annotations

from .contracts_v5 import V5Error
from .source_policy_v5 import SourcePolicy, source_binding_v5

PAIRING_STATUSES = ("paired", "missing_label", "incomplete")
GEOMETRY_STATUSES = ("pass", "fail", "unknown")
GEOMETRY_REASONS = ("SHAPE_MISMATCH", "SPACING_MISMATCH", "AFFINE_MISMATCH", "AXIS_CODE_MISMATCH")


def _field(row: object, key: str, path: str):
    if not isinstance(row, dict):
        raise V5Error("INVALID_TYPE", path)
    if key not in row:
        raise V5Error("MISSING_FIELD", f"{path}.{key}")
    return row[key]


def _own_rows(rows: list, name: str, source_dataset: str) -> list[tuple[str, dict]]:
    if not isinstance(rows, list):
        raise V5Error("INVALID_TYPE", f"$.{name}")
    out = []
    for index, row in enumerate(rows):
        path = f"$.{name}[{index}]"
        if _field(row, "source", path) == source_dataset:
            out.append((path, row))
    return out


def _pairing(rows: list[tuple[str, dict]]) -> dict:
    counts = dict.fromkeys(PAIRING_STATUSES, 0)
    for path, row in rows:
        status = _field(row, "status", path)
        if status not in counts:
            raise V5Error("INVALID_PAIRING_STATUS", f"{path}.status")
        counts[status] += 1
    return dict(sorted(counts.items()))


def _geometry(rows: list[tuple[str, dict]]) -> tuple[dict, dict]:
    counts = dict.fromkeys(GEOMETRY_STATUSES, 0)
    reasons: dict[str, int] = {}
    for path, row in rows:
        status = _field(row, "status", path)
        counts[status if status in ("pass", "fail") else "unknown"] += 1
        for reason in _field(row, "reasons", path) or []:
            key = reason if reason in GEOMETRY_REASONS else "OTHER"
            reasons[key] = reasons.get(key, 0) + 1
    return dict(sorted(counts.items())), dict(sorted(reasons.items()))


def _channel(reports: list) -> dict:
    if not isinstance(reports, list):
        raise V5Error("INVALID_TYPE", "$.channel_reports")
    consistent = unevaluated = 0
    for index, report in enumerate(reports):
        tracks = _field(report, "tracks", f"$.channel_reports[{index}]")
        if not isinstance(tracks, dict):
            raise V5Error("INVALID_TYPE", f"$.channel_reports[{index}].tracks")
        flags = [info["best"].get("consistent") is True
                 for info in tracks.values() if isinstance(info, dict) and info.get("best")]
        if not flags:
            unevaluated += 1
        elif all(flags):
            consistent += 1
    return {"cases_checked": len(reports), "consistent": consistent, "unevaluated": unevaluated}


def _downsampling(reports: list) -> dict:
    if not isinstance(reports, list):
        raise V5Error("INVALID_TYPE", "$.loss_reports")
    factors: set[tuple[int, ...]] = set()
    classes: dict[str, dict[str, int]] = {}
    for index, report in enumerate(reports):
        path = f"$.loss_reports[{index}]"
        factors.add(tuple(int(f) for f in _field(report, "factor", path)))
        for name, info in sorted(_field(report, "classes", path).items()):
            vanished = int(_field(info, "components_vanished", f"{path}.classes"))
            entry = classes.setdefault(name, {"components_vanished": 0, "scans_affected": 0})
            entry["components_vanished"] += vanished
            entry["scans_affected"] += int(vanished > 0)
    return {"factors": [list(f) for f in sorted(factors)], "scans": len(reports),
            "classes": dict(sorted(classes.items()))}


def summarize_source_check_v5(geometry_rows: list[dict], pairing_rows: list[dict], channel_reports: list[dict],
                              loss_reports: list[dict], policy: SourcePolicy) -> dict:
    if not isinstance(policy, SourcePolicy):
        raise V5Error("INVALID_TYPE", "$.policy")
    geometry, reasons = _geometry(_own_rows(geometry_rows, "geometry_rows", policy.source_dataset))
    return {**source_binding_v5(policy),
            "pairing": _pairing(_own_rows(pairing_rows, "pairing_rows", policy.source_dataset)),
            "geometry": geometry, "geometry_reasons": reasons,
            "channel": _channel(channel_reports), "downsampling": _downsampling(loss_reports)}


def _table(header: tuple[str, str], counts: dict) -> list[str]:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | --- |"]
    lines += [f"| {key} | {value} |" for key, value in counts.items()] or ["| — | 0 |"]
    return lines


def render_source_check_md_v5(summary: dict) -> str:
    down = summary["downsampling"]
    factors = ", ".join("×".join(str(f) for f in factor) for factor in down["factors"]) or "—"
    lines = [f"# 来源核验：{summary['source_id']}（{summary['source_version']}）", "",
             "只含计数；不含病例 ID、路径或文件名。", "", "## 配对", ""]
    lines += _table(("状态", "数量"), summary["pairing"])
    lines += ["", "## 几何", ""] + _table(("状态", "数量"), summary["geometry"])
    lines += ["", "### 失败原因", ""] + _table(("原因", "次数"), summary["geometry_reasons"])
    channel = summary["channel"]
    lines += ["", "## SKM 通道变换", "", "| 项 | 数量 |", "| --- | --- |",
              f"| 核验病例 | {channel['cases_checked']} |", f"| 变换一致 | {channel['consistent']} |",
              f"| 无可评估轨道 | {channel['unevaluated']} |"]
    lines += ["", "## 降采样损失", "", f"- 因子：{factors}；扫描数：{down['scans']}", "",
              "| 类别 | 消失连通域 | 受影响扫描 |", "| --- | --- | --- |"]
    lines += [f"| {name} | {info['components_vanished']} | {info['scans_affected']} |"
              for name, info in down["classes"].items()] or ["| — | 0 | 0 |"]
    return "\n".join(lines) + "\n"
