"""Put the four count-only ranking reports of the audit side by side.

Reads rank_R1.md … rank_R4.md of this folder (the archived copies: counts, distributions and hashes only), writes
side_by_side.json (every table row of the four reports, keyed by section) and side_by_side_tables.md (the rows the
results report quotes). No model, no private rows, no server access: run on any machine with

    python reports/v5/ranking_audit_20261115/side_by_side.py reports/v5/ranking_audit_20261115
"""
import json
import sys
from pathlib import Path

RUNS = ("R1", "R2", "R3", "R4")
ROLES = ("train", "development_val")
CLASSES = ("cartilage_lesion", "meniscal_tear")
STATS = (("袋内最高 10% 体素的平均概率（框目标训练的量）", "topk_mean（主要量）"),
         ("袋内平均概率", "mean"),
         ("袋内体素对同一组织其余体素的排序（0.5 = 无差别）", "auc"),
         ("袋内体素进入本组织最高 1% 的比例 ÷ 1%", "enrichment_01"),
         ("袋内体素进入本组织最高 5% 的比例 ÷ 5%", "enrichment_05"),
         ("袋内体素进入本组织最高 10% 的比例 ÷ 10%", "enrichment_10"))
P_PATIENT = "患者等权的平均百分位高于 0.5（随机化分布的正态近似；各框独立换位）"
P_BOX = "框等权的平均百分位高于 0.5（随机化分布的正态近似）"
P_TOP = "百分位 ≥ 0.9 的框数不少于实际（泊松二项，精确）"
ZONES = ("同类主分析框内的宿主组织（袋，正监督）", "同类主分析框内的其余体素（不计正负）", "忽略区内的宿主组织",
         "忽略区内的其余体素", "所有框之外的宿主组织（负监督）", "所有框之外的其他已分割组织（负监督）",
         "所有框之外、没有分割标注的体素（负监督）")


def parse(path: Path) -> dict:
    """{section: {subsection: {row: (train, development_val)}}} of one report; the subsection is the #### heading
    or, in the sections without one, the sentence that introduces the table."""
    out, section, sub, lead = {}, None, "", ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            section, sub, lead = line[4:].strip(), "", ""
        elif line.startswith("#### "):
            sub, lead = line[5:].strip(), ""
        elif section and line.startswith("| "):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 3 and cells[0] not in ("项", "---") and not cells[0].startswith("p（单侧"):
                out.setdefault(section, {}).setdefault(sub or lead, {})[cells[0]] = (cells[1], cells[2])
        elif section and line.strip() and not line.startswith("-"):
            lead = line.strip().split("（")[0].split("：")[0]
    return out


def cell(data: dict, run: str, section: str, sub: str, row: str, role: int) -> str:
    return data[run][section][sub][row][role]


def table(header: list, rows: list) -> list:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def tables(data: dict) -> list:
    lines = ["# 排序审计：四个检查点并列（由 `side_by_side.py` 从四份归档的报告生成）", "",
             "只含计数、分布与 p 值；不含样本或患者 ID、路径或文件名。R1 原运行第 6000 步，R2 原运行第 30000 步，"
             "R3 1% 试验第 6000 步，R4 5% 试验第 6000 步。全部是对所读检查点的诊断；train 上的数值不能代替泛化评价。", ""]
    for group in ("合计", *CLASSES):
        section = f"主分析框对可比摆放：{group}"
        lines += [f"## {group}：支持（四个检查点相同才会生成本表）", ""]
        support = data["R1"][section][""]
        assert all(data[run][section][""] == support for run in RUNS), "support differs between checkpoints"
        lines += table(["项", *ROLES], [[row, *values] for row, values in support.items()])
        for role_index, role in enumerate(ROLES):
            lines += [f"## {group}，{role}：六个统计量的患者等权平均百分位（95% 患者区间）与 p", ""]
            rows = []
            for title, name in STATS:
                sub = f"{group}：{title}"
                rows.append([name, *(
                    "{}（{} – {}）；p {}".format(*(cell(data, run, section, sub, row, role_index) for row in (
                        "患者等权的平均百分位", "患者区间下限（2.5%）", "患者区间上限（97.5%）", P_PATIENT)))
                    for run in RUNS)])
            lines += table(["统计量", *RUNS], rows)
            lines += [f"## {group}，{role}：主要量的逐框分布", ""]
            sub = f"{group}：{STATS[0][0]}"
            rows = [[row, *(cell(data, run, section, sub, row, role_index) for run in RUNS)] for row in (
                "有取值的框", "真实框的取值：中位数", "可比摆放的取值（每个框取中位数）：中位数", "百分位：平均",
                "百分位：第 25 百分位", "百分位：中位数", "百分位：第 75 百分位", "零假设下平均百分位的标准差",
                "百分位 ≥ 0.9 的框", "单框 p ≤ 0.05 的框", "患者数", "平均百分位高于 0.5 的患者", P_BOX, P_TOP)]
            lines += table(["项", *RUNS], rows)
    for group in CLASSES:
        section = f"按监督分区的概率（描述）：{group}"
        for role_index, role in enumerate(ROLES):
            lines += [f"## {group}，{role}：按监督分区（描述）", ""]
            first = next(iter(data["R1"][section]))  # the table of the seven zones opens the section
            rows = [[f"{zone}：平均概率", *(cell(data, run, section, first, f"{zone}：平均概率", role_index)
                                         for run in RUNS)] for zone in ZONES]
            for row in ("袋 对 框外宿主组织：合并", "袋 对 框外宿主组织：每次扫描，中位数",
                        "宿主组织 对 非宿主体素：合并（响应是否跟着解剖走）", "宿主组织 对 非宿主体素：每次扫描，中位数"):
                rows.append([row, *(cell(data, run, section, "分区之间的排序", row, role_index) for run in RUNS)])
            for share in ("0.1%", "1.0%"):
                top = f"每次扫描概率最高的 {share} 体素落在哪些分区"
                for zone in ZONES:
                    rows.append([f"概率最高的 {share} 体素：{zone}：富集",
                                 *(cell(data, run, section, top, f"{zone}：富集", role_index) for run in RUNS)])
            lines += table(["项", *RUNS], rows)
        section = f"扫描一级的响应（描述）：{group}"
        for role_index, role in enumerate(ROLES):
            lines += [f"## {group}，{role}：扫描一级（描述）", ""]
            (sub, rows_of), = data["R1"][section].items()
            lines += table(["项", *RUNS], [[row, *(cell(data, run, section, sub, row, role_index) for run in RUNS)]
                                          for row in rows_of])
    return lines


def main(folder: str) -> None:
    root = Path(folder)
    data = {run: parse(root / f"rank_{run}.md") for run in RUNS}
    assert all(list(data[run]) == list(data["R1"]) for run in RUNS), "the four reports do not have the same sections"
    (root / "side_by_side.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (root / "side_by_side_tables.md").write_text("\n".join(tables(data)), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
