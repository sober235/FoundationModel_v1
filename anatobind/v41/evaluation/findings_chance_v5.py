"""Chance level of the box hit rate (hypothesis H3 of the diagnostics of 2026-11-12; reviewed 2026-11-13).

A box is hit when a finding of its class has its anchor inside it (evaluation/findings_v5.py). A model that reports
many or very large findings on every scan hits boxes whether or not its findings have anything to do with the
lesions. Two controls measure how often that happens. Both are diagnostic: no evaluation rule, threshold or
denominator depends on them. Counts and rates only, no identifiers.

1. RELOCATION ON THE HOST TISSUE, WITHIN THE SCAN (the control to read).
   Null hypothesis: given the findings of a scan, a primary box could equally lie anywhere on its host tissue in
   that scan. For every primary box (cut to the volume), all placements of a box of the same size inside the same
   volume are enumerated. A placement is comparable when the number of voxels of the box's host tissue group
   (reference anatomy) it contains falls in the same octave [2^k, 2^(k+1)) as that of the real box. "Comparable" is
   an equivalence relation, so under the null the real box is uniform over its own class and the test is an exact
   randomisation test. `p_hit` of a box is the share of its comparable placements that the scan's findings of the
   box's class would hit, by the unchanged rule (anchor inside; the largest candidate names the host). No random
   number is drawn. Over the boxes the placements are independent under the null, so the number of hit boxes
   follows a Poisson-binomial distribution with the boxes' `p_hit`; `p_at_least_observed` is its upper tail at the
   observed count.
   The scan, its anatomy, the class, the tissue group and the size of every box are kept; nothing crosses from one
   patient to another, so neither registration nor left/right matters.
   What it cannot tell: lesions are not uniform over a tissue (e.g. weight-bearing zones). A model that learned
   where lesions usually are, without reading the image evidence of this lesion, also beats this null. "Above
   this chance level" therefore means "better than a random place on the same tissue", not "lesion evidence read".
   A box without a voxel of its host tissue in the reference anatomy has no comparable placement and is left out
   (counted as `excluded`).

2. CROSS-PATIENT PAIRING (exploratory only).
   The findings of one scan are scored against the primary boxes of ANOTHER patient's scan, in voxel indices.
   The scans are not registered and a left knee is not mirrored onto a right one, so a box of another scan may
   lie off the tissue or off the knee: the chance level is biased downwards by an unknown amount, and a model that
   merely follows the anatomy of each scan beats it. Its p-value is NOT evidence of lesion localisation.
   `chance_rate`: over all boxes, the mean share of the other patients' scans whose findings hit the box (exact).
   `permutation`: seeded random permutations of the finding sets over the scans in which no scan receives findings
   of its own patient; `p_at_least_observed` = (1 + permutations with a count >= observed) / (1 + n_permutations).

Neither is a false-positive rate: an unmarked place is not a normal place.
"""
from __future__ import annotations

import numpy as np

from ..engine.box_data_v5 import _window
from .findings_v5 import _is_primary, _ratio

KEYS = ("hit", "host_correct")
N_PERMUTATIONS = 1000
SEED = 20261113
MAX_ATTEMPTS_PER_PERMUTATION = 1000


def _sets(findings: list, classes: list) -> dict:
    """Per class: anchors and host groups of one scan's findings, largest first (ties by finding_uid), which is the
    order in which score_findings_v5 chooses the finding that names the host."""
    out = {}
    for cls in classes:
        chosen = sorted((f for f in findings if f["abnormality_class"] == cls),
                        key=lambda f: (-f["voxels"], f["finding_uid"]))
        out[cls] = (np.asarray([f["anchor"] for f in chosen], dtype=np.int64).reshape(len(chosen), 3),
                    [f["host_group"] for f in chosen])
    return out


def _meets(sets: dict, box: dict) -> tuple:
    """(hit, host_correct) of one primary box met by one scan's findings."""
    anchors, hosts = sets[box["abnormality_class"]]
    if not len(hosts):
        return False, False
    start = np.asarray(box["box"]["start"], dtype=np.int64)
    inside = np.all((anchors >= start) & (anchors < start + np.asarray(box["box"]["size"], dtype=np.int64)), axis=1)
    if not inside.any():
        return False, False
    return True, hosts[int(np.argmax(inside))] == box["host"]["tissue_group"]


def _draw(rng: np.random.Generator, patients: list) -> np.ndarray | None:
    """A random permutation of the scans in which no scan receives the findings of its own patient."""
    n = len(patients)
    for _ in range(MAX_ATTEMPTS_PER_PERMUTATION):
        order = rng.permutation(n)
        if all(patients[int(source)] != patients[target] for target, source in enumerate(order)):
            return order
    return None


def chance_hit_control_v5(scans: list, classes: list, *, n_permutations: int = N_PERMUTATIONS,
                          seed: int = SEED) -> dict:
    """`scans`: one {"patient_uid", "findings", "reference"} per scan of one split. Returns counts and rates."""
    patients = [scan["patient_uid"] for scan in scans]
    sets = [_sets(scan["findings"], classes) for scan in scans]
    boxes = [(target, box) for target, scan in enumerate(scans) for box in scan["reference"]["boxes"]
             if _is_primary(box)]
    # met[key][source, b]: box b met by the findings of scan `source`
    met = {key: np.zeros((len(scans), len(boxes)), dtype=bool) for key in KEYS}
    for b, (_, box) in enumerate(boxes):
        for source in range(len(scans)):
            met["hit"][source, b], met["host_correct"][source, b] = _meets(sets[source], box)
    owner = np.asarray([target for target, _ in boxes], dtype=np.int64)
    box_class = [box["abnormality_class"] for _, box in boxes]
    other = np.asarray([[patients[source] != patients[target] for target, _ in boxes]
                        for source in range(len(scans))], dtype=bool).reshape(len(scans), len(boxes))
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(n_permutations):
        order = _draw(rng, patients)
        if order is None:               # e.g. one patient owns more than half of the scans
            orders = None
            break
        orders.append(order)

    def block(cls: str | None) -> dict:
        columns = np.asarray([cls is None or name == cls for name in box_class], dtype=bool)
        n_boxes = int(columns.sum())
        out = {"n_boxes": n_boxes}
        for key in KEYS:
            observed = int(met[key][owner[columns], np.flatnonzero(columns)].sum()) if n_boxes else 0
            pairs = other[:, columns]
            support = pairs.sum(axis=0)                           # other patients' scans per box
            usable = support > 0
            chance = float(((met[key][:, columns] & pairs).sum(axis=0)[usable] / support[usable]).mean()) \
                if usable.any() else None
            permutation = None
            if orders and n_boxes:
                counts = np.asarray([int(met[key][order[owner[columns]], np.flatnonzero(columns)].sum())
                                     for order in orders])
                at_least = int((counts >= observed).sum())
                permutation = {"mean_rate": float(counts.mean()) / n_boxes,
                               "rate_p2_5": float(np.percentile(counts, 2.5)) / n_boxes,
                               "rate_p97_5": float(np.percentile(counts, 97.5)) / n_boxes,
                               "max": int(counts.max()), "n_at_least_observed": at_least,
                               "p_at_least_observed": (1 + at_least) / (1 + len(orders))}
            out[key] = {"observed": observed, "observed_rate": _ratio(observed, n_boxes), "chance_rate": chance,
                        "permutation": permutation}
        return out

    return {"n_scans": len(scans), "n_patients": len(set(patients)),
            "n_cross_patient_pairs": sum(1 for a in patients for b in patients if a != b),
            "n_permutations": int(n_permutations), "seed": int(seed),
            "permutations_drawn": None if orders is None else len(orders),
            "overall": block(None), "by_class": {cls: block(cls) for cls in classes}}


# ---------------------------------------------------------------------------
# Relocation on the host tissue, within the scan


def _octave(count: np.ndarray | int):
    """k with 2^k <= count < 2^(k+1); -1 for 0."""
    return np.where(np.asarray(count) > 0, np.floor(np.log2(np.maximum(np.asarray(count), 1))).astype(np.int64), -1)


def placement_counts_v5(tissue: np.ndarray, size: np.ndarray) -> tuple:
    """Every placement of a box of `size` inside the volume that can contain a voxel of `tissue` (which has one):
    (low, high, counts). The placements start at low + (i, j, k) with low <= start <= high, and counts[i, j, k] is
    the number of tissue voxels the placement holds."""
    shape = np.asarray(tissue.shape, dtype=np.int64)
    size = np.asarray(size, dtype=np.int64)
    where = np.nonzero(tissue)
    low = np.maximum(np.asarray([axis.min() for axis in where]) - size + 1, 0)
    high = np.minimum(np.asarray([axis.max() for axis in where]), shape - size)
    crop = tissue[tuple(slice(int(a), int(b + n)) for a, b, n in zip(low, high, size))]
    total = np.pad(crop.astype(np.int32).cumsum(0).cumsum(1).cumsum(2), ((1, 0), (1, 0), (1, 0)))
    n0, n1, n2 = (int(v) for v in high - low + 1)
    s0, s1, s2 = (int(v) for v in size)

    def corner(a: int, b: int, c: int) -> np.ndarray:
        return total[a:a + n0, b:b + n1, c:c + n2]

    counts = (corner(s0, s1, s2) - corner(0, s1, s2) - corner(s0, 0, s2) - corner(s0, s1, 0)
              + corner(0, 0, s2) + corner(0, s1, 0) + corner(s0, 0, 0) - corner(0, 0, 0))
    return low, high, counts


def relocation_rows_v5(*, findings: list, reference: dict, anatomy_reference: np.ndarray, group_ids: dict) -> list:
    """One row per primary box of one scan: observed hit / host_correct and their probabilities over the comparable
    placements of the box in this scan. `group_ids`: tissue group -> anatomy class ids of the reference anatomy."""
    anatomy = np.asarray(anatomy_reference)
    shape = tuple(int(n) for n in anatomy.shape)
    rows = []
    for box in reference["boxes"]:
        if not _is_primary(box):
            continue
        cls, group = box["abnormality_class"], box["host"]["tissue_group"]
        row = {"abnormality_class": cls, "tissue_group": group, "status": "no_host_tissue", "hit": None,
               "host_correct": None, "p_hit": None, "p_host_correct": None, "comparable_placements": 0,
               "host_voxels": 0}
        rows.append(row)
        window = _window(box["box"], shape)
        if window is None:
            continue
        origin = np.asarray([w.start for w in window], dtype=np.int64)
        size = np.asarray([w.stop - w.start for w in window], dtype=np.int64)
        tissue = np.isin(anatomy, list(group_ids.get(group, [])))
        own = int(tissue[window].sum())
        row["host_voxels"] = own
        if own == 0:
            continue
        low, high, counts = placement_counts_v5(tissue, size)
        comparable = _octave(counts) == int(_octave(own))
        # state of every placement: 0 no finding of the class inside, 1 hit, 2 hit and the largest names the host
        state = np.zeros(counts.shape, dtype=np.int8)
        ordered = sorted((f for f in findings if f["abnormality_class"] == cls),
                         key=lambda f: (-f["voxels"], f["finding_uid"]))
        for finding in reversed(ordered):                     # the largest candidate is written last
            anchor = np.asarray(finding["anchor"], dtype=np.int64)
            first, last = np.maximum(anchor - size + 1, low) - low, np.minimum(anchor, high) - low
            if np.any(first > last):
                continue
            state[tuple(slice(int(a), int(b) + 1) for a, b in zip(first, last))] = \
                2 if finding["host_group"] == group else 1
        here = int(state[tuple(int(v) for v in origin - low)])
        n = int(comparable.sum())
        row.update(status="ok", hit=here > 0, host_correct=here == 2, comparable_placements=n,
                   p_hit=float((comparable & (state > 0)).sum()) / n,
                   p_host_correct=float((comparable & (state == 2)).sum()) / n)
    return rows


def _upper_tail(probabilities: list, observed: int) -> tuple:
    """Poisson-binomial: (P(X >= observed), 2.5 % and 97.5 % quantiles of X) for independent Bernoulli trials."""
    distribution = np.ones(1)
    for p in probabilities:
        distribution = np.convolve(distribution, [1.0 - p, p])
    cumulative = np.cumsum(distribution)
    quantile = [int(np.searchsorted(cumulative, q - 1e-12)) for q in (0.025, 0.975)]
    return float(min(1.0, distribution[observed:].sum())), quantile[0], quantile[1]


def summarize_relocation_v5(rows: list, classes: list) -> dict:
    """`rows`: the box rows of every scan of one split (relocation_rows_v5), flattened."""
    def block(cls: str | None) -> dict:
        selected = [row for row in rows if cls is None or row["abnormality_class"] == cls]
        usable = [row for row in selected if row["status"] == "ok"]
        out = {"n_boxes": len(selected), "n_usable": len(usable), "excluded": len(selected) - len(usable),
               "comparable_placements_median": float(np.median([row["comparable_placements"] for row in usable]))
               if usable else None}
        for key, p_key in (("hit", "p_hit"), ("host_correct", "p_host_correct")):
            observed = sum(1 for row in usable if row[key])
            item = {"observed": observed, "observed_rate": _ratio(observed, len(usable)), "chance_rate": None,
                    "chance_rate_p2_5": None, "chance_rate_p97_5": None, "p_at_least_observed": None}
            if usable:
                tail, lower, upper = _upper_tail([row[p_key] for row in usable], observed)
                item.update(chance_rate=float(np.mean([row[p_key] for row in usable])),
                            chance_rate_p2_5=lower / len(usable), chance_rate_p97_5=upper / len(usable),
                            p_at_least_observed=tail)
            out[key] = item
        return out

    return {"overall": block(None), "by_class": {cls: block(cls) for cls in classes}}


def _cell(value) -> str:
    return "NA" if value is None else str(value) if isinstance(value, int) else f"{value:.3f}"


def render_chance_md_v5(by_role: dict) -> str:
    """`by_role[role]` = {"relocation": summarize_relocation_v5(...), "cross_patient": chance_hit_control_v5(...)}.
    Counts and rates only: no patient, sample or finding identifiers."""
    lines = ["# 偶然命中水平的对照（仅诊断）", "",
             "两种对照都不改变任何评价规则、阈值或分母；都不是误报率：医生没有标注不等于正常。", "",
             "## 一、同一次扫描内，沿所在组织挪动框（应当读的对照）", "",
             "- 零假设：给定这次扫描的所见，主分析框可以等可能地落在同一次扫描里它的所在组织的任何位置。",
             "- 做法：对每个主分析框，枚举同一体积内所有同样大小的摆放；所含所在组织（参照解剖）体素数与真实框处于同一个"
             "二倍区间 [2^k, 2^(k+1)) 的摆放算作可比。偶然命中概率 = 可比摆放中，按不变的命中规则会被本次扫描的所见命中的"
             "比例。不抽随机数。命中框数在零假设下服从泊松二项分布，p = 命中数不少于实际的概率。",
             "- 不跨患者：不需要配准，也不受左右膝影响。框的类别、组织组、大小都不变。",
             "- 它不能说明的：病变在组织上的分布本来就不均匀。模型如果只学到“病变通常在哪里”，没有读到这一处病变的影像"
             "证据，也会高于这个偶然水平。所以“高于偶然”只表示“比同一组织上随机的位置好”。",
             "- 参照解剖里不含所在组织体素的框没有可比的摆放，不计入（列为“未计入”）。", ""]
    for role, control in by_role.items():
        lines += [f"### {role}", "",
                  "| 范围 | 口径 | 主分析框 | 计入 | 未计入 | 实际命中 | 实际命中率 | 偶然命中率 | 零假设下的 2.5%–97.5% | p | "
                  "可比摆放数（中位数） |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        relocation = control["relocation"]
        for name, block in [("合计", relocation["overall"]), *relocation["by_class"].items()]:
            for key, label in (("hit", "命中"), ("host_correct", "命中且所在组织正确")):
                item = block[key]
                interval = "NA" if item["chance_rate"] is None else \
                    f"{item['chance_rate_p2_5']:.3f}–{item['chance_rate_p97_5']:.3f}"
                p = "NA" if item["p_at_least_observed"] is None else f"{item['p_at_least_observed']:.4g}"
                lines.append(f"| {name} | {label} | {block['n_boxes']} | {block['n_usable']} | {block['excluded']} | "
                             f"{item['observed']} | {_cell(item['observed_rate'])} | {_cell(item['chance_rate'])} | "
                             f"{interval} | {p} | {_cell(block['comparable_placements_median'])} |")
        lines.append("")
    first = next(iter(by_role.values()))["cross_patient"]
    lines += ["## 二、跨患者配对（仅探索性，不能据此声称学到了病变定位）", "",
              "- 做法：把一次扫描的所见，按不变的规则，去对另一名患者的扫描上的主分析框（体素下标不变）。",
              "- 为什么只算探索性：扫描之间没有配准，左右膝没有镜像，别人的框在本次扫描里可能落在组织之外。偶然水平因此"
              "被低估，低估多少不知道；只是跟着每次扫描的解剖走的模型也会高于它。",
              f"- 偶然命中率是全部跨患者配对的平均；置换 {first['n_permutations']} 次（种子 {first['seed']}），"
              "p = (1 + 命中数不少于实际的次数) / (1 + 置换次数)。", ""]
    for role, control in by_role.items():
        cross = control["cross_patient"]
        lines += [f"### {role}", "",
                  f"- 扫描 {cross['n_scans']}，患者 {cross['n_patients']}，跨患者的有序扫描配对 "
                  f"{cross['n_cross_patient_pairs']}，完成的置换 {_cell(cross['permutations_drawn'])}", "",
                  "| 范围 | 口径 | 框数 | 实际命中 | 实际命中率 | 偶然命中率 | 置换的 2.5%–97.5% | 置换中不少于实际的次数 | p |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, block in [("合计", cross["overall"]), *cross["by_class"].items()]:
            for key, label in (("hit", "命中"), ("host_correct", "命中且所在组织正确")):
                item, permutation = block[key], block[key]["permutation"] or {}
                interval = "NA" if not permutation else \
                    f"{permutation['rate_p2_5']:.3f}–{permutation['rate_p97_5']:.3f}"
                lines.append(f"| {name} | {label} | {block['n_boxes']} | {item['observed']} | "
                             f"{_cell(item['observed_rate'])} | {_cell(item['chance_rate'])} | {interval} | "
                             f"{_cell(permutation.get('n_at_least_observed'))} | "
                             f"{_cell(permutation.get('p_at_least_observed'))} |")
        lines.append("")
    return "\n".join(lines)
