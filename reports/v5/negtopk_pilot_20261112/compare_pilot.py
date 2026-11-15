"""Side-by-side tables of two evaluate-only summaries of the negative-top-k pilot (2026-11-12): the original run's
checkpoint and the pilot run's checkpoint at the same step. Reads the two summary.json files as fetched (time keys
removed on the server) and prints Markdown; with --json it writes the values the tables show.
Only counts, rates, distributions and hashes; no identifier, no path, no time.
Usage: compare_pilot.py <old summary.json> <new summary.json> [--json <out.json>]"""
import json
import sys

KNEE, BRAIN = "knee_qdess", "brain_dwi_adc"
ROLES = ("train", "development_val")
STAGES = (("no_response", "框内没有达到阈值的响应"), ("below_min_voxels", "有响应，但连通块都小于体素数下限"),
          ("anchor_outside", "有保留的连通块触及框，但定位点在框外"), ("hit_host_wrong", "命中，所在组织不对"),
          ("hit_host_correct", "命中且所在组织正确"))
ZONES = (("same_class_primary", "同类主分析框内"), ("ignored", "忽略区内"), ("outside", "所有框之外"))
SIZE_BINS = ("1", "2-9", "10-49", "50-99", "100-999", "1000-9999", "10000+")


def cell(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def get(node, *keys):
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def main(argv):
    old, new = (json.load(open(path, encoding="utf-8")) for path in argv[1:3])
    sides = {"old": old, "new": new}
    for name, summary in sides.items():
        assert summary["status"] == "COMPLETED", (name, summary["status"])
        assert summary["checkpoint"]["weights_unchanged"] is True
        assert summary["evaluation"]["reuse_inference"] is None, "both evaluations use new predictions"
    values, lines = {"checkpoints": {}, "knee": {}, "brain": {}}, []

    def table(title, header, rows, note=None):
        lines.extend([f"### {title}", ""])
        if note:
            lines.extend([note, ""])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| --- |" + " --- |" * (len(header) - 1))
        for row in rows:
            lines.append("| " + " | ".join(cell(v) if not isinstance(v, str) else v for v in row) + " |")
        lines.append("")

    for name, summary in sides.items():
        c = summary["checkpoint"]
        values["checkpoints"][name] = {"plan_id": summary["plan_id"], "plan_sha256": summary["plan_sha256"],
                                       "run_sha256": summary["run"]["run_sha256"], "step": c["step"],
                                       "sha256": c["sha256"], "state_sha256": c["state_sha256"],
                                       "instance_mask_bytes_limit": summary["evaluation"].get("instance_mask_bytes_limit")}
    table("检查点", ["", "旧损失（原运行）", "新损失（试验运行）"], [
        ["计划", old["plan_id"], new["plan_id"]],
        ["检查点所在步", old["checkpoint"]["step"], new["checkpoint"]["step"]],
        ["检查点 sha256", f"`{old['checkpoint']['sha256']}`", f"`{new['checkpoint']['sha256']}`"],
        ["评价代码", f"`{old['evaluation']['code']['git_commit']}`", f"`{new['evaluation']['code']['git_commit']}`"]])

    # 1. box-level results, official reading (predicted anatomy), every primary box in the denominator
    for role in ROLES:
        f = {k: get(s, "members", KNEE, role, "findings") for k, s in sides.items()}
        d = {k: get(s, "members", KNEE, role, "diagnostics") for k, s in sides.items()}
        rows, store = [], {}

        def add(label, getter):
            pair = [getter(f[k], d[k]) for k in ("old", "new")]
            rows.append([label, *pair])
            store[label] = pair

        add("扫描数", lambda x, y: get(x, "overall", "n_scans"))
        add("主分析框", lambda x, y: get(x, "overall", "n_boxes"))
        add("框命中率（患者等权）", lambda x, y: get(x, "overall", "box_sensitivity", "patient_mean"))
        add("框命中率（合并）", lambda x, y: get(x, "overall", "box_sensitivity", "pooled"))
        add("命中的框（计数）", lambda x, y: get(y, "funnel", "overall", "hit"))
        add("命中且所在组织正确（患者等权）", lambda x, y: get(x, "overall", "host_accuracy", "patient_mean"))
        add("命中且所在组织正确（合并）", lambda x, y: get(x, "overall", "host_accuracy", "pooled"))
        add("命中且所在组织正确的框（计数）", lambda x, y: get(y, "funnel", "overall", "host_correct"))
        for threshold in ("0.1", "0.25", "0.5"):
            add(f"IoBB ≥ {threshold} 的框比例", lambda x, y, t=threshold: get(x, "overall", "iobb_at", t))
        for cls, label in (("cartilage_lesion", "软骨病变"), ("meniscal_tear", "半月板撕裂")):
            add(f"{label}：主分析框", lambda x, y, c=cls: get(x, "by_class", c, "n_boxes"))
            add(f"{label}：框命中率（患者等权）", lambda x, y, c=cls: get(x, "by_class", c, "box_sensitivity", "patient_mean"))
            add(f"{label}：框命中率（合并）", lambda x, y, c=cls: get(x, "by_class", c, "box_sensitivity", "pooled"))
            add(f"{label}：命中且所在组织正确（患者等权）",
                lambda x, y, c=cls: get(x, "by_class", c, "host_accuracy", "patient_mean"))
            add(f"{label}：命中且所在组织正确（合并）", lambda x, y, c=cls: get(x, "by_class", c, "host_accuracy", "pooled"))
            for threshold in ("0.1", "0.25", "0.5"):
                add(f"{label}：IoBB ≥ {threshold}", lambda x, y, c=cls, t=threshold: get(x, "by_class", c, "iobb_at", t))
        table(f"膝关节 {role}：框级结果（正式口径：预测解剖；分母是全部主分析框）", ["项", "旧损失", "新损失"], rows)
        values["knee"].setdefault(role, {})["boxes"] = store

        # failure stages
        rows, store = [], {}
        for scope_label, where in (("合计", ("overall",)), ("软骨病变", ("by_class", "cartilage_lesion")),
                                   ("半月板撕裂", ("by_class", "meniscal_tear"))):
            for stage, label in STAGES:
                pair = [get(d[k], "funnel", *where, "stages", stage) for k in ("old", "new")]
                rows.append([f"{scope_label}：{label}", *pair])
                store[f"{scope_label}:{stage}"] = pair
        table(f"膝关节 {role}：失败点（每个框只计一次，取链路上最先失败的一步）", ["项", "旧损失", "新损失"], rows)
        values["knee"][role]["stages"] = store

        # responses inside the positive bags, and the components that reach the boxes
        rows, store = [], {}
        for key, label in (("max_bag", "袋内最高概率"), ("topk_mean_bag", "袋内前 10% 体素的平均概率"),
                           ("positive_fraction_bag", "袋内达到阈值的体素比例"), ("max_box", "框内最高概率")):
            for q in ("p10", "p50"):
                pair = [get(d[k], "response", "overall", key, q) for k in ("old", "new")]
                rows.append([f"{label}：{'第 10 百分位' if q == 'p10' else '中位数'}", *pair])
                store[f"{key}:{q}"] = pair
        for level in ("0.5", "0.9"):
            pair = [get(d[k], "response", "overall", "boxes_with_max_at_least", level) for k in ("old", "new")]
            rows.append([f"框内最高概率 ≥ {level} 的框", *pair])
            store[f"max_box_at_least:{level}"] = pair
        pair = [get(d[k], "funnel", "overall", "responded") for k in ("old", "new")]
        rows.append(["框内有达到阈值的响应的框", *pair])
        store["responded"] = pair
        table(f"膝关节 {role}：正袋内的响应（全部主分析框）", ["项", "旧损失", "新损失"], rows,
              "袋 = 主分析框内的参照宿主组织（训练时的正袋）。概率是恢复到原始网格后的概率。")
        values["knee"][role]["bag_response"] = store

        rows, store = [], {}
        for group, label in (("hit", "命中的连通块"), ("anchor_outside", "触及框但定位点在框外的最大连通块")):
            for key, sub in (("voxels", "体素数"), ("inside_fraction", "落在框内的比例")):
                for q in ("p10", "p50", "p90"):
                    pair = [get(d[k], "chain", "overall", group, key, q) for k in ("old", "new")]
                    rows.append([f"{label}：{sub}（{q}）", *pair])
                    store[f"{group}:{key}:{q}"] = pair
            pair = [get(d[k], "chain", "overall", group, "voxels", "n") for k in ("old", "new")]
            rows.append([f"{label}：个数", *pair])
            store[f"{group}:n"] = pair
        table(f"膝关节 {role}：触及主分析框的连通块", ["项", "旧损失", "新损失"], rows)
        values["knee"][role]["chain"] = store

        # positives on the host tissues by zone; components; components off every host tissue
        rows, store = [], {}
        for cls, label in (("cartilage_lesion", "软骨病变"), ("meniscal_tear", "半月板撕裂")):
            for zone, zone_label in ZONES:
                for key, sub in (("fraction_pooled", "宿主组织上达到阈值的体素比例（合并）"),):
                    pair = [get(d[k], "zones", cls, zone, key) for k in ("old", "new")]
                    rows.append([f"{label}：{zone_label}：{sub}", *pair])
                    store[f"{cls}:{zone}:{key}"] = pair
                pair = [get(d[k], "zones", cls, zone, "fraction_per_scan", "p50") for k in ("old", "new")]
                rows.append([f"{label}：{zone_label}：同上（每次扫描，中位数）", *pair])
                store[f"{cls}:{zone}:per_scan_p50"] = pair
                pair = [get(d[k], "zones", cls, zone, "positive_on_host") for k in ("old", "new")]
                rows.append([f"{label}：{zone_label}：宿主组织上达到阈值的体素", *pair])
                store[f"{cls}:{zone}:positive_on_host"] = pair
        table(f"膝关节 {role}：宿主组织上的阳性体素（参照解剖，仅诊断）", ["项", "旧损失", "新损失"], rows,
              "区域互不重叠：同类主分析框内 → 忽略区内 → 其余（训练中该类别的负监督区）。框外没有标注不等于正常。")
        values["knee"][role]["zones"] = store

        rows, store = [], {}
        for cls, label in (("cartilage_lesion", "软骨病变"), ("meniscal_tear", "半月板撕裂")):
            for key, sub in (("count", "连通块（过滤前）"), ("kept", "通过体素数过滤的连通块（所见）"),
                             ("positive_voxels", "达到阈值的体素")):
                pair = [get(d[k], "components", cls, key) for k in ("old", "new")]
                rows.append([f"{label}：{sub}", *pair])
                store[f"{cls}:components:{key}"] = pair
            for q in ("p50", "max"):
                pair = [get(d[k], "components", cls, "kept_per_scan", q) for k in ("old", "new")]
                rows.append([f"{label}：每次扫描的所见数（{q}）", *pair])
                store[f"{cls}:kept_per_scan:{q}"] = pair
            for key, sub in (("count", "不在可归属组织上的连通块"), ("kept", "其中成为所见的"),
                             ("share_of_components", "占全部连通块的比例"),
                             ("share_of_positive_voxels", "占达到阈值的体素的比例")):
                pair = [get(d[k], "off_tissue", cls, key) for k in ("old", "new")]
                rows.append([f"{label}：{sub}", *pair])
                store[f"{cls}:off_tissue:{key}"] = pair
            for size in SIZE_BINS:
                pair = [get(d[k], "off_tissue", cls, "size_histogram", size) for k in ("old", "new")]
                rows.append([f"{label}：不在可归属组织上、体素数 {size} 的连通块", *pair])
                store[f"{cls}:off_tissue:size:{size}"] = pair
        table(f"膝关节 {role}：连通块与不在可归属组织上的连通块（参照解剖，仅诊断）", ["项", "旧损失", "新损失"], rows)
        values["knee"][role]["components"] = store

        rows, store = [], {}
        for key, label in (("total", "所见总数"), ("in_reference", "定位点在同类主分析框内"),
                           ("in_ignored", "定位点在忽略区内"), ("outside", "其余（异类主分析框内或所有参照框之外）")):
            pair = [get(f[k], "overall", "findings", key) for k in ("old", "new")]
            rows.append([label, *pair])
            store[key] = pair
        pair = [get(f[k], "overall", "findings_outside_reference_per_scan") for k in ("old", "new")]
        rows.append(["每次扫描在同类主分析框与忽略区之外的所见", *pair])
        store["outside_per_scan"] = pair
        table(f"膝关节 {role}：所见的位置（描述，未匹配的所见不是误报）", ["项", "旧损失", "新损失"], rows)
        values["knee"][role]["locations"] = store

        dice = {k: get(s, "members", KNEE, role, "anatomy_dice") for k, s in sides.items()}
        names = sorted(dice["old"])
        rows = [[name, get(dice["old"], name, "mean"), get(dice["new"], name, "mean")] for name in names]
        table(f"膝关节 {role}：组织 Dice（患者等权）", ["组织", "旧损失", "新损失"], rows)
        values["knee"][role]["anatomy_dice"] = {row[0]: row[1:] for row in rows}

    # brain, development_val, the training driver's module metrics
    module = {k: get(s, "members", BRAIN, "development_val", "module") for k, s in sides.items()}
    rows, store = [], {}
    for task, classes in (module["old"] or {}).get("segmentation", {}).items():
        for cls, fields in classes.items():
            for field in fields:
                pair = [get(module[k], "segmentation", task, cls, field, "mean") for k in ("old", "new")]
                rows.append([f"{task} {cls} {field}", *pair])
                store[f"{task}:{cls}:{field}"] = pair
    for field in ("pooled_sensitivity", "fp_per_scan"):
        pair = [get(module[k], "detection", field, "mean") for k in ("old", "new")]
        rows.append([f"病灶检测 {field}", *pair])
        store[field] = pair
    pair = [get(module[k], "n_patients") for k in ("old", "new")]
    rows.append(["患者数", *pair])
    store["n_patients"] = pair
    table("脑 development_val：梗死分割与病灶检测（患者等权，训练驱动自己的模块指标）", ["项", "旧损失", "新损失"], rows)
    values["brain"] = store

    print("\n".join(lines))
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as handle:
            json.dump(values, handle, ensure_ascii=False, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv)
