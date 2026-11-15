"""Threshold-free reading of the abnormality probabilities against the readers' boxes (researcher's decision of
2026-11-14): no acceptance threshold, no connected component, no anchor. Synthetic volumes, CPU."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anatobind.v41.evaluation.findings_ranking_v5 import (
    BINS, STATS, ZONES, histogram_auc_v5, ranking_scan_v5, render_ranking_md_v5, summarize_ranking_v5)

CLASSES = ["cartilage_lesion", "meniscal_tear"]
GROUP_IDS = {"femoral_cartilage": [1], "meniscus": [2, 3]}
GROUPS_BY_CLASS = {"cartilage_lesion": ["femoral_cartilage"], "meniscal_tear": ["meniscus"]}


def _box(cls, group, start, size, analysis="primary", status="included"):
    return {"status": status, "analysis": analysis, "abnormality_class": cls, "box": {"start": start, "size": size},
            "host": {"tissue_group": group}}


def _slab(shape=(20, 12, 4)):
    """Femoral cartilage in the low half of axis 0, meniscus (two classes of one group) in the high half."""
    anatomy = np.zeros(shape, dtype=np.int64)
    anatomy[2:9, 2:10, 1:3] = 1
    anatomy[11:15, 2:10, 1:3] = 2
    anatomy[15:18, 2:10, 1:3] = 3
    return anatomy


def _strip(length=60):
    """One long strip of femoral cartilage."""
    anatomy = np.zeros((length, 12, 4), dtype=np.int64)
    anatomy[:, 2:10, 1:3] = 1
    return anatomy


def _scan(probability, boxes, anatomy, *, patient="p0", key=(7, 0), n_placements=100000, eligible=None):
    return ranking_scan_v5(
        probability=probability, reference={"boxes": boxes}, anatomy_reference=anatomy,
        anatomy_eligible=np.ones(anatomy.shape, dtype=bool) if eligible is None else eligible, classes=CLASSES,
        group_ids=GROUP_IDS, groups_by_class=GROUPS_BY_CLASS, topk_fraction=0.1, patient_uid=patient, seed_key=key,
        n_placements=n_placements)


def _random(anatomy, seed=3):
    return np.random.default_rng(seed).random((len(CLASSES), *anatomy.shape), dtype=np.float32)


def _window(box):
    return tuple(slice(a, a + n) for a, n in zip(box["box"]["start"], box["box"]["size"]))


def _direct(probability, anatomy, box):
    """The statistics of one box, each from its definition."""
    c = CLASSES.index(box["abnormality_class"])
    tissue = np.isin(anatomy, GROUP_IDS[box["host"]["tissue_group"]])
    inside = np.zeros(anatomy.shape, dtype=bool)
    inside[_window(box)] = True
    bag, rest = probability[c][tissue & inside].astype(np.float64), probability[c][tissue & ~inside].astype(np.float64)
    everything = probability[c][tissue].astype(np.float64)
    top = np.sort(bag)[-max(1, math.ceil(0.1 * bag.size)):]
    greater = (bag[:, None] > rest[None, :]).sum() + 0.5 * (bag[:, None] == rest[None, :]).sum()
    out = {"mean": bag.mean(), "topk_mean": top.mean(), "auc": greater / (bag.size * rest.size)}
    for name, q in (("enrichment_01", 0.01), ("enrichment_05", 0.05), ("enrichment_10", 0.10)):
        # a voxel is in the top q of its tissue when fewer than q·N voxels of the tissue lie above it (mid-rank)
        above = (everything[None, :] > bag[:, None]).sum(axis=1) + 0.5 * ((everything[None, :] == bag[:, None]).sum(axis=1) - 1)
        out[name] = float((above < q * everything.size).mean()) / q
    return out


def _brute_pool(probability, anatomy, box, stat):
    """The statistic of every comparable placement, one by one (the real box is one of them)."""
    size = box["box"]["size"]
    tissue = np.isin(anatomy, GROUP_IDS[box["host"]["tissue_group"]])

    def count(start):
        return int(tissue[tuple(slice(a, a + n) for a, n in zip(start, size))].sum())

    octave = int(np.floor(np.log2(count(box["box"]["start"]))))
    values = []
    for start in np.ndindex(*[n - m + 1 for n, m in zip(anatomy.shape, size)]):
        held = count(start)
        if held == 0 or int(np.floor(np.log2(held))) != octave:
            continue
        values.append(_direct(probability, anatomy, dict(box, box={"start": list(start), "size": size}))[stat])
    return np.asarray(values)


def test_the_statistics_of_a_box_are_those_of_their_definitions():
    anatomy = _slab()
    probability = _random(anatomy)
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [3, 3, 0], [4, 3, 2]),
             _box("meniscal_tear", "meniscus", [12, 1, 1], [5, 4, 3])]
    row = _scan(probability, boxes, anatomy)
    assert [b["status"] for b in row["boxes"]] == ["ok", "ok"]
    for box, measured in zip(boxes, row["boxes"]):
        expected = _direct(probability, anatomy, box)
        assert set(measured["stats"]) == set(STATS)
        for stat in STATS:
            assert measured["stats"][stat]["value"] == pytest.approx(expected[stat], rel=1e-6), stat
    first = row["boxes"][0]
    assert (first["box_voxels"], first["bag_voxels"], first["group_voxels"]) == (4 * 3 * 2, 4 * 3 * 1, 7 * 8 * 2)


def test_with_few_placements_the_pool_is_every_comparable_placement():
    anatomy = _slab()
    probability = _random(anatomy, seed=11)
    box = _box("cartilage_lesion", "femoral_cartilage", [3, 3, 0], [4, 3, 2])
    measured = _scan(probability, [box], anatomy)["boxes"][0]
    for stat in ("topk_mean", "mean", "auc", "enrichment_10"):
        pool = _brute_pool(probability, anatomy, box, stat)
        own = _direct(probability, anatomy, box)[stat]
        entry = measured["stats"][stat]
        assert measured["exhaustive"] is True and measured["pool"] == measured["comparable_placements"] == len(pool)
        below, equal = (pool < own - 1e-12).sum(), (np.abs(pool - own) <= 1e-12).sum()
        assert entry["percentile"] == pytest.approx((below + 0.5 * equal) / len(pool))
        assert entry["p_at_least"] == pytest.approx((pool >= own - 1e-12).sum() / len(pool))
        assert entry["pool_median"] == pytest.approx(np.median(pool), rel=1e-6)


def test_with_many_placements_a_seeded_sample_of_them_is_drawn_and_the_box_itself_is_added():
    anatomy = _strip()
    probability = _random(anatomy, seed=5)
    box = _box("cartilage_lesion", "femoral_cartilage", [10, 4, 1], [3, 3, 2])
    first = _scan(probability, [box], anatomy, n_placements=40)["boxes"][0]
    assert first["exhaustive"] is False and first["pool"] == 41 and first["comparable_placements"] > 41
    assert first == _scan(probability, [box], anatomy, n_placements=40)["boxes"][0]          # same key, same draw
    other = _scan(probability, [box], anatomy, n_placements=40, key=(8, 0))["boxes"][0]
    assert other["stats"]["mean"]["value"] == first["stats"]["mean"]["value"]              # the box does not move
    assert other["stats"]["mean"]["pool_median"] != first["stats"]["mean"]["pool_median"]  # its comparisons do
    assert 1 / 41 <= first["stats"]["mean"]["p_at_least"] <= 1.0


def test_a_box_off_its_tissue_is_left_out_and_an_ignore_region_is_no_box():
    anatomy = _slab()
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [-2, 2, 1], [6, 3, 2]),       # cut to [0, 4)
             _box("meniscal_tear", "meniscus", [0, 0, 0], [2, 2, 1]),                      # no meniscus voxel inside
             _box("cartilage_lesion", "femoral_cartilage", [3, 3, 1], [2, 2, 2], analysis="secondary")]
    row = _scan(_random(anatomy), boxes, anatomy)
    assert len(row["boxes"]) == 2
    assert row["boxes"][0]["status"] == "ok" and row["boxes"][0]["bag_voxels"] == 2 * 3 * 2
    assert row["boxes"][1]["status"] == "no_host_tissue" and row["boxes"][1]["stats"] is None
    block = summarize_ranking_v5([row], CLASSES)["boxes"]["overall"]
    assert (block["n_boxes"], block["n_usable"], block["excluded"]) == (2, 1, 1)


def test_a_bag_that_is_the_whole_tissue_has_no_ranking_against_the_rest():
    anatomy = np.zeros((6, 6, 4), dtype=np.int64)
    anatomy[2:4, 2:4, 1:3] = 1
    box = _box("cartilage_lesion", "femoral_cartilage", [1, 1, 0], [4, 4, 4])
    measured = _scan(_random(anatomy), [box], anatomy)["boxes"][0]
    assert measured["status"] == "ok" and measured["bag_voxels"] == measured["group_voxels"] == 8
    assert measured["stats"]["auc"]["value"] is None and measured["stats"]["auc"]["percentile"] is None
    assert measured["stats"]["mean"]["value"] is not None


def _six_boxes(step=10):
    return [_box("cartilage_lesion", "femoral_cartilage", [step * i + 2, 4, 1], [3, 3, 2]) for i in range(6)]


def test_a_response_at_the_lesions_ranks_the_boxes_at_the_top_of_their_placements():
    anatomy, boxes = _strip(240), _six_boxes(step=40)
    probability = 0.05 * _random(anatomy)
    for box in boxes:
        probability[(0, *_window(box))] += 0.9
    block = summarize_ranking_v5([_scan(probability, boxes, anatomy)], CLASSES)["boxes"]["overall"]
    for stat in STATS:
        entry = block["stats"][stat]
        assert entry["mean_percentile"] > 0.9 and entry["p_normal"] < 0.001, stat
        assert entry["top_decile"]["observed"] >= 4 and entry["top_decile"]["p_at_least_observed"] < 0.001, stat
        assert entry["boxes_p_at_most_0.05"] >= 4, stat
    # the mean over the bag is highest where the box covers the whole response; the top of the bag (what the box
    # objective trains) is as high in every placement that holds a part of it, so those placements tie with the box
    assert block["stats"]["mean"]["top_decile"]["observed"] == 6
    assert block["stats"]["topk_mean"]["mean_percentile"] < block["stats"]["mean"]["mean_percentile"]


def test_a_response_all_over_the_tissue_ranks_the_boxes_like_any_other_placement():
    anatomy = _strip()
    flat = np.full((2, *anatomy.shape), 0.02, dtype=np.float32)
    flat[0][anatomy == 1] = 0.97                                    # the whole tissue responds, lesion or not
    block = summarize_ranking_v5([_scan(flat, _six_boxes(), anatomy)], CLASSES)["boxes"]["overall"]
    entry = block["stats"]["topk_mean"]
    assert entry["mean_percentile"] == pytest.approx(0.5) and entry["p_normal"] == 1.0     # every placement ties
    assert entry["top_decile"]["observed"] == 0
    noisy = 0.5 + 0.4 * _random(anatomy, seed=9)                    # a response unrelated to the boxes
    entry = summarize_ranking_v5([_scan(noisy, _six_boxes(), anatomy)], CLASSES)["boxes"]["overall"]["stats"]["mean"]
    assert 0.2 < entry["mean_percentile"] < 0.8 and entry["p_normal"] > 0.01


def test_under_the_null_the_p_values_are_not_small_more_often_than_they_say():
    """Boxes drawn uniformly over the placements that hold tissue (hence uniform within each octave class) on one
    fixed probability map: P(p <= a) must not exceed a, up to the noise of 400 simulated splits."""
    anatomy = _slab((24, 14, 4))
    probability = _random(anatomy, seed=21)
    size = [3, 3, 2]
    starts = [start for start in np.ndindex(*[n - m + 1 for n, m in zip(anatomy.shape, size)])
              if (anatomy[tuple(slice(a, a + n) for a, n in zip(start, size))] == 1).any()]
    measured = [_scan(probability, [_box("cartilage_lesion", "femoral_cartilage", list(start), size)], anatomy)
                for start in starts]                                  # every box the simulation can draw, once
    rng = np.random.default_rng(5)
    normal, decile, mean = [], [], []
    for _ in range(400):
        drawn = [measured[int(i)] for i in rng.integers(len(starts), size=10)]
        rows = [dict(row, patient_uid=f"p{i}") for i, row in enumerate(drawn)]
        entry = summarize_ranking_v5(rows, CLASSES, n_bootstrap=0)["boxes"]["overall"]["stats"]["topk_mean"]
        normal.append(entry["p_normal"])
        decile.append(entry["top_decile"]["p_at_least_observed"])
        mean.append(entry["mean_percentile"])
        assert entry["patients"]["ci95_low"] is None                  # no resampling was asked for
    assert np.mean(mean) == pytest.approx(0.5, abs=0.02)
    for alpha in (0.01, 0.05, 0.1, 0.25):
        slack = 3 * np.sqrt(alpha * (1 - alpha) / 400)
        assert (np.asarray(normal) <= alpha).mean() <= alpha + slack + 0.01      # a normal approximation
        assert (np.asarray(decile) <= alpha).mean() <= alpha + slack             # exact


def test_patients_are_the_unit_of_the_interval_and_two_scans_of_one_patient_count_once():
    anatomy = _strip()
    rows = []
    for index, patient in enumerate(["p0", "p0", "p1", "p2"]):
        probability = 0.05 * _random(anatomy, seed=index)
        boxes = _six_boxes()[index:index + 2]
        for box in boxes:
            probability[(0, *_window(box))] += 0.9
        rows.append(_scan(probability, boxes, anatomy, patient=patient, key=(7, index)))
    block = summarize_ranking_v5(rows, CLASSES, n_bootstrap=200)["boxes"]["overall"]
    assert block["n_boxes"] == 8 and block["n_patients"] == 3
    patients = block["stats"]["topk_mean"]["patients"]
    assert patients["n"] == 3 and patients["above_half"] == 3
    assert 0.5 < patients["ci95_low"] <= patients["mean"] <= patients["ci95_high"] <= 1.0
    again = summarize_ranking_v5(rows, CLASSES, n_bootstrap=200)["boxes"]["overall"]["stats"]["topk_mean"]["patients"]
    assert again == patients                                                    # the resampling is seeded
    by_class = summarize_ranking_v5(rows, CLASSES, n_bootstrap=200)["boxes"]["by_class"]
    assert by_class["cartilage_lesion"]["n_boxes"] == 8 and by_class["meniscal_tear"]["n_boxes"] == 0
    assert by_class["meniscal_tear"]["stats"]["topk_mean"]["mean_percentile"] is None


# ---------------------------------------------------------------------------
# Zones of the supervision, without a threshold


def _zoned():
    anatomy = _slab()
    boxes = [_box("cartilage_lesion", "femoral_cartilage", [3, 3, 0], [4, 3, 2]),
             _box("meniscal_tear", "meniscus", [12, 1, 1], [5, 4, 3]),
             _box("cartilage_lesion", "femoral_cartilage", [6, 7, 1], [3, 3, 3], analysis="secondary")]
    return anatomy, boxes


def test_the_zones_split_the_eligible_volume_as_the_supervision_does():
    anatomy, boxes = _zoned()
    eligible = np.ones(anatomy.shape, dtype=bool)
    eligible[0] = False
    row = _scan(_random(anatomy), boxes, anatomy, eligible=eligible)
    zones = row["zones"]["cartilage_lesion"]
    assert set(zones) == set(ZONES) and all(len(zone["hist"]) == BINS for zone in zones.values())
    assert sum(zone["voxels"] for zone in zones.values()) == int(eligible.sum())
    assert all(sum(zone["hist"]) == zone["voxels"] for zone in zones.values())
    primary, ignored = np.zeros(anatomy.shape, dtype=bool), np.zeros(anatomy.shape, dtype=bool)
    primary[_window(boxes[0])] = True
    ignored[_window(boxes[2])] = True
    cartilage = anatomy == 1
    assert zones["bag"]["voxels"] == int((primary & cartilage).sum())
    assert zones["box_rest"]["voxels"] == int((primary & ~cartilage & eligible).sum())
    assert zones["ignored_host"]["voxels"] == int((ignored & ~primary & cartilage).sum())
    assert zones["ignored_rest"]["voxels"] == int((ignored & ~primary & ~cartilage).sum())
    outside = ~(primary | ignored) & eligible
    # the three parts of the class's negative support: its own tissue, other segmented tissue (here the meniscus,
    # with the primary box of the other class), and everything that is no segmented tissue
    assert zones["host_outside"]["voxels"] == int((outside & cartilage).sum())
    assert zones["other_tissue"]["voxels"] == int((outside & (anatomy > 1)).sum())
    assert zones["background"]["voxels"] == int((outside & (anatomy == 0)).sum())
    probability = _random(anatomy)
    assert zones["bag"]["sum"] == pytest.approx(float(probability[0][primary & cartilage].sum()), rel=1e-5)


def test_the_area_between_two_histograms_is_the_rank_statistic_of_the_values():
    rng = np.random.default_rng(1)
    a, b = rng.beta(2, 5, 4000), rng.beta(2, 3, 5000)
    anatomy = np.zeros((90, 10, 10), dtype=np.int64)
    anatomy[:40] = 1                                             # 4000 voxels of cartilage
    probability = np.zeros((2, 90, 10, 10), dtype=np.float32)
    probability[0][:40] = a.reshape(40, 10, 10)
    probability[0][40:] = b.reshape(50, 10, 10)
    zones = _scan(probability, [], anatomy)["zones"]["cartilage_lesion"]
    from scipy.stats import mannwhitneyu
    exact = mannwhitneyu(a.astype(np.float32), b.astype(np.float32)).statistic / (a.size * b.size)
    assert histogram_auc_v5(zones["host_outside"]["hist"], zones["background"]["hist"]) == pytest.approx(exact, abs=2e-3)
    assert histogram_auc_v5(zones["host_outside"]["hist"], zones["host_outside"]["hist"]) == pytest.approx(0.5)
    assert histogram_auc_v5(zones["bag"]["hist"], zones["background"]["hist"]) is None            # an empty zone


def test_following_the_tissue_and_following_the_boxes_are_read_apart():
    anatomy = _strip()
    boxes = _six_boxes()
    on_tissue = 0.02 + 0.01 * _random(anatomy)
    on_tissue[0][anatomy == 1] = 0.9 + 0.05 * _random(anatomy, seed=4)[0][anatomy == 1]     # anatomy, not lesions
    summary = summarize_ranking_v5([_scan(on_tissue, boxes, anatomy)], CLASSES)["zones"]["cartilage_lesion"]
    assert summary["auc"]["host_vs_nonhost"]["pooled"] > 0.99
    assert 0.4 < summary["auc"]["bag_vs_host_outside"]["pooled"] < 0.6
    at_boxes = 0.02 + 0.01 * _random(anatomy)
    for box in boxes:
        at_boxes[(0, *_window(box))][anatomy[_window(box)] == 1] = 0.9
    summary = summarize_ranking_v5([_scan(at_boxes, boxes, anatomy)], CLASSES)["zones"]["cartilage_lesion"]
    assert summary["auc"]["bag_vs_host_outside"]["pooled"] > 0.99
    top = summary["top_volume"]["0.01"]
    assert top["share"]["bag"] > 0.9 and top["enrichment"]["bag"] > 10      # the highest 1 % of the volume is in the bags


def test_the_negative_support_is_counted_by_tissue_and_by_the_boxes_of_the_scan():
    anatomy, boxes = _zoned()
    rows = [_scan(_random(anatomy), boxes, anatomy, patient="p0", key=(7, 0)),
            _scan(_random(anatomy, seed=4), boxes[2:], anatomy, patient="p1", key=(7, 1)),      # an ignore region only
            _scan(_random(anatomy, seed=5), [], anatomy, patient="p2", key=(7, 2))]              # no box at all
    assert [row["scan"]["cartilage_lesion"]["group"] for row in rows] == ["with_primary", "other_boxes_only", "no_box"]
    assert rows[0]["scan"]["meniscal_tear"]["group"] == "with_primary"
    summary = summarize_ranking_v5(rows, CLASSES)
    negative = summary["zones"]["cartilage_lesion"]["negative_support"]
    total = sum(sum(row["zones"]["cartilage_lesion"][zone]["voxels"] for zone in ("host_outside", "other_tissue", "background"))
                for row in rows)
    assert negative["voxels"] == total
    assert sum(negative["by_zone"][zone]["voxels"] for zone in ("host_outside", "other_tissue", "background")) == total
    assert sum(negative["by_scan_group"][group]["voxels"] for group in ("with_primary", "other_boxes_only", "no_box")) == total
    assert negative["by_scan_group"]["no_box"]["scans"] == 1
    assert sum(item["share"] for item in negative["by_zone"].values()) == pytest.approx(1.0)
    scans = summary["scans"]["cartilage_lesion"]
    assert {group: scans["groups"][group]["n"] for group in scans["groups"]} == {
        "with_primary": 1, "other_boxes_only": 1, "no_box": 1}
    host = rows[2]["scan"]["cartilage_lesion"]
    assert host["host_voxels"] == int((anatomy == 1).sum())
    values = np.sort(_random(anatomy, seed=5)[0][anatomy == 1].astype(np.float64))
    assert host["host_mean"] == pytest.approx(values.mean(), rel=1e-6)
    assert host["host_top_mean"] == pytest.approx(values[-math.ceil(0.01 * values.size):].mean(), rel=1e-6)


def test_scans_with_a_box_are_told_from_scans_without_when_the_response_knows_the_lesion():
    anatomy = _strip()
    rows = []
    for index in range(8):
        boxes = _six_boxes()[:1] if index < 4 else []
        probability = 0.05 * _random(anatomy, seed=index)
        for box in boxes:
            probability[(0, *_window(box))] += 0.9
        rows.append(_scan(probability, boxes, anatomy, patient=f"p{index}", key=(7, index)))
    scans = summarize_ranking_v5(rows, CLASSES, n_bootstrap=200)["scans"]["cartilage_lesion"]
    told = scans["auc_host_top_mean"]["with_primary_vs_no_box"]
    assert (told["n_with"], told["n_without"], told["auc"]) == (4, 4, 1.0)
    assert told["ci95_low"] == 1.0 and told["ci95_high"] == 1.0
    assert scans["auc_host_top_mean"]["with_primary_vs_rest"]["auc"] == 1.0


def test_the_report_carries_counts_and_no_identifier():
    anatomy, boxes = _zoned()
    summary = summarize_ranking_v5([_scan(_random(anatomy), boxes, anatomy, patient="patient-secret")], CLASSES,
                                   n_bootstrap=50)
    text = render_ranking_md_v5({"train": summary, "development_val": summary})
    assert "patient-secret" not in text and "uid" not in text
    for phrase in ("不用阈值", "train", "development_val", "cartilage_lesion", "meniscal_tear", "可比摆放", "百分位",
                   "弱标注", "不是体素级", "不等于正常", "患者", "宿主组织", "仅诊断"):
        assert phrase in text, phrase
