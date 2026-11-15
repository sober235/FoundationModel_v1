"""Threshold-free reading of the abnormality probabilities against the readers' boxes (researcher's decision of
2026-11-14; frozen in reports/v5/ranking_audit_prereg_20261114.md).

The official chain reads a probability map through a threshold, connected components and an anchor. Every checkpoint
so far hits the boxes no more often than a random place on the same tissue (findings_chance_v5), which leaves open
whether the probabilities themselves know where the lesions are. This module asks that without any of the three
steps. Diagnostic only: nothing here is an official metric, and no evaluation rule depends on it.

WEAK LABELS. A box is not a lesion mask: the readers did not say which voxels of a box are lesion, and a place
without a box is not a normal place. No voxel is counted as a confirmed positive or negative; nothing here is a
voxel-level AUROC of lesions. What is measured is whether the response is higher where the readers drew a box.

1. BOXES AGAINST THEIR COMPARABLE PLACEMENTS (the part to read).
   For every primary box, its *bag* is the reference host tissue inside the box, as in training (box_data_v5).
   Statistics of the class's probability on the bag:
     topk_mean       mean of the highest ceil(topk_fraction·n) probabilities: the quantity the box objective trains;
     mean            mean probability;
     auc             rank statistic of the bag against all other voxels of the same tissue group in the scan
                     (Mann-Whitney, ties one half); None when the bag is the whole tissue;
     enrichment_q    share of the bag's voxels among the highest q of the tissue group in the scan, divided by q
                     (q = 0.01, 0.05, 0.10; a voxel is among them when fewer than q·N voxels lie above it, ties
                     counted one half).
   The same statistics are computed for placements of the box elsewhere in the scan: same size, inside the volume,
   holding a number of host-tissue voxels in the same octave [2^k, 2^(k+1)) as the real box: the equivalence
   classes of findings_chance_v5. With at most `n_placements` comparable placements all of them are used
   (`exhaustive`); otherwise `n_placements` are drawn uniformly with replacement by a seeded generator and the real
   box is added. Either way the real box is, under the null "the box could equally lie anywhere in its class", one
   exchangeable member of its pool, so:
     percentile       mid-rank position of the box in its pool, (below + equal/2) / pool: 0.5 expected, 1 the top;
     p_at_least       share of the pool at least as high as the box: a valid randomisation p-value of that box;
     null_variance    variance of the percentile of a uniformly drawn pool member (1/12 without ties);
     p_top_decile     share of the pool whose own percentile is >= 0.9.
   Nothing crosses from one patient to another. The unit of inference is the box (and the patient), never the
   voxel: voxels are spatially correlated, and the null distribution comes from the same map.
   What it cannot tell: a response that follows where lesions usually are on a tissue, without reading this
   lesion, also ranks above a uniform placement.

2. ZONES OF THE SUPERVISION (description). Per class the eligible volume is split, first match first:
     bag            host tissue of a primary box of the class inside that box (the positive support);
     box_rest       the rest of those boxes (neither positive nor negative in training);
     ignored_host   the class's host tissues inside an ignore region; ignored_rest: the rest of the ignore regions;
     host_outside   the class's host tissues outside all of these; other_tissue: other segmented tissue outside;
     background     no segmented tissue, outside. The last three are the class's negative support in training.
   Each zone keeps its voxel count, the sum of its probabilities and a histogram over BINS equal steps of the
   logit, from which rank statistics between zones and the make-up of the highest part of the volume follow.

3. SCANS. Per class: the mean and the top 1 % mean of the probability on the class's host tissues, and whether the
   scan has a primary box of the class, only other boxes, or no box at all.

Rows carry a patient id; summaries and the report carry counts and distributions only.

AMENDMENT OF 2026-11-15 (reports/v5/ranking_audit_prereg_amendment_20261115.md; frozen before any real-data job).
The bag statistics and the pools are unchanged. What is read changed:
  * the group result is patient-equal: the boxes of a patient are averaged first; the effect is that mean minus the
    null expectation 0.5, its interval resamples patients (pointwise, not simultaneous), and its one-sided p uses
    the randomisation variance of the same patient-equal mean (`patients.p_normal`; boxes relocate independently
    under the null). `patients.p_normal_dependent` assumes the boxes of one patient move together (an upper bound of
    the variance): a sensitivity value, never used for the reading;
  * the family is the development_val comparisons of the checkpoints read (checkpoint x class, FAMILY_SIZE = 8),
    adjusted by Holm (ranking_family_v5); train is diagnostic and pooled classes are descriptive;
  * a comparison reads RANKING_EVIDENCE_DETECTED, NO_RANKING_EVIDENCE_DETECTED or NOT_INTERPRETABLE (read_comparison_v5);
  * the private row of a scan keeps each box's pool values, its seed key and the grid, so that every number of the
    reading can be recomputed without the model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata

from ..engine.box_data_v5 import _is_primary, _window
from .findings_chance_v5 import _octave, _upper_tail, placement_counts_v5
from .findings_diagnostics_v5 import _distribution

N_PLACEMENTS = 1000
N_BOOTSTRAP = 10000
SEED = 20261114
ENRICHMENT = (("enrichment_01", 0.01), ("enrichment_05", 0.05), ("enrichment_10", 0.10))
STATS = ("topk_mean", "mean", "auc", *(name for name, _ in ENRICHMENT))
PRIMARY_STAT = "topk_mean"
ZONES = ("bag", "box_rest", "ignored_host", "ignored_rest", "host_outside", "other_tissue", "background")
NEGATIVE_ZONES = ("host_outside", "other_tissue", "background")
HOST_ZONES = ("bag", "ignored_host", "host_outside")
SCAN_GROUPS = ("with_primary", "other_boxes_only", "no_box")
TOP_VOLUME = (0.001, 0.01)
# the reading of the amendment of 2026-11-15
NULL_PERCENTILE = 0.5
FAMILY_ROLE, DIAGNOSTIC_ROLE = "development_val", "train"
FAMILY_SIZE = 8                 # four checkpoints x two lesion classes on development_val
FAMILY_ALPHA = 0.05
MIN_PATIENTS = 10               # fewer usable patients: the support is inadequate
MAX_EXCLUDED_SHARE = 0.2        # more of the class's primary boxes without a bag: the support is inadequate
MIN_POOL = 20                   # a smaller pool cannot place its box below 0.05
MAX_SMALL_POOL_SHARE = 0.5      # more of the usable boxes with such a pool: the control says too little
DETECTED, NOT_DETECTED, NOT_INTERPRETABLE = (
    "RANKING_EVIDENCE_DETECTED", "NO_RANKING_EVIDENCE_DETECTED", "NOT_INTERPRETABLE")
HOST_TOP_FRACTION = 0.01
LOGIT_LIMIT, LOGIT_STEP = 16.0, 0.025
BINS = int(round(2 * LOGIT_LIMIT / LOGIT_STEP))


# ---------------------------------------------------------------------------
# One scan


def _bins(probability: np.ndarray) -> np.ndarray:
    """Histogram bin of each probability: equal steps of the logit on [-LOGIT_LIMIT, LOGIT_LIMIT), clipped."""
    p = np.clip(probability.astype(np.float64), 1e-12, 1.0 - 1e-12)
    index = np.floor((np.log(p) - np.log1p(-p) + LOGIT_LIMIT) / LOGIT_STEP)
    return np.clip(index, 0, BINS - 1).astype(np.int64)


def _bag_stats(values: np.ndarray, ranks: np.ndarray, total: int, topk_fraction: float) -> list:
    """The STATS of one bag: `values` its probabilities, `ranks` their average ranks (1 = lowest) among the `total`
    voxels of the tissue group in the scan."""
    n = values.size
    k = max(1, math.ceil(topk_fraction * n))
    top = np.partition(values, n - k)[n - k:]
    auc = None if total == n else (float(ranks.sum()) - n * (n + 1) / 2.0) / (n * (total - n))
    above = total - ranks                                  # voxels of the tissue above this one, ties one half
    return [float(top.mean()), float(values.mean()), auc,
            *(float((above < q * total).mean()) / q for _, q in ENRICHMENT)]


def _against_pool(own, pool: list) -> dict:
    """The box's value against its pool (which holds the box itself); members without a value are left out."""
    if own is None:
        return {"value": None, "percentile": None, "p_at_least": None, "null_variance": None, "p_top_decile": None,
                "pool_median": None, "pool_values": None}
    values = np.asarray([value for value in pool if value is not None], dtype=np.float64)
    size = values.size
    percentiles = (rankdata(values, method="average") - 0.5) / size
    below, equal = int((values < own).sum()), int((values == own).sum())
    return {"value": float(own), "percentile": (below + 0.5 * equal) / size,
            "p_at_least": float((values >= own).sum()) / size,
            "null_variance": float(((percentiles - 0.5) ** 2).mean()),
            "p_top_decile": float((percentiles >= 0.9).mean()), "pool_median": float(np.median(values)),
            "pool_values": [float(value) for value in np.sort(values)]}


def _tissue_ranks(probability: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """Average rank (1 = lowest) of every tissue voxel's probability among the tissue voxels; 0 elsewhere.
    float32 holds half-integer ranks exactly far beyond the size of a tissue."""
    ranks = np.zeros(tissue.shape, dtype=np.float32)
    ranks[tissue] = rankdata(probability[tissue].astype(np.float64), method="average")
    return ranks


def _box_row(box: dict, probability: np.ndarray, tissue: np.ndarray, ranks, shape: tuple, topk_fraction: float,
             rng: np.random.Generator, n_placements: int) -> dict:
    """`ranks`: a callable giving _tissue_ranks(probability, tissue), asked only when the box has a bag."""
    row = {"abnormality_class": box["abnormality_class"], "tissue_group": box["host"]["tissue_group"],
           "status": "no_host_tissue", "box_voxels": 0, "bag_voxels": 0, "group_voxels": int(tissue.sum()),
           "comparable_placements": 0, "pool": 0, "exhaustive": None, "stats": None}
    window = _window(box["box"], shape)
    if window is None:
        return row
    size = np.asarray([w.stop - w.start for w in window], dtype=np.int64)
    origin = np.asarray([w.start for w in window], dtype=np.int64)
    own = int(tissue[window].sum())
    row["box_voxels"] = int(size.prod())
    if own == 0:
        return row
    total = row["group_voxels"]
    ranks = ranks()
    low, _, counts = placement_counts_v5(tissue, size)
    starts = np.argwhere(_octave(counts) == int(_octave(own))) + low        # every comparable placement

    def stats_at(start: np.ndarray) -> list:
        region = tuple(slice(int(a), int(a + n)) for a, n in zip(start, size))
        held = tissue[region]
        return _bag_stats(probability[region][held].astype(np.float64), ranks[region][held].astype(np.float64), total,
                          topk_fraction)

    real = stats_at(origin)
    exhaustive = len(starts) <= n_placements
    if exhaustive:                                          # the real box is one of them
        pool = [stats_at(start) for start in starts]
    else:
        pool = [stats_at(starts[int(i)]) for i in rng.integers(len(starts), size=n_placements)] + [real]
    row.update(status="ok", bag_voxels=own, comparable_placements=int(len(starts)), pool=len(pool),
               exhaustive=bool(exhaustive),
               stats={name: _against_pool(real[k], [member[k] for member in pool]) for k, name in enumerate(STATS)})
    return row


def scan_group_v5(boxes: list, cls: str) -> str:
    """A scan by its boxes, for one class: it has a primary box of the class, only other boxes (ignore regions or
    primary boxes of another class), or no box at all."""
    if any(_is_primary(box) and box["abnormality_class"] == cls for box in boxes):
        return "with_primary"
    return "other_boxes_only" if boxes else "no_box"


def ranking_scan_v5(*, probability: np.ndarray, reference: dict, anatomy_reference: np.ndarray,
                    anatomy_eligible: np.ndarray, classes: list, group_ids: dict, groups_by_class: dict,
                    topk_fraction: float, patient_uid: str, seed_key: tuple,
                    n_placements: int = N_PLACEMENTS, grid: dict | None = None) -> dict:
    """One scan's private row. `probability`: [classes, *shape], restored to the grid of the reference anatomy;
    `group_ids`: tissue group -> anatomy class ids; `groups_by_class`: abnormality class -> its host tissue groups;
    `seed_key`: integers that name this scan in the split (the placements drawn depend on nothing else);
    `grid`: the geometry of the original grid, kept in the row as given."""
    anatomy = np.asarray(anatomy_reference)
    shape = tuple(int(n) for n in anatomy.shape)
    eligible = np.asarray(anatomy_eligible, dtype=bool)
    probability = np.asarray(probability)
    if probability.shape != (len(classes), *shape) or eligible.shape != shape:
        raise ValueError("GRID_MISMATCH")
    if not (isinstance(topk_fraction, (int, float)) and 0 < topk_fraction <= 1) or n_placements < 1:
        raise ValueError("INVALID_ARGUMENT")
    boxes = list(reference["boxes"])
    primary = [box for box in boxes if _is_primary(box)]
    tissue_of = {group: np.isin(anatomy, list(ids)) & eligible for group, ids in group_ids.items()}

    box_rows, ranked = [], {}
    for ordinal, box in enumerate(primary):
        c, group = classes.index(box["abnormality_class"]), box["host"]["tissue_group"]
        tissue = tissue_of.get(group)
        if tissue is None:
            tissue = np.zeros(shape, dtype=bool)

        def ranks(c=c, group=group, tissue=tissue) -> np.ndarray:        # once per class and tissue group
            if (c, group) not in ranked:
                ranked[(c, group)] = _tissue_ranks(probability[c], tissue)
            return ranked[(c, group)]

        rng = np.random.default_rng([int(v) for v in seed_key] + [ordinal])
        box_rows.append(_box_row(box, probability[c], tissue, ranks, shape, topk_fraction, rng, n_placements))
    del ranked

    in_ignored = np.zeros(shape, dtype=bool)
    for box in boxes:
        window = None if _is_primary(box) else _window(box["box"], shape)
        if window is not None:
            in_ignored[window] = True
    zones, scan = {}, {}
    for c, name in enumerate(classes):
        allowed = sorted({i for group in groups_by_class.get(name, []) for i in group_ids[group]})
        host = np.isin(anatomy, allowed) & eligible
        in_primary, bag = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        for box in primary:
            window = _window(box["box"], shape) if box["abnormality_class"] == name else None
            if window is None:
                continue
            in_primary[window] = True
            tissue = tissue_of.get(box["host"]["tissue_group"])
            if tissue is not None:
                bag[window] |= tissue[window]
        zone = np.full(shape, ZONES.index("background"), dtype=np.int64)
        zone[anatomy > 0] = ZONES.index("other_tissue")
        zone[host] = ZONES.index("host_outside")
        zone[in_ignored] = ZONES.index("ignored_rest")
        zone[in_ignored & host] = ZONES.index("ignored_host")
        zone[in_primary] = ZONES.index("box_rest")
        zone[bag] = ZONES.index("bag")
        values = probability[c][eligible].astype(np.float64)
        index = zone[eligible]
        histogram = np.bincount(index * BINS + _bins(values), minlength=len(ZONES) * BINS).reshape(len(ZONES), BINS)
        sums = np.bincount(index, weights=values, minlength=len(ZONES))
        zones[name] = {label: {"voxels": int(histogram[z].sum()), "sum": float(sums[z]),
                               "hist": [int(v) for v in histogram[z]]} for z, label in enumerate(ZONES)}
        on_host = np.sort(probability[c][host].astype(np.float64))
        scan[name] = {"group": scan_group_v5(boxes, name),
                      "host_voxels": int(on_host.size),
                      "host_mean": float(on_host.mean()) if on_host.size else None,
                      "host_top_mean": float(on_host[-max(1, math.ceil(HOST_TOP_FRACTION * on_host.size)):].mean())
                      if on_host.size else None}
    for ordinal, row in enumerate(box_rows):
        row["seed_key"] = [int(v) for v in seed_key] + [ordinal]
    return {"patient_uid": patient_uid, "classes": list(classes), "shape": list(shape), "grid": grid,
            "eligible_voxels": int(eligible.sum()), "topk_fraction": float(topk_fraction),
            "n_placements": int(n_placements), "boxes": box_rows, "zones": zones, "scan": scan}


# ---------------------------------------------------------------------------
# Summaries


def histogram_auc_v5(first: list, second: list):
    """P(a value of `first` is above a value of `second`) + half P(same bin), from two histograms over the same
    bins; None when one is empty."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if not a.sum() or not b.sum():
        return None
    below = np.cumsum(b) - b
    return float((a * (below + 0.5 * b)).sum() / (a.sum() * b.sum()))


def _rank_auc(first: list, second: list):
    """The same for two lists of values."""
    if not first or not second:
        return None
    ranks = rankdata(np.asarray([*first, *second], dtype=np.float64), method="average")
    n, m = len(first), len(second)
    return float((ranks[:n].sum() - n * (n + 1) / 2.0) / (n * m))


def _resample(patients: list, statistic, n_bootstrap: int, seed: int) -> tuple:
    """2.5 % and 97.5 % of `statistic(patients drawn with replacement)`; resamples without a value are left out."""
    if n_bootstrap <= 0 or not patients:
        return None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        value = statistic([patients[int(i)] for i in rng.integers(len(patients), size=len(patients))])
        if value is not None:
            values.append(value)
    if not values:
        return None, None
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def _stat_block(boxes: list, stat: str, n_bootstrap: int, seed: int) -> dict:
    """`boxes`: (patient, box row) of the usable boxes. One statistic over them."""
    entries = [(patient, row["stats"][stat]) for patient, row in boxes if row["stats"][stat]["value"] is not None]
    out = {"n": len(entries), "value": _distribution([e["value"] for _, e in entries]),
           "pool_median": _distribution([e["pool_median"] for _, e in entries]),
           "percentile": _distribution([e["percentile"] for _, e in entries]),
           "mean_percentile": None, "null_sd": None, "z": None, "p_normal": None,
           "top_decile": {"observed": 0, "expected": None, "p_at_least_observed": None},
           "boxes_p_at_most_0.05": sum(1 for _, e in entries if e["p_at_least"] <= 0.05),
           "patients": {"n": 0, "mean": None, "effect": None, "ci95_low": None, "ci95_high": None, "above_half": 0,
                        "null_sd": None, "z": None, "p_normal": None, "null_sd_dependent": None,
                        "p_normal_dependent": None}}
    if not entries:
        return out
    percentiles = np.asarray([e["percentile"] for _, e in entries], dtype=np.float64)
    mean = float(percentiles.mean())
    sd = math.sqrt(sum(e["null_variance"] for _, e in entries)) / len(entries)      # of the mean, under the null
    z = (mean - 0.5) / sd if sd > 0 else None
    observed = int((percentiles >= 0.9).sum())
    tail, _, _ = _upper_tail([e["p_top_decile"] for _, e in entries], observed)
    by_patient = {}
    for patient, entry in entries:
        by_patient.setdefault(patient, []).append(entry)
    means = [float(np.mean([e["percentile"] for e in group])) for _, group in sorted(by_patient.items())]
    low, high = _resample(means, lambda drawn: float(np.mean(drawn)), n_bootstrap, seed)
    # the patient-equal mean under the null: each box's percentile has mean 0.5 and its own null variance. Boxes
    # relocate independently (the stated null); "dependent" lets the boxes of one patient move together, which
    # bounds the variance of that patient's mean from above (sensitivity only)
    patient_mean = float(np.mean(means))
    independent = math.sqrt(sum(sum(e["null_variance"] for e in group) / len(group) ** 2
                                for group in by_patient.values())) / len(by_patient)
    dependent = math.sqrt(sum((sum(math.sqrt(e["null_variance"]) for e in group) / len(group)) ** 2
                              for group in by_patient.values())) / len(by_patient)

    def upper(null_sd: float) -> tuple:
        if not null_sd > 0:
            return None, 1.0
        score = (patient_mean - NULL_PERCENTILE) / null_sd
        return score, 0.5 * math.erfc(score / math.sqrt(2.0))

    patient_z, patient_p = upper(independent)
    out.update(mean_percentile=mean, null_sd=sd, z=z,
               p_normal=1.0 if z is None else 0.5 * math.erfc(z / math.sqrt(2.0)),
               top_decile={"observed": observed, "expected": float(np.mean([e["p_top_decile"] for _, e in entries])),
                           "p_at_least_observed": tail},
               patients={"n": len(means), "mean": patient_mean, "effect": patient_mean - NULL_PERCENTILE,
                         "ci95_low": low, "ci95_high": high,
                         "above_half": sum(1 for value in means if value > 0.5),
                         "null_sd": independent, "z": patient_z, "p_normal": patient_p,
                         "null_sd_dependent": dependent, "p_normal_dependent": upper(dependent)[1]})
    return out


def _box_block(rows: list, cls: str | None, n_bootstrap: int, seed: int) -> dict:
    selected = [(row["patient_uid"], box) for row in rows for box in row["boxes"]
                if cls is None or box["abnormality_class"] == cls]
    usable = [(patient, box) for patient, box in selected if box["status"] == "ok"]
    return {"n_boxes": len(selected), "n_usable": len(usable), "excluded": len(selected) - len(usable),
            "n_patients": len({patient for patient, _ in selected}),
            "exhaustive_boxes": sum(1 for _, box in usable if box["exhaustive"]),
            "small_pool_boxes": sum(1 for _, box in usable if box["pool"] < MIN_POOL),
            **{key: _distribution([box[key] for _, box in usable])
               for key in ("comparable_placements", "pool", "bag_voxels", "box_voxels")},
            "stats": {stat: _stat_block(usable, stat, n_bootstrap, seed) for stat in STATS}}


def _ratio(numerator: float, denominator: float):
    return None if not denominator else float(numerator) / float(denominator)


def _top_volume(items: list, q: float) -> dict:
    """Which zones hold the highest q of each scan's eligible volume, added over the scans. A bin that straddles
    the cut is shared between the zones in proportion to their counts in it."""
    top, volume = np.zeros(len(ZONES)), np.zeros(len(ZONES))
    for zones in items:
        histogram = np.asarray([zones[label]["hist"] for label in ZONES], dtype=np.float64)
        per_bin = histogram.sum(axis=0)
        need = q * per_bin.sum()
        if need <= 0:
            continue
        above = np.cumsum(per_bin[::-1])[::-1] - per_bin              # voxels in higher bins
        cut = int(np.flatnonzero(above < need)[0])                    # the lowest bin that is needed
        top += histogram[:, cut + 1:].sum(axis=1) + histogram[:, cut] * (need - above[cut]) / per_bin[cut]
        volume += histogram.sum(axis=1)
    share = {label: _ratio(top[z], top.sum()) for z, label in enumerate(ZONES)}
    return {"share": share,
            "enrichment": {label: None if share[label] is None or not volume[z] else
                           share[label] / (volume[z] / volume.sum()) for z, label in enumerate(ZONES)}}


def _merged(zones: dict, labels: tuple) -> np.ndarray:
    return np.sum([np.asarray(zones[label]["hist"], dtype=np.float64) for label in labels], axis=0)


def _zone_block(rows: list, cls: str) -> dict:
    items = [row["zones"][cls] for row in rows]
    groups = [row["scan"][cls]["group"] for row in rows]
    voxels = {label: sum(zones[label]["voxels"] for zones in items) for label in ZONES}
    total = sum(voxels.values())
    negative = sum(voxels[label] for label in NEGATIVE_ZONES)
    by_group = {group: sum(zones[label]["voxels"] for zones, of in zip(items, groups) if of == group
                           for label in NEGATIVE_ZONES) for group in SCAN_GROUPS}
    pairs = {"bag_vs_host_outside": (("bag",), ("host_outside",)),
             "host_vs_nonhost": (HOST_ZONES, ("other_tissue", "background"))}
    auc = {}
    for key, (first, second) in pairs.items():
        per_scan = [histogram_auc_v5(_merged(zones, first), _merged(zones, second)) for zones in items]
        pooled = histogram_auc_v5(np.sum([_merged(zones, first) for zones in items], axis=0),
                                  np.sum([_merged(zones, second) for zones in items], axis=0)) if items else None
        auc[key] = {"pooled": pooled, "per_scan": _distribution(per_scan),
                    "scans": sum(1 for value in per_scan if value is not None)}
    return {"zones": {label: {"voxels": voxels[label], "share_of_volume": _ratio(voxels[label], total),
                              "mean_probability": _ratio(sum(zones[label]["sum"] for zones in items), voxels[label])}
                      for label in ZONES},
            "negative_support": {
                "voxels": negative,
                "by_zone": {label: {"voxels": voxels[label], "share": _ratio(voxels[label], negative)}
                            for label in NEGATIVE_ZONES},
                "by_scan_group": {group: {"voxels": by_group[group], "share": _ratio(by_group[group], negative),
                                          "scans": sum(1 for of in groups if of == group)} for group in SCAN_GROUPS}},
            "auc": auc, "top_volume": {format(q, "g"): _top_volume(items, q) for q in TOP_VOLUME}}


def _scan_block(rows: list, cls: str, n_bootstrap: int, seed: int) -> dict:
    scans = [(row["patient_uid"], row["scan"][cls]) for row in rows if row["scan"][cls]["host_voxels"]]
    by_patient = {}
    for patient, scan in scans:
        by_patient.setdefault(patient, []).append(scan)
    patients = [by_patient[patient] for patient in sorted(by_patient)]

    def told(without: tuple):
        def statistic(drawn: list):
            flat = [scan for group in drawn for scan in group]
            return _rank_auc([scan["host_top_mean"] for scan in flat if scan["group"] == "with_primary"],
                             [scan["host_top_mean"] for scan in flat if scan["group"] in without])
        low, high = _resample(patients, statistic, n_bootstrap, seed)
        return {"n_with": sum(1 for _, scan in scans if scan["group"] == "with_primary"),
                "n_without": sum(1 for _, scan in scans if scan["group"] in without),
                "auc": statistic(patients), "ci95_low": low, "ci95_high": high}

    return {"n_scans": len(scans),
            "groups": {group: {"n": sum(1 for _, scan in scans if scan["group"] == group),
                               **{key: _distribution([scan[key] for _, scan in scans if scan["group"] == group])
                                  for key in ("host_mean", "host_top_mean")}} for group in SCAN_GROUPS},
            "auc_host_top_mean": {"with_primary_vs_no_box": told(("no_box",)),
                                  "with_primary_vs_rest": told(("other_boxes_only", "no_box"))}}


def summarize_ranking_v5(rows: list, classes: list, *, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Counts, distributions and tests over a split's rows (ranking_scan_v5); no patient or sample identifier.
    Intervals resample patients with a seeded generator; `n_bootstrap` 0 leaves them out."""
    pools = [box["pool"] for row in rows for box in row["boxes"] if box["status"] == "ok"]
    return {"n_scans": len(rows), "n_patients": len({row["patient_uid"] for row in rows}), "classes": list(classes),
            "n_bootstrap": int(n_bootstrap), "seed": int(seed), "largest_pool": max(pools) if pools else 0,
            "boxes": {"overall": _box_block(rows, None, n_bootstrap, seed),
                      "by_class": {cls: _box_block(rows, cls, n_bootstrap, seed) for cls in classes}},
            "zones": {cls: _zone_block(rows, cls) for cls in classes},
            "scans": {cls: _scan_block(rows, cls, n_bootstrap, seed) for cls in classes}}


# ---------------------------------------------------------------------------
# Report

_STAT_LABELS = {"topk_mean": "袋内最高 10% 体素的平均概率（框目标训练的量）", "mean": "袋内平均概率",
                "auc": "袋内体素对同一组织其余体素的排序（0.5 = 无差别）",
                "enrichment_01": "袋内体素进入本组织最高 1% 的比例 ÷ 1%",
                "enrichment_05": "袋内体素进入本组织最高 5% 的比例 ÷ 5%",
                "enrichment_10": "袋内体素进入本组织最高 10% 的比例 ÷ 10%"}
_ZONE_LABELS = {"bag": "同类主分析框内的宿主组织（袋，正监督）", "box_rest": "同类主分析框内的其余体素（不计正负）",
                "ignored_host": "忽略区内的宿主组织", "ignored_rest": "忽略区内的其余体素",
                "host_outside": "所有框之外的宿主组织（负监督）", "other_tissue": "所有框之外的其他已分割组织（负监督）",
                "background": "所有框之外、没有分割标注的体素（负监督）"}
_GROUP_LABELS = {"with_primary": "有该类主分析框", "other_boxes_only": "只有其他的框", "no_box": "没有任何框"}


def _cell(value) -> str:
    if value is None:
        return "NA"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def _p(value) -> str:
    return "NA" if value is None else f"{value:.4g}"


def render_ranking_md_v5(by_role: dict) -> str:
    """Counts and distributions of each split side by side; no patient, sample or finding identifier."""
    roles = list(by_role)
    classes = list(next(iter(by_role.values()))["classes"]) if roles else []
    first = next(iter(by_role.values())) if roles else {"n_bootstrap": 0, "seed": 0}

    def table(header: str, rows: list, cell=_cell) -> list:
        lines = [f"| {header} | " + " | ".join(roles) + " |", "| --- |" + " --- |" * len(roles)]
        for label, getter in rows:
            lines.append(f"| {label} | " + " | ".join(cell(getter(by_role[role])) for role in roles) + " |")
        return lines

    lines = ["# 不用阈值的框内响应排序诊断（仅诊断）", "",
             "本报告只含计数与分布；不含样本或患者 ID、路径或文件名。全部是对所读检查点的诊断，不是正式指标；"
             "train 上的数值只用于诊断，不能代替泛化评价。", "",
             "- 不用阈值、不连通块、不取定位点：直接读恢复到原始网格的异常概率。",
             "- **弱标注**：框不是病变轮廓，框内哪些体素是病变没有标注；没有框的地方不等于正常。这里没有任何体素被当作"
             "确认的阳性或阴性，下面的量都不是体素级的病变 AUROC，只回答“医生画框的地方，响应是否更高”。",
             "- 袋：主分析框内的宿主组织（参照解剖），与训练时的正监督相同。",
             "- 可比摆放：同一次扫描内、同样大小、所含宿主组织体素数与真实框处于同一个二倍区间的全部摆放；"
             "超过上限时用带种子的均匀抽样，真实框计入。百分位 = 真实框在自己的可比摆放里的中位秩位置，"
             "零假设（框可以等可能地落在其中任何一个）下的期望是 0.5。",
             "- 推断的单位是框与患者，不是体素（体素在空间上相关）。患者区间：对患者重抽样 "
             f"{first['n_bootstrap']} 次（种子 {first['seed']}）的 2.5%–97.5%。",
             "- 它不能说明的：模型如果只学到“病变通常在组织的哪一带”，没有读到这一处病变的影像证据，百分位也会高于 0.5。",
             ""]

    def scopes():
        yield "合计", None
        for name in classes:
            yield name, name

    def block(summary: dict, name: str | None) -> dict:
        return summary["boxes"]["overall"] if name is None else summary["boxes"]["by_class"][name]

    for title, name in scopes():
        lines += [f"## 主分析框对可比摆放：{title}", ""]
        lines += table("项", [
            ("主分析框", lambda s: block(s, name)["n_boxes"]),
            ("计入", lambda s: block(s, name)["n_usable"]),
            ("未计入（参照解剖里框内没有宿主组织）", lambda s: block(s, name)["excluded"]),
            ("患者", lambda s: block(s, name)["n_patients"]),
            ("可比摆放数：中位数", lambda s: block(s, name)["comparable_placements"]["p50"]),
            ("可比摆放数：最小", lambda s: block(s, name)["comparable_placements"]["min"]),
            ("用了全部可比摆放的框", lambda s: block(s, name)["exhaustive_boxes"]),
            ("袋的体素数：中位数", lambda s: block(s, name)["bag_voxels"]["p50"])])
        lines.append("")
        for stat in STATS:
            def entry(summary: dict, stat=stat) -> dict:
                return block(summary, name)["stats"][stat]

            lines += [f"### {title}：{_STAT_LABELS[stat]}", ""]
            lines += table("项", [
                ("有取值的框", lambda s: entry(s)["n"]),
                ("真实框的取值：中位数", lambda s: entry(s)["value"]["p50"]),
                ("可比摆放的取值（每个框取中位数）：中位数", lambda s: entry(s)["pool_median"]["p50"]),
                ("百分位：平均", lambda s: entry(s)["mean_percentile"]),
                ("百分位：第 25 百分位", lambda s: entry(s)["percentile"]["p25"]),
                ("百分位：中位数", lambda s: entry(s)["percentile"]["p50"]),
                ("百分位：第 75 百分位", lambda s: entry(s)["percentile"]["p75"]),
                ("零假设下平均百分位的标准差", lambda s: entry(s)["null_sd"]),
                ("百分位 ≥ 0.9 的框", lambda s: entry(s)["top_decile"]["observed"]),
                ("百分位 ≥ 0.9 的框：零假设下的期望比例", lambda s: entry(s)["top_decile"]["expected"]),
                ("单框 p ≤ 0.05 的框", lambda s: entry(s)["boxes_p_at_most_0.05"]),
                ("患者数", lambda s: entry(s)["patients"]["n"]),
                ("患者等权的平均百分位", lambda s: entry(s)["patients"]["mean"]),
                ("患者区间下限（2.5%）", lambda s: entry(s)["patients"]["ci95_low"]),
                ("患者区间上限（97.5%）", lambda s: entry(s)["patients"]["ci95_high"]),
                ("平均百分位高于 0.5 的患者", lambda s: entry(s)["patients"]["above_half"]),
                ("效应量（患者等权的平均百分位 − 0.5）", lambda s: entry(s)["patients"]["effect"])])
            lines += table("p（单侧，越小越不像偶然）", [
                ("患者等权的平均百分位高于 0.5（随机化分布的正态近似；各框独立换位）",
                 lambda s: entry(s)["patients"]["p_normal"]),
                ("同上，假设同一患者的框一起移动（敏感性，不进入判读）",
                 lambda s: entry(s)["patients"]["p_normal_dependent"]),
                ("框等权的平均百分位高于 0.5（随机化分布的正态近似）", lambda s: entry(s)["p_normal"]),
                ("百分位 ≥ 0.9 的框数不少于实际（泊松二项，精确）",
                 lambda s: entry(s)["top_decile"]["p_at_least_observed"])], cell=_p)
            lines.append("")
    for name in classes:
        lines += [f"## 按监督分区的概率（描述）：{name}", "",
                  "分区互不重叠，按训练时的监督划分；宿主组织取参照解剖。“负监督”三行是该类别在训练中被当作负样本的全部"
                  "体素：框外没有标注不等于正常，这只是训练规则。", ""]
        rows = []
        for zone in ZONES:
            rows += [(f"{_ZONE_LABELS[zone]}：体素", lambda s, zone=zone: s["zones"][name]["zones"][zone]["voxels"]),
                     (f"{_ZONE_LABELS[zone]}：占体积", lambda s, zone=zone: s["zones"][name]["zones"][zone]["share_of_volume"]),
                     (f"{_ZONE_LABELS[zone]}：平均概率",
                      lambda s, zone=zone: s["zones"][name]["zones"][zone]["mean_probability"])]
        lines += table("项", rows, cell=lambda v: "NA" if v is None else str(v) if isinstance(v, int) else f"{v:.5f}")
        lines += ["", "负监督体素的构成（整卷体素计数，不是训练时裁块里的计数）：", ""]
        lines += table("项", [
            ("负监督体素", lambda s: s["zones"][name]["negative_support"]["voxels"]),
            *[(f"其中 {_ZONE_LABELS[zone]}：占比",
               lambda s, zone=zone: s["zones"][name]["negative_support"]["by_zone"][zone]["share"])
              for zone in NEGATIVE_ZONES],
            *[item for group in SCAN_GROUPS for item in (
                (f"来自“{_GROUP_LABELS[group]}”的扫描：扫描数",
                 lambda s, group=group: s["zones"][name]["negative_support"]["by_scan_group"][group]["scans"]),
                (f"来自“{_GROUP_LABELS[group]}”的扫描：占比",
                 lambda s, group=group: s["zones"][name]["negative_support"]["by_scan_group"][group]["share"]))]],
            cell=lambda v: "NA" if v is None else str(v) if isinstance(v, int) else f"{v:.5f}")
        lines += ["", "分区之间的排序（由概率直方图算出，同一格内算一半；0.5 = 无差别；描述，不作检验）：", ""]
        lines += table("项", [
            ("袋 对 框外宿主组织：合并", lambda s: s["zones"][name]["auc"]["bag_vs_host_outside"]["pooled"]),
            ("袋 对 框外宿主组织：每次扫描，中位数",
             lambda s: s["zones"][name]["auc"]["bag_vs_host_outside"]["per_scan"]["p50"]),
            ("袋 对 框外宿主组织：有袋的扫描数", lambda s: s["zones"][name]["auc"]["bag_vs_host_outside"]["scans"]),
            ("宿主组织 对 非宿主体素：合并（响应是否跟着解剖走）",
             lambda s: s["zones"][name]["auc"]["host_vs_nonhost"]["pooled"]),
            ("宿主组织 对 非宿主体素：每次扫描，中位数",
             lambda s: s["zones"][name]["auc"]["host_vs_nonhost"]["per_scan"]["p50"])])
        for q in TOP_VOLUME:
            key = format(q, "g")
            lines += ["", f"每次扫描概率最高的 {q:.1%} 体素落在哪些分区（各扫描相加；富集 = 占比 ÷ 该分区占体积的比例）：", ""]
            rows = []
            for zone in ZONES:
                rows += [(f"{_ZONE_LABELS[zone]}：占比",
                          lambda s, zone=zone, key=key: s["zones"][name]["top_volume"][key]["share"][zone]),
                         (f"{_ZONE_LABELS[zone]}：富集",
                          lambda s, zone=zone, key=key: s["zones"][name]["top_volume"][key]["enrichment"][zone])]
            lines += table("项", rows)
        lines.append("")
    for name in classes:
        lines += [f"## 扫描一级的响应（描述）：{name}", "",
                  f"每次扫描在该类别宿主组织上的平均概率与最高 {HOST_TOP_FRACTION:.0%} 体素的平均概率，按扫描有没有框分组。"
                  "没有框不等于正常，所以“区分”一行只是弱标注下的描述，不是检出性能。", ""]
        rows = [("有宿主组织的扫描", lambda s: s["scans"][name]["n_scans"])]
        for group in SCAN_GROUPS:
            rows += [(f"{_GROUP_LABELS[group]}：扫描数", lambda s, group=group: s["scans"][name]["groups"][group]["n"]),
                     (f"{_GROUP_LABELS[group]}：宿主组织平均概率，中位数",
                      lambda s, group=group: s["scans"][name]["groups"][group]["host_mean"]["p50"]),
                     (f"{_GROUP_LABELS[group]}：宿主组织最高 1% 的平均概率，中位数",
                      lambda s, group=group: s["scans"][name]["groups"][group]["host_top_mean"]["p50"])]
        for key, label in (("with_primary_vs_no_box", "有该类主分析框 对 没有任何框"),
                           ("with_primary_vs_rest", "有该类主分析框 对 其余全部扫描")):
            rows += [(f"区分（最高 1% 的平均概率的排序）：{label}",
                      lambda s, key=key: s["scans"][name]["auc_host_top_mean"][key]["auc"]),
                     (f"{label}：患者区间下限（2.5%）",
                      lambda s, key=key: s["scans"][name]["auc_host_top_mean"][key]["ci95_low"]),
                     (f"{label}：患者区间上限（97.5%）",
                      lambda s, key=key: s["scans"][name]["auc_host_top_mean"][key]["ci95_high"])]
        lines += table("项", rows)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# The reading of the amendment of 2026-11-15: one family of development comparisons


def holm_adjust_v5(pvalues: list, family_size: int | None = None) -> list:
    """Holm's step-down adjusted p-values, in the order given. `family_size`: the number of comparisons of the
    family when some of them have no p-value here (they still count: the adjustment is then conservative)."""
    m = len(pvalues) if family_size is None else int(family_size)
    if m < len(pvalues) or any(not (0.0 <= p <= 1.0) for p in pvalues):
        raise ValueError("INVALID_ARGUMENT")
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    adjusted, running = [None] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def support_problems_v5(block: dict) -> list:
    """Why a class's boxes of one split cannot carry a reading (frozen thresholds); empty when they can."""
    entry = block["stats"][PRIMARY_STAT]
    problems = []
    if entry["patients"]["n"] < MIN_PATIENTS:
        problems.append("FEW_PATIENTS")
    if block["n_boxes"] and block["excluded"] > MAX_EXCLUDED_SHARE * block["n_boxes"]:
        problems.append("MANY_BOXES_WITHOUT_BAG")
    if block["n_usable"] and block["small_pool_boxes"] > MAX_SMALL_POOL_SHARE * block["n_usable"]:
        problems.append("SMALL_POOLS")
    if entry["patients"]["n"] and not (entry["patients"]["null_sd"] or 0) > 0:
        problems.append("NO_NULL_VARIANCE")
    if entry["patients"]["n"] and entry["patients"]["ci95_low"] is None:
        problems.append("NO_INTERVAL")
    return problems


def read_comparison_v5(patients: dict, holm_p, problems: list) -> str:
    """DETECTED: the effect is positive, the patient interval lies above the null, and the Holm-adjusted p is at
    most FAMILY_ALPHA. NOT_INTERPRETABLE: `problems` is not empty. NOT_DETECTED: anything else (which is not a
    proof that nothing is there)."""
    if problems or holm_p is None:
        return NOT_INTERPRETABLE
    detected = patients["effect"] > 0 and patients["ci95_low"] > NULL_PERCENTILE and holm_p <= FAMILY_ALPHA
    return DETECTED if detected else NOT_DETECTED


def _comparison(block: dict) -> dict:
    entry = block["stats"][PRIMARY_STAT]
    patients = entry["patients"]
    return {"n_boxes": block["n_boxes"], "n_usable": block["n_usable"], "excluded": block["excluded"],
            "small_pool_boxes": block["small_pool_boxes"], "exhaustive_boxes": block["exhaustive_boxes"],
            "pool": dict(block["pool"]), "n_patients": patients["n"], "mean": patients["mean"],
            "effect": patients["effect"], "ci95_low": patients["ci95_low"], "ci95_high": patients["ci95_high"],
            "above_half": patients["above_half"], "p": patients["p_normal"],
            "p_dependent": patients["p_normal_dependent"], "box_mean": entry["mean_percentile"],
            "box_p": entry["p_normal"], "top_decile": dict(entry["top_decile"])}


def ranking_family_v5(entries: list, *, family_size: int = FAMILY_SIZE) -> dict:
    """The comparisons of several checkpoints side by side. `entries`: [{"label", "checkpoint_sha256", "step",
    "plan_id", "problems": [...], "ranking": {role: summarize_ranking_v5(...)}}], every one read with the same
    settings. The family is label x class on FAMILY_ROLE (its p-values Holm-adjusted over `family_size`
    comparisons); DIAGNOSTIC_ROLE and the pooled classes are listed without adjustment and without a reading.
    `problems` of an entry (a failed job, a failed preflight) make all its comparisons NOT_INTERPRETABLE; an entry
    whose job left no result (`ranking` None) has no row and still counts in `family_size`. The
    pools of a box do not depend on the checkpoint, so the entries must agree on the counts of every block; when
    they do not, nothing is interpretable."""
    read = [entry for entry in entries if entry.get("ranking")]
    if not read or len({entry["label"] for entry in entries}) != len(entries) \
            or any(FAMILY_ROLE not in entry["ranking"] for entry in read):
        raise ValueError("INVALID_ARGUMENT")
    classes = list(read[0]["ranking"][FAMILY_ROLE]["classes"])
    if len(entries) * len(classes) > family_size:
        raise ValueError("FAMILY_TOO_SMALL")
    counts = ("n_boxes", "n_usable", "excluded", "small_pool_boxes", "exhaustive_boxes", "pool")
    rows, same_support = [], True
    for entry in read:                                  # an entry without a result has no row; it still counts
        for role, summary in entry["ranking"].items():
            for scope, block in (("pooled", summary["boxes"]["overall"]),
                                 *((name, summary["boxes"]["by_class"][name]) for name in classes)):
                row = {"label": entry["label"], "role": role, "class": scope, **_comparison(block),
                       "problems": [*entry.get("problems", []),
                                    *(support_problems_v5(block) if scope != "pooled" else [])]}
                first = next((r for r in rows if (r["role"], r["class"]) == (role, scope)), None)
                if first is not None and any(first[key] != row[key] for key in counts):
                    same_support = False
                rows.append(row)
    family = [row for row in rows if row["role"] == FAMILY_ROLE and row["class"] != "pooled"]
    if not same_support:
        for row in family:
            row["problems"].append("SUPPORT_DIFFERS_BETWEEN_CHECKPOINTS")
    readable = [row for row in family if not row["problems"] and row["p"] is not None]
    for row, adjusted in zip(readable, holm_adjust_v5([row["p"] for row in readable], family_size)):
        row["p_holm"] = adjusted
    for row in rows:
        in_family = row["role"] == FAMILY_ROLE and row["class"] != "pooled"
        row.setdefault("p_holm", None)
        row["family"] = in_family
        row["reading"] = read_comparison_v5(row, row["p_holm"], row["problems"]) if in_family else None
    return {"kind": "ranking_family", "primary_stat": PRIMARY_STAT, "null_percentile": NULL_PERCENTILE,
            "family_role": FAMILY_ROLE, "family_size": int(family_size), "alpha": FAMILY_ALPHA,
            "thresholds": {"min_patients": MIN_PATIENTS, "max_excluded_share": MAX_EXCLUDED_SHARE,
                           "min_pool": MIN_POOL, "max_small_pool_share": MAX_SMALL_POOL_SHARE},
            "classes": classes, "same_support": same_support,
            "entries": [{key: entry.get(key) for key in ("label", "checkpoint_sha256", "step", "plan_id", "problems")}
                        for entry in entries],
            "comparisons": rows}


_READING_LABELS = {DETECTED: "检出排序证据", NOT_DETECTED: "未检出排序证据", NOT_INTERPRETABLE: "不可判读", None: "—"}


def render_ranking_family_md_v5(family: dict) -> str:
    """The family of comparisons; counts, effects, intervals and p-values only."""
    def line(row: dict, with_reading: bool) -> str:
        cells = [row["label"], row["class"], f"{row['n_usable']}/{row['n_boxes']}", str(row["n_patients"]),
                 _cell(row["mean"]), "NA" if row["effect"] is None else f"{row['effect']:+.3f}",
                 f"{_cell(row['ci95_low'])} – {_cell(row['ci95_high'])}", _p(row["p"])]
        if with_reading:
            cells += [_p(row["p_holm"]), _p(row["p_dependent"]),
                      _READING_LABELS[row["reading"]] + (f"（{'、'.join(row['problems'])}）" if row["problems"] else "")]
        else:
            cells += [_p(row["p_dependent"])]
        return "| " + " | ".join(cells) + " |"

    thresholds = family["thresholds"]
    lines = ["# 不用阈值的排序审计：四个检查点的比较族（仅诊断）", "",
             "本报告只含计数、效应量、区间与 p 值；不含样本或患者 ID、路径或文件名。全部是对所读检查点的诊断，不是正式指标。", "",
             f"- 主要量：`{family['primary_stat']}` 的百分位（真实框在同一次扫描的可比摆放里的中位秩位置）。"
             f"零假设下的期望是 {family['null_percentile']}：这是**排序百分位**，不是病变概率的阈值。",
             "- 患者等权：先在患者内对框取平均，再对患者取平均。效应量 = 患者平均百分位 − 0.5。",
             "- 区间：对患者重抽样的 95% 区间，逐个比较给出（不是同时区间）。",
             "- p：患者等权平均百分位高于 0.5 的单侧 p（随机化分布的正态近似；零假设下各框独立地换位）。"
             "“同患者相依”一列假设同一患者的框一起移动（方差的上界），只作敏感性参考，不进入判读。",
             f"- 比较族：{family['family_role']} 上“检查点 × 病变类别”共 {family['family_size']} 个比较，"
             f"Holm 校正，族错误率 {family['alpha']}。train 只作诊断，两类合计只作描述；它们不校正、不判读。",
             f"- 判读：**检出排序证据** = 效应量为正，患者区间下限高于 0.5，且 Holm 校正后的 p ≤ {family['alpha']}；"
             "**未检出排序证据** = 没有达到上面的条件（不等于证明没有）；**不可判读** = 支持不足、对照假设不成立或实现问题"
             f"（可用患者少于 {thresholds['min_patients']}；没有袋的框超过 {thresholds['max_excluded_share']:.0%}；"
             f"可比摆放池小于 {thresholds['min_pool']} 的框超过 {thresholds['max_small_pool_share']:.0%}；"
             "作业或预检未通过；各检查点读到的框与摆放池不一致）。",
             "- 即使检出，也只表示“在所述解剖对照下，框处的概率更高”，不等于证明了病变特异的定位。",
             "- 各检查点读到的框与摆放池" + ("一致。" if family["same_support"] else "**不一致**。"), "",
             "## 读取的检查点", "", "| 代号 | 计划 | 步 | 检查点 sha256 | 问题 |", "| --- | --- | --- | --- | --- |"]
    for entry in family["entries"]:
        lines.append(f"| {entry['label']} | {entry['plan_id']} | {entry['step']} | `{entry['checkpoint_sha256']}` | "
                     f"{'、'.join(entry['problems'] or []) or '无'} |")
    rows = family["comparisons"]
    lines += ["", f"## 比较族：{family['family_role']}（判读）", "",
              "| 检查点 | 类别 | 计入框/主分析框 | 患者 | 患者平均百分位 | 效应量 | 95% 患者区间 | p | Holm p | "
              "p（同患者相依） | 判读 |", "| --- |" + " --- |" * 10]
    lines += [line(row, True) for row in rows if row["family"]]
    for title, chosen in ((f"两类合计：{family['family_role']}（描述，不校正、不判读）",
                           lambda r: r["role"] == family["family_role"] and r["class"] == "pooled"),
                          ("train（诊断，不校正、不判读；含训练过的框，不能代替泛化评价）",
                           lambda r: r["role"] != family["family_role"])):
        selected = [row for row in rows if chosen(row)]
        if selected:
            lines += ["", f"## {title}", "",
                      "| 检查点 | 类别 | 计入框/主分析框 | 患者 | 患者平均百分位 | 效应量 | 95% 患者区间 | p | "
                      "p（同患者相依） |", "| --- |" + " --- |" * 8]
            lines += [line(row, False) for row in selected]
    return "\n".join(lines) + "\n"
