"""Box supervision targets for the unified model (design 2026-11-10 §5.2). Synthetic arrays only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anatobind.v41.annotation_v5 import load_skm_box_policy_v5
from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.box_data_v5 import box_bags_v5, load_box_references_v5
from tests.v5.factories import make_split_records, make_split_registry

SCOPE = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_skm_joint_early.yaml").read_text()))
POLICY = load_skm_box_policy_v5(Path("configs/v5/skm_box_policy_v1.yaml"))
SHAPE = (8, 8, 4)


def _raises(code, path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (code, path)


def _anatomy():
    arrays = {name: np.zeros(SHAPE, dtype=bool) for name in SCOPE.anatomy_classes}
    arrays["femoral_cartilage"][0:4] = True
    arrays["meniscus_medial"][4:8, 0:4] = True
    return arrays


def _box(status, analysis, cls, group, start, size):
    return {"status": status, "analysis": analysis, "abnormality_class": cls, "box": {"start": start, "size": size},
            "host": None if group is None else {"tissue_group": group}}


def _reference(boxes):
    return {"sample_uid": "s-0", "policy_sha256": POLICY["policy_sha256"], "boxes": boxes}


def test_primary_box_becomes_a_bag_on_its_host_tissue_and_other_boxes_are_ignored():
    reference = _reference([
        _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [2, 0, 0], [4, 2, 2]),
        _box("included", "secondary", "cartilage_lesion", "femoral_cartilage", [0, 6, 0], [2, 2, 1]),
        _box("excluded", None, None, None, [7, 7, 3], [5, 5, 5]),              # runs out of the volume
        _box("out_of_scope", None, None, None, [4, 4, 0], [1, 1, 1])])
    out = box_bags_v5(reference, SHAPE, SCOPE, _anatomy(), POLICY)
    lesion, tear = (SCOPE.abnormality_classes.index(n) for n in ("cartilage_lesion", "meniscal_tear"))
    expected = np.zeros(SHAPE, dtype=np.int32)
    expected[2:4, 0:2, 0:2] = 1                                               # box ∩ femoral cartilage
    assert out["bags"].dtype == np.int32 and np.array_equal(out["bags"][lesion], expected)
    assert not out["bags"][tear].any() and out["n_bags"] == {"cartilage_lesion": 1, "meniscal_tear": 0}
    assert np.array_equal(out["e_pos"][lesion], expected > 0)
    assert int(out["e_neg"][lesion].sum()) == 256 - (16 + 4 + 1 + 1)          # whole primary box + three ignore boxes
    assert int(out["e_neg"][tear].sum()) == 256 - (4 + 1 + 1)                 # ignore boxes only


def test_overlapping_bags_of_one_class_go_to_the_smaller_one():
    reference = _reference([
        _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [0, 0, 0], [4, 4, 1]),
        _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [2, 2, 0], [2, 2, 1])])
    bags = box_bags_v5(reference, SHAPE, SCOPE, _anatomy(), POLICY)["bags"][0]
    assert int((bags == 2).sum()) == 4 and int((bags == 1).sum()) == 12


def test_a_scan_without_primary_boxes_has_only_negative_support():
    out = box_bags_v5(_reference([]), SHAPE, SCOPE, _anatomy(), POLICY)
    assert not out["bags"].any() and not out["e_pos"].any() and out["e_neg"].all()


def test_positive_centers_keep_bags_with_even_ids():
    from anatobind.v41.engine.box_data_v5 import box_positive_centers_v5
    from anatobind.v41.engine.data_v5 import TrainingSampleV5
    from tests.v5.factories import make_training_sample_v5

    base = make_training_sample_v5(SCOPE, "s-0", shape=SHAPE)
    bags = np.zeros((2, *SHAPE), dtype=np.int32)
    bags[0, 7, 7, 3] = 2                                                      # an even bag id
    sample = TrainingSampleV5(observation=base.observation, targets={"A": np.zeros(SHAPE, np.int64), "U": bags},
                              e_pos={"A": base.e_pos["A"], "U": bags > 0}, e_neg=base.e_neg)
    assert box_positive_centers_v5(sample).tolist() == [[7, 7, 3]]


def test_grid_and_class_errors_are_named():
    empty_host = _reference([_box("included", "primary", "meniscal_tear", "meniscus", [0, 0, 0], [2, 2, 2])])
    _raises("BOX_HOST_EMPTY", "$.boxes[0]", box_bags_v5, empty_host, SHAPE, SCOPE, _anatomy(), POLICY)
    unknown = _reference([_box("included", "primary", "other_class", "meniscus", [4, 0, 0], [2, 2, 2])])
    _raises("BOX_CLASS_UNKNOWN", "$.boxes[0].abnormality_class", box_bags_v5, unknown, SHAPE, SCOPE, _anatomy(), POLICY)
    _raises("GRID_MISMATCH", "$.anatomy", box_bags_v5, _reference([]), (8, 8, 5), SCOPE, _anatomy(), POLICY)


def test_references_are_read_by_frozen_role_only(tmp_path):
    records = make_split_records(["train", "development_val", "locked_test"])
    registry = make_split_registry(records)
    path = tmp_path / "box_references.jsonl"
    path.write_text("".join(json.dumps({"sample_uid": f"s-{i}", "policy_sha256": "0" * 64, "boxes": []}) + "\n"
                            for i in range(3)))
    assert set(load_box_references_v5(path, registry, ["train"])) == {"s-0"}
    assert set(load_box_references_v5(path, registry, ["train", "development_val"])) == {"s-0", "s-1"}
    _raises("LOCKED_TEST_NOT_READABLE", "$.roles[0]", load_box_references_v5, path, registry, ["locked_test"])


def test_a_box_sample_has_released_anatomy_and_bag_targets(tmp_path):
    from anatobind.v41.engine.box_data_v5 import load_box_training_sample_v5
    from tests.v5.factories import SYNTH_LABEL_MAPS, make_recipe, make_scope, make_supervision, write_volume_case

    scope = validate_scope_v5(make_scope())
    record = write_volume_case(tmp_path, scope, uid="s-0")                    # tissue_a on x < 4, tissue_b on x 4..5
    policy = {"policy_sha256": "p" * 64, "host_tissues": {"T": {"group": "g", "anatomy": ["tissue_a"]}}}
    reference = {"sample_uid": "s-0", "policy_sha256": "p" * 64,
                 "boxes": [_box("included", "primary", "target_a", "g", [2, 0, 0], [4, 2, 2])]}
    regions = {"review": {}, "visibility": {}}

    def load(recipe=None, reference=reference, policy=policy):
        return load_box_training_sample_v5(tmp_path, record, scope, make_supervision(scope),
                                           recipe or make_recipe(model={"tasks": ["A", "U"]}), SYNTH_LABEL_MAPS,
                                           regions, reference, policy)

    sample = load()
    assert set(sample.targets) == {"A", "U"} and sample.targets["U"].dtype == np.int32
    assert sample.targets["U"].shape == (len(scope.abnormality_classes), 8, 8, 8)
    assert np.array_equal(sample.e_pos["U"], sample.targets["U"] > 0)
    assert int((sample.targets["U"][0] == 1).sum()) == 8                      # box ∩ tissue_a: x 2..3, y 0..1, z 0..1
    assert int(sample.e_neg["U"][0].sum()) == 512 - 16 and sample.e_neg["U"][1].all()
    _raises("BOX_DOWNSAMPLE_NOT_SUPPORTED", "$.recipe.downsample_factor", load,
            make_recipe(model={"tasks": ["A", "U"]}, downsample_factor=[2, 2, 1]))
    _raises("BOX_REFERENCE_MISMATCH", "$.reference.sample_uid", load, reference=dict(reference, sample_uid="s-9"))
    _raises("BOX_POLICY_MISMATCH", "$.reference.policy_sha256", load, policy=dict(policy, policy_sha256="q" * 64))
