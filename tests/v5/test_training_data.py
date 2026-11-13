"""v5.1 observation loading, label decoding, eligibility-aware patches and sampling (plan_v5 T09b; L2, L4, L5)."""
from __future__ import annotations

import copy
import dataclasses
import io

import numpy as np
import pytest
import torch
import yaml

from anatobind.v41.contracts_v5 import V5Error, validate_sample_v5, validate_scope_v5
from anatobind.v41.engine.data_v5 import (
    PatchSamplerV5, decode_targets_v5, extract_patch_v5, load_label_maps_v5, load_observation_v5,
    load_training_sample_v5, load_training_set_v5, make_batch_v5, positive_centers_v5)
from anatobind.v41.engine.volume_io_v5 import VolumeV5
from anatobind.v41.models.interface_v5 import ModelInputV5
from tests.v5.factories import (
    CONFIG_DIR_V5, SYNTH_LABEL_MAPS, make_module_scope, make_recipe, make_sample_v5, make_split_registry,
    make_supervision, make_training_sample_v5, write_nifti, write_skm_h5, write_volume_case)

NO_REGIONS = {"review": {}, "visibility": {}}


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _scope(task="A", **overrides):
    return validate_scope_v5(make_module_scope(task, **overrides))


def _recipe(scope, **overrides):
    tasks = ["A"] if scope.anatomy_classes else ["U"]
    return make_recipe(model={"tasks": tasks}, **overrides)


def _with(record, **fields):
    body = dataclasses.asdict(record)
    body.update(fields)
    return validate_sample_v5(body)


def _annotation(**fields):
    return validate_sample_v5(make_sample_v5(annotation=fields)).annotation_records[0]


def test_record_modalities_must_equal_scope_inputs(tmp_path):
    scope = _scope("A")
    record = write_volume_case(tmp_path, scope)
    observation = load_observation_v5(tmp_path, record, scope, _recipe(scope))
    assert list(observation.channels) == ["synthetic_e1", "synthetic_e2"]
    swapped = _with(record, model_modalities=["synthetic_e2", "synthetic_e1"])
    _raises("MODALITY_SCOPE_MISMATCH", "$.model_modalities", load_observation_v5, tmp_path, swapped, scope,
            _recipe(scope))
    unavailable = _with(record, available_modality_mask=[True, False])
    _raises("INPUT_MODALITY_UNAVAILABLE", "$.available_modality_mask[1]", load_observation_v5, tmp_path,
            unavailable, scope, _recipe(scope))
    # Files listed in the other order are still read by modality name, never by position.
    reordered = _with(record, modality_names=["synthetic_e2", "synthetic_e1"],
                      image_paths=list(reversed(record.image_paths)))
    again = load_observation_v5(tmp_path, reordered, scope, _recipe(scope))
    for name in scope.input_modalities:
        assert np.array_equal(again.channels[name], observation.channels[name])


def test_observation_loader_reads_no_labels(tmp_path):
    scope = _scope("A")
    record = write_volume_case(tmp_path, scope, label=False)
    observation = load_observation_v5(tmp_path, record, scope, _recipe(scope))
    assert observation.sample_uid == "case-1" and observation.original_shape == (8, 8, 8)
    assert observation.valid_fov.all() and observation.to_original == []
    assert observation.grid["axis_order"] == "xyz" and observation.grid["spacing_mm"] == [1.0, 1.0, 2.0]
    assert observation.grid["units_reliable"] is False
    for channel in observation.channels.values():
        assert channel.dtype == np.float32
        assert abs(float(channel.mean())) < 1e-5 and abs(float(channel.std()) - 1.0) < 1e-4
    _raises("VOLUME_MISSING", "$.ref", load_training_sample_v5, tmp_path, record, scope, make_supervision(scope),
            _recipe(scope), SYNTH_LABEL_MAPS, NO_REGIONS)


def test_observation_downsamples_by_stride_and_records_transform(tmp_path):
    scope = _scope("A")
    record = write_volume_case(tmp_path, scope, physical_geometry_status="verified",
                               geometry_evidence={"index_alignment": None, "physical_geometry": "synthetic"})
    observation = load_observation_v5(tmp_path, record, scope, _recipe(scope, downsample_factor=[2, 2, 1]))
    assert observation.valid_fov.shape == (4, 4, 8) and observation.original_shape == (8, 8, 8)
    assert observation.to_original == [{"op": "downsample", "factor": [2, 2, 1], "input_shape": [8, 8, 8]}]
    assert observation.grid["shape"] == [4, 4, 8] and observation.grid["spacing_mm"] == [2.0, 2.0, 2.0]
    assert np.allclose(observation.grid["affine"], np.diag([2.0, 2.0, 2.0, 1.0]))
    assert observation.grid["units_reliable"] is True
    flat = write_volume_case(tmp_path, scope, uid="flat")
    write_nifti(tmp_path / flat.image_paths[0], np.ones((8, 8, 8), dtype=np.float32), np.diag([1.0, 1.0, 2.0, 1.0]))
    _raises("DEGENERATE_CHANNEL", "$.channels[0]", load_observation_v5, tmp_path, flat, scope, _recipe(scope))


def test_label_maps_decode_channels_and_values(tmp_path):
    maps = load_label_maps_v5(CONFIG_DIR_V5 / "label_maps_v5.yaml")
    skm_scope = yaml.safe_load((CONFIG_DIR_V5 / "scope_skm_a.yaml").read_text())
    assert maps["skm_a6_h5_channels_v1"]["encoding"] == "channels"
    assert list(maps["skm_a6_h5_channels_v1"]["classes"]) == skm_scope["anatomy_classes"]
    assert list(maps["skm_a6_h5_channels_v1"]["classes"].values()) == [0, 1, 2, 3, 4, 5]
    assert maps["isles_binary_v1"] == {"encoding": "values", "classes": {"ischemic_infarct": 1}}
    classes = ["tissue_a", "tissue_b"]
    seg = np.zeros((2, 2, 2, 2), dtype=bool)
    seg[0, 0, 0, 0] = seg[1, 1, 1, 1] = True
    by_channel = _annotation(task="A", class_scope=classes, label_map_version="synthetic_a_channels_v1")
    decoded = decode_targets_v5(VolumeV5(array=seg, header=None), by_channel, classes, SYNTH_LABEL_MAPS)
    assert decoded.dtype == bool and decoded.shape == (2, 2, 2, 2)
    assert decoded[0, 0, 0, 0] and decoded[1, 1, 1, 1] and int(decoded.sum()) == 2
    values = np.array([[[0, 1], [2, 0]], [[1, 1], [0, 2]]], dtype=np.int16)
    by_value = _annotation(task="A", class_scope=classes, label_map_version="synthetic_a_values_v1")
    decoded = decode_targets_v5(VolumeV5(array=values, header=None), by_value, classes, SYNTH_LABEL_MAPS)
    assert np.array_equal(decoded[0], values == 1) and np.array_equal(decoded[1], values == 2)
    unknown = _annotation(task="A", class_scope=classes, label_map_version="other_v9")
    _raises("LABEL_MAP_UNKNOWN", "$.label_map_version", decode_targets_v5, VolumeV5(values, None), unknown, classes,
            SYNTH_LABEL_MAPS)
    seg[0, 0, 0, 1] = True
    _raises("LABEL_OVERLAP", "$.label", decode_targets_v5, VolumeV5(seg, None), by_channel, classes, SYNTH_LABEL_MAPS)
    _raises("LABEL_VALUE_UNEXPECTED", "$.label", decode_targets_v5, VolumeV5(values + 5 * (values == 2), None),
            by_value, classes, SYNTH_LABEL_MAPS)
    (tmp_path / "bad.yaml").write_text("broken_v1:\n  encoding: channels\n  classes: {tissue_a: 0, tissue_b: 0}\n")
    _raises("INVALID_LABEL_MAP", "$.broken_v1", load_label_maps_v5, tmp_path / "bad.yaml")


def test_lost_positive_blocks_have_no_negative_eligibility(tmp_path):
    scope = _scope("U")
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[1, 1, 1] = 1
    record = write_volume_case(tmp_path, scope, label=label)
    supervision = make_supervision(scope)
    full = load_training_sample_v5(tmp_path, record, scope, supervision, _recipe(scope), SYNTH_LABEL_MAPS, NO_REGIONS)
    assert full.targets["U"].shape == (1, 8, 8, 8) and full.targets["U"][0, 1, 1, 1]
    assert full.e_neg["U"].all() and full.e_pos["U"].all()
    down = load_training_sample_v5(tmp_path, record, scope, supervision, _recipe(scope, downsample_factor=[2, 2, 2]),
                                   SYNTH_LABEL_MAPS, NO_REGIONS)
    assert down.targets["U"].shape == (1, 4, 4, 4) and not down.targets["U"].any()
    assert not down.e_neg["U"][0, 0, 0, 0] and int((~down.e_neg["U"]).sum()) == 1
    a_scope = _scope("A")
    a_label = np.zeros((8, 8, 8), dtype=np.int16)
    a_label[:4] = 1
    a_label[5, 5, 5] = 2
    a_record = write_volume_case(tmp_path, a_scope, uid="case-a", label=a_label)
    a_down = load_training_sample_v5(tmp_path, a_record, a_scope, make_supervision(a_scope),
                                     _recipe(a_scope, downsample_factor=[2, 2, 2]), SYNTH_LABEL_MAPS, NO_REGIONS)
    assert a_down.targets["A"].dtype == np.int64 and a_down.targets["A"][0, 0, 0] == 1
    assert not a_down.e_neg["A"][1, 2, 2, 2] and a_down.e_neg["A"][0, 2, 2, 2]
    softmax_eligible = a_down.e_pos["A"].all(axis=0) & a_down.e_neg["A"].all(axis=0)
    assert not softmax_eligible[2, 2, 2] and int((~softmax_eligible).sum()) == 1


def test_review_regions_are_strided_with_the_image(tmp_path):
    scope = _scope("U")
    record = write_volume_case(tmp_path, scope)
    body = dataclasses.asdict(record)
    body["annotation_records"][0].update(completeness="region_complete", review_region_path="review/r1")
    regional = validate_sample_v5(body)
    review = np.zeros((8, 8, 8), dtype=bool)
    review[:4] = True
    sample = load_training_sample_v5(tmp_path, regional, scope, make_supervision(scope),
                                     _recipe(scope, downsample_factor=[2, 1, 1]), SYNTH_LABEL_MAPS,
                                     {"review": {"review/r1": review}, "visibility": {}})
    assert sample.e_pos["U"].shape == (1, 4, 8, 8)
    assert sample.e_pos["U"][0, :2].all() and not sample.e_pos["U"][0, 2:].any()
    _raises("GRID_MISMATCH", "$.regions.review", load_training_sample_v5, tmp_path, regional, scope,
            make_supervision(scope), _recipe(scope), SYNTH_LABEL_MAPS,
            {"review": {"review/r1": review[:4]}, "visibility": {}})


def test_patch_padding_has_no_eligibility_or_fov():
    scope = _scope("A")
    sample = make_training_sample_v5(scope, "s-0", shape=(6, 6, 6))
    patch = extract_patch_v5(sample, (0, 0, 0), (4, 4, 4))
    assert patch["valid_fov"].shape == (4, 4, 4)
    assert not patch["valid_fov"][:2].any() and not patch["valid_fov"][:, :2].any()
    assert patch["valid_fov"][2:, 2:, 2:].all()
    assert not patch["e_pos"]["A"][:, :2].any() and not patch["e_neg"]["A"][:, :, :, :2].any()
    assert patch["e_pos"]["A"][:, 2:, 2:, 2:].all()
    for name, channel in patch["channels"].items():
        assert (channel[:2] == 0).all()
        assert np.array_equal(channel[2:, 2:, 2:], sample.observation.channels[name][:2, :2, :2])
    assert np.array_equal(patch["targets"]["A"][2:, 2:, 2:], sample.targets["A"][:2, :2, :2])
    assert (patch["targets"]["A"][:2] == 0).all()
    assert patch["grid"]["shape"] == [4, 4, 4] and patch["to_original"][0]["op"] == "crop"
    inside = extract_patch_v5(sample, (3, 3, 3), (4, 4, 4))
    assert inside["valid_fov"].all() and np.array_equal(inside["targets"]["A"], sample.targets["A"][1:5, 1:5, 1:5])


def _draws(sampler, fov, n):
    out = []
    for _ in range(n):
        uid = sampler.next_uid()
        out.append((uid, sampler.draw_center(uid, fov[uid])))
    return out


def test_sampler_is_deterministic_and_resumable():
    centers = {"s-a": np.array([[1, 1, 1], [2, 2, 2]]), "s-b": np.zeros((0, 3), dtype=np.int64)}
    fov = {"s-a": np.ones((4, 4, 4), dtype=bool), "s-b": np.ones((3, 3, 3), dtype=bool)}
    sampler = PatchSamplerV5(["s-b", "s-a"], centers, 0.5, 3)
    first = _draws(sampler, fov, 10)
    state = copy.deepcopy(sampler.state_dict())
    tail = _draws(sampler, fov, 5)
    assert _draws(PatchSamplerV5(["s-a", "s-b"], centers, 0.5, 3), fov, 10) == first
    resumed = PatchSamplerV5(["s-a", "s-b"], centers, 0.5, 99)
    resumed.load_state_dict(state)
    assert _draws(resumed, fov, 5) == tail
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    reloaded = PatchSamplerV5(["s-a", "s-b"], centers, 0.5, 7)
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    assert _draws(reloaded, fov, 5) == tail
    epoch = [uid for uid, _ in _draws(PatchSamplerV5(["s-a", "s-b"], centers, 0.5, 3), fov, 4)]
    assert sorted(epoch[:2]) == ["s-a", "s-b"] and sorted(epoch[2:]) == ["s-a", "s-b"]
    for bad in ({"positive_fraction": 1.5}, {"seed": -1}, {"sample_uids": []}, {"sample_uids": ["s-a", "s-a"]}):
        args = {"sample_uids": ["s-a", "s-b"], "positive_centers": centers, "positive_fraction": 0.5, "seed": 3}
        args.update(bad)
        with pytest.raises(V5Error):
            PatchSamplerV5(**args)


def test_positive_centers_come_from_eligible_positives():
    scope = validate_scope_v5(make_module_scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"]))
    sample = make_training_sample_v5(scope, "s-0", shape=(4, 4, 4))
    targets = {"A": np.zeros((4, 4, 4), dtype=np.int64), "U": np.zeros((1, 4, 4, 4), dtype=bool)}
    targets["A"][0, 0, 0] = targets["A"][1, 1, 1] = 1
    targets["U"][0, 2, 2, 2] = targets["U"][0, 3, 3, 3] = True
    e_pos = {key: value.copy() for key, value in sample.e_pos.items()}
    e_neg = {key: value.copy() for key, value in sample.e_neg.items()}
    e_neg["A"][1, 1, 1, 1] = False
    e_pos["U"][0, 3, 3, 3] = False
    sample = dataclasses.replace(sample, targets=targets, e_pos=e_pos, e_neg=e_neg)
    assert positive_centers_v5(sample).tolist() == [[0, 0, 0], [2, 2, 2]]
    sampler = PatchSamplerV5(["s-0"], {"s-0": positive_centers_v5(sample)}, 1.0, 0)
    fov = {"s-0": sample.observation.valid_fov}
    assert {center for _, center in _draws(sampler, fov, 20)} <= {(0, 0, 0), (2, 2, 2)}
    partial = np.zeros((4, 4, 4), dtype=bool)
    partial[1:3, 1:3, 1:3] = True
    empty = PatchSamplerV5(["s-0"], {"s-0": np.zeros((0, 3), dtype=np.int64)}, 1.0, 0)
    draws = _draws(empty, {"s-0": partial}, 30)
    assert all(partial[center] for _, center in draws)
    _raises("EMPTY_FOV", "$.valid_fov", empty.draw_center, "s-0", np.zeros((4, 4, 4), dtype=bool))


def test_batch_is_built_in_scope_order_with_eligibility():
    scope = validate_scope_v5(make_module_scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"]))
    samples = {uid: make_training_sample_v5(scope, uid, shape=(6, 6, 6), seed=i) for i, uid in enumerate(["s-0", "s-1"])}
    sampler = PatchSamplerV5(sorted(samples), {uid: positive_centers_v5(s) for uid, s in samples.items()}, 0.5, 1)
    recipe = make_recipe(model={"tasks": ["A", "U"]}, batch_size=3, patch_size=[4, 4, 4])
    batch, targets, eligibility = make_batch_v5(samples, sampler, recipe, scope)
    assert isinstance(batch, ModelInputV5) and tuple(batch.image.shape) == (3, 2, 4, 4, 4)
    assert targets["A"].dtype == torch.int64 and tuple(targets["A"].shape) == (3, 4, 4, 4)
    assert targets["U"].dtype == torch.bool and tuple(targets["U"].shape) == (3, 1, 4, 4, 4)
    assert tuple(eligibility["A"]["e_pos"].shape) == (3, 2, 4, 4, 4) and eligibility["U"]["e_neg"].dtype == torch.bool
    _raises("TARGET_MISSING", "$.samples.targets.U", make_batch_v5,
            {"s-0": dataclasses.replace(samples["s-0"], targets={"A": samples["s-0"].targets["A"]})},
            PatchSamplerV5(["s-0"], {}, 0.0, 1), recipe, scope)


def test_locked_test_samples_never_loaded(tmp_path):
    scope = _scope("A")
    kept = write_volume_case(tmp_path, scope, uid="s-0", split="train")
    hidden = validate_sample_v5(dataclasses.asdict(write_volume_case(tmp_path / "elsewhere", scope, uid="s-1",
                                                                    split="train")))
    registry = make_split_registry([kept, hidden], official_split={"synthetic:s-0": "train", "synthetic:s-1": "test"})
    assert hidden.split == "train" and registry["assignment"]["s-1"] == "locked_test"
    samples = load_training_set_v5(tmp_path, [kept, hidden], registry, scope, make_supervision(scope), _recipe(scope),
                                   SYNTH_LABEL_MAPS, {})
    assert list(samples) == ["s-0"]
    assert samples["s-0"].observation.sample_uid == "s-0"


def test_grids_are_audited_with_the_registered_tolerance(tmp_path):
    scope = _scope("A")
    record = write_volume_case(tmp_path, scope)
    supervision = make_supervision(scope)
    recipe = _recipe(scope, grid_tolerance={"atol_affine": 1e-4, "atol_spacing": 1e-4})
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[:4] = 1
    base = np.diag([1.0, 1.0, 2.0, 1.0])
    drift, shifted = base.copy(), base.copy()
    drift[:3, 3] = 5e-5     # float-precision drift, as between SPIDER image and mask (research report §4.2)
    shifted[0, 3] = 0.5     # half a voxel
    write_nifti(tmp_path / "lab" / "case-1.nii.gz", label, drift)
    sample = load_training_sample_v5(tmp_path, record, scope, supervision, recipe, SYNTH_LABEL_MAPS, NO_REGIONS)
    assert (sample.targets["A"][:4] == 1).all() and (sample.targets["A"][4:] == 0).all()
    write_nifti(tmp_path / "lab" / "case-1.nii.gz", label, shifted)
    _raises("GRID_MISMATCH", "$.annotation_records[0].path", load_training_sample_v5, tmp_path, record, scope,
            supervision, recipe, SYNTH_LABEL_MAPS, NO_REGIONS)
    # A label without a header cannot be checked against an image that has one: refused, not trusted.
    seg = np.zeros((8, 8, 8, 2), dtype=np.uint8)
    write_skm_h5(tmp_path / "lab" / "case-1.h5", label.astype(np.float32), label.astype(np.float32), seg)
    annotation = dict(dataclasses.asdict(record.annotation_records[0]), path="lab/case-1.h5#seg",
                      label_map_version="synthetic_a_channels_v1")
    headerless = _with(record, annotation_records=[annotation])
    _raises("GRID_MISMATCH", "$.annotation_records[0].path", load_training_sample_v5, tmp_path, headerless, scope,
            supervision, recipe, SYNTH_LABEL_MAPS, NO_REGIONS)
    # Channels are audited against the first channel the same way.
    write_nifti(tmp_path / record.image_paths[1], np.arange(512, dtype=np.float32).reshape(8, 8, 8), shifted)
    _raises("GRID_MISMATCH", "$.image_paths[1]", load_observation_v5, tmp_path, record, scope, recipe)
    # The tolerance is a registered recipe value; there is no default.
    _raises("RECIPE_VALUE_MISSING", "$.grid_tolerance.atol_affine", load_observation_v5, tmp_path, record, scope,
            _recipe(scope, grid_tolerance={"atol_affine": None}))
    _raises("INVALID_RECIPE_VALUE", "$.grid_tolerance.atol_spacing", load_observation_v5, tmp_path, record, scope,
            _recipe(scope, grid_tolerance={"atol_spacing": -1.0}))


def test_patch_chain_lists_ops_in_application_order():
    # Chains list operations in the order they were applied to the original grid; restoring walks them backwards.
    scope = _scope("A")
    sample = make_training_sample_v5(scope, "s-0", shape=(6, 6, 6))
    down = {"op": "downsample", "factor": [2, 2, 2], "input_shape": [12, 12, 12]}
    observation = dataclasses.replace(sample.observation, to_original=[down],
                                      grid=dict(sample.observation.grid, transform_chain=[down]))
    patch = extract_patch_v5(dataclasses.replace(sample, observation=observation), (3, 3, 3), (4, 4, 4))
    crop = {"op": "crop", "start": [1, 1, 1], "input_shape": [6, 6, 6]}
    assert patch["to_original"] == [down, crop] and patch["grid"]["transform_chain"] == [down, crop]


def test_headerless_image_grid_must_match_record_geometry(tmp_path):
    """H5 images carry no header, so the model grid takes spacing/affine from the record; a record geometry
    whose shape differs from the array describes another grid and is refused, not used."""
    scope = _scope("A")
    record = write_volume_case(tmp_path, scope)
    recipe = _recipe(scope)
    ramp = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    write_skm_h5(tmp_path / "img" / "case-1.h5", ramp, ramp * 2.0 + 1.0, np.zeros((8, 8, 8, 2), dtype=np.uint8))
    headerless = _with(record, image_paths=["img/case-1.h5#echo1", "img/case-1.h5#echo2"])
    observation = load_observation_v5(tmp_path, headerless, scope, recipe)
    assert observation.original_header is None and observation.grid["spacing_mm"] == [1.0, 1.0, 2.0]
    stale = _with(headerless, image_geometry={**record.image_geometry, "shape": [8, 8, 6]})
    _raises("GRID_MISMATCH", "$.image_geometry.shape", load_observation_v5, tmp_path, stale, scope, recipe)
