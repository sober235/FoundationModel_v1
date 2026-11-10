"""v5.1 acceptance, state composition and the perception output chain (plan_v5 T09c; Proposal §2.1 rules 1-5)."""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from anatobind.v41.contracts_v5 import (
    ModuleOutput, PerceptionOutput, V5Error, array_sha256, canonical_sha256, validate_module_output_v5,
    validate_perception_output_v5, validate_scope_v5)
from anatobind.v41.engine.data_v5 import load_observation_v5, original_geometry_v5
from anatobind.v41.engine.infer_v5 import predict_volume_v5, restore_probabilities_v5
from anatobind.v41.models.interface_v5 import build_model_input_v5
from anatobind.v41.perception import (
    apply_acceptance_v5, extract_instances_v5, infer_and_save_v5, run_perception_v5, validate_acceptance_policy_v5)
from anatobind.v41.perception_io import load_output_v5
from anatobind.v41.state_rule_v5 import REASON_BITS, derive_state_map_v5
from tests.v5.factories import (
    CONFIG_DIR_V5, StubPerceptionModel, make_acceptance_policy, make_dev_split, make_model_input_v5,
    make_module_scope, make_original_geometry, make_recipe, make_scope, write_volume_case)

REPO = Path(__file__).resolve().parents[2]
SHAPE = (6, 6, 6)
UNCERTAIN, OUT_OF_FOV = REASON_BITS["anatomy_uncertain"], REASON_BITS["out_of_fov"]


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _joint(**overrides):
    return validate_scope_v5(make_scope(**overrides))


def _fields(output) -> dict:
    return {field.name: getattr(output, field.name) for field in dataclasses.fields(output)}


def _constant(values):
    def logits(batch):
        size = (batch.image.shape[0], len(values), *batch.image.shape[2:])
        return torch.tensor(values, dtype=torch.float32).view(1, -1, *[1] * (batch.image.ndim - 2)).expand(size).clone()
    return logits


def _tissue_logits(batch):
    """Channel 0 is 1 in tissue_a, 0 in tissue_b and 0.5 where the anatomy should be uncertain (uniform softmax)."""
    a = batch.image[:, 0]
    return torch.stack([torch.zeros_like(a), 16 * (a - 0.5), 16 * (0.5 - a)], dim=1)


def _lesion_logits(batch):
    """Channel 1 marks target_a; target_b is never present."""
    lesion = batch.image[:, 1]
    return torch.stack([16 * lesion - 8, torch.full_like(lesion, -8.0)], dim=1)


def _host_relations(instances, anatomy_map, anatomy_accepted, scope):
    relations, reasons = [], {}
    for instance in instances:
        tissues = sorted({int(t) for t in np.unique(anatomy_map[instance["mask"] & anatomy_accepted]) if t > 0})
        relations.append({"instance_uid": instance["instance_uid"], "predicate": scope.relation_contract["predicate"],
                          "tissue_ids": tissues, "status": "bound" if tissues else "no_host", "accepted": True,
                          "completion_role": scope.relation_contract["completion_role"]})
        reasons[instance["instance_uid"]] = 0
    return relations, reasons


def _channels(uncertain=None):
    tissue = np.zeros(SHAPE, dtype=np.float32)
    tissue[:3] = 1.0
    lesion = np.zeros(SHAPE, dtype=np.float32)
    lesion[1:3, 1:3, 1:3] = 1.0      # 8 voxels in tissue_a
    lesion[4, 4, 4] = 1.0            # 1 voxel in tissue_b: below min_voxels, stays in the voxel map
    if uncertain is not None:
        tissue[uncertain] = 0.5
        lesion[uncertain] = 1.0
    return {"synthetic_e1": tissue, "synthetic_e2": lesion}


def _kwargs(scope, shape=SHAPE, **overrides):
    args = {"observation_uid": "obs-synth-1", "checkpoint_sha256": "b" * 64,
            "original_geometry": make_original_geometry(shape), "acceptance_policy": make_acceptance_policy(scope),
            "window": (4, 4, 4)[:len(shape)], "stride": (2, 2, 2)[:len(shape)], "precision": "fp32"}
    args.update(overrides)
    return args


def test_acceptance_policy_template_fails_closed():
    scope = _joint()
    template = yaml.safe_load((CONFIG_DIR_V5 / "acceptance_policy.template.yaml").read_text())
    _raises("ACCEPTANCE_VALUE_MISSING", "$.policy_id", validate_acceptance_policy_v5, template, scope)
    policy = make_acceptance_policy(scope)
    assert validate_acceptance_policy_v5(policy, scope) is policy
    _raises("INVALID_ACCEPTANCE_VALUE", "$.policy_role", validate_acceptance_policy_v5,
            make_acceptance_policy(scope, policy_role="final"), scope)
    short = make_acceptance_policy(scope)
    short["abnormality"]["thresholds"] = {"target_a": 0.5}
    _raises("INVALID_ACCEPTANCE_VALUE", "$.abnormality.thresholds", validate_acceptance_policy_v5, short, scope)
    _raises("INVALID_ACCEPTANCE_VALUE", "$.abnormality.thresholds.target_b", validate_acceptance_policy_v5,
            make_acceptance_policy(scope, abnormality={"thresholds": {"target_b": 0.0}}), scope)
    _raises("INVALID_ACCEPTANCE_VALUE", "$.abnormality.reject_band.target_a", validate_acceptance_policy_v5,
            make_acceptance_policy(scope, abnormality={"reject_band": {"target_a": [0.6, 0.4]}}), scope)
    _raises("INVALID_ACCEPTANCE_VALUE", "$.instances.connectivity", validate_acceptance_policy_v5,
            make_acceptance_policy(scope, instances={"connectivity": 18}), scope)
    _raises("ACCEPTANCE_VALUE_MISSING", "$.anatomy.min_confidence", validate_acceptance_policy_v5,
            make_acceptance_policy(scope, anatomy={"min_confidence": None}), scope)
    _raises("UNKNOWN_FIELD", "$.<unknown>", validate_acceptance_policy_v5, dict(policy, extra=1), scope)
    # A value for a task the scope does not have would be registered but never used: refused.
    u_only = validate_scope_v5(make_module_scope("U"))
    _raises("INVALID_ACCEPTANCE_VALUE", "$.anatomy.min_confidence", validate_acceptance_policy_v5,
            make_acceptance_policy(u_only, anatomy={"min_confidence": 0.5}), u_only)


def test_acceptance_reasons_per_task():
    scope = _joint()
    policy = make_acceptance_policy(scope, anatomy={"min_confidence": 0.6},
                                    abnormality={"reject_band": {"target_a": [0.4, 0.6]}})
    shape = (2, 3, 1)
    anatomy = np.zeros((3, *shape), dtype=np.float32)
    anatomy[:, 0, 0, 0] = [0.1, 0.8, 0.1]       # tissue_a, confident
    anatomy[:, 0, 1, 0] = [0.4, 0.35, 0.25]     # max 0.4 < 0.6: uncertain
    anatomy[:, 0, 2, 0] = [0.05, 0.05, 0.9]     # tissue_b
    anatomy[:, 1, 0, 0] = [0.05, 0.05, 0.9]     # not represented
    anatomy[:, 1, 1, 0] = [0.05, 0.9, 0.05]     # outside the FOV
    anatomy[:, 1, 2, 0] = [0.6, 0.2, 0.2]       # background, confident
    abnormality = np.zeros((2, *shape), dtype=np.float32)
    abnormality[0, :, :, 0] = [[0.9, 0.55, 0.45], [0.9, 0.9, 0.1]]
    abnormality[1, :, :, 0] = [[0.55, 0.2, 0.2], [0.2, 0.2, 0.2]]
    represented = np.ones(shape, dtype=bool)
    represented[1, 0, 0] = False
    fov = np.ones(shape, dtype=bool)
    fov[1, 1, 0] = False
    out = apply_acceptance_v5({"anatomy": anatomy, "abnormality": abnormality}, represented, fov, policy, scope)
    a, u = out["anatomy"], out["abnormality"]
    assert a["map"][..., 0].tolist() == [[1, -1, 2], [-1, -1, 0]]
    assert a["accepted"][..., 0].tolist() == [[True, False, True], [False, False, True]]
    assert a["reason"][..., 0].tolist() == [[0, UNCERTAIN, 0], [OUT_OF_FOV, OUT_OF_FOV, 0]]
    band = REASON_BITS["abnormality_uncertain"]
    assert u["map"][0, ..., 0].tolist() == [[True, True, False], [False, False, False]]   # a rejected positive stays
    assert u["accepted"][0, ..., 0].tolist() == [[True, False, False], [False, False, True]]
    assert u["reason"][0, ..., 0].tolist() == [[0, band, band], [OUT_OF_FOV, OUT_OF_FOV, 0]]
    assert u["map"][1, ..., 0].tolist() == [[True, False, False], [False, False, False]]
    assert u["accepted"][1, ..., 0].tolist() == [[True, True, True], [False, False, True]]
    assert a["map"].dtype.kind == "i" and u["map"].dtype == bool and a["reason"].dtype == np.uint8


def test_instances_follow_policy_connectivity_and_size():
    scope = _joint()
    positive = np.zeros((2, *SHAPE), dtype=bool)
    positive[0, 0, 0, 0] = positive[0, 1, 1, 1] = True     # diagonal neighbours: one 26-component, two 6-components
    positive[0, 4:6, 4:6, 4] = True
    positive[1, 3, 0, 0] = True
    accepted = np.ones_like(positive)
    accepted[1, 3, 0, 0] = False                              # a rejected positive never forms an instance
    six = extract_instances_v5(positive, accepted, scope, make_acceptance_policy(scope, instances={"connectivity": 6}))
    assert [(i["class_name"], i["bbox"]) for i in six] == [("target_a", [0, 0, 0, 1, 1, 1]),
                                                         ("target_a", [1, 1, 1, 2, 2, 2]),
                                                         ("target_a", [4, 4, 4, 6, 6, 5])]
    big = extract_instances_v5(positive, accepted, scope,
                               make_acceptance_policy(scope, instances={"connectivity": 26, "min_voxels": 3}))
    assert [(i["instance_uid"], i["bbox"], int(i["mask"].sum())) for i in big] == [("inst-0001", [4, 4, 4, 6, 6, 5], 4)]
    assert all(i["accepted"] is True and i["candidate_set_hash"] == six[0]["candidate_set_hash"] for i in six)
    assert big[0]["candidate_set_hash"] != six[0]["candidate_set_hash"]
    assert [i["instance_uid"] for i in six] == ["inst-0001", "inst-0002", "inst-0003"]


def test_instances_refuse_to_allocate_beyond_the_mask_budget(monkeypatch):
    """Every instance carries a full-volume mask, so the kept components times the volume is bounded before any
    mask is allocated (a 40-step model on a 512x512x160 knee volume gave about 200,000 components)."""
    from anatobind.v41 import perception

    scope = _joint()
    positive = np.zeros((2, *SHAPE), dtype=bool)
    positive[0, 0, 0, 0] = positive[0, 2, 2, 2] = positive[1, 4, 4, 4] = True          # three 1-voxel components
    positive[0, 4:6, 4:6, 0] = True                                                    # one 4-voxel component
    accepted = np.ones_like(positive)
    volume = int(np.prod(SHAPE))
    monkeypatch.setattr(perception, "INSTANCE_MASK_BYTES_LIMIT", 3 * volume)
    with pytest.raises(V5Error) as error:
        extract_instances_v5(positive, accepted, scope, make_acceptance_policy(scope, instances={"connectivity": 26}))
    assert (error.value.code, error.value.path) == ("INSTANCE_LIMIT_EXCEEDED", "$.abnormal_instances")
    kept = extract_instances_v5(positive, accepted, scope,
                                make_acceptance_policy(scope, instances={"connectivity": 26, "min_voxels": 4}))
    assert [int(i["mask"].sum()) for i in kept] == [4]            # the size filter is applied before the budget
    monkeypatch.setattr(perception, "INSTANCE_MASK_BYTES_LIMIT", 4 * volume)
    assert len(extract_instances_v5(positive, accepted, scope,
                                    make_acceptance_policy(scope, instances={"connectivity": 26}))) == 4


def test_module_output_marks_disabled_tasks():
    scope = validate_scope_v5(make_module_scope("A"))
    model = StubPerceptionModel(scope, ("A",), anatomy_fn=_constant([0.0, 3.0, 0.0]))
    batch = make_model_input_v5(scope, SHAPE)
    output = run_perception_v5(model, batch, scope, **_kwargs(scope))
    assert isinstance(output, ModuleOutput)
    assert output.enabled_tasks == ["A"] and output.disabled_tasks == ["U", "S", "R"]
    assert output.abnormality_map is None and set(output.acceptance) == {"anatomy"} and (output.anatomy_map == 1).all()
    assert output.provenance == {
        "input_sha256": canonical_sha256([array_sha256(batch.image[0, i].numpy()) for i in range(2)]),
        "model_sha256": "b" * 64, "output_policy_version": "synthetic_acceptance_v1"}
    assert output.geometry == make_original_geometry(SHAPE) and output.observation_uid == "obs-synth-1"
    validate_module_output_v5(_fields(output), scope)


def test_joint_chain_uses_state_rule_and_validates():
    scope = _joint()
    model = StubPerceptionModel(scope, ("A", "U"), anatomy_fn=_tissue_logits, abnormality_fn=_lesion_logits)
    batch = make_model_input_v5(scope, SHAPE, channels=_channels())
    policy = make_acceptance_policy(scope, instances={"min_voxels": 2})
    _raises("RELATION_FN_REQUIRED", "$.relation_fn", run_perception_v5, model, batch, scope,
            **_kwargs(scope, acceptance_policy=policy))
    assert model.calls == 0
    output = run_perception_v5(model, batch, scope, relation_fn=_host_relations,
                               **_kwargs(scope, acceptance_policy=policy))
    assert isinstance(output, PerceptionOutput)
    state, reason = derive_state_map_v5(output.anatomy_map, output.acceptance["anatomy"], output.reason["anatomy"],
                                        output.abnormality_map, output.acceptance["abnormality"],
                                        output.reason["abnormality"], scope.state_tissue_ids)
    assert np.array_equal(output.tissue_state_map, state) and np.array_equal(output.reason["state"], reason)
    assert (output.tissue_state_map[1:3, 1:3, 1:3] == 1).all() and output.tissue_state_map[4, 4, 4] == 1
    assert output.tissue_state_map[0, 0, 0] == 0 and output.tissue_state_map[5, 0, 0] == 0
    assert [(i["class_name"], i["bbox"]) for i in output.abnormal_instances] == [("target_a", [1, 1, 1, 3, 3, 3])]
    assert output.tissue_relations[0]["tissue_ids"] == [1] and output.tissue_relations[0]["status"] == "bound"
    assert len(output.state_policy_hash) == 64 and output.provenance["model_sha256"] == "b" * 64
    validate_perception_output_v5(_fields(output), scope)


def test_a_uncertainty_does_not_erase_u():
    scope = _joint()
    region = (slice(4, 6), slice(4, 6), slice(0, 2))     # anatomy uncertain here, with a lesion not touching others
    model = StubPerceptionModel(scope, ("A", "U"), anatomy_fn=_tissue_logits, abnormality_fn=_lesion_logits)
    output = run_perception_v5(model, make_model_input_v5(scope, SHAPE, channels=_channels(uncertain=region)), scope,
                               relation_fn=_host_relations, **_kwargs(scope))
    assert not output.acceptance["anatomy"][region].any() and (output.anatomy_map[region] == -1).all()
    assert output.abnormality_map[0][region].all() and output.acceptance["abnormality"][0][region].all()
    assert (output.tissue_state_map[region] == 2).all()
    assert (output.reason["state"][region] & UNCERTAIN).all()
    kept = [i for i in output.abnormal_instances if i["mask"][region].all()]
    assert len(kept) == 1
    relation = next(r for r in output.tissue_relations if r["instance_uid"] == kept[0]["instance_uid"])
    assert relation["status"] == "no_host" and relation["tissue_ids"] == []


def test_3d_only_model_rejects_2d_scope_before_forward():
    model = StubPerceptionModel(_joint(), ("A", "U"), anatomy_fn=_tissue_logits, abnormality_fn=_lesion_logits)
    flat = _joint(observation_mode="2d")
    _raises("MODE_NOT_SUPPORTED", "$.observation_mode", run_perception_v5, model, make_model_input_v5(flat, (6, 6)),
            flat, relation_fn=_host_relations, **_kwargs(flat, shape=(6, 6)))
    assert model.calls == 0


def test_inference_entry_takes_no_reference_inputs():
    forbidden = ("target", "reference", "eligib", "review", "visib", "truth")
    for fn in (run_perception_v5, infer_and_save_v5, predict_volume_v5, restore_probabilities_v5,
               apply_acceptance_v5, extract_instances_v5):
        names = [name.lower() for name in inspect.signature(fn).parameters]
        assert not [n for n in names if any(word in n for word in forbidden)], fn.__name__


def test_observation_chain_restores_to_original_grid(tmp_path):
    scope = validate_scope_v5(make_module_scope("A"))
    record = write_volume_case(tmp_path / "data", scope)
    observation = load_observation_v5(tmp_path / "data", record, scope, make_recipe(downsample_factor=[2, 2, 1]))
    batch = build_model_input_v5(observation.channels, scope, observation.valid_fov, observation.grid,
                                 observation.to_original)
    geometry = original_geometry_v5(observation, record)
    assert geometry == {"shape": [8, 8, 8], "axis_order": "xyz", "spacing_mm": [1.0, 1.0, 2.0],
                        "affine": np.diag([1.0, 1.0, 2.0, 1.0]).tolist(), "units_reliable": False,
                        "transform_chain": []}
    model = StubPerceptionModel(scope, ("A",), anatomy_fn=_constant([0.0, 3.0, 0.0]))
    root = tmp_path / "private"
    registry, run = make_dev_split(scope, [record.sample_uid])
    output, path = infer_and_save_v5(model, batch, scope, out_dir=root / "outputs", private_root=root, repo=REPO,
                                     run=run, registry=registry,
                                     **_kwargs(scope, observation_uid=record.sample_uid, original_geometry=geometry))
    assert output.anatomy_map.shape == (8, 8, 8) and (output.anatomy_map == 1).all() and output.geometry == geometry
    loaded = load_output_v5(path, scope)
    assert np.array_equal(loaded.anatomy_map, output.anatomy_map) and loaded.geometry == geometry


def test_model_grid_fov_maps_back_conservatively():
    # Original x = 0..7 was downsampled by 2; coarse x = 3 is outside the FOV. Original x reads coarse x / 2, so
    # x = 5 (between coarse 2 and 3), 6 and 7 (edge value of coarse 3) are out of the FOV; x <= 4 is inside.
    scope = validate_scope_v5(make_module_scope("A"))
    chain = [{"op": "downsample", "factor": [2, 1, 1], "input_shape": [8, 4, 4]}]
    fov = np.ones((4, 4, 4), dtype=bool)
    fov[3] = False
    grid = dict(make_original_geometry((4, 4, 4)), transform_chain=chain)
    batch = build_model_input_v5({name: np.zeros((4, 4, 4), dtype=np.float32) for name in scope.input_modalities},
                                 scope, fov, grid, chain)
    model = StubPerceptionModel(scope, ("A",), anatomy_fn=_constant([0.0, 3.0, 0.0]))
    output = run_perception_v5(model, batch, scope, **_kwargs(scope, original_geometry=make_original_geometry((8, 4, 4))))
    assert (output.reason["anatomy"][5:] == OUT_OF_FOV).all() and (output.anatomy_map[5:] == -1).all()
    assert (output.reason["anatomy"][:5] == 0).all() and (output.anatomy_map[:5] == 1).all()


def test_inference_reads_only_roles_the_run_registered(tmp_path):
    """The formal inference entry reads an observation only through the run's frozen split (L4): locked_test and
    roles the run did not register are refused before the model runs or anything is written."""
    scope = validate_scope_v5(make_module_scope("A"))
    batch = make_model_input_v5(scope, SHAPE)
    model = StubPerceptionModel(scope, ("A",), anatomy_fn=_constant([0.0, 3.0, 0.0]))
    roles = {"obs-locked": "locked_test"}
    registry, run = make_dev_split(scope, ["obs-dev", "obs-locked"], roles)
    _, train_only = make_dev_split(scope, ["obs-dev", "obs-locked"], roles, readable_roles=["train"])
    other_registry, _ = make_dev_split(scope, ["obs-dev"])
    root = tmp_path / "private"

    def infer(uid, run_, registry_):
        return infer_and_save_v5(model, batch, scope, out_dir=root / "outputs", private_root=root, repo=REPO,
                                 run=run_, registry=registry_, **_kwargs(scope, observation_uid=uid))

    _raises("LOCKED_TEST_NOT_READABLE", "$.sample_uid", infer, "obs-locked", run, registry)
    _raises("ROLE_NOT_READABLE", "$.sample_uid", infer, "obs-dev", train_only, registry)
    _raises("SPLIT_ASSIGNMENT_MISSING", "$.sample_uid", infer, "obs-9", run, registry)
    _raises("SPLIT_HASH_MISMATCH", "$.split.split_sha256", infer, "obs-dev", run, other_registry)
    assert model.calls == 0 and not (root / "outputs").exists()
    output, path = infer("obs-dev", run, registry)
    assert output.observation_uid == "obs-dev" and path.is_file()
