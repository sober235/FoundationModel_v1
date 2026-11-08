"""v5.1 residual U-Net module baselines (plan_v5 T10; Proposal §6.1, §7.1). CPU, synthetic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.engine.train_v5 import build_optimizer_v5, training_loss_v5
from anatobind.v41.models.interface_v5 import PerceptionModelV5
from anatobind.v41.models.resunet_v5 import (
    AOnlyModelV5, ResidualBlockV5, SharedAUModelV5, UOnlyModelV5, build_model_v5, count_parameters_v5)
from tests.v5.factories import make_model_input_v5, make_module_scope, make_recipe, make_supervision


def _raises(error_code, error_path, fn, *args, **kwargs):
    with pytest.raises(V5Error) as err:
        fn(*args, **kwargs)
    assert (err.value.code, err.value.path) == (error_code, error_path)


def _recipe(tasks, strides=((1, 1, 1), (2, 2, 2)), **overrides):
    values = {"model": {"tasks": list(tasks), "widths": [4, 8], "strides": [list(s) for s in strides],
                        "blocks_per_stage": 1},
              "patch_size": [8, 8, 8], "inference": {"window": [8, 8, 8], "stride": [4, 4, 4]}}
    values.update(overrides)
    return make_recipe(**values)


def _au_scope():
    return validate_scope_v5(make_module_scope("A", scope_id="synthetic_au", abnormality_classes=["target_a"]))


def _batch(scope, shape=(8, 8, 8), seed=0):
    rng = np.random.default_rng(seed)
    return make_model_input_v5(scope, shape, channels={name: rng.standard_normal(shape).astype(np.float32)
                                                       for name in scope.input_modalities})


def test_models_implement_interface_and_shapes():
    scope = _au_scope()
    batch = _batch(scope)
    for tasks, cls in ((["A"], AOnlyModelV5), (["U"], UOnlyModelV5), (["A", "U"], SharedAUModelV5)):
        model = build_model_v5(_recipe(tasks), scope)
        assert type(model) is cls and isinstance(model, PerceptionModelV5) and model.supported_modes == ("3d",)
        assert model.tasks == tuple(tasks)
        output = model(batch)
        assert (output.anatomy_logits is None) == ("A" not in tasks)
        assert (output.abnormality_logits is None) == ("U" not in tasks)
        if "A" in tasks:
            assert tuple(output.anatomy_logits.shape) == (1, 3, 8, 8, 8)
        if "U" in tasks:
            assert tuple(output.abnormality_logits.shape) == (1, 1, 8, 8, 8)
        assert count_parameters_v5(model) == sum(p.numel() for p in model.parameters()) > 0
    # Anisotropic strides (in-plane first) keep the output on the input grid.
    flat_first = build_model_v5(_recipe(["A"], strides=((1, 1, 1), (1, 2, 2)), patch_size=[3, 8, 8],
                                        inference={"window": [3, 8, 8], "stride": [3, 4, 4]}), scope)
    assert tuple(flat_first(_batch(scope, (3, 8, 8))).anatomy_logits.shape) == (1, 3, 3, 8, 8)
    # Initialisation follows recipe seeds.init and leaves the global RNG alone.
    state = torch.get_rng_state()
    first, again = build_model_v5(_recipe(["A"]), scope), build_model_v5(_recipe(["A"]), scope)
    other = build_model_v5(_recipe(["A"], seeds={"init": 7, "sampler": 1}), scope)
    assert torch.equal(torch.get_rng_state(), state)
    pairs = list(zip(first.state_dict().values(), again.state_dict().values(), other.state_dict().values()))
    assert all(torch.equal(a, b) for a, b, _ in pairs) and not all(torch.equal(a, c) for a, _, c in pairs)


def test_residual_block_layout():
    block = ResidualBlockV5(4, 8, (1, 2, 2))
    assert block.conv1.kernel_size == (3, 3, 3) and block.conv1.stride == (1, 2, 2)
    assert isinstance(block.norm1, torch.nn.InstanceNorm3d) and block.norm1.affine
    assert isinstance(block.act, torch.nn.LeakyReLU) and block.act.negative_slope == 0.01
    assert block.conv2.kernel_size == (3, 3, 3) and block.conv2.stride == (1, 1, 1)
    assert isinstance(block.norm2, torch.nn.InstanceNorm3d) and block.norm2.affine
    assert block.skip.kernel_size == (1, 1, 1) and block.skip.stride == (1, 2, 2)
    assert isinstance(ResidualBlockV5(8, 8, (1, 1, 1)).skip, torch.nn.Identity)
    out = block(torch.zeros(2, 4, 4, 8, 8))
    assert tuple(out.shape) == (2, 8, 4, 4, 4)


def test_shared_au_has_one_encoder_two_decoders():
    scope = _au_scope()
    model = build_model_v5(_recipe(["A", "U"]), scope)
    encoder = {id(p) for p in model.encoder.parameters()}
    side = {task: {id(p) for p in model.decoders[task].parameters()} | {id(p) for p in model.heads[task].parameters()}
            for task in ("A", "U")}
    assert encoder and side["A"] and side["U"]
    assert not encoder & side["A"] and not encoder & side["U"] and not side["A"] & side["U"]
    every = [id(p) for p in model.parameters()]
    assert len(every) == len(set(every)) == len(encoder | side["A"] | side["U"])
    assert set(build_model_v5(_recipe(["A"]), scope).decoders) == {"A"}


def test_models_reject_2d_mode():
    flat = validate_scope_v5(make_module_scope("A", observation_mode="2d"))
    _raises("MODE_NOT_SUPPORTED", "$.scope.observation_mode", build_model_v5, _recipe(["A"]), flat)
    model = build_model_v5(_recipe(["A"]), validate_scope_v5(make_module_scope("A")))
    _raises("MODE_NOT_SUPPORTED", "$.observation_mode", model, make_model_input_v5(flat, (8, 8)))


def test_patch_divisibility_checked():
    scope = validate_scope_v5(make_module_scope("A"))
    _raises("PATCH_NOT_DIVISIBLE", "$.patch_size", build_model_v5, _recipe(["A"], patch_size=[8, 7, 8]), scope)
    _raises("PATCH_NOT_DIVISIBLE", "$.inference.window", build_model_v5,
            _recipe(["A"], inference={"window": [8, 8, 7], "stride": [4, 4, 4]}), scope)
    _raises("MODULE_TASKS_INVALID", "$.tasks", build_model_v5, _recipe(["U"]), scope)
    _raises("INVALID_RECIPE_VALUE", "$.model.architecture", build_model_v5,
            _recipe(["A"], model={"tasks": ["A"], "widths": [4, 8], "strides": [[1, 1, 1], [2, 2, 2]],
                                  "blocks_per_stage": 1, "architecture": "unet2d"}), scope)


@pytest.mark.parametrize("task", ["A", "U"])
def test_cpu_synthetic_overfit(task):
    scope = validate_scope_v5(make_module_scope(task))
    shape = (16, 16, 16)
    recipe = _recipe([task], patch_size=list(shape), inference={"window": list(shape), "stride": [8, 8, 8]},
                     optimizer={"name": "adamw", "lr": 0.01, "weight_decay": 0.0})
    model = build_model_v5(recipe, scope)
    cube = np.zeros(shape, dtype=np.float32)
    cube[4:12, 4:12, 4:12] = 1.0
    rng = np.random.default_rng(0)
    batch = make_model_input_v5(scope, shape, channels={name: (cube + 0.1 * rng.standard_normal(shape))
                                                        .astype(np.float32) for name in scope.input_modalities})
    if task == "A":
        target = torch.from_numpy(cube.astype(np.int64))[None]
        eligible = torch.ones((1, len(scope.anatomy_classes), *shape), dtype=torch.bool)
    else:
        target = torch.from_numpy(cube.astype(bool))[None, None]
        eligible = torch.ones((1, 1, *shape), dtype=torch.bool)
    optimizer = build_optimizer_v5(model, recipe)
    supervision = make_supervision(scope)
    losses = []
    for _ in range(100):
        loss, _ = training_loss_v5(model(batch), {task: target}, {task: {"e_pos": eligible, "e_neg": eligible}},
                                   supervision)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if losses[-1] < losses[0] / 2:
            break
    assert losses[-1] < losses[0] / 2
