"""One shared encoder, per-protocol stems, decoders and heads (design 2026-11-10 §5.1). CPU only."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error, validate_scope_v5
from anatobind.v41.models.interface_v5 import PerceptionModelV5
from anatobind.v41.models.unified_v5 import UnifiedModelV5, build_unified_model_v5
from tests.v5.factories import make_model_input_v5, make_module_scope, make_scope

KNEE = validate_scope_v5(make_scope())                       # joint: 2 tissues, 2 abnormality classes
BRAIN = validate_scope_v5(make_module_scope("U", input_modalities=["synthetic_dwi", "synthetic_adc"]))
SHAPE = (4, 8, 8)
MODEL = {"architecture": "resunet3d", "widths": [4, 8], "strides": [[1, 1, 1], [1, 2, 2]], "blocks_per_stage": 1,
         "stem_channels": 3}


def _model():
    return UnifiedModelV5({"knee": (KNEE, ("A", "U")), "brain": (BRAIN, ("U",))}, MODEL["widths"], MODEL["strides"],
                          MODEL["blocks_per_stage"], MODEL["stem_channels"])


def _plan(**member):
    base = {"tasks": ["U"], "patch_size": list(SHAPE), "inference": {"window": list(SHAPE), "stride": list(SHAPE)}}
    return {"shared": {"model": dict(MODEL), "seeds": {"init": 7, "sampler": 8}},
            "members": {"knee": dict(base, tasks=["A", "U"]), "brain": dict(base, **member)}}


def test_each_view_is_a_perception_model_with_its_own_outputs():
    model = _model()
    assert model.protocol_ids == ("brain", "knee")
    knee, brain = model.view("knee"), model.view("brain")
    assert isinstance(knee, PerceptionModelV5) and knee.scope is KNEE and brain.tasks == ("U",)
    out = knee(make_model_input_v5(KNEE, SHAPE))
    assert out.anatomy_logits.shape == (1, 3, *SHAPE) and out.abnormality_logits.shape == (1, 2, *SHAPE)
    out = brain(make_model_input_v5(BRAIN, SHAPE))
    assert out.anatomy_logits is None and out.abnormality_logits.shape == (1, 1, *SHAPE)


def test_every_tensor_is_in_the_state_once_and_the_encoder_is_shared():
    model = _model()
    state = model.state_dict()
    assert len({id(t) for t in state.values()}) == len(state)
    assert not any(".encoder." in key or ".stem" in key for key in state if key.startswith("members."))
    assert {key.split(".")[0] for key in state} == {"stems", "encoder", "members"}
    loss = model.view("brain")(make_model_input_v5(BRAIN, SHAPE)).abnormality_logits.sum()
    loss.backward()
    grads = {name: p.grad is not None and bool(p.grad.abs().sum() > 0) for name, p in model.named_parameters()}
    assert any(v for k, v in grads.items() if k.startswith("encoder."))
    assert not any(v for k, v in grads.items() if k.startswith(("members.knee.", "stems.knee.")))


def test_a_protocol_refuses_another_protocols_channels():
    with pytest.raises(V5Error) as err:                       # brain channels given to the knee protocol
        make_model_input_v5(KNEE, SHAPE, channels={n: np.zeros(SHAPE, np.float32) for n in BRAIN.input_modalities})
    assert err.value.code == "INPUT_MODALITY_MISSING"
    with pytest.raises(V5Error) as err:
        _model().view("spine")
    assert (err.value.code, err.value.path) == ("PROTOCOL_UNKNOWN", "$.protocol_id")


def test_build_is_seeded_and_checks_divisibility():
    scopes = {"knee": KNEE, "brain": BRAIN}
    torch.manual_seed(123)
    before = torch.get_rng_state()
    a, b = build_unified_model_v5(_plan(), scopes), build_unified_model_v5(_plan(), scopes)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    with pytest.raises(V5Error) as err:
        build_unified_model_v5(_plan(patch_size=[4, 7, 8]), scopes)
    assert (err.value.code, err.value.path) == ("PATCH_NOT_DIVISIBLE", "$.members.brain.patch_size")
