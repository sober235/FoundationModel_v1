"""Unified model: one shared encoder, per-protocol stems, decoders and heads (Proposal_v5 §0.6; design §5.1).

The scan protocol picks the stem (a 1x1x1 convolution from that protocol's channels) and the decoders and heads;
every protocol runs through the same encoder. Each protocol's *view* is a PerceptionModelV5 on that protocol's
ActiveScope, so the frozen input/output checks, full-volume inference and the output chain apply unchanged.

State layout: `stems.<protocol>.*`, `encoder.*`, `members.<protocol>.decoders.<task>.*`,
`members.<protocol>.heads.<task>.*`. A view registers only its own decoders and heads; the shared stem and encoder
are plain attributes of the view, so every tensor is in the unified state exactly once.
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn

from ..contracts_v5 import ActiveScope, V5Error
from .interface_v5 import ModelOutputV5, PerceptionModelV5
from .resunet_v5 import DecoderV5, EncoderV5

_PROTOCOL_ID = re.compile(r"[a-z][a-z0-9_]*")


class MemberModelV5(PerceptionModelV5):
    """One protocol's view of the unified model."""

    supported_modes = ("3d",)

    def __init__(self, scope: ActiveScope, tasks: tuple, stem: nn.Module, encoder: nn.Module, widths: list,
                 strides: list):
        super().__init__(scope, tasks)
        outputs = {"A": 1 + len(scope.anatomy_classes), "U": len(scope.abnormality_classes)}
        self.decoders = nn.ModuleDict({task: DecoderV5(widths, strides) for task in self.tasks})
        self.heads = nn.ModuleDict({task: nn.Conv3d(widths[0], outputs[task], 1) for task in self.tasks})
        object.__setattr__(self, "_shared", (stem, encoder))        # not registered: owned by UnifiedModelV5

    def _forward_v5(self, batch) -> ModelOutputV5:
        stem, encoder = self._shared
        features = encoder(stem(batch.image))
        logits = {task: self.heads[task](self.decoders[task](features)) for task in self.tasks}
        return ModelOutputV5(anatomy_logits=logits.get("A"), abnormality_logits=logits.get("U"))


class UnifiedModelV5(nn.Module):
    def __init__(self, members: dict, widths: list, strides: list, blocks_per_stage: int, stem_channels: int):
        super().__init__()
        if not isinstance(members, dict) or not members \
                or any(not isinstance(pid, str) or not _PROTOCOL_ID.fullmatch(pid) for pid in members):
            raise V5Error("PROTOCOL_INVALID", "$.members")
        widths = [int(w) for w in widths]
        strides = [tuple(int(s) for s in stride) for stride in strides]
        if not widths or len(strides) != len(widths) or any(len(s) != 3 for s in strides):
            raise V5Error("INVALID_RECIPE_VALUE", "$.model.strides")
        self.protocol_ids = tuple(sorted(members))
        self.stems = nn.ModuleDict({pid: nn.Conv3d(len(members[pid][0].input_modalities), int(stem_channels), 1)
                                    for pid in self.protocol_ids})
        self.encoder = EncoderV5(int(stem_channels), widths, strides, int(blocks_per_stage))
        self.members = nn.ModuleDict({pid: MemberModelV5(members[pid][0], tuple(members[pid][1]), self.stems[pid],
                                                         self.encoder, widths, strides)
                                      for pid in self.protocol_ids})

    def view(self, protocol_id: str) -> MemberModelV5:
        if protocol_id not in self.protocol_ids:
            raise V5Error("PROTOCOL_UNKNOWN", "$.protocol_id")
        return self.members[protocol_id]


def build_unified_model_v5(plan: dict, scopes: dict) -> UnifiedModelV5:
    """The unified model of a plan, initialised under shared.seeds.init like build_model_v5 (forked RNG)."""
    model = plan["shared"]["model"]
    factor = [math.prod(int(stride[axis]) for stride in model["strides"]) for axis in range(3)]
    for pid in sorted(plan["members"]):
        member = plan["members"][pid]
        for vector, path in ((member["patch_size"], "patch_size"), (member["inference"]["window"], "inference.window")):
            if any(int(size) % f for size, f in zip(vector, factor)):
                raise V5Error("PATCH_NOT_DIVISIBLE", f"$.members.{pid}.{path}")
    members = {pid: (scopes[pid], tuple(plan["members"][pid]["tasks"])) for pid in plan["members"]}
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(plan["shared"]["seeds"]["init"])
        return UnifiedModelV5(members, model["widths"], model["strides"], model["blocks_per_stage"],
                              model["stem_channels"])
