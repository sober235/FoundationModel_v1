"""v5.1 residual U-Net module baselines (plan_v5 T10; Proposal §6.1, §7.1).

Encoder stage i holds ``blocks_per_stage`` residual blocks; the first one downsamples with stride
``model.strides[i]`` (anisotropic allowed, e.g. in-plane only; stage 0 is usually [1, 1, 1]). Each decoder level
upsamples with a transposed convolution whose kernel equals that level's stride, concatenates the encoder
features of the same level and runs one residual block; a 1x1x1 head gives the logits. No deep supervision.
A-only and U-only models have one decoder; the shared-AU model has one encoder and two independent decoders.
Arrays are (x, y, z); a model sees only the ModelInputV5 fields, never labels or eligibility (GC8).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..contracts_v5 import ActiveScope, V5Error
from ..engine.registration_v5 import validate_recipe_v5
from .interface_v5 import ModelOutputV5, PerceptionModelV5

_NEGATIVE_SLOPE = 0.01
_UNIT = (1, 1, 1)


class ResidualBlockV5(nn.Module):
    """conv3 → IN → LeakyReLU → conv3 → IN, plus the skip, then LeakyReLU.

    The skip is a 1x1x1 convolution with the block's stride when the channel count or the stride changes, the
    identity otherwise. Convolutions followed by an affine InstanceNorm carry no bias (the norm's shift absorbs it).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = _UNIT):
        super().__init__()
        stride = tuple(int(s) for s in stride)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(_NEGATIVE_SLOPE)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        if in_channels == out_channels and stride == _UNIT:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))
        return self.act(y + self.skip(x))


class EncoderV5(nn.Module):
    """Returns the features of every stage, finest first."""

    def __init__(self, in_channels: int, widths: list, strides: list, blocks_per_stage: int):
        super().__init__()
        stages, channels = [], in_channels
        for width, stride in zip(widths, strides):
            blocks = [ResidualBlockV5(channels, width, stride)]
            blocks += [ResidualBlockV5(width, width) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            channels = width
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class DecoderV5(nn.Module):
    """From the coarsest features back to stage 0: transposed conv up, concat the skip, one residual block."""

    def __init__(self, widths: list, strides: list):
        super().__init__()
        self.ups, self.blocks = nn.ModuleList(), nn.ModuleList()
        for level in range(len(widths) - 2, -1, -1):
            stride = tuple(strides[level + 1])
            self.ups.append(nn.ConvTranspose3d(widths[level + 1], widths[level], stride, stride=stride))
            self.blocks.append(ResidualBlockV5(2 * widths[level], widths[level]))

    def forward(self, features: list) -> torch.Tensor:
        x = features[-1]
        for up, block, skip in zip(self.ups, self.blocks, reversed(features[:-1])):
            x = block(torch.cat([up(x), skip], dim=1))
        return x


class _ResUNetV5(PerceptionModelV5):
    supported_modes = ("3d",)
    TASKS: tuple = ()

    def __init__(self, scope: ActiveScope, widths: list, strides: list, blocks_per_stage: int):
        super().__init__(scope, self.TASKS)
        widths = [int(w) for w in widths]
        strides = [tuple(int(s) for s in stride) for stride in strides]
        if not widths or len(strides) != len(widths) or any(len(s) != 3 for s in strides):
            raise V5Error("INVALID_RECIPE_VALUE", "$.model.strides")
        outputs = {"A": 1 + len(scope.anatomy_classes), "U": len(scope.abnormality_classes)}
        self.encoder = EncoderV5(len(scope.input_modalities), widths, strides, int(blocks_per_stage))
        self.decoders = nn.ModuleDict({task: DecoderV5(widths, strides) for task in self.tasks})
        self.heads = nn.ModuleDict({task: nn.Conv3d(widths[0], outputs[task], 1) for task in self.tasks})

    def _forward_v5(self, batch) -> ModelOutputV5:
        features = self.encoder(batch.image)
        logits = {task: self.heads[task](self.decoders[task](features)) for task in self.tasks}
        return ModelOutputV5(anatomy_logits=logits.get("A"), abnormality_logits=logits.get("U"))


class AOnlyModelV5(_ResUNetV5):
    """Encoder + A decoder; head 1+K channels (softmax, channel 0 background)."""
    TASKS = ("A",)


class UOnlyModelV5(_ResUNetV5):
    """Encoder + U decoder; head C_U channels (sigmoid)."""
    TASKS = ("U",)


class SharedAUModelV5(_ResUNetV5):
    """One shared encoder, two independent decoders and heads (plain shared-encoder baseline, §6.1)."""
    TASKS = ("A", "U")


_CLASSES = {("A",): AOnlyModelV5, ("U",): UOnlyModelV5, ("A", "U"): SharedAUModelV5}


def build_model_v5(recipe: dict, scope: ActiveScope) -> PerceptionModelV5:
    """Pick the class from recipe model.tasks; initialise under recipe seeds.init.

    patch_size and inference.window must be divisible, per axis, by the product of that axis's strides. The
    weights are drawn from the CPU generator seeded with seeds.init inside a forked RNG, so the same recipe gives
    the same weights and the caller's RNG state is left untouched.
    """
    recipe = validate_recipe_v5(recipe)
    model = recipe["model"]
    factor = [math.prod(stride[axis] for stride in model["strides"]) for axis in range(3)]
    for vector, path in ((recipe["patch_size"], "$.patch_size"), (recipe["inference"]["window"], "$.inference.window")):
        if any(size % f for size, f in zip(vector, factor)):
            raise V5Error("PATCH_NOT_DIVISIBLE", path)
    cls = _CLASSES[tuple(sorted(model["tasks"]))]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(recipe["seeds"]["init"])
        return cls(scope, model["widths"], model["strides"], model["blocks_per_stage"])


def count_parameters_v5(model: nn.Module) -> int:
    """Number of parameter values, for the capacity-matching report."""
    return int(sum(p.numel() for p in model.parameters()))
