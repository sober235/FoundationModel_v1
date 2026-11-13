"""v5.1 observations, training samples, eligibility-aware patches and sampling (plan_v5 T09b; L2, L4, L5).

``load_observation_v5`` is shared by training and inference and reads image channels only: never labels,
review or visibility files (GC8). Channels are taken by modality name in the scope's order, and the record's
model modalities must equal the scope's inputs (L2). Downsampling is strided sampling anchored at the origin, the
same for image, labels and regions; a block whose positive vanished on the lattice gets no negative eligibility
(L5). Training samples come only from the frozen split's train assignment (L4). Arrays are (x, y, z).
Transform chains list operations in the order they were applied to the original grid; restoring walks them backwards.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ..contracts_v5 import ActiveScope, SampleV5, V5Error, array_sha256, canonical_sha256
from ..eligibility import annotation_channel_v5, build_eligibility_v5
from ..geometry import audit_geometry
from ..geometry_v5 import downsample_loss_report_v5
from ..models.interface_v5 import ModelInputV5, build_model_input_v5, collate_model_inputs_v5
from .registration_v5 import select_samples_v5, validate_recipe_v5, validate_supervision_v5
from .volume_io_v5 import VolumeV5, read_volume_v5

LABEL_ENCODINGS = ("channels", "values")
AXIS_ORDER = "xyz"


@dataclass(frozen=True, eq=False)
class ObservationV5:
    sample_uid: str
    patient_uid: str
    channels: dict          # modality name -> float32 array on the model grid, scope order
    valid_fov: np.ndarray   # bool, model grid
    grid: dict              # GEOMETRY_KEYS of the model grid
    to_original: list       # transform chain from the model grid back to the original grid
    original_shape: tuple
    original_header: dict | None  # VolumeV5 header of the first channel (None for H5)


@dataclass(frozen=True, eq=False)
class TrainingSampleV5:
    observation: ObservationV5
    targets: dict   # "A": int64 map (0 background, 1..K); "U": bool [C_U, ...]
    e_pos: dict     # task -> bool [classes, ...]
    e_neg: dict


def _classes(scope: ActiveScope, task: str) -> list[str]:
    return list(scope.anatomy_classes if task == "A" else scope.abnormality_classes)


def _steps(factor) -> tuple:
    return tuple(slice(None, None, int(f)) for f in factor)


def _same_grid(reference: dict | None, header: dict | None, reference_shape: tuple, shape: tuple,
               tolerance: dict) -> bool:
    """Shapes must match; when either side has a header both must, and they must agree within the recipe's
    registered tolerance (v41 audit_geometry). H5 volumes carry no header and are compared by shape only."""
    if tuple(reference_shape) != tuple(shape) or (reference is None) != (header is None):
        return False
    return reference is None or audit_geometry(reference, header, atol_affine=tolerance["atol_affine"],
                                               atol_spacing=tolerance["atol_spacing"])["status"] == "pass"


# ---------------------------------------------------------------------------
# Label maps


def _check_label_map(name: str, entry: object) -> dict:
    if not (isinstance(entry, dict) and set(entry) == {"encoding", "classes"}
            and entry["encoding"] in LABEL_ENCODINGS and isinstance(entry["classes"], dict) and entry["classes"]):
        raise V5Error("INVALID_LABEL_MAP", f"$.{name}")
    codes = list(entry["classes"].values())
    minimum = 0 if entry["encoding"] == "channels" else 1
    if not all(isinstance(k, str) and k for k in entry["classes"]) or not all(
            isinstance(c, int) and not isinstance(c, bool) and c >= minimum for c in codes) \
            or len(set(codes)) != len(codes):
        raise V5Error("INVALID_LABEL_MAP", f"$.{name}")
    return entry


def load_label_maps_v5(path: Path) -> dict:
    maps = yaml.safe_load(Path(path).read_text())
    if not isinstance(maps, dict) or not maps:
        raise V5Error("INVALID_LABEL_MAP", "$")
    return {name: _check_label_map(name, entry) for name, entry in maps.items()}


def decode_targets_v5(volume: VolumeV5, annotation, classes: list[str], label_maps: dict) -> np.ndarray:
    """bool [len(classes), *spatial]; a class absent from the map is all-False unless the annotation claims it."""
    entry = label_maps.get(annotation.label_map_version) if isinstance(label_maps, dict) else None
    if entry is None:
        raise V5Error("LABEL_MAP_UNKNOWN", "$.label_map_version")
    entry = _check_label_map(annotation.label_map_version, entry)
    array = volume.array
    if entry["encoding"] == "channels":
        if array.dtype != bool or array.ndim != 4:
            raise V5Error("LABEL_ENCODING_MISMATCH", "$.label")
        spatial = array.shape[:3]
    else:
        if array.dtype.kind not in "iu" or array.ndim != 3:
            raise V5Error("LABEL_ENCODING_MISMATCH", "$.label")
        spatial = array.shape
        present = set(np.unique(array).tolist()) - {0}
        if not present <= set(entry["classes"].values()):
            raise V5Error("LABEL_VALUE_UNEXPECTED", "$.label")
    out = np.zeros((len(classes), *spatial), dtype=bool)
    for k, name in enumerate(classes):
        code = entry["classes"].get(name)
        if code is None:
            if name in annotation.class_scope:
                raise V5Error("LABEL_MAP_CLASS_MISSING", f"$.classes[{k}]")
            continue
        if entry["encoding"] == "channels":
            if code >= array.shape[3]:
                raise V5Error("LABEL_MAP_CLASS_MISSING", f"$.classes[{k}]")
            out[k] = array[..., code]
        else:
            out[k] = array == code
    if annotation.task == "A" and (out.sum(axis=0) > 1).any():
        raise V5Error("LABEL_OVERLAP", "$.label")
    return out


# ---------------------------------------------------------------------------
# Observations and training samples


def _model_grid(header: dict | None, record: SampleV5, shape: tuple, factor: tuple, to_original: list) -> dict:
    geometry = header if header is not None else (record.image_geometry or {})
    spacing, affine = geometry.get("spacing_mm"), geometry.get("affine")
    return {"shape": list(shape), "axis_order": AXIS_ORDER,
            "spacing_mm": None if spacing is None else [float(s) * f for s, f in zip(spacing, factor)],
            "affine": None if affine is None else (np.asarray(affine, dtype=float) @ np.diag([*factor, 1])).tolist(),
            "units_reliable": record.physical_geometry_status == "verified",
            "transform_chain": copy.deepcopy(to_original)}


def original_geometry_v5(observation: ObservationV5, record: SampleV5) -> dict:
    """GEOMETRY_KEYS of the observation's original grid: restored outputs are written on it (T09c)."""
    if not isinstance(observation, ObservationV5) or not isinstance(record, SampleV5) \
            or observation.sample_uid != record.sample_uid:
        raise V5Error("INVALID_TYPE", "$.observation")
    ones = (1,) * len(observation.original_shape)
    return _model_grid(observation.original_header, record, observation.original_shape, ones, [])


def load_observation_v5(root: Path, record: SampleV5, scope: ActiveScope, recipe: dict) -> ObservationV5:
    if not isinstance(record, SampleV5):
        raise V5Error("INVALID_TYPE", "$.record")
    if not isinstance(scope, ActiveScope):
        raise V5Error("INVALID_TYPE", "$.scope")
    validate_recipe_v5(recipe)
    if list(record.model_modalities) != list(scope.input_modalities):
        raise V5Error("MODALITY_SCOPE_MISMATCH", "$.model_modalities")
    if scope.observation_mode != "3d" or record.observation_mode != "3d":
        raise V5Error("MODE_NOT_SUPPORTED", "$.observation_mode")
    channels: dict[str, np.ndarray] = {}
    header = None
    for name in scope.input_modalities:
        index = record.modality_names.index(name)
        if not record.available_modality_mask[index]:
            raise V5Error("INPUT_MODALITY_UNAVAILABLE", f"$.available_modality_mask[{index}]")
        volume = read_volume_v5(root, record.image_paths[index], "image")
        if volume.array.ndim != 3 or (channels and not _same_grid(
                header, volume.header, next(iter(channels.values())).shape, volume.array.shape,
                recipe["grid_tolerance"])):
            raise V5Error("GRID_MISMATCH", f"$.image_paths[{index}]")
        if not np.isfinite(volume.array).all():
            raise V5Error("NON_FINITE_INPUT", f"$.image_paths[{index}]")
        if not channels:
            header = volume.header
        channels[name] = volume.array
    shape = next(iter(channels.values())).shape
    if header is None and record.image_geometry is not None \
            and list(record.image_geometry.get("shape") or []) != list(shape):
        # headerless (H5) channels take spacing/affine from the record: only when it describes this grid
        raise V5Error("GRID_MISMATCH", "$.image_geometry.shape")
    factor = tuple(int(f) for f in recipe["downsample_factor"])
    fov = np.ones(shape, dtype=bool)
    to_original: list[dict] = []
    if any(f != 1 for f in factor):
        steps = _steps(factor)
        channels = {name: array[steps] for name, array in channels.items()}
        fov = fov[steps]
        to_original = [{"op": "downsample", "factor": list(factor), "input_shape": list(shape)}]
    normalized = {}
    for index, (name, array) in enumerate(channels.items()):
        values = array[fov].astype(np.float64)
        mean, std = values.mean(), values.std()
        if not std > 0:
            raise V5Error("DEGENERATE_CHANNEL", f"$.channels[{index}]")
        out = np.zeros(array.shape, dtype=np.float32)
        out[fov] = ((array[fov] - mean) / std).astype(np.float32)
        normalized[name] = out
    return ObservationV5(sample_uid=record.sample_uid, patient_uid=record.patient_uid, channels=normalized,
                         valid_fov=fov, grid=_model_grid(header, record, fov.shape, factor, to_original),
                         to_original=to_original, original_shape=tuple(shape), original_header=header)


def _strided_regions(regions: object, shape: tuple, steps: tuple) -> tuple[dict, dict]:
    if not (isinstance(regions, dict) and set(regions) == {"review", "visibility"}
            and all(isinstance(regions[k], dict) for k in regions)):
        raise V5Error("INVALID_TYPE", "$.regions")
    out = []
    for kind in ("review", "visibility"):
        strided = {}
        for key, region in regions[kind].items():
            region = np.asarray(region)
            if region.dtype != bool or region.shape != tuple(shape):
                raise V5Error("GRID_MISMATCH", f"$.regions.{kind}")
            strided[key] = region[steps]
        out.append(strided)
    return out[0], out[1]


def _task_targets(root: Path, record: SampleV5, task: str, classes: list[str], label_maps: dict,
                  observation: ObservationV5, tolerance: dict) -> np.ndarray:
    shape = observation.original_shape
    full = np.zeros((len(classes), *shape), dtype=bool)
    for index, annotation in enumerate(record.annotation_records):
        if annotation_channel_v5(annotation, task) != "expert" or not set(classes) & set(annotation.class_scope):
            continue
        if annotation.path is None:
            raise V5Error("ANNOTATION_PATH_MISSING", f"$.annotation_records[{index}].path")
        volume = read_volume_v5(root, annotation.path, "label")
        decoded = decode_targets_v5(volume, annotation, classes, label_maps)
        if not _same_grid(observation.original_header, volume.header, shape, decoded.shape[1:], tolerance):
            raise V5Error("GRID_MISMATCH", f"$.annotation_records[{index}].path")
        for k, name in enumerate(classes):
            if name in annotation.class_scope:
                full[k] |= decoded[k]
    if task == "A" and (full.sum(axis=0) > 1).any():
        raise V5Error("LABEL_OVERLAP", "$.annotation_records")
    return full


def load_training_sample_v5(root: Path, record: SampleV5, scope: ActiveScope, supervision: dict, recipe: dict,
                            label_maps: dict, regions: dict) -> TrainingSampleV5:
    validate_supervision_v5(supervision, scope)
    observation = load_observation_v5(root, record, scope, recipe)
    factor = tuple(int(f) for f in recipe["downsample_factor"])
    steps = _steps(factor)
    review, visibility = _strided_regions(regions, observation.original_shape, steps)
    policy = {key: supervision[key] for key in ("track", "allow_unknown_visibility", "reference_policy_id")}
    targets, e_pos, e_neg = {}, {}, {}
    for task in sorted(supervision["tasks"]):
        classes = _classes(scope, task)
        full = _task_targets(root, record, task, classes, label_maps, observation, recipe["grid_tolerance"])
        down = full[(slice(None), *steps)]
        positive, negative = build_eligibility_v5(record, task, classes, {}, review, visibility,
                                                  observation.valid_fov, policy, "expert")
        if any(f != 1 for f in factor):
            negative &= ~downsample_loss_report_v5(full, factor, classes)["lost_positive_mask"]
        if task == "A":
            targets[task] = np.where(down.any(axis=0), down.argmax(axis=0) + 1, 0).astype(np.int64)
        else:
            targets[task] = down
        e_pos[task], e_neg[task] = positive, negative
    return TrainingSampleV5(observation=observation, targets=targets, e_pos=e_pos, e_neg=e_neg)


def load_training_set_v5(root: Path, records: list[SampleV5], registry: dict, scope: ActiveScope, supervision: dict,
                         recipe: dict, label_maps: dict, regions: dict) -> dict[str, TrainingSampleV5]:
    """Training samples keyed by sample_uid, read only for the frozen split's train assignment (L4)."""
    if not isinstance(regions, dict):
        raise V5Error("INVALID_TYPE", "$.regions")
    empty = {"review": {}, "visibility": {}}
    return {record.sample_uid: load_training_sample_v5(root, record, scope, supervision, recipe, label_maps,
                                                       regions.get(record.sample_uid, empty))
            for record in select_samples_v5(records, registry, ["train"])}


def training_data_sha256_v5(samples: dict) -> str:
    """Digest of exactly what training reads, uids sorted: per sample every array (channels in scope order,
    valid_fov, targets, e_pos, e_neg), the model and original grids and the transform chain. Records, label maps
    and regions enter through the arrays they produce, so editing any of them changes the digest."""
    if not isinstance(samples, dict) or not samples:
        raise V5Error("INVALID_TYPE", "$.samples")
    digests = {}
    for uid in sorted(samples):
        sample = samples[uid]
        if not isinstance(sample, TrainingSampleV5):
            raise V5Error("INVALID_TYPE", "$.samples")
        observation = sample.observation
        digests[uid] = canonical_sha256({
            "patient_uid": observation.patient_uid,
            "channels": [[name, array_sha256(array)] for name, array in observation.channels.items()],
            "valid_fov": array_sha256(observation.valid_fov), "grid": observation.grid,
            "to_original": observation.to_original, "original_shape": list(observation.original_shape),
            "original_header": observation.original_header,
            **{key: {task: array_sha256(array) for task, array in getattr(sample, key).items()}
               for key in ("targets", "e_pos", "e_neg")}})
    return canonical_sha256(digests)


# ---------------------------------------------------------------------------
# Sampling, patches and batches


def positive_centers_v5(sample: TrainingSampleV5) -> np.ndarray:
    """Eligible positive voxels: A foreground where every class is eligible; U any class positive with e_pos."""
    mask = np.zeros(sample.observation.valid_fov.shape, dtype=bool)
    if "A" in sample.targets:
        mask |= (sample.targets["A"] > 0) & sample.e_pos["A"].all(axis=0) & sample.e_neg["A"].all(axis=0)
    if "U" in sample.targets:
        mask |= (sample.targets["U"] & sample.e_pos["U"]).any(axis=0)
    return np.argwhere(mask)


class PatchSamplerV5:
    """Seeded patch centers; every random draw comes from one PCG64 stream so its state resumes exactly."""

    def __init__(self, sample_uids: list[str], positive_centers: dict, positive_fraction: float, seed: int):
        if not (isinstance(sample_uids, list) and sample_uids and all(isinstance(u, str) for u in sample_uids)
                and len(set(sample_uids)) == len(sample_uids)):
            raise V5Error("INVALID_SAMPLER", "$.sample_uids")
        if not (isinstance(positive_fraction, (int, float)) and not isinstance(positive_fraction, bool)
                and 0 <= positive_fraction <= 1):
            raise V5Error("INVALID_SAMPLER", "$.positive_fraction")
        if not (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0):
            raise V5Error("INVALID_SAMPLER", "$.seed")
        if not isinstance(positive_centers, dict) or not set(positive_centers) <= set(sample_uids):
            raise V5Error("INVALID_SAMPLER", "$.positive_centers")
        self._uids = sorted(sample_uids)
        self._centers = {uid: np.asarray(centers, dtype=np.int64) for uid, centers in positive_centers.items()}
        self._fraction = float(positive_fraction)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._epoch = 0
        self._cursor = 0
        self._order = [int(i) for i in self._rng.permutation(len(self._uids))]

    def next_uid(self) -> str:
        if self._cursor == len(self._order):
            self._epoch += 1
            self._cursor = 0
            self._order = [int(i) for i in self._rng.permutation(len(self._uids))]
        uid = self._uids[self._order[self._cursor]]
        self._cursor += 1
        return uid

    def draw_center(self, uid: str, valid_fov: np.ndarray) -> tuple[int, ...]:
        positives = self._centers.get(uid)
        if self._rng.random() < self._fraction and positives is not None and len(positives):
            return tuple(int(v) for v in positives[int(self._rng.integers(len(positives)))])
        flat = np.asarray(valid_fov, dtype=bool).ravel()
        count = int(np.count_nonzero(flat))
        if count == 0:
            raise V5Error("EMPTY_FOV", "$.valid_fov")
        k = int(self._rng.integers(count))
        index = k if count == flat.size else int(np.flatnonzero(flat)[k])
        return tuple(int(v) for v in np.unravel_index(index, np.shape(valid_fov)))

    def state_dict(self) -> dict:
        return {"bit_generator": copy.deepcopy(self._rng.bit_generator.state), "epoch": self._epoch,
                "cursor": self._cursor, "order": list(self._order)}

    def load_state_dict(self, state: dict) -> None:
        if not (isinstance(state, dict) and set(state) == {"bit_generator", "epoch", "cursor", "order"}
                and sorted(state["order"]) == list(range(len(self._uids)))):
            raise V5Error("INVALID_SAMPLER_STATE", "$.sampler")
        self._rng.bit_generator.state = copy.deepcopy(state["bit_generator"])
        self._epoch, self._cursor = int(state["epoch"]), int(state["cursor"])
        self._order = [int(i) for i in state["order"]]


def _crop(array: np.ndarray, start: tuple, size: tuple, leading: int = 0) -> np.ndarray:
    out = np.zeros(array.shape[:leading] + tuple(size), dtype=array.dtype)
    source, target = [], []
    for s, n, dim in zip(start, size, array.shape[leading:]):
        low, high = max(s, 0), min(s + n, dim)
        if high <= low:
            return out
        source.append(slice(low, high))
        target.append(slice(low - s, high - s))
    lead = (slice(None),) * leading
    out[lead + tuple(target)] = array[lead + tuple(source)]
    return out


def extract_patch_v5(sample: TrainingSampleV5, center: tuple, patch_size: tuple) -> dict:
    """Patch around `center`; outside the volume everything is zero and nothing is in FOV or eligible."""
    observation = sample.observation
    shape = observation.valid_fov.shape
    if len(center) != len(shape) or len(patch_size) != len(shape):
        raise V5Error("GRID_MISMATCH", "$.patch_size")
    start = tuple(int(c) - int(n) // 2 for c, n in zip(center, patch_size))
    size = tuple(int(n) for n in patch_size)
    grid = dict(copy.deepcopy(observation.grid), shape=list(size))
    if grid["affine"] is not None:
        affine = np.asarray(grid["affine"], dtype=float)
        affine[:3, 3] += affine[:3, :3] @ np.asarray(start, dtype=float)
        grid["affine"] = affine.tolist()
    crop = {"op": "crop", "start": list(start), "input_shape": list(shape)}
    grid["transform_chain"] = [*copy.deepcopy(observation.to_original), crop]   # application order
    return {
        "channels": {name: _crop(array, start, size) for name, array in observation.channels.items()},
        "valid_fov": _crop(observation.valid_fov, start, size),
        "targets": {task: _crop(t, start, size, leading=t.ndim - len(shape)) for task, t in sample.targets.items()},
        "e_pos": {task: _crop(e, start, size, leading=1) for task, e in sample.e_pos.items()},
        "e_neg": {task: _crop(e, start, size, leading=1) for task, e in sample.e_neg.items()},
        "grid": grid,
        "to_original": list(grid["transform_chain"]),
    }


def make_batch_v5(samples: dict, sampler: PatchSamplerV5, recipe: dict,
                  scope: ActiveScope) -> tuple[ModelInputV5, dict, dict]:
    """One batch of `batch_size` patches, single process so the sampler state is the whole data state."""
    tasks = sorted(recipe["model"]["tasks"])
    patch_size = tuple(recipe["patch_size"])
    items, targets, e_pos, e_neg = [], {t: [] for t in tasks}, {t: [] for t in tasks}, {t: [] for t in tasks}
    for _ in range(recipe["batch_size"]):
        uid = sampler.next_uid()
        sample = samples.get(uid)
        if sample is None:
            raise V5Error("SAMPLE_UNKNOWN", "$.samples")
        for task in tasks:
            if task not in sample.targets or task not in sample.e_pos or task not in sample.e_neg:
                raise V5Error("TARGET_MISSING", f"$.samples.targets.{task}")
        patch = extract_patch_v5(sample, sampler.draw_center(uid, sample.observation.valid_fov), patch_size)
        items.append(build_model_input_v5(patch["channels"], scope, patch["valid_fov"], patch["grid"],
                                          patch["to_original"]))
        for task in tasks:
            targets[task].append(patch["targets"][task])
            e_pos[task].append(patch["e_pos"][task])
            e_neg[task].append(patch["e_neg"][task])
    batch = collate_model_inputs_v5(items)
    return (batch, {task: torch.from_numpy(np.stack(targets[task])) for task in tasks},
            {task: {"e_pos": torch.from_numpy(np.stack(e_pos[task])),
                    "e_neg": torch.from_numpy(np.stack(e_neg[task]))} for task in tasks})
