# 统一病变所见模型（本地实现）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一个 checkpoint 按扫描协议处理膝关节 qDESS 与脑 DWI+ADC 两种三维影像，输出"位置框 + 所在结构 + 疑似异常"的所见，并能在开发集上按医生的原标注评价；本计划只到合成数据上的 CPU 验证。

**Architecture:** 共享一个残差 U-Net 编码器，每个协议有自己的输入层、解码器和输出头；每个协议的"视图"是一个现有接口的 `PerceptionModelV5`，所以整卷推理、输出链、模块评价、运行登记和曝光登记都原样复用。膝关节的病变只有框，用多示例损失训练（框内宿主组织上"至少有"，框外为负，非主分析框为忽略区）。每个训练步各取两个协议的一个批次，梯度累加后更新一次。

**Tech Stack:** Python 3.11（服务器）/ 3.9（本机），PyTorch，NumPy，SciPy，PyYAML，pytest。不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-10-unified-lesion-finding-design.md`（执行者两份都要读）。背景规格 `Proposal_v5.md`，现有工程计划 `plan_v5.md`。

## Global Constraints

- 只在 `anatobind/v41/` 内原位新增文件；不改动已冻结接口的行为：`models/interface_v5.py`、`engine/train_v5.py`、`engine/registration_v5.py`、`perception.py`、`gates_v5.py` 只允许被调用，不允许改语义。任务里明确写出的新增函数除外。
- 错误一律 `V5Error(code, path)`，不带输入值、病例 ID 或路径。
- 私有产物只经 `paths.resolve_private_output` / `write_private_json` 写出：目录 700、文件 600、不覆盖。
- 进仓库的只有代码、合成测试、脱敏汇总；不出现私有路径、病例 ID、数据文件名。服务器在文档里叫 `amax`。
- 训练与模型选择只读冻结划分的 `train` 与 `development_val`；`locked_test` 一律拒绝（`LOCKED_TEST_NOT_READABLE`）。
- 框不转成体素真值；医生没标的地方不写成"正常"；框外预测不叫误报。
- 所有科学结论保持 `NOT_EVALUATED`；`core_target_validated` 保持 `null`；G-JOINT 的规则不动。
- 计划文件里的数值都是提案，真实运行前由研究者批准；批准文件由研究者本人签，执行方不填 `approved_by`。
- 本计划全部在本机 CPU 上用合成数据完成；不连接 amax，不读真实数据。
- 提交信息用英文祈使句，不带任何 AI 署名或 Co-Authored-By；只做本地提交，推送须先问研究者。
- 每个任务结束时全套测试必须通过：`python3 -m pytest tests -q -p no:cacheprovider`（当前基线 759 passed, 2 skipped）。
- 测试文件开头的写法、`_raises(code, path, fn, ...)` 辅助函数和合成工厂沿用 `tests/v5/test_training_core.py` 与 `tests/v5/factories.py`。

## Review Focus

1. **补丁边界切断了框**：框只有一部分落在训练补丁里。期望：损失有限、可反传，不报错。测试在任务 3。
2. **没有主分析框的扫描**（真实训练集 86 例里有 40 例）：期望：能正常成为训练样本，U 只有负约束，采样不崩。测试在任务 2 和任务 6。
3. **越界或退化的框**（被排除的框可能零尺寸或超出体积）：期望：构造忽略区时裁剪到体积内，不报错、不产生负索引切片。测试在任务 2。
4. **推理没有任何所见，或所见处没有被接受的组织**：期望：输出"未见所列异常"的固定句式或"所在结构未能确定"，评价把对应参照框记为未命中，不崩。测试在任务 7、任务 8。
5. **协议与影像不匹配**（把脑的通道交给膝的协议）：期望：在进入网络之前以 `INPUT_MODALITY_MISSING` 拒绝，不静默运行。测试在任务 4。

---

### Task 1: 记录目标修订（规格与状态）

**Files:**
- Modify: `Proposal_v5.md`（在 §0.5 之后新增 §0.6；§1.1 末尾加一段指向 §0.6）
- Modify: `plan_v5.md`（第一处 `SHA256 \`...\`` 更新为新的规格哈希；修订说明加一行）
- Modify: `reports/v5/status.md`（YAML 头 `spec_sha256`；正文新增"## 2026-11-10 目标修订"一节）
- Modify: `reports/v5/traceability.md`（末尾加一行指向本计划）
- Test: `tests/v5/test_status_docs.py`

**Interfaces:**
- Consumes: 无。
- Produces: 文档事实。后续任务引用"Proposal §0.6"。

- [ ] **Step 1: 写失败测试**，加到 `tests/v5/test_status_docs.py` 末尾：

```python
def test_goal_revision_of_20261110_is_recorded_without_claiming_validation():
    proposal = Path("Proposal_v5.md").read_text(encoding="utf-8")
    assert "### 0.6 2026-11-10 研究者决定：按原标注粒度验收" in proposal
    for phrase in ("同一个 checkpoint 处理膝关节 qDESS 与脑 DWI+ADC", "只做三维", "不再要求体素级的病变边界验证",
                   "医生没有标注不等于正常", "`core_target_validated` 保持 `null`"):
        assert phrase in proposal
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "## 2026-11-10 目标修订" in status
    assert "docs/superpowers/plans/2026-11-10-unified-lesion-finding.md" in status
    meta = _front_matter()
    assert meta["core_target_validated"] is None and meta["scientific_conclusion"] == "NOT_EVALUATED"
    design = Path("docs/superpowers/specs/2026-11-10-unified-lesion-finding-design.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, design)
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_status_docs.py -q -p no:cacheprovider`，预期新测试 FAIL（找不到 §0.6）。

- [ ] **Step 3: 写 Proposal §0.6**。内容逐条对应设计稿 §1–§4，必须包含 Step 1 断言的五个短语，并写明：
  - 这是研究者 2026-11-10 的决定，不是执行方改目标；
  - §1.1 的体素级八项能力不再是当前版本的验收目标，相关条文保留作历史与后续扩展；
  - T12–T18 中依赖体素级参照、二维模式或 H2 的部分暂停，不删除任务编号；
  - 新工作的计划是本文件，设计是设计稿。

- [ ] **Step 4: 更新哈希与状态**。`shasum -a 256 Proposal_v5.md` 的结果写入 `status.md` 的 `spec_sha256` 和 `plan_v5.md` 的 `SHA256` 处；`status.md` 新增一节，列出：决定内容三条、计划与设计稿路径、"八项能力与 H1–H3 暂停"、"G-JOINT 规则不变"。

- [ ] **Step 5: 运行全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider      # 760 passed, 2 skipped
git add Proposal_v5.md plan_v5.md reports/v5/status.md reports/v5/traceability.md tests/v5/test_status_docs.py docs/superpowers
git commit -m "Record the 2026-11-10 goal revision: validate at the granularity of the released annotations"
```

---

### Task 2: 框监督目标

**Files:**
- Create: `anatobind/v41/engine/box_data_v5.py`
- Test: `tests/v5/test_box_data.py`

**Interfaces:**
- Consumes: `engine.data_v5.load_training_sample_v5`、`TrainingSampleV5`；`engine.registration_v5.READABLE_ROLES`；`annotation_v5.load_skm_box_policy_v5`；框参照的一行（`real_t11_v5` 写出的 `box_references.jsonl`，字段 `sample_uid, policy_sha256, boxes[*].{status, analysis, abnormality_class, box.{start,size}, host.tissue_group}`）。
- Produces:
  - `box_bags_v5(reference: dict, shape: tuple, scope: ActiveScope, anatomy: dict, box_policy: dict) -> dict`，返回 `{"bags": int32 [C_U, *shape], "e_pos": bool [C_U, *shape], "e_neg": bool [C_U, *shape], "n_bags": {class: int}}`。
  - `load_box_references_v5(path: Path, registry: dict, roles: list[str]) -> dict[str, dict]`。
  - `load_box_training_sample_v5(root, record, scope, supervision, recipe, label_maps, regions, reference, box_policy) -> TrainingSampleV5`，其中 `targets["U"]` 是 `bags`（int32），`e_pos["U"]`、`e_neg["U"]` 已与 `valid_fov` 相与。
  - `box_positive_centers_v5(sample: TrainingSampleV5) -> np.ndarray`：采样用的正中心。A 前景（与 `positive_centers_v5` 相同的条件）并上 `((targets["U"] > 0) & e_pos["U"]).any(axis=0)`。不能直接用现有的 `positive_centers_v5`：它对 U 做 `targets & e_pos` 的按位与，示例编号为偶数的体素会被丢掉。

`box_bags_v5` 的规则（签名和测试不足以唯一确定，所以写明）：

1. 类别通道顺序为 `scope.abnormality_classes`。框的体素范围是 `[start, start+size)`，先裁剪到 `[0, shape)`；裁剪后为空的框没有任何效果。
2. **主分析框**（`status == "included"` 且 `analysis == "primary"`）：示例区 = 框 ∩ 宿主组织组的参照组织。组织组到 A 类名的映射取 `box_policy["host_tissues"]` 中 `group` 等于该框 `host.tissue_group` 的条目的 `anatomy` 列表。示例区的体素写入该类通道，值为该框在 `reference["boxes"]` 里的位置加 1。
3. 同一类别的两个示例区重叠时，体素归示例区体素数较少的框；相等时归编号小的框。
4. `e_pos[c] = bags[c] > 0`。
5. `e_neg[c]` 初始全真，然后置假：类别 c 的每个主分析框的**整个框**；以及**所有非主分析框**（secondary、excluded、out_of_scope）的整个框，对每个类别都置假。
6. 错误：`anatomy` 缺少需要的类名或形状不等于 `shape` → `GRID_MISMATCH`，`$.anatomy`；主分析框的类别不在 scope → `BOX_CLASS_UNKNOWN`，`$.boxes[i].abnormality_class`；主分析框的示例区为空 → `BOX_HOST_EMPTY`，`$.boxes[i]`（导入时已保证非空，为空说明网格不一致）。

`load_box_references_v5`：`roles` 必须是 `READABLE_ROLES` 的非空子集，否则 `LOCKED_TEST_NOT_READABLE`，`$.roles[i]`（与 `select_samples_v5` 相同的码和路径）；只返回冻结登记 `registry["assignment"]` 中角色属于 `roles` 的样本的参照；文件里其他样本的行读入后立即丢弃，不返回。重复的 `sample_uid` → `DUPLICATE_SAMPLE`，`$.references`。

`load_box_training_sample_v5`：
- `recipe["downsample_factor"] != [1, 1, 1]` → `BOX_DOWNSAMPLE_NOT_SUPPORTED`，`$.recipe.downsample_factor`；
- `reference["sample_uid"] != record.sample_uid` → `BOX_REFERENCE_MISMATCH`，`$.reference.sample_uid`；
- `reference["policy_sha256"] != box_policy["policy_sha256"]` → `BOX_POLICY_MISMATCH`，`$.reference.policy_sha256`；
- A 的目标与资格由 `load_training_sample_v5` 得到，传入的 supervision 是原 supervision 把 `tasks` 限制为 `{"A": ...}` 的副本；`anatomy` 由 `targets["A"] == k+1` 得到；
- 返回的样本包含 A 与 U 两个任务。

- [ ] **Step 1: 写失败测试** `tests/v5/test_box_data.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_box_data.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError: anatobind.v41.engine.box_data_v5`。

- [ ] **Step 3: 实现 `box_bags_v5` 与 `load_box_references_v5`**，按上面的规则。重叠处理的做法：先算每个主分析框的示例区体素数，按（体素数降序、编号降序）依次写入，后写的覆盖先写的。

- [ ] **Step 4: 运行确认通过**：同 Step 2 的命令，预期 6 passed（`box_positive_centers_v5` 一并实现）。

- [ ] **Step 5: 为 `load_box_training_sample_v5` 写失败测试**，追加到同一文件。用 `tests.v5.factories.write_volume_case` 在 `tmp_path` 写一个合成的联合 scope 病例（`make_scope()` 的合成联合 scope，形状 `(8, 8, 8)`），再手工给出一个主分析框参照；合成 scope 没有真实的框策略，所以测试里用最小的策略字典 `{"policy_sha256": "p" * 64, "host_tissues": {"T": {"group": "g", "anatomy": ["tissue_a"]}}}`。断言：

```python
    assert set(sample.targets) == {"A", "U"} and sample.targets["U"].dtype == np.int32
    assert sample.targets["U"].shape == (len(scope.abnormality_classes), 8, 8, 8)
    assert np.array_equal(sample.e_pos["U"], sample.targets["U"] > 0)
    _raises("BOX_DOWNSAMPLE_NOT_SUPPORTED", "$.recipe.downsample_factor", load_box_training_sample_v5, ...)  # factor [2, 2, 1]
    _raises("BOX_REFERENCE_MISMATCH", "$.reference.sample_uid", load_box_training_sample_v5, ...)            # other uid
    _raises("BOX_POLICY_MISMATCH", "$.reference.policy_sha256", load_box_training_sample_v5, ...)            # other hash
```

- [ ] **Step 6: 实现 `load_box_training_sample_v5`，运行确认通过**，预期 7 passed。

- [ ] **Step 7: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/engine/box_data_v5.py tests/v5/test_box_data.py
git commit -m "Add box supervision targets: bags on the host tissue, negatives outside boxes, ignore regions"
```

---

### Task 3: 框多示例损失

**Files:**
- Modify: `anatobind/v41/losses.py`（文件末尾新增，不改现有函数）
- Test: `tests/v5/test_box_loss.py`

**Interfaces:**
- Consumes: 任务 2 的 `bags / e_pos / e_neg` 约定。
- Produces: `BOX_SPEC_KEYS = ("kind", "topk_fraction", "positive_weight", "negative_weight", "box_policy_sha256")`；`box_mil_objective_v5(logits: Tensor [B, C, *spatial], bags: Tensor [B, C, *spatial] 整型, e_pos: Tensor bool, e_neg: Tensor bool, spec: dict) -> Tensor`（标量）。

公式（记 `sp(x) = softplus(x)`）：

- 负项：对每个有 `e_neg` 体素的 (类别 c)，取这些体素上 `sp(logit)` 的均值；再对有支持的类别取均值，得 `L_neg`。
- 正项：对每个 (b, c, 示例编号 k)，取 `bags[b, c] == k` 且 `e_pos[b, c]` 的体素，数量 n；取其中 logit 最大的 `m = max(1, ceil(topk_fraction · n))` 个，示例损失为这 m 个体素上 `sp(-logit)` 的均值；对所有示例取均值，得 `L_pos`。
- 返回 `positive_weight · L_pos + negative_weight · L_neg`；缺少的一项记 0。两项都没有支持时返回可反传的零：`logits.masked_fill(torch.ones_like(logits, dtype=torch.bool), 0).sum() * 0.0`。
- 校验：`spec["kind"] != "box_mil"` → `UNKNOWN_LOSS_KIND`，`$.spec.kind`；`topk_fraction` 不在 (0, 1] 或权重为负或两权重之和 ≤ 0 → `INVALID_LOSS_WEIGHTS`，`$.spec`；四个张量形状不一致 → `GRID_MISMATCH`，`$.logits`；被选中的 logit 含非有限值 → `NON_FINITE_LOGITS`，`$.logits`。

- [ ] **Step 1: 写失败测试** `tests/v5/test_box_loss.py`：

```python
"""Box multiple-instance objective (design 2026-11-10 §5.2). Hand-computed values."""
from __future__ import annotations

import math

import pytest
import torch

from anatobind.v41.contracts_v5 import V5Error
from anatobind.v41.losses import box_mil_objective_v5

SPEC = {"kind": "box_mil", "topk_fraction": 0.5, "positive_weight": 1.0, "negative_weight": 1.0,
        "box_policy_sha256": "p" * 64}


def _case(values, bag_ids, negative):
    logits = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1).requires_grad_()
    bags = torch.tensor(bag_ids, dtype=torch.int32).reshape(1, 1, 1, 1, -1)
    e_neg = torch.tensor(negative, dtype=torch.bool).reshape(1, 1, 1, 1, -1)
    return logits, bags, bags > 0, e_neg


def test_value_matches_the_hand_computation():
    logits, bags, e_pos, e_neg = _case([2.0, -1.0, 0.0, 3.0], [1, 1, 0, 0], [False, False, True, True])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)
    positive = math.log1p(math.exp(-2.0))                                  # top 1 of the 2 bag voxels
    negative = (math.log(2.0) + math.log1p(math.exp(3.0))) / 2
    assert float(loss) == pytest.approx(positive + negative, abs=1e-6)     # 1.997795
    loss.backward()
    assert logits.grad.reshape(-1)[1] == 0                                 # the bag voxel outside the top-k


def test_a_bag_cut_by_the_patch_border_still_gives_a_finite_loss():
    logits, bags, e_pos, e_neg = _case([-4.0, 0.5, 0.2, 0.1], [7, 0, 0, 0], [False, False, False, False])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)          # one bag voxel left, no negatives
    assert float(loss) == pytest.approx(math.log1p(math.exp(4.0)), abs=1e-6)


def test_no_support_gives_a_differentiable_zero_even_with_infinite_logits():
    logits, bags, e_pos, e_neg = _case([float("inf"), 1.0], [0, 0], [False, False])
    loss = box_mil_objective_v5(logits, bags, e_pos, e_neg, SPEC)
    loss.backward()
    assert float(loss) == 0.0 and not logits.grad.any()


def test_two_bags_average_and_weights_scale_each_part():
    logits, bags, e_pos, e_neg = _case([0.0, 0.0, 0.0], [1, 2, 0], [False, False, True])
    spec = dict(SPEC, topk_fraction=1.0, positive_weight=2.0, negative_weight=0.5)
    assert float(box_mil_objective_v5(logits, bags, e_pos, e_neg, spec)) == pytest.approx(2.5 * math.log(2.0), abs=1e-6)


@pytest.mark.parametrize("change,code,path", [
    ({"kind": "sigmoid"}, "UNKNOWN_LOSS_KIND", "$.spec.kind"),
    ({"topk_fraction": 0.0}, "INVALID_LOSS_WEIGHTS", "$.spec"),
    ({"positive_weight": 0.0, "negative_weight": 0.0}, "INVALID_LOSS_WEIGHTS", "$.spec")])
def test_invalid_specs_are_refused(change, code, path):
    logits, bags, e_pos, e_neg = _case([0.0], [0], [True])
    with pytest.raises(V5Error) as err:
        box_mil_objective_v5(logits, bags, e_pos, e_neg, dict(SPEC, **change))
    assert (err.value.code, err.value.path) == (code, path)
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_box_loss.py -q -p no:cacheprovider`，预期 `ImportError: cannot import name 'box_mil_objective_v5'`。
- [ ] **Step 3: 实现 `box_mil_objective_v5`**，按上面的公式；top-m 用 `torch.topk`。
- [ ] **Step 4: 运行确认通过**，预期 7 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/losses.py tests/v5/test_box_loss.py
git commit -m "Add the box multiple-instance objective for box-level abnormality supervision"
```

---

### Task 4: 统一模型

**Files:**
- Create: `anatobind/v41/models/unified_v5.py`
- Test: `tests/v5/test_unified_model.py`

**Interfaces:**
- Consumes: `models.resunet_v5.EncoderV5 / DecoderV5`，`models.interface_v5.PerceptionModelV5 / ModelOutputV5`。
- Produces:
  - `class MemberModelV5(PerceptionModelV5)`：一个协议的视图。`_forward_v5` = 该协议的输入层 → 共享编码器 → 自己的解码器与输出头。
  - `class UnifiedModelV5(torch.nn.Module)`：`__init__(self, members: dict[str, tuple[ActiveScope, tuple]], widths: list, strides: list, blocks_per_stage: int, stem_channels: int)`；属性 `protocol_ids: tuple[str, ...]`（排序后）；方法 `view(self, protocol_id: str) -> MemberModelV5`。
  - `build_unified_model_v5(plan: dict, scopes: dict[str, ActiveScope]) -> UnifiedModelV5`：读 `plan["shared"]["model"]`、`plan["shared"]["seeds"]["init"]` 和每个成员的 `tasks / patch_size / inference.window`。

结构约定：

- 参数名：`stems.<pid>.*`（`Conv3d(通道数, stem_channels, 1)`）、`encoder.*`（`EncoderV5(stem_channels, widths, strides, blocks_per_stage)`）、`members.<pid>.decoders.<task>.*`、`members.<pid>.heads.<task>.*`。
- 视图只把自己的解码器和输出头注册为子模块；共享的输入层和编码器用 `object.__setattr__` 存成普通属性，不注册。这样 `UnifiedModelV5.state_dict()` 里每个张量只出现一次。
- 协议号必须匹配 `[a-z][a-z0-9_]*`，否则 `PROTOCOL_INVALID`，`$.members`；`view` 遇到未知协议 → `PROTOCOL_UNKNOWN`，`$.protocol_id`。
- `build_unified_model_v5`：每个成员的 `patch_size` 和 `inference.window` 必须能被该轴步长的乘积整除，否则 `PATCH_NOT_DIVISIBLE`，`$.members.<pid>.patch_size` 或 `$.members.<pid>.inference.window`；初始化与 `build_model_v5` 相同的做法（`torch.random.fork_rng` 内以 `seeds.init` 播种），同一计划得到逐位相同的权重，调用者的随机数状态不变。

- [ ] **Step 1: 写失败测试** `tests/v5/test_unified_model.py`：

```python
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
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_unified_model.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 3: 实现 `unified_v5.py`**，按结构约定。
- [ ] **Step 4: 运行确认通过**，预期 4 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/models/unified_v5.py tests/v5/test_unified_model.py
git commit -m "Add the unified model: one shared encoder with per-protocol stems, decoders and heads"
```

---

### Task 5: 统一计划与运行登记

**Files:**
- Create: `anatobind/v41/engine/unified_registration_v5.py`
- Create: `configs/v5/proposals/unified_knee_brain.yaml`
- Modify: `tests/v5/factories.py`（新增 `make_unified_plan`、`make_unified_members`）
- Test: `tests/v5/test_unified_registration.py`

**Interfaces:**
- Consumes: `engine.registration_v5.{validate_recipe_v5, validate_supervision_v5, register_run_v5, write_run_v5, run_sha256_v5}`；`perception.validate_acceptance_policy_v5`；`evaluation.policy.validate_metric_policy_v5`；`engine.smoke_v5.RESUME_TOLERANCE_KEYS`；任务 3 的 `BOX_SPEC_KEYS`。
- Produces:
  - `PLAN_KEYS = ("plan_id", "shared", "members", "smoke", "budget")`
  - `SHARED_KEYS = ("model", "seeds", "optimizer", "scheduler", "precision", "selection_rule", "checkpoint_every", "budget")`
  - `MEMBER_KEYS = ("scope_id", "scope_hash", "source", "lineage_id", "tasks", "patch_size", "batch_size", "downsample_factor", "grid_tolerance", "normalization", "sampling", "inference", "loss_weight", "supervision", "box_supervision", "acceptance_policy", "metric_policy", "finding_policy")`
  - `SMOKE_KEYS = ("steps", "pause_at", "checkpoint_every", "subset_count", "subset_salt", "tolerances", "max_wall_hours")`
  - `validate_unified_plan_v5(plan: dict, scopes: dict[str, ActiveScope], sources: dict[str, dict]) -> dict`
  - `unified_plan_sha256_v5(plan: dict) -> str`（`canonical_sha256(plan)`）
  - `member_recipe_v5(plan: dict, protocol_id: str, run_kind: str) -> dict`（`run_kind` 为 `"baseline"` 或 `"smoke"`）
  - `register_unified_run_v5(*, run_id: str, run_kind: str, plan: dict, plan_sha256: str, member_runs: dict[str, dict], environment: dict, code: dict, created_at: str, fixture_mode: bool) -> dict`
  - `write_unified_run_v5(unified_run: dict, member_runs: dict, registries: dict, private_root: Path, repo: Path) -> Path`

规则：

- `member_recipe_v5` 返回一个能通过 `validate_recipe_v5` 的字典：`recipe_id = f"{plan_id}.{pid}"`（smoke 加后缀 `.smoke`）；`model` = 共享的 `architecture / widths / strides / blocks_per_stage` 加上成员的 `tasks`（不含 `stem_channels`）；`seeds / optimizer / scheduler / precision / selection_rule` 取共享；`patch_size / batch_size / downsample_factor / grid_tolerance / normalization / sampling / inference` 取成员；baseline：`readable_roles = ["train", "development_val"]`、`checkpoint_every` 与 `budget` 取共享；smoke：`readable_roles = ["train"]`、`checkpoint_every = smoke.checkpoint_every`、`budget = {"max_steps": smoke.steps, "max_wall_hours": smoke.max_wall_hours}`。
- `validate_unified_plan_v5` 的错误码与路径：键集合不符或类型不符 → `UNIFIED_PLAN_INVALID`，路径指到出错的键；成员集合与 `scopes`、`sources` 的键集合不同 → `UNIFIED_PLAN_INVALID`，`$.members`；`scope_id / scope_hash` 与 scope 不符 → `UNIFIED_SCOPE_MISMATCH`，`$.members.<pid>.scope_hash`；`source` 不符 → `UNIFIED_SOURCE_MISMATCH`，`$.members.<pid>.source`。每个成员还要满足：`tasks` 非空且是该 scope 可训练任务的子集，并等于 `supervision.tasks` 的键集合；`member_recipe_v5` 的两种结果都通过 `validate_recipe_v5`；`supervision` 通过 `validate_supervision_v5`；`acceptance_policy` 通过 `validate_acceptance_policy_v5`；`metric_policy` 在 `"development"` 阶段没有被阻断的指标；`loss_weight` 为正数；`box_supervision` 为 `None` 或键集合等于 `BOX_SPEC_KEYS`，非空时要求 `"U" in tasks` 且 `downsample_factor == [1, 1, 1]`；`finding_policy` 的键为 `("host", "iobb_thresholds")`，`host` 为 `"majority_accepted_anatomy"`（要求 `"A" in tasks`）或 `"none"`。`smoke`：`0 < pause_at < steps`，`pause_at` 与 `steps` 都是 `checkpoint_every` 的倍数，`subset_count` 的键等于成员集合且值为正整数，`tolerances` 的键等于 `RESUME_TOLERANCE_KEYS`。`budget = {"max_wall_hours": x}` 且 `x >= shared.budget.max_wall_hours`。
- `register_unified_run_v5`：`run_kind` 只能是 `"smoke"` 或 `"baseline"`；每个成员运行必须是 `register_run_v5` 的结果，且 `run_id == f"{run_id}.{pid}"`、`run_kind` 相同、`recipe_sha256 == canonical_sha256(member_recipe_v5(plan, pid, run_kind))`、`supervision_sha256 == canonical_sha256(成员 supervision)`、`scope` 与成员的 `scope_id / scope_hash` 一致；任一不符 → `UNIFIED_MEMBER_MISMATCH`，`$.member_runs.<pid>`。`plan_sha256` 与计划不符 → `UNIFIED_PLAN_HASH_MISMATCH`，`$.plan_sha256`。返回：

```python
{"schema_version": "v5.1", "run_id": run_id, "run_kind": f"unified_{run_kind}", "plan_id": plan["plan_id"],
 "plan_sha256": plan_sha256,
 "members": {pid: {"run_id": ..., "run_sha256": run_sha256_v5(member_run), "scope": {...}, "source": {...},
                   "split_sha256": member_run["split"]["split_sha256"]} for pid in sorted(members)},
 "code": dict(code), "environment": copy.deepcopy(environment), "fixture_mode": fixture_mode,
 "created_at": created_at}
```

- `write_unified_run_v5`：按协议号排序，对每个成员调用现有的 `write_run_v5(member_run, registries[pid], private_root, repo)`（这一步写出各自的曝光），然后把统一运行写到 `runs/<run_id>/run.json`；已存在则 `PrivatePathError("PRIVATE_FILE_EXISTS")`；返回统一运行的目录。

`configs/v5/proposals/unified_knee_brain.yaml`（提案，未批准；文件头注释写明这一点和"任何改动都会改变哈希"）：

```yaml
plan_id: unified_knee_brain_v1
shared:
  model:
    architecture: resunet3d
    widths: [32, 64, 128, 256, 320]
    strides: [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    blocks_per_stage: 2
    stem_channels: 16
  seeds: {init: 20261110, sampler: 20261111}
  optimizer: {name: adamw, lr: 0.001, weight_decay: 0.0001}
  scheduler: {name: poly, exponent: 0.9}
  precision: fp32
  selection_rule: last
  checkpoint_every: 2000
  budget: {max_steps: 30000, max_wall_hours: 10.0}
members:
  knee_qdess:
    scope_id: skm_joint_early
    scope_hash: f6e4b5f6e3c95a49f5f8cb45591b2a2c437b8db2d432551e1bd1e9e81935ec9f
    source: {source_id: skm_tea_joint_early, source_version: skm_tea_annotations_v1.0.0}
    lineage_id: v51_skm_joint_early
    tasks: [A, U]
    patch_size: [192, 192, 48]
    batch_size: 2
    downsample_factor: [1, 1, 1]
    grid_tolerance: {atol_affine: 0.001, atol_spacing: 0.001}
    normalization: zscore_volume
    sampling: {positive_fraction: 0.5}
    inference: {window: [192, 192, 48], stride: [96, 96, 24]}
    loss_weight: 1.0
    supervision:
      policy_id: skm_joint_early_supervision_v1
      scope_id: skm_joint_early
      reference_policy_id: skm_box_reference_v1
      track: released_reference
      allow_unknown_visibility: false
      channel: expert
      tasks:
        A: {kind: softmax, ce_weight: 1.0, dice_weight: 1.0, task_weight: 1.0}
        U: {kind: sigmoid, ce_weight: 1.0, dice_weight: 1.0, task_weight: 1.0}   # weights unused: box_supervision replaces the U loss
    box_supervision:
      kind: box_mil
      topk_fraction: 0.1
      positive_weight: 1.0
      negative_weight: 1.0
      box_policy_sha256: 09b1003e63fb5e8517be7626ba7e522dba9a7397a53dcd4ef910226bc734f2a9
    acceptance_policy:
      policy_id: skm_joint_early_engineering_v1
      policy_role: engineering
      anatomy: {min_confidence: 0.0}
      abnormality:
        thresholds: {cartilage_lesion: 0.5, meniscal_tear: 0.5}
        reject_band: {cartilage_lesion: none, meniscal_tear: none}
      instances: {connectivity: 26, min_voxels: 1}
    metric_policy: {metrics: [dice], surface_tolerance_mm: null, iou_match_threshold: null, fp_budget_per_scan: null, class_aware: null, size_strata_voxels: null}
    finding_policy: {host: majority_accepted_anatomy, iobb_thresholds: [0.1, 0.25, 0.5]}
  brain_dwi_adc:
    scope_id: isles_u
    scope_hash: 0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb
    source: {source_id: isles_2022, source_version: isles2022_bids1.6.0}
    lineage_id: v51_isles_u_b2
    tasks: [U]
    patch_size: [112, 112, 64]
    batch_size: 2
    downsample_factor: [1, 1, 1]
    grid_tolerance: {atol_affine: 0.001, atol_spacing: 0.001}
    normalization: zscore_volume
    sampling: {positive_fraction: 0.67}
    inference: {window: [112, 112, 64], stride: [56, 56, 32]}
    loss_weight: 1.0
    supervision:
      policy_id: isles_u_supervision_v1
      scope_id: isles_u
      reference_policy_id: isles_released_reference_v1
      track: released_reference
      allow_unknown_visibility: true
      channel: expert
      tasks:
        U: {kind: sigmoid, ce_weight: 1.0, dice_weight: 1.0, task_weight: 1.0}
    box_supervision: null
    acceptance_policy:
      policy_id: isles_u_engineering_v1
      policy_role: engineering
      anatomy: {min_confidence: null}
      abnormality:
        thresholds: {ischemic_infarct: 0.5}
        reject_band: {ischemic_infarct: none}
      instances: {connectivity: 26, min_voxels: 1}
    metric_policy: {metrics: [dice, hd95, nsd, lesion_detection], surface_tolerance_mm: 2.0, iou_match_threshold: 0.1, fp_budget_per_scan: null, class_aware: true, size_strata_voxels: [10, 100, 1000]}
    finding_policy: {host: none, iobb_thresholds: null}
smoke:
  steps: 40
  pause_at: 20
  checkpoint_every: 20
  subset_count: {knee_qdess: 2, brain_dwi_adc: 2}
  subset_salt: v51-smoke-unified-20261110
  tolerances: {loss_atol: 0.0001, loss_rtol: 0.001, param_rtol: 0.001}
  max_wall_hours: 2.0
budget: {max_wall_hours: 14.0}
```

脑成员的监督、接受策略和指标策略与 `baseline_isles_u.yaml` 逐字段相同，这样脑部指标可以与基线直接对照。

- [ ] **Step 1: 在 `tests/v5/factories.py` 末尾新增两个工厂**：

  - `make_unified_plan(scopes: dict, **overrides) -> dict`：对每个 scope 生成一个成员（`patch_size [4, 8, 8]`、`batch_size 1`、共享模型取任务 4 测试里的 `MODEL`、`shared.budget.max_steps 4`、`checkpoint_every 2`、smoke `steps 4 / pause_at 2 / checkpoint_every 2`），联合 scope 的成员带 `box_supervision`（`box_policy_sha256 = "p" * 64`）和 `host: majority_accepted_anatomy`，其余成员 `box_supervision: None`、`host: none`；`source` 用 `SYNTH_SOURCE`；监督用 `make_supervision(scope)`，接受策略用 `make_acceptance_policy(scope)`，指标策略用 `make_metric_policy()`。
  - `make_unified_members(plan, scopes, run_id, run_kind="baseline") -> tuple[dict, dict, dict]`：返回 `(member_runs, registries, records)`，三者都按协议号索引。每个成员用 `make_split_records(["train", "train", "development_val"])`（两个成员各自一套记录与登记，样本号可以相同）、`make_split_registry(records, lineage_id=成员的 lineage_id)`、`make_gate_decisions(scope, run_kind)`，smoke 时另给 `make_approvals(scope, ["resources"])`，再调用 `register_run_v5(run_id=f"{run_id}.{pid}", ...)`。
  - `make_unified_plan` 里成员的 `lineage_id` 为 `f"lin-{pid}"`；`overrides` 用现有的 `_merge` 深合并。

- [ ] **Step 2: 写失败测试** `tests/v5/test_unified_registration.py`，至少包含：

```python
def test_the_proposed_plan_file_is_valid_and_bound_to_the_real_scopes():
    plan = yaml.safe_load(Path("configs/v5/proposals/unified_knee_brain.yaml").read_text())
    scopes = {"knee_qdess": _scope("skm_joint_early"), "brain_dwi_adc": _scope("isles_u")}
    policies = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))
    sources = {"knee_qdess": source_binding_v5(policies["skm_tea_joint_early"]),
               "brain_dwi_adc": source_binding_v5(policies["isles_2022"])}
    assert validate_unified_plan_v5(plan, scopes, sources) is plan
    baseline = yaml.safe_load(Path("configs/v5/proposals/baseline_isles_u.yaml").read_text())
    brain = plan["members"]["brain_dwi_adc"]
    for key in ("supervision", "acceptance_policy", "metric_policy"):
        assert brain[key] == baseline[key]                     # brain metrics stay comparable with the baseline
    box_policy = load_skm_box_policy_v5(Path("configs/v5/skm_box_policy_v1.yaml"))
    assert plan["members"]["knee_qdess"]["box_supervision"]["box_policy_sha256"] == box_policy["policy_sha256"]
    build_unified_model_v5(plan, scopes)                       # patch and window divide the shared strides


def test_member_recipes_are_valid_recipes_of_both_kinds():
    plan = make_unified_plan(SCOPES)
    recipe = member_recipe_v5(plan, "knee", "baseline")
    assert validate_recipe_v5(recipe) is recipe and recipe["recipe_id"] == f"{plan['plan_id']}.knee"
    assert recipe["model"]["tasks"] == ["A", "U"] and "stem_channels" not in recipe["model"]
    smoke = member_recipe_v5(plan, "knee", "smoke")
    assert smoke["readable_roles"] == ["train"] and smoke["budget"]["max_steps"] == plan["smoke"]["steps"]


@pytest.mark.parametrize("edit,code,path", [
    (lambda p: p["members"]["knee"].update(scope_hash="0" * 64), "UNIFIED_SCOPE_MISMATCH", "$.members.knee.scope_hash"),
    (lambda p: p["members"]["knee"].update(source={"source_id": "x", "source_version": "y"}),
     "UNIFIED_SOURCE_MISMATCH", "$.members.knee.source"),
    (lambda p: p["members"]["brain"].update(box_supervision=dict(p["members"]["knee"]["box_supervision"]),
                                             downsample_factor=[2, 2, 1]),
     "UNIFIED_PLAN_INVALID", "$.members.brain.box_supervision"),
    (lambda p: p["members"]["brain"]["finding_policy"].update(host="majority_accepted_anatomy"),
     "UNIFIED_PLAN_INVALID", "$.members.brain.finding_policy.host"),
    (lambda p: p["smoke"].update(pause_at=3), "UNIFIED_PLAN_INVALID", "$.smoke.pause_at"),
    (lambda p: p.pop("budget"), "UNIFIED_PLAN_INVALID", "$")])
def test_invalid_plans_are_refused(edit, code, path): ...


def test_a_unified_run_binds_its_member_runs_and_writes_their_exposure(tmp_path):
    plan = make_unified_plan(SCOPES)
    member_runs, registries, _ = make_unified_members(plan, SCOPES, "run-u")
    run = register_unified_run_v5(run_id="run-u", run_kind="baseline", plan=plan,
                                  plan_sha256=unified_plan_sha256_v5(plan), member_runs=member_runs,
                                  environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE),
                                  created_at=SYNTH_CREATED_AT, fixture_mode=True)
    assert run["run_kind"] == "unified_baseline" and sorted(run["members"]) == ["brain", "knee"]
    assert run["members"]["knee"]["run_sha256"] == run_sha256_v5(member_runs["knee"])
    run_dir = write_unified_run_v5(run, member_runs, registries, tmp_path, REPO)
    assert json.loads((run_dir / "run.json").read_text()) == run
    for pid in ("knee", "brain"):
        exposure = json.loads((tmp_path / "runs" / f"run-u.{pid}" / "exposure.json").read_text())
        assert exposure["run_id"] == f"run-u.{pid}" and exposure["entries"]
    with pytest.raises(PrivatePathError):
        write_unified_run_v5(run, member_runs, registries, tmp_path, REPO)


def test_a_member_run_of_another_recipe_is_refused():
    plan = make_unified_plan(SCOPES)
    member_runs, _, _ = make_unified_members(plan, SCOPES, "run-u")
    other = make_unified_plan(SCOPES, shared={"optimizer": {"lr": 0.5}})
    _raises("UNIFIED_MEMBER_MISMATCH", "$.member_runs.brain", register_unified_run_v5, run_id="run-u",
            run_kind="baseline", plan=other, plan_sha256=unified_plan_sha256_v5(other), member_runs=member_runs,
            environment=dict(SYNTH_ENVIRONMENT), code=dict(SYNTH_CODE), created_at=SYNTH_CREATED_AT, fixture_mode=True)
```

- [ ] **Step 3: 运行确认失败**：`python3 -m pytest tests/v5/test_unified_registration.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 4: 写 `unified_knee_brain.yaml` 并实现 `unified_registration_v5.py`**。
- [ ] **Step 5: 运行确认通过**，预期 10 passed。
- [ ] **Step 6: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/engine/unified_registration_v5.py configs/v5/proposals/unified_knee_brain.yaml tests/v5/factories.py tests/v5/test_unified_registration.py
git commit -m "Add the unified plan, its member recipes and the unified run record"
```

---

### Task 6: 统一训练循环

**Files:**
- Create: `anatobind/v41/engine/train_unified_v5.py`
- Test: `tests/v5/test_unified_training.py`

**Interfaces:**
- Consumes: 任务 2 `box_positive_centers_v5`；任务 3 `box_mil_objective_v5`；任务 4 `UnifiedModelV5 / build_unified_model_v5`；任务 5 的统一运行与成员运行；`engine.train_v5.{build_optimizer_v5, build_scheduler_v5, build_scaler_v5, autocast_v5, save_checkpoint_v5, load_checkpoint_v5, _rng_state, _set_rng_state, _append_jsonl, _checkpoint_steps, _to_device, STOP_CODES}`；`engine.data_v5.{PatchSamplerV5, make_batch_v5, positive_centers_v5, training_data_sha256_v5}`；`losses.masked_objective_v5`。
- Produces:
  - `unified_loss_v5(output: ModelOutputV5, targets: dict, eligibility: dict, supervision: dict, box_supervision: dict | None) -> tuple[Tensor, dict]`：A 用 `masked_objective_v5`；U 在 `box_supervision` 非空时用 `box_mil_objective_v5`，否则用 `masked_objective_v5`；总损失是各任务 `task_weight · L` 之和；`parts[task] = {"loss": float, "support": int}`，框监督的 U 的 `support` 为示例数加负体素数。
  - `train_unified_v5(unified_run: dict, model: UnifiedModelV5, members: dict[str, dict], plan: dict, run_dir: Path, private_root: Path, repo: Path, resume: tuple | None, *, pause_at: int | None = None) -> dict`，返回 `{status, step, last_checkpoint, last_checkpoint_sha256, failure_code}`，取值与 `train_v5` 相同。
  - `members[pid] = {"run": 成员运行, "samples": dict[str, TrainingSampleV5], "recipe": 成员 recipe, "supervision": dict, "box_supervision": dict | None, "loss_weight": float}`。

循环（签名和测试不足以唯一确定）：

1. 启动前检查，顺序如下，任何一项失败都不写文件：
   - `run_dir/run.json` 存在且等于 `unified_run`，否则 `RUN_NOT_REGISTERED`，`$.run_dir`；
   - `set(members) == set(unified_run["members"]) == set(model.protocol_ids)`，否则 `UNIFIED_MEMBER_MISMATCH`，`$.members`；
   - 每个成员：`run_sha256_v5(members[pid]["run"])` 等于统一运行里登记的值，`canonical_sha256(recipe)` 等于成员运行的 `recipe_sha256`，否则 `RUN_MISMATCH`，`$.members.<pid>`；成员运行目录 `runs/<成员 run_id>/exposure.json` 里角色为 `training` 的样本号集合等于 `samples` 的键集合，否则 `SAMPLE_NOT_REGISTERED`，`$.members.<pid>.samples`；
   - 所有成员 recipe 的 `budget` 与 `checkpoint_every` 相同，否则 `UNIFIED_MEMBER_MISMATCH`，`$.members`；
   - 全新启动时模型权重逐位等于 `build_unified_model_v5(plan, 各视图的 scope)`，否则 `MODEL_INIT_MISMATCH`，`$.model`；
   - `metrics.jsonl / failure.json / checkpoints / data.json` 的存在性规则、`pause_at` 的合法性、续训的三项检查，与 `train_v5` 相同（同样的错误码）。
2. 每个成员一个 `PatchSamplerV5`，种子为 `seeds.sampler + i`，i 是协议号排序后的序号。正中心：有 `box_supervision` 的成员用任务 2 的 `box_positive_centers_v5`，其余用 `positive_centers_v5`。优化器、调度器、缩放器各一个，作用于整个统一模型。
3. 每一步：`optimizer.zero_grad`；按协议号排序，对每个成员取一个批次（`make_batch_v5(samples, sampler, recipe, view.scope)`），在 `autocast_v5` 下前向，检查 logit 有限，算 `unified_loss_v5`，乘以 `loss_weight` 后 `scaler.scale(...).backward()`；全部成员完成后 `scaler.step(optimizer)`、`scaler.update()`、`scheduler.step()` 各一次。
4. `metrics.jsonl` 每步一行：`{"event": "step", "step", "loss"（加权和）, "lr", "members": {pid: {"loss": 未加权, "tasks": parts}}}`。
5. 检查点沿用 `save_checkpoint_v5(path, state, unified_run, ...)`：`state["sampler"] = {pid: sampler.state_dict()}`，`state["data_sha256"] = canonical_sha256({pid: training_data_sha256_v5(samples)})`，其余键同 `train_v5`。
6. 非有限 logit 或损失、显存不足、超出墙钟预算：写 `failure.json`，返回 `FAILED`，不重试。

- [ ] **Step 1: 写失败测试** `tests/v5/test_unified_training.py`。公共夹具：`SCOPES = {"knee": 联合合成 scope, "brain": U 模块合成 scope}`；`plan = make_unified_plan(SCOPES)`；`member_runs, registries, records = make_unified_members(plan, SCOPES, "run-u")`；注册并 `write_unified_run_v5`；样本用 `make_training_sample_v5(scope, uid, shape=(4, 8, 8), seed=i)`，膝成员的 `targets["U"]` 换成 int32 的示例编号图（把随机布尔图乘 1 即可），至少一个膝样本的 `targets["U"]` 全零（没有主分析框的扫描）。测试：

```python
def test_training_completes_and_logs_every_member(fixture):
    result = train_unified_v5(fixture.run, fixture.model(), fixture.members, fixture.plan, fixture.run_dir,
                              fixture.root, REPO, None)
    assert (result["status"], result["step"], result["failure_code"]) == ("COMPLETED", 4, None)
    lines = [json.loads(x) for x in (fixture.run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [x for x in lines if x["event"] == "step"]
    assert [x["step"] for x in steps] == [1, 2, 3, 4]
    assert set(steps[0]["members"]) == {"brain", "knee"} and set(steps[0]["members"]["knee"]["tasks"]) == {"A", "U"}
    assert sorted(p.name for p in (fixture.run_dir / "checkpoints").glob("*.pt")) == ["step_00000002.pt", "step_00000004.pt"]


def test_pause_and_resume_equals_the_continuous_run_bitwise_on_cpu(fixture, second_fixture):
    full = train_unified_v5(..., None)
    paused = train_unified_v5(..., None, pause_at=2)                       # second registered run directory
    assert paused["status"] == "PAUSED"
    resumed = train_unified_v5(..., (paused["last_checkpoint"], paused["last_checkpoint_sha256"]))
    a = load_checkpoint_v5(full["last_checkpoint"], run_a, full["last_checkpoint_sha256"], "cpu")
    b = load_checkpoint_v5(resumed["last_checkpoint"], run_b, resumed["last_checkpoint_sha256"], "cpu")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]) and a["sampler"] == b["sampler"]


def test_binding_and_init_are_checked_before_anything_is_written(fixture):
    members = copy.deepcopy(fixture.members)
    members["knee"]["samples"]["other"] = make_training_sample_v5(SCOPES["knee"], "other", shape=(4, 8, 8))
    _raises("SAMPLE_NOT_REGISTERED", "$.members.knee.samples", train_unified_v5, fixture.run, fixture.model(),
            members, fixture.plan, fixture.run_dir, fixture.root, REPO, None)
    model = fixture.model()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    _raises("MODEL_INIT_MISMATCH", "$.model", train_unified_v5, fixture.run, model, fixture.members, fixture.plan,
            fixture.run_dir, fixture.root, REPO, None)
    assert not (fixture.run_dir / "data.json").exists()


def test_a_non_finite_loss_stops_the_run_with_a_failure_record(fixture, monkeypatch):
    monkeypatch.setattr(train_unified_module, "box_mil_objective_v5",
                        lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    result = train_unified_v5(fixture.run, fixture.model(), fixture.members, fixture.plan, fixture.run_dir,
                              fixture.root, REPO, None)
    assert (result["status"], result["failure_code"]) == ("FAILED", "NON_FINITE_LOSS")
    assert json.loads((fixture.run_dir / "failure.json").read_text())["code"] == "NON_FINITE_LOSS"


def test_unified_loss_uses_the_box_objective_only_where_box_supervision_is_given():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 2, 2, 2)
    bags = torch.zeros(1, 1, 2, 2, 2, dtype=torch.int32)
    bags[0, 0, 0, 0, 0] = 2
    eligibility = {"U": {"e_pos": bags > 0, "e_neg": bags == 0}}
    supervision = {"tasks": {"U": {"kind": "sigmoid", "ce_weight": 1.0, "dice_weight": 1.0, "task_weight": 3.0}}}
    output = ModelOutputV5(anatomy_logits=None, abnormality_logits=logits)
    boxed, parts = unified_loss_v5(output, {"U": bags}, eligibility, supervision, BOX_SPEC)
    assert float(boxed) == pytest.approx(3.0 * float(box_mil_objective_v5(logits, bags, bags > 0, bags == 0, BOX_SPEC)))
    assert parts["U"]["support"] == 1 + 7                                  # one bag, seven negative voxels
    plain, _ = unified_loss_v5(output, {"U": bags > 0}, eligibility, supervision, None)
    assert float(plain) == pytest.approx(3.0 * float(masked_objective_v5(logits, bags > 0, bags > 0, bags == 0,
                                                                         supervision["tasks"]["U"])))
```

  `BOX_SPEC` 取任务 3 测试里的 `SPEC`。

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_unified_training.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 3: 实现 `train_unified_v5.py`**。
- [ ] **Step 4: 运行确认通过**，预期 5 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/engine/train_unified_v5.py tests/v5/test_unified_training.py
git commit -m "Add the unified training loop: one batch per protocol each step, one optimizer step"
```

---

### Task 7: 协议表与所见输出

**Files:**
- Create: `configs/v5/protocols_v5.yaml`
- Create: `anatobind/v41/findings_v5.py`
- Test: `tests/v5/test_findings.py`

**Interfaces:**
- Consumes: 现有输出链的结果：联合 scope 的 `PerceptionOutput`（`abnormal_instances[*].{instance_uid, class_name, mask, bbox}`、`tissue_relations[*].{instance_uid, tissue_ids, status}`），模块 scope 的 `ModuleOutput`（`abnormality_map`、`acceptance["abnormality"]`）；`perception.extract_instances_v5`；`annotation_v5.load_skm_box_policy_v5`。
- Produces:
  - `load_protocols_v5(path: Path, config_dir: Path) -> dict[str, dict]`：每个协议 `{"scope": ActiveScope, "source_id": str, "description": str, "groups": dict[str, list[str]] | None, "groups_by_class": dict[str, list[str]] | None, "display": dict[str, str]}`。`groups` 与 `groups_by_class` 由该协议的 `box_policy` 文件导出（`host_tissues[*].{group, anatomy}` 与 `tissue_groups_by_class`），没有 `box_policy` 时为 `None`。
  - `host_relation_fn_v5(groups: dict, groups_by_class: dict)`：返回可交给 `run_perception_v5(relation_fn=...)` 的函数。
  - `findings_from_output_v5(output, scope: ActiveScope, acceptance_policy: dict, groups: dict | None) -> list[dict]`
  - `render_findings_zh_v5(findings: list[dict], scope: ActiveScope, display: dict) -> list[str]`
  - 一条所见：`{"finding_uid": str, "abnormality_class": str, "box": {"start": [x, y, z], "size": [dx, dy, dz]}, "anchor": [x, y, z], "voxels": int, "host_group": str | None, "host_status": "bound" | "undetermined" | "not_reported"}`。

规则：

- `host_relation_fn_v5`：对每个实例，取实例掩膜内被接受的预测组织体素；只看该实例类别在 `groups_by_class` 里允许的组织组；体素最多的组胜出（并列取组名字典序最小的）。结果：`tissue_ids` 为胜出组里在实例内实际出现的 A 类编号（升序），`status = "bound"`；没有任何允许组的体素时 `tissue_ids = []`、`status = "unknown"`。`predicate` 与 `completion_role` 取 `scope.relation_contract`，`accepted = True`，原因码 0。
- `findings_from_output_v5`：联合输出用它自带的实例与关系；模块输出先用 `extract_instances_v5(abnormality_map, acceptance["abnormality"], scope, acceptance_policy)` 得到实例。`box.start` 为实例外接框的起点，`size` 为各轴长度；`anchor` 为实例体素中离质心欧氏距离最近的一个（并列取 (x, y, z) 字典序最小的）；`host_group`：`groups` 为 `None` 时 `None` 且 `host_status = "not_reported"`；否则关系 `status == "bound"` 时取包含其 `tissue_ids` 的组、`host_status = "bound"`，其余情况 `None`、`"undetermined"`。所见按 `finding_uid`（即实例号）排序。
- `render_findings_zh_v5` 的固定句式：
  - `bound`：`在{display[组]}上，疑似{display[类别]}`
  - `undetermined`：`疑似{display[类别]}，所在结构未能确定`
  - `not_reported`：`疑似{display[类别]}`
  - 没有所见时返回一句：`未见所列异常（仅限：{各类别的显示名，顿号分隔}）。医生没有标注不等于正常。`
- `display` 缺少需要的名字 → `PROTOCOL_DISPLAY_MISSING`，`$.display`。

`configs/v5/protocols_v5.yaml`：

```yaml
# Scan protocols the unified model accepts (design 2026-11-10 §5.1). The protocol picks the model's stem,
# decoders and heads; it carries no private path. display gives the wording of a finding.
knee_qdess:
  scope: scope_skm_joint_early.yaml
  source_id: skm_tea_joint_early
  description: 膝关节 qDESS 双回波
  box_policy: skm_box_policy_v1.yaml
  display:
    cartilage_lesion: 软骨病变
    meniscal_tear: 半月板撕裂
    femoral_cartilage: 股骨软骨
    patellar_cartilage: 髌软骨
    tibial_cartilage: 胫骨软骨
    meniscus: 半月板
brain_dwi_adc:
  scope: scope_isles_u.yaml
  source_id: isles_2022
  description: 脑 DWI 与 ADC
  box_policy: null
  display:
    ischemic_infarct: 缺血性梗死
```

- [ ] **Step 1: 写失败测试** `tests/v5/test_findings.py`：

```python
def test_protocol_table_resolves_scopes_groups_and_wording():
    protocols = load_protocols_v5(Path("configs/v5/protocols_v5.yaml"), Path("configs/v5"))
    knee, brain = protocols["knee_qdess"], protocols["brain_dwi_adc"]
    assert knee["scope"].scope_id == "skm_joint_early" and brain["scope"].scope_id == "isles_u"
    assert knee["groups"]["tibial_cartilage"] == ["tibial_cartilage_medial", "tibial_cartilage_lateral"]
    assert knee["groups_by_class"] == {"cartilage_lesion": ["femoral_cartilage", "patellar_cartilage", "tibial_cartilage"],
                                       "meniscal_tear": ["meniscus"]}
    assert brain["groups"] is None and brain["display"] == {"ischemic_infarct": "缺血性梗死"}


def test_host_is_the_allowed_group_with_most_accepted_voxels():
    # SCOPE = real skm_joint_early; anatomy_map: femoral_cartilage (id 2) on x < 4, meniscus_medial (id 5) elsewhere
    # one cartilage_lesion instance covering 6 femoral voxels and 10 meniscus voxels
    relations, reasons = host_relation_fn_v5(GROUPS, GROUPS_BY_CLASS)([instance], anatomy_map, accepted, SCOPE)
    assert relations[0]["status"] == "bound" and relations[0]["tissue_ids"] == [2]     # meniscus is not allowed for cartilage
    assert relations[0]["predicate"] == "located_in_tissue_group" and reasons == {"inst-0001": 0}
    relations, _ = host_relation_fn_v5(GROUPS, GROUPS_BY_CLASS)([instance], anatomy_map, np.zeros_like(accepted), SCOPE)
    assert relations[0]["status"] == "unknown" and relations[0]["tissue_ids"] == []


def test_a_finding_carries_box_anchor_host_and_a_fixed_sentence():
    # joint output built with tests.v5.factories.make_joint_output on the real scope: one cartilage_lesion instance
    # on voxels x 1..2, y 1..3, z 0 (6 voxels), bound to femoral_cartilage
    findings = findings_from_output_v5(output, SCOPE, POLICY, GROUPS)
    assert findings == [{"finding_uid": "inst-0001", "abnormality_class": "cartilage_lesion",
                         "box": {"start": [1, 1, 0], "size": [2, 3, 1]}, "anchor": [1, 2, 0], "voxels": 6,
                         "host_group": "femoral_cartilage", "host_status": "bound"}]
    assert render_findings_zh_v5(findings, SCOPE, DISPLAY) == ["在股骨软骨上，疑似软骨病变"]


def test_undetermined_host_and_no_findings_have_their_own_wording():
    undetermined = [dict(FINDING, host_group=None, host_status="undetermined")]
    assert render_findings_zh_v5(undetermined, SCOPE, DISPLAY) == ["疑似软骨病变，所在结构未能确定"]
    assert render_findings_zh_v5([], SCOPE, DISPLAY) == [
        "未见所列异常（仅限：软骨病变、半月板撕裂）。医生没有标注不等于正常。"]


def test_a_module_output_gives_findings_without_a_host():
    # ModuleOutput for isles_u (make_module_output) with one 8-voxel accepted component
    findings = findings_from_output_v5(output, BRAIN_SCOPE, BRAIN_POLICY, None)
    assert [(f["host_group"], f["host_status"], f["voxels"]) for f in findings] == [(None, "not_reported", 8)]
    assert render_findings_zh_v5(findings, BRAIN_SCOPE, {"ischemic_infarct": "缺血性梗死"}) == ["疑似缺血性梗死"]
```

  质心定位点的取值在测试里要可手算：上面 6 个体素的质心是 (1.5, 2, 0)，离它最近的体素有 (1, 2, 0) 和 (2, 2, 0)，取字典序小的 (1, 2, 0)。

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_findings.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 3: 写 `protocols_v5.yaml`，实现 `findings_v5.py`**。
- [ ] **Step 4: 运行确认通过**，预期 5 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add configs/v5/protocols_v5.yaml anatobind/v41/findings_v5.py tests/v5/test_findings.py
git commit -m "Add the protocol table and findings: box, anchor, host tissue group and fixed wording"
```

---

### Task 8: 所见评价

**Files:**
- Create: `anatobind/v41/evaluation/findings_v5.py`
- Test: `tests/v5/test_findings_eval.py`

**Interfaces:**
- Consumes: 任务 7 的所见字典；框参照的一行（任务 2 的字段）；`evaluation.segmentation` 中 `module_metrics_v5` 所用的 Dice 计算。
- Produces:
  - `score_findings_v5(findings: list[dict], reference: dict, patient_uid: str, classes: list[str], iobb_thresholds: list[float]) -> dict`（一次扫描一行）
  - `summarize_findings_v5(rows: list[dict]) -> dict`
  - `render_findings_eval_md_v5(summary: dict) -> str`（只含计数与指标）
  - `anatomy_dice_v5(anatomy_map: np.ndarray, truth: np.ndarray, eligible: np.ndarray, classes: list[str]) -> dict[str, float | None]`

规则：

- 参照框只取 `status == "included"` 且 `analysis == "primary"` 的；其余框是忽略框。
- 对每个参照框 r（类别 c，组织组 g）：候选 = 类别为 c 且 `anchor` 落在 r 内的所见。`hit = 有候选`。选中的所见 = 候选里 `voxels` 最大的（并列取 `finding_uid` 小的）。`host_correct = hit 且 选中所见的 host_group == g`。`iobb` = 候选中 `|所见框 ∩ r| / |所见框|` 的最大值；没有候选时为 `None`。
- 一行：`{"patient_uid", "boxes": [{"abnormality_class", "tissue_group", "hit", "host_correct", "iobb"}], "findings": {"total", "in_reference", "in_ignored", "outside"}, "tissue_positive": [{"abnormality_class", "tissue_group", "predicted": bool}]}`。
  - `in_reference`：定位点落在同类别的某个参照框内；否则若落在任一忽略框内记 `in_ignored`；否则 `outside`。
  - `tissue_positive`：扫描中每个至少有一个参照框的 (类别, 组织组) 一条；`predicted` = 存在该类别且 `host_group` 等于该组的所见。
- 汇总，按类别和合计分别给出，全部固定分母（未命中计入分母）：
  - `box_sensitivity`：先在患者内算命中比例，再对有参照框的患者等权平均；同时给出 `pooled`（命中数 / 框数）。
  - `host_accuracy`：归属正确数 / 全部参照框数，同样给患者等权与 pooled。
  - `iobb_at[τ]`：`iobb >= τ` 的参照框比例（`None` 记不达标）。
  - `tissue_positive_recall`：`predicted` 为真的条数 / 总条数。
  - `findings_outside_reference_per_scan`：`outside` 的每扫描均值，字段说明固定为"描述性计数，不是误报"。
  - 支持数：`n_patients`、`n_patients_with_boxes`、`n_boxes`。分母为 0 的指标为 `None`。
- `anatomy_dice_v5`：每类在 `eligible` 内的 Dice；参照与预测都为空时为 `None`（与模块评价相同的空值规则）。

- [ ] **Step 1: 写失败测试** `tests/v5/test_findings_eval.py`：

```python
REFERENCE = {"sample_uid": "s-0", "policy_sha256": "p" * 64, "boxes": [
    _box("included", "primary", "cartilage_lesion", "femoral_cartilage", [0, 0, 0], [4, 4, 2]),      # r0
    _box("included", "primary", "meniscal_tear", "meniscus", [10, 10, 0], [2, 2, 2]),                # r1: missed
    _box("included", "secondary", "cartilage_lesion", "patellar_cartilage", [20, 0, 0], [2, 2, 2])]}  # ignored
FINDINGS = [
    _finding("inst-0001", "cartilage_lesion", [1, 1, 0], [2, 2, 1], anchor=[1, 1, 0], voxels=4, host="femoral_cartilage"),
    _finding("inst-0002", "cartilage_lesion", [2, 2, 0], [6, 6, 1], anchor=[3, 3, 0], voxels=9, host="tibial_cartilage"),
    _finding("inst-0003", "cartilage_lesion", [20, 0, 0], [1, 1, 1], anchor=[20, 0, 0], voxels=1, host=None),
    _finding("inst-0004", "meniscal_tear", [30, 30, 0], [1, 1, 1], anchor=[30, 30, 0], voxels=1, host="meniscus")]


def test_one_scan_is_scored_against_the_doctors_primary_boxes():
    row = score_findings_v5(FINDINGS, REFERENCE, "synthetic:p0", ["cartilage_lesion", "meniscal_tear"], [0.1, 0.25, 0.5])
    hit, missed = row["boxes"]
    assert (hit["hit"], hit["host_correct"]) == (True, False)          # the larger candidate names the wrong tissue
    assert hit["iobb"] == pytest.approx(1.0)                           # inst-0001 lies wholly inside r0
    assert (missed["hit"], missed["host_correct"], missed["iobb"]) == (False, False, None)
    assert row["findings"] == {"total": 4, "in_reference": 2, "in_ignored": 1, "outside": 1}
    assert row["tissue_positive"] == [
        {"abnormality_class": "cartilage_lesion", "tissue_group": "femoral_cartilage", "predicted": True},
        {"abnormality_class": "meniscal_tear", "tissue_group": "meniscus", "predicted": True}]


def test_summary_keeps_missed_boxes_in_every_denominator_and_weights_patients_equally():
    second = score_findings_v5([], {"sample_uid": "s-1", "policy_sha256": "p" * 64, "boxes": [REFERENCE["boxes"][0]]},
                               "synthetic:p1", ["cartilage_lesion", "meniscal_tear"], [0.1, 0.25, 0.5])
    third = score_findings_v5([], {"sample_uid": "s-2", "policy_sha256": "p" * 64, "boxes": []}, "synthetic:p2",
                              ["cartilage_lesion", "meniscal_tear"], [0.1, 0.25, 0.5])
    summary = summarize_findings_v5([score_findings_v5(FINDINGS, REFERENCE, "synthetic:p0",
                                                       ["cartilage_lesion", "meniscal_tear"], [0.1, 0.25, 0.5]),
                                     second, third])
    total = summary["overall"]
    assert (total["n_patients"], total["n_patients_with_boxes"], total["n_boxes"]) == (3, 2, 3)
    assert total["box_sensitivity"] == {"patient_mean": pytest.approx(0.25), "pooled": pytest.approx(1 / 3)}
    assert total["host_accuracy"]["pooled"] == 0.0
    assert total["iobb_at"] == {"0.1": pytest.approx(1 / 3), "0.25": pytest.approx(1 / 3), "0.5": pytest.approx(1 / 3)}
    assert total["findings_outside_reference_per_scan"] == pytest.approx(1 / 3)
    assert summary["by_class"]["meniscal_tear"]["box_sensitivity"]["pooled"] == 0.0


def test_no_reference_boxes_gives_none_not_zero_or_one():
    summary = summarize_findings_v5([score_findings_v5([], {"sample_uid": "s", "policy_sha256": "p" * 64, "boxes": []},
                                                       "synthetic:p", ["cartilage_lesion"], [0.5])])
    assert summary["overall"]["box_sensitivity"] == {"patient_mean": None, "pooled": None}


def test_report_has_counts_and_metrics_only_and_does_not_call_extra_findings_false_positives():
    summary = summarize_findings_v5([score_findings_v5(FINDINGS, REFERENCE, "synthetic:p0",
                                                       ["cartilage_lesion", "meniscal_tear"], [0.1, 0.25, 0.5])])
    text = render_findings_eval_md_v5(summary)
    assert "描述性计数，不是误报" in text and "NOT_EVALUATED" in text
    assert "synthetic:p0" not in text and "inst-0001" not in text


def test_anatomy_dice_matches_a_hand_value_and_is_none_when_both_are_empty():
    # truth class 1 on 4 voxels, prediction on 2 of them plus 2 others: Dice = 2*2 / (4+4) = 0.5; class 2 empty in both
    assert anatomy_dice_v5(pred, truth, np.ones(SHAPE, bool), ["tissue_a", "tissue_b"]) == {"tissue_a": 0.5, "tissue_b": None}
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_findings_eval.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 3: 实现 `evaluation/findings_v5.py`**。
- [ ] **Step 4: 运行确认通过**，预期 5 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/evaluation/findings_v5.py tests/v5/test_findings_eval.py
git commit -m "Add findings evaluation against the released boxes with fixed denominators"
```

---

### Task 9: 统一驱动

**Files:**
- Create: `anatobind/v41/real_unified_v5.py`
- Test: `tests/v5/test_real_unified.py`

**Interfaces:**
- Consumes: 任务 2–8 的全部产出；`real_t10_v5.{training_smoke_evidence_v5, gbase_evidence_v5}`；`gates_v5.evaluate_gate_v5`；`engine.registration_v5.{register_run_v5, select_samples_v5, select_subset_v5}`；`engine.smoke_v5.compare_resume_v5`；`perception.infer_and_save_v5`；`engine.module_eval_v5.{freeze_module_reference_v5, module_metrics_v5, summarize_module_metrics_v5}`；`real_baseline_v5._reference`。
- Produces:
  - `check_unified_approval_v5(approval: object, plan_sha256: str, scope: ActiveScope, fixture_mode: bool) -> dict`：与 `check_baseline_approval_v5` 相同的检查，但比较 `approval["unified_plan_sha256"]`，不符 → `UNIFIED_PLAN_NOT_APPROVED`，`$.approvals.resources.unified_plan_sha256`。
  - `load_member_samples_v5(*, plan, protocol_id, scope, records, registry, raw, label_maps, roles, box_references, box_policy, subset=None) -> dict[str, TrainingSampleV5]`：有 `box_supervision` 的成员用 `load_box_training_sample_v5`，否则用 `load_training_sample_v5`；样本只取冻结登记中 `roles` 的。
  - `run_unified_smoke_v5(**kw) -> dict`、`run_unified_training_v5(**kw) -> dict`、`render_unified_md_v5(summary: dict) -> str`、`main(argv) -> int`。
  - 命令行子命令：`plan-hash`、`smoke`、`gbase-gate`、`train`。

行为：

- **`smoke`**：每个成员登记一个 `smoke` 类成员运行（样本为 `select_subset_v5(records, registry, count=smoke.subset_count[pid], salt=smoke.subset_salt, strata=None)`；需要 G-DATA 的 PASS 决定和按 scope 绑定、写明 `unified_plan_sha256` 的 `resources` 批准）。做两次执行：连续执行到 `smoke.steps`；另一个运行号在 `pause_at` 暂停后续训。用 `compare_resume_v5` 比较两次的最终检查点与日志。再用最终检查点对每个成员的一个训练样本做一次整卷推理并生成所见。汇总 `status` 为 `PASS` 的条件：两次执行都 `COMPLETED`、续训比较 `passed`、每个成员的推理输出通过输出校验。写 `summary.json` 和 `unified_smoke.md`；每个成员的 `training_smoke` 证据用 `training_smoke_evidence_v5(summary, markdown, scope, source)` 生成，写入 `smoke_evidence.<pid>.json`。
- **`gbase-gate`**：对一个成员，用 `gbase_evidence_v5(scope=..., source=..., plan={"scope_hash": ..., "recipe": member_recipe_v5(plan, pid, "baseline"), "supervision": 成员 supervision}, gdata_decision=..., split_markdown=..., development_groups=..., smoke_evidence=...)` 生成五项证据，再对该成员的每个任务评估 `G-BASE-A` / `G-BASE-U`，批准从批准文件的 `<scope_id>.G-BASE-<task>.resources` 读取。输出 `gbase_decision.<task>.json`。
- **`train`**：
  1. 读数据之前：计划哈希、每个成员的批准（`check_unified_approval_v5`）、冒烟证据与计划和 scope 的绑定；任何一项失败都不读数据。
  2. 登记成员运行（`baseline` 类，`gate_decisions` 为该成员各任务的 G-BASE 决定）与统一运行，写出曝光。
  3. 膝成员的框参照用 `load_box_references_v5(path, registry, ["train"])` 读训练部分；评价阶段再用 `["development_val"]` 读开发部分。两次读取分开，训练阶段不持有开发集的框。
  4. 训练（`train_unified_v5`），按哈希重新加载最终检查点。
  5. 开发集评价，每个成员每个 `development_val` 扫描一次整卷推理（`infer_and_save_v5`，膝成员传 `relation_fn=host_relation_fn_v5(...)`）：脑成员走现有模块评价；膝成员算 `anatomy_dice_v5`、`findings_from_output_v5`、`score_findings_v5`。参照在任何预测之前冻结。
  6. 汇总并写 `summary.json` 与 `unified_report.md`。
  7. 训练失败、超时、显存不足或超出 `plan.budget.max_wall_hours`：停在当前阶段并记录，不重试，不出部分结果（与 `run_baseline_v5` 相同）。
- **`render_unified_md_v5`**：只含配置、计数、指标、哈希。必须包含：每个成员的运行哈希与划分哈希；脑成员的模块指标表；膝成员的每类 Dice、所见评价表、每条所见句式的条数（按句式种类计数，不列逐例内容）；与基线的并列对照（基线数值从 `--baseline-summary <pid>=<路径>` 给出的基线 `summary.json` 读取，未给出时该列写"未提供"）；固定说明"读 train 与 development_val；locked_test 未读；`scientific_conclusion: NOT_EVALUATED`；框外预测为描述性计数，不是误报；医生没有标注不等于正常"。

- [ ] **Step 1: 写失败测试** `tests/v5/test_real_unified.py`。合成数据用 `tests.v5.factories.write_volume_case` 在 `tmp_path` 写两个 scope 各 3 个病例（2 个 train、1 个 development_val，形状 `(8, 8, 8)`），膝成员另写一份 `box_references.jsonl`（其中一个训练病例没有主分析框）。计划用 `make_unified_plan`（`patch_size [4, 8, 8]`）。测试：

```python
def test_smoke_passes_and_gives_evidence_bound_to_each_member(env):
    summary = run_unified_smoke_v5(**env.smoke_kwargs)
    assert summary["status"] == "PASS" and summary["resume"]["passed"] and summary["resume"]["bitwise_parameters"]
    for pid, scope in env.scopes.items():
        evidence = env.smoke_evidence(summary, pid)
        assert evidence["status"] == "PASS" and evidence["scope_hash"] == scope.scope_hash


def test_training_runs_end_to_end_and_reports_both_members(env):
    summary = run_unified_training_v5(**env.train_kwargs)
    assert summary["status"] == "COMPLETED" and summary["training"]["step"] == 4
    assert set(summary["members"]) == {"brain", "knee"}
    assert summary["members"]["brain"]["module"] is not None
    knee = summary["members"]["knee"]
    assert set(knee) >= {"anatomy_dice", "findings", "sentences"} and knee["findings"]["overall"]["n_patients"] == 1
    text = render_unified_md_v5(summary)
    assert "NOT_EVALUATED" in text and "不是误报" in text and "医生没有标注不等于正常" in text
    assert str(env.raw) not in text and "s-0" not in text


def test_nothing_is_read_before_the_plan_is_approved(env, monkeypatch):
    monkeypatch.setattr(real_unified_module, "load_member_samples_v5", _fail_if_called)
    approvals = copy.deepcopy(env.approvals)
    approvals["knee"]["unified_plan_sha256"] = "0" * 64
    _raises("UNIFIED_PLAN_NOT_APPROVED", "$.approvals.resources.unified_plan_sha256", run_unified_training_v5,
            **dict(env.train_kwargs, approvals=approvals))


def test_training_never_holds_development_boxes(env, monkeypatch):
    seen = []
    original = real_unified_module.load_box_references_v5
    monkeypatch.setattr(real_unified_module, "load_box_references_v5",
                        lambda path, registry, roles: seen.append(list(roles)) or original(path, registry, roles))
    run_unified_training_v5(**env.train_kwargs)
    assert seen == [["train"], ["development_val"]]


def test_plan_hash_command_prints_the_hash(tmp_path, capsys):
    assert main(["plan-hash", "--plan", "configs/v5/proposals/unified_knee_brain.yaml", "--protocols",
                 "configs/v5/protocols_v5.yaml", "--policies", "configs/v5/sources_v5.yaml"]) == 0
    assert re.fullmatch(r"[0-9a-f]{64}\n", capsys.readouterr().out)
```

- [ ] **Step 2: 运行确认失败**：`python3 -m pytest tests/v5/test_real_unified.py -q -p no:cacheprovider`，预期 `ModuleNotFoundError`。
- [ ] **Step 3: 实现 `real_unified_v5.py`**。命令行参数：四个子命令都要 `--plan --protocols --policies`；`smoke` 与 `train` 另要 `--raw --approvals --run-id --device --out`，以及可重复的 `--member <pid>:records=<路径>,split-registry=<路径>,gdata-decision=<路径>[,box-references=<路径>][,gbase-a=<路径>][,gbase-u=<路径>][,smoke-evidence=<路径>]`；`train` 另有可重复的 `--baseline-summary <pid>=<路径>`；`gbase-gate` 要 `--protocol --gdata-decision --split-summary --smoke-evidence --approvals --out`。`--out` 必须是不存在的目录，以 700 创建。
- [ ] **Step 4: 运行确认通过**，预期 5 passed。
- [ ] **Step 5: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add anatobind/v41/real_unified_v5.py tests/v5/test_real_unified.py
git commit -m "Add the unified driver: smoke, G-BASE evidence, training and the development report"
```

---

### Task 10: 状态与文档收尾

**Files:**
- Modify: `reports/v5/status.md`（"2026-11-10 目标修订"一节补充实现情况）
- Modify: `reports/v5/traceability.md`
- Create: `reports/v5/unified_local_report_20261110.md`
- Test: `tests/v5/test_status_docs.py`

**Interfaces:**
- Consumes: 任务 1–9 的提交号与测试数。
- Produces: 入库的状态记录。

- [ ] **Step 1: 写失败测试**，加到 `tests/v5/test_status_docs.py`：

```python
def test_unified_local_report_separates_code_checks_from_real_runs():
    text = Path("reports/v5/unified_local_report_20261110.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    for phrase in ("只做了合成数据验证", "没有在 amax 上运行", "NOT_EVALUATED", "提案，未批准",
                   "unified_knee_brain.yaml", "定位点用质心体素"):
        assert phrase in text
    for name in ("test_box_data.py", "test_box_loss.py", "test_unified_model.py", "test_unified_registration.py",
                 "test_unified_training.py", "test_findings.py", "test_findings_eval.py", "test_real_unified.py"):
        assert Path(f"tests/v5/{name}").exists() and name in text
```

- [ ] **Step 2: 运行确认失败**，预期找不到报告文件。
- [ ] **Step 3: 写报告**：交付的文件与提交号、每个测试文件的通过数、执行方所做的选择清单（设计稿里标"执行方定"的各条，加上本计划里的数值提案）、未做的事（真实运行、DICOM 入口、独立测试）、下一步需要研究者做的决定。更新 `status.md` 与 `traceability.md`。
- [ ] **Step 4: 全套测试并提交**

```bash
python3 -m pytest tests -q -p no:cacheprovider
git add reports/v5 tests/v5/test_status_docs.py
git commit -m "Record the local implementation of the unified lesion-finding model"
```

---

## 本计划之后（不在本计划内执行）

### 计划二：amax 真实运行（需研究者逐项同意）

顺序如下，命令以任务 9 的驱动为准，计划一完成后单独成文：

1. 同步代码到服务器，CPU 全套测试。
2. 冻结联合 scope 的正式划分（现有 `real_t08_v5`，lineage `v51_skm_joint_early`）。曝光未知，官方 test 的 36 组进入 `excluded`。
3. 联合 scope 的 G-DATA 证据（现有 `real_gdata_v5`，带 `--box-import`），研究者签署 data_use，再评估 G-DATA 关卡。
4. 研究者批准统一冒烟的资源（两个成员各一条，写明 `unified_plan_sha256`）。
5. 统一冒烟（1 张卡）。确认膝成员的 A 标注在联合 scope 的记录下资格正常，确认显存与单步耗时，再定正式预算。
6. 两个成员的 G-BASE 证据与关卡；研究者批准训练资源。
7. 统一训练（1 张卡，估计 6–10 小时）与开发集报告；与两个基线并列对照。
8. 脱敏报告入库。

### 计划三：DICOM 入口

读取 DICOM 序列并转成模型输入，需要真实 DICOM 样本来核对方向、回波拆分和归一化是否与训练一致。服务器上的两套数据看起来都不是 DICOM 原始文件；有了样本再立计划。

## 需要研究者确认的执行方选择

1. 定位点用实例的质心体素，而不是 S7 原文的概率峰值点（输出链不保存概率）。
2. 训练时把医生框外的宿主组织当作负例；评价时框外预测只作描述性计数。
3. "不应明显变差"不设数值界限，报告并列给出差值。
4. 两个协议共用膝基线的步长；每步各取一个批次。
5. 统一训练经 G-BASE-A / G-BASE-U 放行，G-JOINT 不动。
6. 计划文件 `unified_knee_brain.yaml` 里的数值：`stem_channels 16`、`topk_fraction 0.1`、正负权重各 1.0、30000 步、冒烟 40 步。
