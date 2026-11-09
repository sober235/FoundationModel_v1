# T11 前期：SKM-TEA 原有异常框与 tissue_id 导入（准入材料，2026-11-09）

- 计划：`plan_v5.md` T11（前期原标签导入）；规格：`Proposal_v5.md` §2.2 `skm_joint_early`、§5.0。
- 完成情况：
  - 导入代码已写好，合成数据验证已通过；
  - scope 实例与来源绑定已写成提案。
- 没有做的事：
  - 没有在真实数据上运行导入；
  - SPIDER 不在本轮范围内。
- T11 与 T10 并行推进，不是 T10 的前置条件。T12、T13 没有开始。
- 科学结论：NOT_EVALUATED。前期结果只能登记到 `early_phase_status`，不能使 `core_target_validated` 为 true。

## 1. 交付物

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `anatobind/v41/annotation_v5.py` | 单次扫描的框导入（`import_skm_boxes_v5`）、发布表校验、计数汇总（只有计数） | 代码完成 |
| `configs/v5/skm_box_policy_v1.yaml` | 把 S1–S7 写成冻结规则，策略 sha256 为 `09b1003e…` | 取值是研究者 2026-11-07 已采纳的 S1–S7 |
| `configs/v5/proposals/scope_skm_joint_early.yaml` | ActiveScope 提案，scope_hash 为 `f6e4b5f6e3c95a49f5f8cb45591b2a2c437b8db2d432551e1bd1e9e81935ec9f` | **提案，未批准** |
| `configs/v5/proposals/source_skm_tea_joint_early.yaml` | 联合 scope 自己的来源绑定，source_id 为 `skm_tea_joint_early` | **提案，未批准** |
| `tests/v5/test_annotation_v5.py` | 22 项合成测试 | 全部通过 |

## 2. 语义如何落实（S1–S7）

| 规则 | 实现 |
| --- | --- |
| 框级 U 不转成体素真值 | 每个框保持为框，`reference_granularity: box`。导入时不生成掩膜和资格图，输出是纯 JSON（有测试） |
| 无框不等于阴性 | 组织状态只有 `positive` 和 `annotation_unknown` 两种取值。无框的扫描或组织记 `annotation_unknown`；`inspected` 不产生阴性（有测试） |
| S 只评价有支持的阳性 | 只有被纳入的主分析框才能使某组织组为 `positive`。1 级病变单列在 `secondary` 中，不进主分析（S2） |
| R 用显式的组织归属 | 用框的 `tissue_id` 去查发布文件里的 `tissues` 名称表，得到组织组，并记 `assignment: released_tissue_id`。不按整数 id 直接对应，也不按框的位置推断（有测试：组织表打乱 id 后，归属仍随名称） |
| S1 / S3 | 主分析：软骨病变 2A/2B/3，加半月板撕裂（不含挤出）。挤出、韧带撕裂、积液记为 `out_of_scope`；积液的关系记 `no_host`，ACL/PCL 记 `out_of_scope` |
| S4 | 只排除置信度为 0 的框。策略校验写死这个已采纳的值，改成其他阈值会被拒绝（`SKM_POLICY_NOT_ADOPTED`） |
| S5 | 按固定顺序排除并计数：类别与组织不符 → 置信度 0 → 零尺寸 → 非整数坐标 → 越界 → 框内没有宿主组织（按参照 A 判断）。被排除的框仍是未知，不会变成正常 |
| S6 | 主分析用 4 个组织组。次要分析中，半月板和胫骨软骨的内外侧由“框 ∩ 参照 A”导出，不涉及模型输出 |
| S7 | 导入只准备评价所需的参照，不实现评价，评价属于 T12/T13 |

- 发布文件中出现策略没有定义的类别名或组织名时，直接报错（`SKM_CATEGORY_UNKNOWN` / `SKM_TISSUE_UNKNOWN`），不猜。
- 参照 A 的网格与框的网格不一致时，报 `GRID_MISMATCH`。

## 3. 新的联合 scope 需要自己的绑定和证据

- scope 提案 `skm_joint_early`：
  - A 为 A6；U 为 `cartilage_lesion` 和 `meniscal_tear`；R 为宿主组织组；
  - `reference_policy_id` 就是框策略的 `skm_box_reference_v1`。
- `completion_role` 有意设为 `auxiliary_spatial`：
  - 原标签只有框级 U 和组织级 R，无法完成核心的宿主绑定；
  - 因此 G-JOINT 始终会被 `CORE_RELATION_REQUIRED` 阻断（有测试）。
- 它的 scope_hash 与 skm_a 不同，来源也另起 source_id。所以 skm_a 的 G-DATA 证据和批准都不能用在这里，必须重新产生。

## 4. 准入还缺什么（按顺序）

1. **研究者批准两份提案**，即 scope 实例与来源绑定。批准后，它们分别移入 `configs/v5/scope_skm_joint_early.yaml` 和 `sources_v5.yaml`。
2. **决定导入读哪些划分**。推荐只导入 train 和 development_val，locked_test 的框保持封存，等 G-LOCKED 时再处理。
   - 理由：导入时要读参照 A 来判断框内是否有宿主组织，等于读取标签。
   - SKM 官方 test 的曝光状态以 T08 账本为准（P6）。
3. **补两段代码**：
   - 真实导入的命令行：读取发布 JSON，以及每个扫描的 H5 分割；只输出计数。
   - G-DATA 链守卫中 joint 任务的规则：要求来源策略同时登记了 A 和 U。目前守卫检查的是 `"joint" in annotation_modalities`。
4. **在 amax 上运行真实导入**，产出：
   - 联合 scope 的来源核验摘要（填入 `index_evidence`）；
   - 六项 G-DATA 证据（任务为 joint）。
5. **研究者签署 G-DATA 的 data_use**，绑定 `skm_joint_early` 的 scope_hash 和 `skm_tea_joint_early` 的来源。

真实导入跑完后，计数应与研究报告中的级联一致，可作一致性核对。研究报告按 train/val/test 列出的目标框数：

| 阶段 | train | val | test | 合计 |
| --- | --- | --- | --- | --- |
| 类别与组织一致 | 157 | 77 | 84 | 318 |
| 置信度 ≥ 1 | 155 | 75 | 84 | 314 |
| 非零尺寸 | 153 | 71 | 83 | 307 |
| 框内含宿主组织 | 146 | 69 | 82 | 297 |
| 主分析（去掉 1 级和挤出） | 106 | 53 | 63 | 222 |

导入的级联还多两步：非整数坐标和越界。研究报告记录的越界框只有 1 个，而且不是目标框，所以这两步预计不改变上表的计数。

## 5. 本轮没做的事

- 没有在真实数据上运行导入。原因是 scope 和来源都还是提案，联合 scope 的 data_use 也没有批准。
- SPIDER 的分级导入没有做：本轮数据边界不含 SPIDER。
- 没有做 T12、T13，也没有做评价代码。
