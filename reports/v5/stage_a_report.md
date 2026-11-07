# v5.1 阶段 A（T00–T06c）完成报告

- 计划：`plan_v5.md` 阶段 A；规格：`Proposal_v5.md`
- 分支：`restart-dev`，提交范围 `0ca68b5..14c8a9d`（任务提交 `79ad7ed`…`a9d9f73`，终审修复 `d791931`，plan T06b 同步 `14c8a9d`）
- 科学结论：NOT_EVALUATED（阶段 A 只交付契约、资格与评估器，未接触真实数据）
- 逐任务状态见 `reports/v5/status.md`
- **2026-11-06 补记：** 本报告中的部分内容已被 `reports/v5/status.md` 与 plan_v5.md §2A 取代——第 33 项（gate 证据未绑定 scope）、M11（测试批准人）、M13（traceability 未更新）已由补修 FX1/FX4 处理；“待研究者决定 D1–D4”已由研究者决定。以 status.md 为准。

## 结果

- 全套测试 319 passed；`tests/v41` 仍为 144 passed，v41 函数未改动。
- 测试命令：`env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python3 -m pytest -q -p no:cacheprovider tests`
- 执行方式：按计划逐任务 TDD（每步先确认 RED 再实现）。
- 终审：由独立评审（opus，全新上下文）对整个分支做一次评审。
  - 6 个 Important（I1–I6），加上 4 个按实际影响上调为 Important 的问题（M1、M4、M10、M15），已在一轮修复中处理，每项 RED→GREEN。
  - M5、M6 并入 I1、I2 一起修复。
  - 其余 14 个 minor 延后，见下文。

## 终审修复

| 编号 | 问题 | 验证测试 |
|---|---|---|
| I1+M5 | outcome/reference 标志接受非 bool，状态与标志可不一致 | `test_numpy_bool_outcome_counts_as_success`、`test_outcome_flags_must_be_bool_and_match_status`、`test_reference_flags_must_be_bool`、`test_exclusion_reason_on_joint_lesion_rejected` |
| I2+M6 | 从未预测的组织 F1 为 None 而非 0；接受 str 作为预测集合 | `test_never_predicted_tissue_has_zero_f1`、`test_string_predicted_set_rejected` |
| I3 | 配对统计未检查扫描配对 | `test_scan_missing_on_one_side_rejected`（UNPAIRED_SCAN）、`test_scan_missing_for_one_seed_rejected`（MISSING_SCAN） |
| I4 | StateReference 中 unknown/不可观察的可见性可能变成 normal；缺 review region 未报错 | `test_unknown_or_unobservable_visibility_is_never_normal`、`test_region_complete_without_review_region_rejected` |
| I5 | 参照未要求 expert_voxel A/U，A 未要求覆盖 state 组织 | `test_reference_needs_expert_voxel_a_and_u_covering_scope` |
| I6 | StateReference 数组未复制、未只读，评分前未校验哈希 | `test_reference_arrays_are_copied_and_read_only`、`test_tampered_state_reference_rejected` |
| M1 | NSD 容差非有限值时静默得到 0/1 | `test_surface_tolerance_must_be_finite_positive` |
| M4 | V5Error 回显未知键名（可能是病例 ID/路径） | `test_unknown_key_names_are_not_echoed`、`test_unknown_output_key_is_not_echoed` |
| M10 | BCE 跨类合并，未按类归一化 | `test_bce_is_normalized_per_class` |
| M15 | 概率图被静默二值化 | `test_probability_maps_rejected`、`test_non_bool_masks_rejected` |

## 执行中的裁定

按时间顺序；括号内为判错的代价。

### 执行方式

1. 计划标题是 `### T00：` 而非 `Task N`，标准脚本匹配不到；改用自写脚本按 `### <ID>：` 提取任务说明（无代价，只影响提取）。

### T01a

2. 模块 scope 也拒绝非空的 relation_type/relation_predicate（带悬空 predicate 的模块配置会被拒）。
3. joint 模板中 relation_type 和 reference_policy_id 也置为 null（多两个字段待填）。
4. make_sample_v5 复用 `tests.v41.conftest._sample`（改 v41 测试数据会连带影响 v5）。
5. skm_a 的 reference_policy_id 定为 `skm_a6_released_reference_v1`，normal_claim_scope 文案按 Proposal §2.2 写（改名即可）。

### T01b

6. derive_state_map_v5 的三道检查：state 组织为空报 STATE_CLASSES_REQUIRED；C=0 报 JOINT_REQUIRES_U；网格不一致报 GRID_MISMATCH（C=0 的调用方须改用 ModuleOutput）。
7. contracts_v5 在函数内延迟导入 state_rule_v5，避免循环导入（无代价）。
8. 新增错误码 UNKNOWN_CLASS、INVALID_BBOX、INVALID_TYPE（改名）。
9. make_joint_output 只给一个数组时另一个取默认值；2d 取固定的一张 z 切片（仅影响测试数据工厂）。

### T02

10. observable_target 配 allow_unknown_visibility=True 时报 UNKNOWN_VISIBILITY_NOT_ALLOWED，不静默忽略（变成配置报错而非无操作）。
11. 新增错误码 UNKNOWN_TRACK、UNKNOWN_CHANNEL、UNKNOWN_LOSS_KIND、VISIBILITY_REGION_MISSING、REVIEW_REGION_MISSING、TARGET_MISSING、GRID_MISMATCH（改名）。
12. Dice 为逐类 soft Dice，在有支持的类上平均；BCE/CE 在监督位置上取均值（损失量级不同，不影响支持规则）。

### T03

13. class_aware 非 bool 报 INVALID_POLICY；策略中多余的键忽略（改名）。

### T04

14. 校验参照集时同时重算联合集合 G，防止 G 被篡改（对正常调用方无代价）。
15. 关系评分 R 的分母为「有 R 参照 ∧ 可评 ∧ 可见性合格」；predicted_sets 缺键算未预测；未知 uid 报错（R 计数与 J 的 G 集不同）。原「F1 无定义时为 None」已被终审 I2 修正为 0。
16. 新增 DUPLICATE_PATIENT、INVALID_PREDICTION_STATUS、MISSING_FIELD；truth_relation_set 冻结为排序列表（改名）。

### T05

17. 未登记 seed 报 UNREGISTERED_SEED，不静默丢弃；另有 INVALID_SEED_POLICY、INVALID_SEED、DUPLICATE_ROW（调用方须先自行过滤）。

### T06a

18. 区域可评估时始终计算 Dice，即使策略未要求（结果中会多出未请求的 Dice 值）。
19. 表面距离的失败原因累加记录；BOTH_EMPTY 提前返回；spacing 格式错误报 INVALID_SPACING；锁定阶段 metrics 为 null 报错（原因列表更长）。
20. `test_metric_policy.py` 过滤 DeprecationWarning（会掩盖 SciPy 2.0 可能破坏固定版本源码的信号；生产封装同样屏蔽）。

### T06b

21. 私有的 `_channel_v5` 改为公开的 `annotation_channel_v5` 并复用，专家/伪标签通道只有一处定义（一次改名）。
22. 参照哈希额外覆盖 single_slice_readable 和 eligibility_reason（仅标注资格变化也会改变哈希，这是预期行为）。
23. U 完整性取 target_complete/region_complete 专家标注的 class_scope 并集，须覆盖全部异常类别；positive_only 不计入。
24. 拒绝模块 scope，并校验参照数组的类型与形状（改名）。

### T06c

25. size_strata_voxels 须为非空、严格递增的整数列表；voxel_volume_mm3 为 None 或正的有限值；分层为半开区间；零扫描时 fp_per_scan 为 None（改名）。

### 终审修复

26. StateReference 输入契约改为按标注路径索引的 review_regions/visibility_regions 字典，不再接受调用方自给的数组；plan T06b 文字已同步（T12 必须传区域字典）。
27. LesionOutcome 状态须与标志一致：matched ⇒ detected ∧ accepted ∧ relation_correct≠None；missed ⇒ ¬detected；rejected ⇒ detected ∧ ¬accepted；no_relation ⇒ detected ∧ relation_correct=None（生产方须选与标志相符的状态）。
28. 扫描配对：同侧每个 seed 覆盖同一批扫描，两侧每个患者的扫描集合一致（崩溃的扫描须显式记录，不能丢弃）。
29. 专家训练通道保留计划规则（reference_role ≠ unverified），只有 StateReference 要求 expert_voxel（标为 pseudo_only 的人工标注仍会进入专家训练通道）。

### 评审未下结论、由执行者裁定的 11 项

30. A 被拒的实例下接受 no_host 可通过校验，留给 T13（T13 前此类输出会被放行）。
31. val/test 标注也获得训练支持，划分与暴露控制留给 T08（T08 前不得把 val/test 记录送入训练）。
32. n_negative_patients 统计的是无参照病灶的患者，而非按协议审阅的阴性患者，留给 T12（暂无法计算特异度）。
33. gate 证据未绑定 scope_hash，G-DATA 未按 (source, task) 参数化，留给阶段 B（在此之前证据可能跨 scope 复用）。
34. 患者内聚合为每次观测比值的均值，随 D1 一起决定（D1 定后可能更换估计量）。
35. 病灶匹配计入全部预测实例（全覆盖视角），仅计已接受预测的版本随 T14 加入（T14 前缺这组指标）。
36. A 被拒体素的 state 原因只带解剖位（这些体素的原因不够细）。
37. 匹配平局由 scipy 在按 uid 排序的矩阵上决定（等权平局不一定严格按 ID 字典序）。
38. vendored surface-distance 按设计固定版本（SciPy 2.0 时需重新 vendor 或打补丁；`test_anisotropic_matches_reference` 会发现破坏）。
39. memory.md 带私有路径仍被 Git 跟踪，属研究者决定 D2（决定前暴露情况不变）。
40. 阶段 A 不做 migrate_v41，移到 T07；v41 记录被显式拒绝（T07 前无代价）。

## 延后的 minor（14 项）

| 编号 | 内容 | 去向 |
|---|---|---|
| M2 | ActiveScope/PerceptionOutput 直接引用调用方的列表和字典，校验后再修改会改变 scope_hash | 待定 |
| M3 | PerceptionOutput 缺内部一致性检查（实例 mask ⊆ U 通道、bbox ⊇ mask、tissue_id 唯一、uid 为 str、关系与实例接受一致、spacing 与 units_reliable 一致） | T09c/T13 |
| M7 | canonical_sha256 接受 NaN/inf 和非字符串键 | 待定 |
| M8 | 部分格式错误的输入抛 KeyError/AttributeError/TypeError/IndexError 而非 V5Error | 待定 |
| M9 | validate_sample_v5 接受重复 annotation_uid、不在 modality_names 中的 annotation_modalities | 待定 |
| M11 | gate 在测试之外也接受 `approved_by: synthetic-fixture` | 待定 |
| M12 | joint scope 检查偏松：接受 relation_type 为 null、空 scope_id、辅助 predicate 声明为核心 | 待定 |
| M13 | `reports/v5/traceability.md` 状态列未逐任务更新 | 待定 |
| M14 | 匹配开销约为实例对数 × 体积（256³、20×20 实例约 2 秒） | 阶段 B 裁剪到 bbox 并集 |
| M16 | spacing_mm 轴序（zyx 还是 NIfTI xyz）未定 | 阶段 B |
| M17 | J/R/S 结果不带 checkpoint/scope 标识，汇总无法发现混用 | 待定 |
| M18 | derive_state_map_v5 无 valid_fov 输入，FOV 只经上游接受处理 | T09c |
| M19 | `tests/v5/factories.py` 依赖 `tests.v41.conftest` | 待定 |
| M20 | 病灶/关系汇总按扫描或病灶合并统计，无患者键 | 阶段 C 需患者等权版本 |

## 待研究者决定

- **D1**：T17 的能力标准。
- **D2**：memory.md 带私有路径的处理。
- **D3**：专家标注资源。
- **D4**：真实数据所在主机。

## 下一步

阶段 A 经研究者审阅后，把阶段 B（T07–T10）细化到阶段 A 的粒度，再执行；阶段 B 需要真实数据，在数据服务器上运行（取决于 D4）。
