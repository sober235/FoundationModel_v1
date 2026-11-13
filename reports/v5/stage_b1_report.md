# v5.1 阶段 B1（T07a–T08b）完成报告

- 计划：`plan_v5.md` §2B 阶段 B1；规格：`Proposal_v5.md`
- 分支：`restart-dev`，B1 提交范围 `0bfdc8b..9db2cb3`
  - 计划细化 `dfa0234`
  - 任务提交 `0255e86`…`b959ed8`
  - 终审修复 `39b35ae`…`c69e351`，修复记录 `9db2cb3`
- 本分支同时包含此前未推送的阶段 A 补修 FX1–FX4（`f0e826a..9fb5752`）和 amax 只读预检（`0bfdc8b`）。
- 数据运行状态：NOT_STARTED。B1 只在合成数据上验证，未接触真实数据。真实 T07 运行要等 G-DATA 批准和服务器写入同意。
- 科学结论：NOT_EVALUATED
- 逐任务状态见 `reports/v5/status.md`

## 结果

- 全套测试 414 passed，另有 3 个 v41 原有警告；`tests/v41` 仍为 144 passed。
- 测试命令：`env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python3 -m pytest -q -p no:cacheprovider tests`
- 执行方式：按计划逐任务 TDD，每步先确认 RED 再实现。
- 终审：由独立评审（opus，全新上下文）对 `dfa0234..b959ed8` 做一次评审，结论是“修复后可合并”。
  - 共 1 个 Critical、5 个 Important、9 个 Minor。
  - M3 按实际影响上调为 Important，与 C1、I1–I5 在一轮修复中一并处理，每项 RED→GREEN。
  - M9（warning 过滤范围）改为对第 5 条裁定的更正，见第 26 条。
  - 其余 7 个 minor 延后，见下文。

## 交付内容

| 任务 | 模块 | 测试数 |
|---|---|---|
| T07a 来源政策与 v41 迁移 | `source_policy_v5.py`、`migrate_v5.py`、`configs/v5/sources_v5.template.yaml` | 9 + 20 |
| T07b 几何检查与降采样损失报告 | `geometry_v5.py` | 11 |
| T07c 脱敏来源核验汇总 | `source_check_v5.py` | 9 |
| T08a 曝光角色与 locked_test 阻断 | `exposure_v5.py`、`schemas/v5/sample.schema.json` | 7 |
| T08b 官方划分与分层划分 | `splits_v5.py` | 11 |

## 终审修复

| 编号 | 问题 | 验证测试 |
|---|---|---|
| C1 | 薄结构降采样后，只要有一个体素落在采样点上就不算“消失”，其余丢失的阳性块会被当成阴性（SKM 软骨会遇到）。现改为逐块判断：块内有阳性、采样点为阴性即标为丢失 | `test_downsample_reports_lost_blocks_of_surviving_thin_structure` |
| I1 | 标签重采样遇到正好半个体素的平局时按银行家舍入，奇数标签会消失；现一律向上取整 | `test_resample_half_voxel_tie_rounds_half_up` |
| I2 | 已迁移的 v5 记录可再迁移一次，几何状态可被升级为 verified；现只接受 v41 记录 | `test_only_v41_records_are_migrated` |
| I3 | 分组映射可把同一患者或泄漏组拆进不同划分；现报 `LEAKAGE_GROUP_SPLIT`，报错不带 ID | `test_group_mapping_cannot_split_a_patient_or_leakage_group` |
| I4 | 从 CSV 读回的失败原因是 JSON 字符串，汇总时被逐字符计数；格式错误的通道/损失报告抛非 V5Error；现一律报 `V5Error` | `test_malformed_rows_raise_v5_errors` |
| I5 | Proposal §3.2 要求登记的几何证据没有字段；单位策略没有依据。现样本新增必填 `geometry_evidence`，verified 必须带依据；来源政策新增 `physical_units_evidence` | `test_migration_records_geometry_evidence`、`test_verified_geometry_needs_evidence`、`test_policy_rejects_unknown_or_missing_fields[6]` |
| M3→Important | 训练曝光的 lineage 为空字符串时不阻断，样本会悄悄进入 locked_test；现空白 lineage 视同未命名，阻断所有谱系 | `test_blank_lineage_entry_blocks_like_null` |

修复清单也记在 `plan_v5.md` §2B“B1 终审修复”。

## 执行中的裁定

按时间顺序；括号内为判错的代价。

### 执行方式

1. 计划标题是 `### T07a：` 这种格式，改用自写脚本按 `### <ID>：` 提取任务说明（无代价）。

### T07a

2. 去掉“几何 pass、单位 unknown”这个测试用例，因为 v41 schema 只允许 mm；`units == "mm"` 检查作为防御条件保留（将来 schema 放开单位时要补回该测试）。
3. `status.md` 的 T07 一行到 T07c 时一次性更新，不按子任务更新（无代价）。

### T07b

4. `resample_labels_v5` 返回 (labels, valid_mask)，`restore_grid_v5` 返回 (array, represented_mask)，而不是计划写的裸数组。Proposal §3.2 要求 FOV/填充单独核验；裸数组会把 FOV 外记成标签 0、把裁掉的体素记成已还原（后续调用方要解包）。
5. 仿射往返测试过滤了 matmul 的假警告，已核实是 NumPy 2.0.2 在 macOS Accelerate 上的误报：结果有限，与 einsum 相差 3e-14（隐藏一个警告；准确范围见第 26 条）。

### T07c

6. 把这个过滤移到整个测试文件（`80a64e2`），因为重采样测试也触发同一假警告（无代价）。
7. 汇总总是列出全部状态键（pass/fail/unknown；paired/missing_label/incomplete）；认不出的失败原因记为 `OTHER`；未知配对状态报 `INVALID_PAIRING_STATUS`；没有可评估轨道的通道病例记为 unevaluated。都不回显行内容（汇总键名可能要改）。
8. T07 的 data_run_status 沿用 `NOT_STARTED`，没用计划写的 `TODO`，与已有词表一致（一个词）。
9. 顺带刷新 T00、T01c、T06b 三行过期的测试计数（4→8、18→23、24→25）（无代价）。

### T08a

10. v5 曝光条目的五个字段都必须出现，lineage_id 和 evidence 可以为 null；v41 形状的条目不能通过 v5 校验，避免缺失的 lineage 被误读成 null（产出方要显式写 null）。
11. 新增 v41 曝光的迁移函数 `migrate_exposure_v41` 和两个错误码 `INVALID_EXPOSURE_ROLE`、`INVALID_LINEAGE`，并加测试 `test_blockers_reject_bad_inputs`（多两个错误码）。

### T08b

12. `groups` 按 `{sample_uid: 组}` 理解，而不是读 `SampleV5.leakage_group_uid`。原先估计判错只差一行适配；终审指出这会让同一患者跨划分泄漏，已由 I3 修复。
13. 分层切分点用累计比例四舍五入（半数向上，`floor(n*cum+0.5)`），顺序为 train、development_val、locked_test；比例必须恰好三项、有限、非负、和为 1±1e-9；拒绝名为 `__merged__` 的层（每层边界处可能差 1 组）。
14. 计数列出全部五种划分（含 0）；`split_sha256` 哈希 {assignment, lineage_id, policy}；另加错误码 `MISSING_GROUP`、`INVALID_GROUP`、`DUPLICATE_SAMPLE` 和两项测试（可能改名）。
15. `status.md` 的 T08 code_status 记 PASS，data_run_status 保持 NOT_STARTED；`traceability.md` 的 V5-01、V5-07、V5-08 改为“代码已交付（合成数据），真实运行待做”（措辞）。

### 终审：评审未下结论、由执行者裁定的项

16. `geometry_evidence` 的结构定为 `{index_alignment, physical_geometry}`（string 或 null，必填）。verified 时必须有非空依据；几何检查失败时记为 `v41_geometry_status:fail`；来源政策设了单位策略就必须给 `physical_units_evidence`。Proposal §3.2 只给了字段名（真实记录出现前可能要改 schema）。
17. locked_test 的独立性证据和人工确认（Proposal §3.3）留给 gate 与批准环节，不放进 `freeze_splits_v5`，因为划分本身证明不了独立性（接上之前，冻结的划分不能单独算合格的 locked test）。
18. 反复换种子直到划分有利属于流程治理问题。`split_sha256` 包含政策（含种子），每次换种子都可审计；真实运行前要先登记种子（代码能发现但拦不住）。
19. 把 `lost_positive_mask` 用于阴性资格判断留给 B2 的 T09，计划本来就这样分配（T09 之前没有代码把这些块从阴性中剔除）。
20. lineage 去掉首尾空白后精确匹配，不推断后代谱系；null 或空白 lineage 阻断所有谱系（后代换了新 ID 就不会被父谱系的训练曝光阻断；T10 之前要定好命名规则）。
21. 记录的 `source_release` 与政策的 `source_version` 的绑定推迟到真实运行 T07 时再做，因为真实 v41 的 release 字符串还不知道（在此之前，其他版本的记录也能按这个政策迁移）。
22. 标签取值 QC 的计数不进脱敏汇总，T07c 接口没有这一项；v41 的 `summarize_geometry_qc` 仍在私有侧产出（入库的 md 里没有这组计数）。
23. 保留计划规定的步长取样降采样；C1 修复后，每个丢失的阳性块都会被报告（对阴性判断无代价）。
24. `annotation_modalities` 按任务登记，`index_evidence` 是自由文本，都沿用计划的 SourcePolicy 接口（以后要按标注区分模态时需改政策）。
25. `pad_or_crop_v5`、`restore_grid_v5` 的补零默认值 `fill=0` 保留：它不是 Proposal 的候选值（GC11），且总与显式掩码一起返回（无代价）。
26. 更正第 5 条：模块级过滤（`80a64e2`）实际隐藏 matmul 的除零、溢出、无效值三类警告，不只是除零；断言仍会抓到错误结果（`test_geometry_v5` 中真实的 matmul 溢出警告会被隐藏）。

## 延后的 minor（7 项）

| 编号 | 内容 | 去向 |
|---|---|---|
| M1 | v41 记录几何为 pass 但 `image_geometry` 为 None 时，`migrate_v5` 抛 `AttributeError`；批处理只捕获 V5Error，整批迁移中断 | 真实 T07 运行前 |
| M2 | 来源政策的 `model_modalities` 没有与 scope 的 `input_modalities` 核对（测试数据中两者已不一致） | 待定 |
| M4 | 迁移后保留的 v41 `split` 值在 `freeze_splits_v5` 之前不能当真 | 待定 |
| M5 | 合并了哪些小层、各层多大都没有报告；合并后仍低于最小组数的层也能通过 | 待定 |
| M6 | 来源核验汇总：通道/损失报告不按来源过滤；几何未知状态记为 unknown 而配对未知状态报错，处理不一致；没有比例；来源填错时只得到一份全零汇总，没有提示 | 待定 |
| M7 | 仿射含 NaN 时往返误差返回 NaN；降采样因子接受 bool | 待定 |
| M8 | 来源政策的报错路径里会带出 YAML 的来源键名 | 待定 |

## 待研究者决定

- 4 份仍被 Git 跟踪的 v4.1 文档含私有路径（blocker `PRIVATE_PATHS_TRACKED:v41_docs`），是否脱敏。
- 是否同意在 amax 上建独立工作目录，用服务器解释器跑 `tests/v41 tests/v5`（见 `reports/v5/host_precheck_amax.md`）。
- ISLES-2022 和 SKM-TEA 按来源与版本的 G-DATA 数据使用批准。
- D2 的 Git 历史审查。

## 下一步

B1 经研究者审阅后，把阶段 B2（T09a–c、T10：训练、推理、基线）细化到 B1 的粒度，再执行。B2 需要 GPU、G-BASE 批准和已登记的 amax 执行环境（D4）。
