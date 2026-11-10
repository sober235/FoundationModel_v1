# 导入与划分登记：检查与补齐（2026-11-10）

本报告只含计数、规则与哈希；不含病例 ID、路径或文件名。范围停在导入与登记：没有冻结正式划分，没有训练，没有解封测试。`scientific_conclusion: NOT_EVALUATED`；`core_target_validated` 保持 `null`。

依据：Proposal_v5 §3.1、§3.3、§5.0，plan_v5 §1.6 L8、§2B T08a/T08b；研究者 2026-11-10 关于导入范围的决定。

## 1. 研究者决定（2026-11-10）

1. SKM-TEA 导入读取全部三个官方划分文件及其病例，每个病例保留官方来源划分（`provenance.release_split`）。这取代了当天早些时候"只读 train 和 development_val"的决定。
2. 来源划分记录出处，正式 `split` 控制训练与评价。新登记病例的正式 `split` 保持 `unassigned`，经患者分组、曝光核验和正式划分流程后才冻结。
3. SKM-TEA 沿用官方患者分组：train → train，val → development_val，test 经独立性与曝光核验后合格者进入 locked_test。官方 test 的名称不构成盲测资格。
4. ISLES 先登记可用病例，再按患者组分层划分；比例、分层变量和种子以正式确认的配置为准。
5. 导入、身份核验和管理性 QC 的接触，与训练、模型选择和查看测试结果的曝光分开记录。

## 2. 检查结果：已有行为

| 要求 | 已有实现 | 结论 |
| --- | --- | --- |
| 保留官方来源划分 | `audits.build_registration_records` 写 `provenance.release_split` | 已有，复用 |
| 新登记病例正式 split 为 unassigned | 两个来源的登记记录都写 `split: unassigned` | 已有 |
| 官方划分映射 | `splits_v5.OFFICIAL_MAP`：train → train，val → development_val，test → locked_test | 已有 |
| 官方 test 不自动成为盲测 | 冻结时曝光阻断的组移出 locked_test；`unknown` 对所有谱系阻断 | 已有。真实冻结中两个 scope 的 locked_test 均为 0 |
| 同患者随访归一组 | SKM 的 `leakage_group_uid` 取发布文件的患者号；冻结时同一患者或泄漏组分到两组会被拒绝 | 已有 |
| 回波、轨道不重复计数 | SKM 一个病例一条记录，两个回波在同一个 H5 内，标签轨道只登记 dicom 轨道 | 已有 |
| 同一患者跨官方划分 | 冻结时报 `OFFICIAL_SPLIT_CONFLICT` | 已有，但只在冻结时才发现 |
| 曝光角色分开 | `exposure_v5` 有 `administrative_qc`，不阻断 locked_test | 角色已有；导入没有写接触记录 |
| 加载器按正式角色读取 | `select_samples_v5` 只读冻结登记，不读记录里的 split 字段；请求 locked_test 被拒绝 | 已有 |
| ISLES 划分配置 | `split_rule_isles_u.yaml`：研究者 2026-11-08 登记的 70/15/15、种子 20261108、病灶体积三分位分层 | 已有。Proposal 中的 0.6/0.2/0.2 与 20261101 没有出现在任何函数默认值或 v5 配置里，只在旧版测试的显式输入里 |
| ISLES 配对 | `adapters/isles.pair_assets` 加来源核验的配对计数 | 已有；重复文件会被后一个静默覆盖 |

## 3. 缺口与修改

| 缺口 | 修改 |
| --- | --- |
| 官方划分文件缺失时被静默跳过 | `audits.scan_skm_release`：三个文件逐个检查并报告存在、可读、扫描数、可确认患者数、框数 |
| 病例在同一文件重复、或出现在多个划分时，被静默并入第一个划分，框会重复累加 | 重复与冲突分别计数；冲突病例的 `release_split` 置空并写 `release_split_conflict`，按官方规则无法获得正式划分；框只取第一个文件的 |
| 同一患者跨官方划分只在冻结时才报错 | 登记阶段即计数（`patients_in_multiple_splits`） |
| 没有患者号时，患者号退回为病例号，患者数被高估 | 记录写 `patient_identity: unconfirmed_case_fallback`；患者数只统计可确认的，另列不能确认的扫描数 |
| 本地有影像但发布文件无条目 | 记录写 `release_entry: missing`；两个方向的不一致都计数 |
| 扫描数与患者数没有分开的登记汇总 | `audits.registration_summary` 与 `real_t07_v5 registration-report`：扫描、可确认患者、泄漏组、按来源划分的计数、正式 split、曝光角色 |
| ISLES 与 SKM 的重复文件无记录 | `adapters/isles.pairing_report`、`adapters/skm.pairing_report`：配对成功、各类缺失、重复文件、未登记的其他文件 |
| 框导入只读 train 和 development_val，并依赖冻结划分 | `real_t11_v5`：导入全部登记扫描，结果带 `release_split`；不需要也不分配正式划分 |
| 导入没有接触记录 | `real_t11_v5` 写 `contact_ledger.jsonl`，角色 `administrative_qc`，与运行登记的曝光分开 |
| 联合 G-DATA 证据绑定在冻结划分的读取范围上 | `real_gdata_v5`：改为绑定导入覆盖的记录集；三个官方文件都须存在，且没有官方划分未定的扫描 |

异常标记只在有问题时写入 `provenance`。没有问题的记录与修改前逐字段相同，已有的冻结划分与数据哈希不受影响。

## 4. 实际修改的文件

| 文件 | 修改 |
| --- | --- |
| `anatobind/v41/audits.py` | 新增 `scan_skm_release`、`registration_summary`；`load_skm_annotations` 改为调用前者；登记记录在异常时写标记 |
| `anatobind/v41/adapters/isles.py`、`adapters/skm.py` | 各新增 `pairing_report` |
| `anatobind/v41/real_t07_v5.py` | 新增 `registration_report_v5`、`render_registration_md_v5` 和子命令 `registration-report` |
| `anatobind/v41/real_t11_v5.py` | 导入范围改为全部官方划分；接触台账；不再读取冻结划分 |
| `anatobind/v41/real_gdata_v5.py` | 联合 scope 的框导入绑定改为记录集 |
| `tests/v5/test_registration_report.py` | 新增 16 项 |
| `tests/v5/test_real_t11.py` | 改写为 35 项 |

`splits.py`、`splits_v5.py`、`exposure_v5.py`、`identity.py` 和各配置文件没有改动。

## 5. 验证

### 5.1 代码验证（合成数据）

本地全套测试：758 passed，2 skipped（Python 3.9）。对应五项交付：

| 交付项 | 测试 |
| --- | --- |
| 三个官方划分均被检查和读取，来源划分保留 | `test_all_three_official_files_are_checked_and_read`、`test_import_covers_all_three_official_splits_and_keeps_the_release_split` |
| 扫描与患者分开统计，派生版本不重复计数 | `test_scans_and_confirmable_patients_are_counted_separately`、`test_registration_report_counts_scans_patients_splits_and_exposure` |
| 缺失、重复、冲突、曝光未知有记录 | `test_a_missing_or_unreadable_file_is_recorded_not_skipped`、`test_a_duplicate_case_inside_one_file_...`、`test_a_case_in_two_official_splits_...`、`test_a_patient_in_two_official_splits_...`、`test_a_missing_official_file_is_recorded_and_its_scans_fail` |
| 加载器按正式角色筛选 | `test_loaders_read_by_the_frozen_role_never_by_the_release_split_or_the_record_field` |
| 导入不冻结正式划分 | `test_import_assigns_no_formal_split`、`test_the_formal_split_comes_from_the_freeze_step_not_from_import` |

### 5.2 真实数据验证（amax，只读，2026-11-10）

做了的：把 `scan_skm_release` 的源码原样连同目录计数脚本经标准输入送到服务器执行，只读、只打印计数，没有写任何文件。

没做的：没有在服务器上运行仓库里的 `registration-report`、prepare 或 `import-skm`（服务器代码还在 `ee52ca7`，且写服务器需要研究者同意）。所以框导入的级联计数、接触台账和联合 G-DATA 证据都还没有真实结果。

## 6. 各来源划分的匿名汇总（真实数据，只读）

SKM-TEA 标注 v1.0.0：

| 官方划分文件 | 存在 | 可读 | 扫描 | 可确认患者 | 无患者号的扫描 | 框 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 是 | 是 | 86 | 86 | 0 | 242 |
| val | 是 | 是 | 33 | 33 | 0 | 104 |
| test | 是 | 是 | 36 | 36 | 0 | 130 |
| 合计 | | | 155 | 155 | 0 | 476 |

- 同一文件内重复的病例 0；出现在多个划分的病例 0；出现在多个划分的患者 0；找不到扫描的标注 0。
- 影像文件 155，dicom 轨道标签 155，raw 轨道标签 155；发布文件有而本地无影像 0，本地有而发布文件无条目 0。
- 每名患者一次扫描（时间点全部为 0），所以扫描数等于患者数；这是数据本身的情况，不是计数方式造成的。
- 正式 split：登记阶段全部为 `unassigned`。曝光：155 条记录都带角色 `unknown` 的遗留条目。

ISLES-2022：

| 项 | 计数 |
| --- | --- |
| 病例 | 250 |
| DWI、ADC、mask 齐全 | 250 |
| 缺 DWI / 缺 ADC / 缺 mask | 0 / 0 / 0 |
| 重复文件 | 0 |

- 每例另有 FLAIR，不是模型输入，不登记。
- 250 例都带角色 `unknown` 的遗留曝光条目。几何与标签取值的合格情况见 `source_check_isles_2022.md`（3 例全零 mask，保留为完整阴性）。

## 7. 剩余阻塞项与待完成项

阻塞项：

1. **`LEGACY_EXPOSURE_UNKNOWN`。** 两个来源的全部记录曝光为 unknown，冻结时没有任何组能进入 locked_test。须由研究者核实旧模型的实际曝光。
2. **联合 scope 的真实链未运行。** 登记汇总、prepare、迁移、导入都需要研究者同意在 amax 上写私有目录。
3. **联合 scope 的 G-DATA 数据使用未签。** 导入产生证据后才能签。
4. **G-JOINT 被 `CORE_RELATION_REQUIRED` 阻断。** 前期联合训练走哪个门，待研究者决定。

尚未实现的访问控制（明确列为待完成）：

1. **框参照的按角色读取。** `box_references.jsonl` 现在含三个官方划分的框。读取它的下游代码（T12 参照发布、T13 训练与评价）还没有写；写的时候必须经由运行的冻结划分筛选，训练只取 train，模型选择只取 development_val。目前没有任何代码读这个文件，但也没有代码阻止直接读。
2. **强制隔离。** 同一系统用户下只有流程隔离，没有操作系统级的权限隔离，也没有直接读文件的负向测试（Proposal_v5 §10）。
3. **locked_test 解封流程（T18）。** 批次账本、一次性解封许可和锁定评分都没有实现；现有代码只是拒绝读取 locked_test。
4. **历史或内部验证队列的属性。** 冻结时曝光阻断的官方 test 组被移到 `excluded`，登记里只有组数。"可用于历史或内部验证"这一性质还没有单独的字段记录（Proposal_v5 §4.1 要求作为评价队列属性，不塞进 split 枚举）；属于正式划分冻结环节，本轮不改。
5. **接触台账未并入样本的曝光历史。** 导入的接触写在导入目录的单独台账里，没有合并进各记录的 `exposure_history`。两者分开符合本轮要求；冻结或解封前如何汇总核对，属于后续流程。
