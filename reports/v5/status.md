---
plan_revision: "2026-11-07"
execution_base: "GitHub sober235/FoundationModel_v1 current content (branch restart-dev); code identical to d7cbd5b"
spec_sha256: "532ff42129510d7d11082d351dd7df78a0d8d0466a3a417e861cd1dc62509953"
scientific_conclusion: NOT_EVALUATED
core_target_validated: null
study_package_complete: false
historical_evidence:
  v41_local_tests: "144 passed, Python 3.9.6, 2026-11-06"
  v41_engineering_execution_1094_tests: "historical; code not in this repository"
  skm_a_200step_trial: "historical; code not in this repository"
open_decisions: ["D5"]
decisions:
  D1:
    choice: reference_comparison_with_h3_noninferiority
    decided_at: "2026-11-06"
    implementation_status: TODO   # margins and constraints frozen before independent test; policy instance waits for a frozen joint scope
    run_status: NOT_RUN
    scientific_conclusion: NOT_EVALUATED
  D2:
    choice: untrack_and_ignore_history_review_pending
    decided_at: "2026-11-06"
    implementation_status: PARTIAL   # FX3: memory.md untracked + ignored; 4 tracked v4.1 docs still carry private paths; history not rewritten
    run_status: NOT_APPLICABLE
    scientific_conclusion: NOT_APPLICABLE
  D3:
    choice: fm_data_original_labels_skm_primary_spider_secondary   # supersedes skm_joint_pilot_20_30_dev_scans (2026-11-06), kept for the later voxel-level phase
    decided_at: "2026-11-07"
    implementation_status: TODO   # scope values adopted 2026-11-07 (S1-S7, P1-P6); T11 import and G-DATA (SKM-TEA, SPIDER) pending
    run_status: NOT_RUN
    scientific_conclusion: NOT_EVALUATED
  D4:
    choice: server_amax
    decided_at: "2026-11-06"
    implementation_status: DONE      # 2026-11-08: independent working directory; v5 suite 585 passed on the server interpreter
    run_status: PASS
    scientific_conclusion: NOT_EVALUATED
blockers: ["PRIVATE_PATHS_IN_HISTORY:memory.md", "PRIVATE_PATHS_TRACKED:v41_docs", "BLOCKED_EXPERT_REVIEW",
           "LEGACY_EXPOSURE_UNKNOWN:isles_u,skm_a"]
tasks:
  T00:
    code_status: PASS
    test_evidence: "tests/v5/test_status_docs.py: 8 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01a:
    code_status: PASS
    test_evidence: "tests/v5/test_contracts_scope.py: 17 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01b:
    code_status: PASS
    test_evidence: "tests/v5/test_contracts_outputs.py: 14 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01c:
    code_status: PASS
    test_evidence: "tests/v5/test_gates_v5.py: 23 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T02:
    code_status: PASS
    test_evidence: "tests/v5/test_eligibility_gradients.py: 15 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T03:
    code_status: PASS
    test_evidence: "tests/v5/test_matching_cardinality.py: 50 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T04:
    code_status: PASS
    test_evidence: "tests/v5/test_joint_reference_denominator.py: 21 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T05:
    code_status: PASS
    test_evidence: "tests/v5/test_patient_statistics.py: 12 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06a:
    code_status: PASS
    test_evidence: "tests/v5/test_metric_policy.py: 13 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06b:
    code_status: PASS
    test_evidence: "tests/v5/test_tissue_state_metrics.py: 25 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06c:
    code_status: PASS
    test_evidence: "tests/v5/test_lesion_metrics.py: 5 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T07:
    code_status: PASS
    test_evidence: "tests/v5/test_source_policy.py: 11 passed; test_migrate_v41.py: 23 passed; test_geometry_v5.py: 11 passed; test_source_check.py: 9 passed; test_real_t07.py: 21 passed (L1-L3 closed; real driver)"
    data_run_status: PASS            # 2026-11-08 on amax: ISLES 250/250 and SKM-TEA 155/155 prepared and migrated; reports/v5/source_check_*.md
    approval_status: PENDING         # G-DATA data_use not yet signed
    scientific_conclusion: NOT_EVALUATED
  T08:
    code_status: PASS
    test_evidence: "tests/v5/test_exposure_v5.py: 8 passed; test_splits_v5.py: 12 passed; test_real_t08.py: 5 passed; test_real_gdata.py: 10 passed"
    data_run_status: PASS            # 2026-11-08: splits frozen under the rules registered before the run (L8); reports/v5/split_*.md
    approval_status: PENDING         # written split sign-off pending; locked_test empty (LEGACY_EXPOSURE_UNKNOWN)
    scientific_conclusion: NOT_EVALUATED
  T09a:
    code_status: PASS
    test_evidence: "tests/v5/test_run_registration.py: 76 passed (2026-11-09: smoke subset contract); test_model_interface.py: 9 passed; test_exposure_v5.py: 8 passed; test_splits_v5.py: 12 passed"
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T09b:
    code_status: PASS
    test_evidence: "tests/v5/test_volume_io.py: 14 passed, 1 skipped (SimpleITK comparison runs on amax); test_training_data.py: 14 passed; test_training_core.py: 14 passed (after the B2 final-review fixes)"
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T09c:
    code_status: PASS
    test_evidence: "tests/v5/test_inference_v5.py: 6 passed; test_perception_chain.py: 11 passed; test_perception_io.py: 4 passed; test_training_data.py: 14 passed (chain order; after the B2 final-review fixes)"
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T10:
    code_status: PASS   # 2026-11-09: real smoke driver, G-DATA chain guard, full-step measurement, G-BASE gate, baseline driver (real_baseline_v5)
    test_evidence: "2026-11-09 local (Python 3.9, torch 2.8): tests/v5/test_real_baseline.py: 12 passed; test_real_t10.py: 14 passed; test_real_gdata.py: 21 passed; test_module_baseline.py: 17 passed; test_training_core.py: 21 passed; test_run_registration.py: 76 passed; test_resunet_v5.py: 7 passed. amax (Python 3.11, torch 2.5.1, CPU): 664 passed at 2c2b7ab, 691 passed at 3a478cb"
    data_run_status: PASS      # 2026-11-10: real smokes PASS; G-BASE-U/A PASS after approval C; baselines isles_u and skm_a COMPLETED, reports/v5/module_baseline_*.md
    approval_status: PASS      # approvals A (G-DATA data_use), B (smoke resources), C (G-BASE resources) signed by the researcher 2026-11-09
    scientific_conclusion: NOT_EVALUATED
  T11:
    code_status: PARTIAL   # 2026-11-10: SKM-TEA box import, real import driver (real_t11_v5) and joint G-DATA rule; no SPIDER
    test_evidence: "2026-11-10 local: tests/v5/test_annotation_v5.py: 22 passed; test_real_t11.py: 35 passed; test_registration_report.py: 16 passed (synthetic)"
    data_run_status: PARTIAL   # 2026-11-10 on amax: skm_joint_early registered, prepared, migrated and box-imported (155/155, 0 failures, cascade as expected); reports/v5/real_run_t11_skm_joint_early_20261110.md. Joint split freeze and G-DATA evidence not run; no SPIDER
    approval_status: PENDING   # 2026-11-10: scope_skm_joint_early and its source binding approved; joint G-DATA data_use not signed
    scientific_conclusion: NOT_EVALUATED
  T12:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T13:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T14:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T15:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T16:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T17:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T18a:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T18b:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T18c:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
---

# v5.1 任务状态

本文件 YAML 头是 v5.1 各任务状态的唯一登记处。每个任务分别记录 `code_status`、`test_evidence`、`data_run_status`、`approval_status`、`scientific_conclusion`（Proposal §10）。合成测试只证明软件行为；全部科学结论为 `NOT_EVALUATED`。历史 144 项 v41 测试仅是旧报告，不是 v5 验收证书。

## 已复现的 v41 缺陷（2026-11-06 本仓库实测）

| 位置 | 实测 | 由谁修复 |
| --- | --- | --- |
| `evaluation/instances.py` `match_instances` | 六对 IoU 1/9 合格匹配对五对 IoU 2/3 匹配，返回 5 对（代价 `-(2+IoU)`） | T03 |
| `evaluation/joint.py` `joint_score` | 两个有参照病灶、第二个漏检且 `relation_ok=None` 时分母为 1，`patient_mean_j=1.0` | T04 |
| `evaluation/statistics.py` `paired_patient_bootstrap` | 100 行值 1 的患者对 1 行值 0 的患者，估计为池化 0.990 而非患者等权 0.5；`n_boot=0` 抛 `IndexError` | T05 |
| `eligibility.py` `build_eligibility` | positive-only 以 `target > 0` 给所有类别通道赋资格；正负资格未分开 | T02 |
| `losses.py` | 全 ignore 时返回 `logits.sum() * 0.0`，logit 含 inf 时得 NaN | T02 |
| `evaluation/segmentation.py` | NSD 为无权边界体素比例，HD95 为两向距离合并后的普通分位数 | T06a |

## 研究者决定（2026-11-06；D3 于 2026-11-07 修订）

“决定已作出”不等于“实现已完成”“实际运行已完成”或“科学验证已完成”；四者在 YAML 头 `decisions` 中分别登记。

- **D1 能力标准：** 采用 `reference_comparison`，参照为同一联合场景、同患者划分、同合法信息条件下的专用二维／三维完整系统；保留 H3 的同 checkpoint 双模式、A/U/S/R 八项能力及预注册非劣要求。不给每格叠加尚无依据的绝对临床阈值。非劣容差、最低参照支持、覆盖率、FP 与 false-normal 约束依据专家／开发证据形成，在独立测试前冻结；未冻结前不生成最终能力 PASS。政策实例待 SKM 联合 scope 冻结（有 scope_hash）后建立，`configs/v5/capability_policy.template.yaml` 保持模板。
  - 聚合：患者间等权；J 为患者内冻结病灶集合的成功数／支持数；A/U/S 观测级指标先按观测计算、再患者内汇总、再患者等权；逐指标登记支持集合、患者内权重与空值原因；不得按测试结果选择聚合方式。
  - 预测为空或全部弃权造成的 `None` 不得记作“参照无支持”。当前 `summarize_state_by_patient_v5` 的 `n_patients_without_support` 对 precision 混记了这两种情形，修订放入阶段 C 汇总任务。
  - D1 不阻塞阶段 B 工程；它阻塞最终能力判定与 G-H3-SCI。
- **D2 `memory.md`：** 已停止 Git 跟踪并加入 `.gitignore`（FX3），公开仓库保留脱敏的 `memory.example.md`。**当前跟踪树并不干净**：以下 4 个仍被跟踪的 v4.1 文档含与旧 `memory.md` 相同的私有原始数据根与私有产物根（blocker `PRIVATE_PATHS_TRACKED:v41_docs`）：`AnatoBind_MRI_v4.1_Feasibility_Proposal.md`、`AnatoBind_MRI_v4.1_Supervision_Data_Contract.md`、`docs/archive/v4.1/plan.md`、`reports/v41/reuse_decisions.md`。是否脱敏这些归档文档由研究者决定；未处理前 D2 实现状态为 `PARTIAL`。Git 历史未改写：历史中有私有路径；按 token 格式扫描未发现凭证字符串，但这不是完整的敏感性审查。是否清理历史由研究者另行决定（blocker `PRIVATE_PATHS_IN_HISTORY:memory.md`）。其他机器上的旧克隆 `git pull` 前须先备份本机 `memory.md`。
- **D3 前期联合场景（2026-11-07 修订）：** 前期不引入医生，联合参照只用 FM_data 原有标注（Proposal §0.5、§5.0；核查见 `reports/v5/same_case_label_check_amax.md`）。
  - SKM-TEA 为主场景：A 体素级，U 框级，R 组织级，S 只评价阳性（无框只记“报告未记载”，不作阴性）。
  - SPIDER 为第二场景：A 体素级，S 与 R 组织级，无 U。
  - 前期结果登记为 `early_phase_status`，`core_target_validated` 保持 `null`；体素级核心目标留到后期。
  - 这一决定取代 2026-11-07 核心目标复核中“X0 与 B2 同级并行”的建议。`BLOCKED_EXPERT_REVIEW` 保留，只约束后期体素级参照与核心目标验证。
  - **范围取值（2026-11-07 研究者全部采纳）：** `reports/v5/early_phase_scope_research_20261107.md` 的 S1–S7、P1–P6。
    - SKM：主分析为软骨病变 2A/2B/3 加半月板撕裂（不含挤出），1 级单列；只排除置信度 0 分、类别与组织不符、零尺寸和不含宿主组织的框。
    - SPIDER：输入矢状 T2；主目标为 Pfirrmann ≥ III，终板改变作次要目标。
    - 评价准则的数值仍在开发集上选定，锁定评价前冻结。
  - 实现状态 `TODO`：
    - 2026-11-09 完成了 T11 的 SKM 框与 tissue_id 导入代码；2026-11-10 补了真实导入驱动 `real_t11_v5.py` 和联合 scope 的 G-DATA 规则。导入范围按研究者同日的最终决定：读取全部三个官方划分并保留来源划分，正式划分另行冻结；登记检查与补齐见 `import_registration_report_20261110.md`。真实数据先做了只读的发布文件与配对计数（SKM-TEA 155 扫描 / 155 患者，86/33/36；ISLES 250/250 齐全）。研究者同日同意后，联合数据链在 amax 上完成了登记汇总、prepare、迁移和框导入：155 条全部导入，失败 0，级联计数与预期一致（主分析框 222），见 `real_run_t11_skm_joint_early_20261110.md`。联合 scope 的正式划分冻结和 G-DATA 证据未做，SPIDER 未做。
    - `skm_joint_early` 的 scope 实例与来源绑定由研究者 2026-11-10 批准，已移入 `configs/v5/scope_skm_joint_early.yaml` 与 `sources_v5.yaml`（`t11_skm_joint_early_admission_20261109.md` §6）。G-JOINT 仍被 `CORE_RELATION_REQUIRED` 阻断。
    - 联合 scope 的 G-DATA 证据和批准都还没有。skm_a 的 G-DATA 不能沿用。
- **D3 原决定（2026-11-06，留作后期方案）：** 保留 SKM 联合场景；先准备 20–30 次非锁定开发扫描试点，明确主读、独立复读、仲裁责任，以及异常边界、不确定带与单层可判读规则。人员与预算未落实前 X0 不标完成，相关科学验证保持 `BLOCKED_EXPERT_REVIEW`。框、分级或伪标签不能替代原定联合验证；试点不能形成稳定体素参照时，再按实际资源提出替代场景。
- **D4 执行主机：** 选定 amax 数据服务器。2026-11-06 完成只读预检（`reports/v5/host_precheck_amax.md`）。2026-11-08 经研究者同意写入服务器：代码以 git bundle 克隆到独立工作目录（不动旧 `main` 检出），私有产物放在私有根下的独立子目录；v5 全套测试在服务器解释器（Python 3.11 / PyTorch 2.5.1 / NumPy 2.4 / SciPy 1.17）上 585 passed，SimpleITK 对照测试实际运行。已登记为正式执行环境，blocker `EXECUTION_HOST_UNVERIFIED` 撤销。解释器与目录路径只在本地配置。
- **D5 长期目标定位（2026-11-07 提出，待决定）：** v5.1 是否作为“可迁移 MRI 基础模型”长期目标中的一个能力验证阶段；若是，是否在 v5.1 之后另立复用/迁移/泛化里程碑（Proposal §1.4，plan §1.5）。不改变 v5.1 的任务与完成判定。

## 2026-11-10 目标修订

研究者 2026-11-10 决定（Proposal §0.6）：

- 按医生的原标注粒度验收：膝关节报组织轮廓和带类别、所在组织组的病变框；脑报梗死轮廓。不再要求体素级的病变边界验证。
- 同一个 checkpoint 处理膝关节 qDESS 与脑 DWI+ADC 两种协议；只做三维。
- 成功标准沿用模块基线的指标，与两个基线并列对照。

对现有登记的影响：

- 八项能力与 H1–H3 暂停；T12–T18 中依赖体素级参照、二维模式或 H2 的部分不再推进，任务编号和状态行保留。
- `scientific_conclusion` 仍为 `NOT_EVALUATED`，`core_target_validated` 仍为 `null`。G-JOINT 规则不变，继续阻断。
- 设计：`docs/superpowers/specs/2026-11-10-unified-lesion-finding-design.md`。实施计划：`docs/superpowers/plans/2026-11-10-unified-lesion-finding.md`（本地实现，合成数据验证）。amax 真实运行和 DICOM 入口另立计划。

实现情况（2026-11-10，`unified_local_report_20261110.md`）：

- 本地实现的 10 个任务完成：框监督目标与损失、统一模型、统一计划与运行登记、统一训练循环、协议表与所见输出、所见评价、统一驱动 `real_unified_v5.py`。
- 只做了合成数据验证，全套测试 812 通过、2 跳过；没有在 amax 上运行，没有读真实数据。
- 计划文件 `configs/v5/proposals/unified_knee_brain.yaml` 是提案，未批准。
- 此前登记为"尚未实现"的"框参照按角色读取"已实现：框参照经冻结划分按角色读取，`locked_test` 被拒绝。
- 研究者 2026-11-10 确认了执行方的六项设计选择（报告 §4），实现不变；同日口头同意了计划文件的取值（计划哈希 `602128f3…7defaab`）。写明该哈希的资源批准尚未签署。

计划二第一步（2026-11-10，amax，研究者同日同意；不占 GPU）：

- 服务器代码同步到 `095d24e`；CPU 全套测试 812 通过、2 跳过。
- 联合 scope 的划分已冻结，lineage `v51_skm_joint_early`，规则为研究者 2026-11-08 登记的 SKM-TEA 官方患者划分（`split_rule_skm_a.yaml`）：train 86、development_val 33、excluded 36、locked_test 0；`split_sha256` `32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9`（`split_skm_joint_early.md`）。155 名患者在 `v51_skm_a_b2` 与本划分中的角色一致，冲突 0。
- 联合 scope 的 G-DATA 证据六项全部 PASS，证据 sha256 `d5a567af32752c4f025a77f79a3c67aebe75f134acc4c895d51c8e0e38dd9383`（`gdata_evidence_skm_tea_joint_early.md`）。
- 联合 scope 的数据使用批准已于 2026-11-10 写入：研究者在会话中明确授权（"授权你代签"），由执行方按研究者给出的署名代为写入，批准文件内注明了这一点。此前的批准均由研究者本人写入。
- G-DATA 关卡（`skm_joint_early`，来源 `skm_tea_joint_early`）：engineering_status PASS，阻断项 0；`scientific_conclusion` 仍为 NOT_EVALUATED。
- 统一冒烟的资源批准已于 2026-11-10 写入（两个 scope 各一条，写明计划哈希 `602128f3…7defaab`），同样是研究者明确授权、由执行方代为写入，文件内已注明。
- 统一冒烟第一次运行（2026-11-10，1 张 GPU，约 7 分钟）：**FAIL**。连续训练 40 步和暂停后续训两段都跑完，脑成员的整卷推理已保存；膝成员的整卷推理在提取异常实例时失败（无法再分配内存）。没有产生冒烟证据，G-BASE 未评估，训练未开始。
- 失败原因：膝成员的接受策略 `min_voxels: 1` 把每个孤立的阳性连通块都算作一个实例，而输出链为每个实例保存一张整卷大小的掩膜（膝为 512×512×160）。只训练了 40 步的模型给出数量极多的零散连通块，实例数乘以整卷大小超出了可分配的范围。脑的体积小，此前的模块冒烟没有遇到。这次运行的私有输出保留，不覆盖。
- 诊断（2026-11-10，研究者同意；1 张 GPU，约 2 分钟，只输出计数）：用失败那次冒烟留下的第 40 步模型，对同样 2 个膝训练样本重做推理并统计阳性连通块。每次扫描两类合计约 21 万个连通块，其中 1 个体素的约 6 万个；不小于 50 体素的约 290 个，不小于 100 体素的 28 至 86 个。训练集 106 个主分析框的体素数：最小 144，第 5 百分位 973，中位数 35791。
- 修复一（代码，已完成）：`extract_instances_v5` 在分配任何掩膜之前先核算"保留的实例数 × 整卷体素数"，超过 16 GiB 即报 `INSTANCE_LIMIT_EXCEEDED`。这是内存与磁盘的工程上限，不是正式阈值，不改变任何未超限的结果。
- 修复二（计划取值，研究者 2026-11-10 同意）：膝成员的 `min_voxels` 由 1 改为 50。计划哈希变为 `50ab3e77ebc243f15583f5a6032d133ed5ca8d535383761bec65c2c2f9d1e4c1`；写明旧哈希的冒烟资源批准不再适用。
- 按新哈希重签的冒烟资源批准已写入（研究者授权、执行方代为写入，文件内已注明，并说明取代写着旧哈希的那一份）。服务器代码同步到 `2d8d5db`，服务器上 CPU 全套测试 814 通过、2 跳过。
- 统一冒烟第二次运行（2026-11-10，1 张 GPU，约 7 分钟）：**PASS**（`unified_smoke_20261110.md`）。连续与续训两次执行都完成 40 步，续训比较通过，参数逐位相同；显存峰值 23.42 GiB；单步 1.599 秒。整卷推理：脑 60 处所见，膝 294 处所见（模型只训练了 40 步，这些数字只说明流程走通）。`scientific_conclusion` 仍为 NOT_EVALUATED。
- 按冒烟的单步耗时，30000 步约需 13.3 小时，超过原定 10 小时的训练预算。研究者 2026-11-10 决定加预算：训练 16 小时、整体 20 小时，步数不变。计划哈希变为 `2f43a97185292644b2ab2f1c91a8a9eeb7dba4dbec274200758bac910d3aa71c`；冒烟不重跑（与通过冒烟的计划只差这两个预算值）。G-BASE 资源批准尚未签署，训练未开始。
- 统一模型的 G-BASE 预评估（2026-11-10，不占 GPU，服务器代码 `fd3bc53`）：两个成员的五项证据（g_data、supervision_policy、input_policy、frozen_dev_split、training_smoke）全部 PASS。三个关卡（`skm_joint_early` 的 G-BASE-A 与 G-BASE-U、`isles_u` 的 G-BASE-U）目前都是 BLOCKED，唯一的阻断项是 `APPROVAL_MISSING:resources`。

## 真实 T07/T08 运行（2026-11-08，amax）

研究者 2026-11-08 同意：写 amax（新工作目录与私有子目录）、最多 2 张空闲 GPU、ISLES-2022 与 SKM-TEA 的数据使用（批准文件由研究者本人写入，尚未写入）、L8 划分规则（SKM 官方划分；ISLES 按病灶体积三分位分层 70/15/15，种子 20261108）。

| 项 | ISLES-2022 → `isles_u` | SKM-TEA → `skm_a` |
| --- | --- | --- |
| 来源核验 | 配对 250/250；图像–mask 几何 500/500 一致（`source_check_isles_2022.md`） | 配对 155/155；通道核验 155/155（137 例 x/y 互换，18 例另有 z 反转）（`source_check_skm_tea.md`） |
| 标签取值 | 247 通过，3 例全零 mask（保留为完整阴性） | 155 通过 |
| 迁移 | 250/250；索引与物理几何 verified | 155/155；索引 verified，物理几何 unknown（头文件无单位，毫米指标为 NA） |
| 降采样损失（候选因子） | 2×2×2：147 例共 846 个病灶连通域消失 → 建议不降采样 | 2×2×1：薄软骨与半月板共 151 个连通域消失 → 建议不降采样 |
| 冻结划分 | train 175 / development_val 38 / excluded 37 / locked_test 0（`split_isles_u.md`） | train 86 / development_val 33 / excluded 36 / locked_test 0（`split_skm_a.md`） |
| G-DATA 证据 | 6 项 PASS（`gdata_evidence_isles_2022.md`） | 6 项 PASS（`gdata_evidence_skm_tea.md`） |

- 全部 v41 记录带遗留曝光 `legacy_foundation_model_unverified`（角色 unknown），按 Proposal §3.3 阻断所有 lineage 的 locked_test，规则原本分到 locked_test 的组移到 `excluded`（blocker `LEGACY_EXPOSURE_UNKNOWN:isles_u,skm_a`）。B2 只用开发集，不受影响；G-LOCKED 之前须由研究者核实旧模型的实际曝光，并按同一规则与种子重新冻结。
- §1.6 关闭：L1–L3（`dfbef2d`），L8（规则登记于 `configs/v5/split_rule_*.yaml`，划分随真实 T08 冻结）。
- 本轮上下文、裁定与待办汇总：`reports/v5/real_run_t07_t08_report_20261108.md`。

## T10 真实运行准备与 T11 前期导入（2026-11-09）

本轮与历史分开记录。

- 本轮在 amax 上的运行（数据链复核、预测量、batch 检查）用的是提交 `2c2b7ab`。
- 收尾时把代码同步到 `3a478cb`，服务器 CPU 测试 691 passed，检出无改动，两个 smoke 计划的哈希不变。
- 没有重跑 T07/T08。

| 项 | 本轮结果 | 依据 |
| --- | --- | --- |
| 修复 3a：G-DATA 数据链守卫 | 来源策略、来源核验、冻结划分和迁移记录必须同属一条链。正确输入下证据逐字节不变；用 ISLES 输入评估 skm_a 被 `SCOPE_POLICY_MISMATCH` 拒绝 | `623f90b`；`t10_preapproval_20261109.md` §2 |
| 修复 3b：完整训练步测量 | 测量覆盖 optimizer.step、优化器状态、scaler 和调度器；OOM 按规则记录，不重试；只测前向加反向不能使候选可用 | `6fcb8fe`；`t10_preapproval_20261109.md` §3 |
| smoke 支撑 | 子集契约（`hash_rank_v1`，固定列表加哈希）；检查点写完后受控暂停；过拟合按窗口均值判定；续训对比 | `7fc47de`、`ed52709` |
| 驱动 `real_t10_v5.py` | 子命令：plan-hash、premeasure、batch-check、smoke、gdata-gate、gbase-gate（G-BASE 只读取按关卡分键的 resources 批准） | `3fabf80`、`e5742e1`、`2165233` |
| smoke 计划 | isles_u 和 skm_a 各一份，**提案，未批准**（sha256 为 `02b5c260…`、`93ce6daa…`） | `configs/v5/proposals/smoke_*.yaml` |
| 批准前运行 | 合成张量预测量（每个 scope 用 1 张 A800，候选全部 OK，都含优化器状态）；CPU 真实 batch 检查（每个 scope 读 2 个 train 样本，不注册运行） | `t10_preapproval_20261109.md` |
| 待批准材料 | 一次性提交 A：G-DATA data_use；B：smoke resources；C：smoke PASS 后的 G-BASE resources 与基线预算 | `t10_approval_package_20261109.md` |
| T11 前期（SKM） | 框和 tissue_id 导入代码，22 项合成测试；联合 scope 与来源绑定当时为提案，2026-11-10 获批准 | `0883395`；`t11_skm_joint_early_admission_20261109.md` |

- 基线驱动 `real_baseline_v5.py` 已与两份基线计划提案一起提交，见下文“基线驱动与批准 C 材料”。

### 真实 smoke（2026-11-09，amax，批准 A、B 由研究者签署后）

- 研究者签署：批准 A（G-DATA data_use）和批准 B（smoke resources）的文件由研究者写入，签名 CONGCONG，日期 2026-11-09。研究者同时授权把 ISLES-2022、SKM-TEA 用于本研究。
- G-DATA 关卡：isles_u 和 skm_a 均为 PASS（代码 `d99c981`）。
- 环境：1 张空闲 A800；Python 3.11、torch 2.5.1；启动时设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，与代码中确定性模式的设置相同。
- **第一次 isles_u（代码 `d99c981`）：失败，记录保留在私有目录。**
  - 失败位置：续训加载检查点时报 `TypeError: RNG state must be a torch.ByteTensor`。
  - 原因：检查点按运行设备加载，CPU 上的 RNG 状态被搬到了 GPU。合成测试只在 CPU 上跑，没有覆盖到。
  - 修复：`b4d9630`，加回归测试。另加 CUDA 端到端 smoke 测试（`5fea585`），在 amax GPU 上通过。
  - 计划、判定标准和哈希都没变。

| 项 | isles_u（代码 `5fea585`，重跑） | skm_a（代码 `5fea585`） |
| --- | --- | --- |
| 结果 | PASS，用时 0.057 h（预算 2 h） | PASS，用时 0.332 h（预算 2 h） |
| 按规则选中的候选 | 112×112×64，b2，fp32 | 192×192×48，b2，fp32 |
| 过拟合 | 损失下降 0.943（≥ 0.5），Dice 0.929（≥ 0.5） | 损失下降 0.584（≥ 0.5），Dice 0.607（≥ 0.6，余量小） |
| 续训对比（第 300 步暂停） | 计数器全部一致，参数逐位一致 | 计数器全部一致，参数逐位一致 |
| 全体积推理 | 通过校验，按哈希重新加载，形状同原网格 | 同左 |
| 报告 | `gpu_measurement_isles_u.md`（training_smoke sha256 `bf4ebff6cad2d0991327ca9252a2a9af53dc05de76208d27634b93b4cc507f73`） | `gpu_measurement_skm_a.md`（training_smoke sha256 `0fc50ac8ae0b2af99bba498c340214c5ad2518487c944577a617f5f5d0e64038`） |

- 读取的数据：每个 scope 只读子集规则选出的 2 个 train 样本，并登记为 run_kind `smoke` 的曝光。development_val 和 locked_test 都没有读。
- 冻结的 patch、batch 和精度：按 `first_ok_within_memory` 规则选中的候选（见上表），将作为基线计划提案的取值。
- G-BASE 预评估：两个 scope 的五项证据（g_data、supervision_policy、input_policy、frozen_dev_split、training_smoke）全部 PASS。门控为 BLOCKED，唯一阻断是 `APPROVAL_MISSING:resources`，即批准 C。smoke 批准不能打开这个门控。
- T10 的 `data_run_status` 当时记为 PARTIAL：smoke 已完成，真实基线没有运行。2026-11-10 两个基线完成后改为 PASS，见下文"真实基线"。

### 基线驱动与批准 C 材料（2026-11-09，未运行）

- 驱动 `anatobind/v41/real_baseline_v5.py`：
  - 子命令 `plan-hash`、`baseline`；
  - 随机初始化，用全部 train 训练，选择规则 `last`；
  - 在全部 development_val 上整卷推理，按患者等权汇总；locked_test 不读。
- 拒绝条件：
  - 计划必须是已通过 smoke 的配方加上规则选中的候选，其他改动一律拒绝；
  - 批准 C 必须按关卡分键并写明计划 sha256，否则在读取数据之前拒绝；
  - 训练失败或超出整体预算时，停在当前阶段并记录，不重试，不出部分结果。
- 测试：`tests/v5/test_real_baseline.py`，12 项，全部基于合成数据。
- 基线计划（**提案，未批准**）：
  - isles_u：`baseline_isles_u.yaml`，sha256 `96040004…`；
  - skm_a：`baseline_skm_a.yaml`，sha256 `3809a1ce…`。
- 待签材料：`t10_approval_c_20261109.md`，含批准文件格式、预算、资源和执行命令。
- 状态：两个基线 2026-11-10 完成，见下节。

### 真实基线（2026-11-10，amax，批准 C 签署后）

- 批准 C：`gbase_resources_20261109.yaml`，签名 CONGCONG，2026-11-09；写入前重新计算的两份计划哈希与材料一致。该文件由执行代理在研究者本次会话的明确授权下写入，属一次性例外，已在报告中记录。
- G-BASE-U、G-BASE-A 均 PASS，无阻断（代码 `c7cd0c1`）。

| 项 | isles_u | skm_a |
| --- | --- | --- |
| 结果 | COMPLETED，1.477 h（预算 5 h），A800 × 1 | COMPLETED，4.331 h（预算 8 h），A100 × 1（计划写 A800，研究者启动前同意改用） |
| 训练 | 随机初始化，train 175，30000 步，选择 last | 随机初始化，train 86，20000 步，选择 last |
| 开发集 | 38 患者；Dice 0.752，HD95 15.8 mm，NSD 0.881，病灶灵敏度 0.641，FP/scan 2.34 | 33 患者；Dice 0.831–0.887（六类） |
| 报告 | `module_baseline_isles_u.md`（sha256 `93467cb664be1eb44666f900d77bdea8993c5f5ae38d3dff26a9f906bf1d7409`） | `module_baseline_skm_a.md`（sha256 `6171e1381e78f42fc4725e9ff87f2400a217ce07bba1f1f47c86dd137a27aee5`） |

- 读 train（`training` 曝光）与 development_val（`model_selection` 曝光）；locked_test 未读。无失败、无重试、计划未改。
- 汇总：`t10_baseline_report_20261110.md`。T10 `data_run_status` 改为 PASS；科学结论仍 NOT_EVALUATED，不形成阈值。

## 阶段 A 补修（plan_v5.md §2A，外部审查后）

| 任务 | 提交 | 内容 | 测试 |
| --- | --- | --- | --- |
| FX1 | `c6427e9` | 准入证据绑定 scope_hash、来源（id、版本）与任务；测试专用批准人在非 fixture 模式下被拒 | `tests/v5/test_gates_v5.py` |
| FX2 | `f246926` | StateReference 按参照组织身份应用 A 的输入可见性（新原因位 `a_not_visible`=128） | `tests/v5/test_tissue_state_metrics.py` |
| FX3 | `fa15697` | `memory.md` 停止跟踪、加入忽略，新增 `memory.example.md` | `tests/v5/test_status_docs.py::test_memory_md_untracked_and_ignored` |
| FX4 | `4a5ae80` | D1–D4 决定登记；README、traceability、plan 同步 | `tests/v5/test_status_docs.py::test_decisions_recorded_with_separate_statuses` |

补修经独立终审（With fixes：Important 4、Minor 6）后一轮修复：G-DATA 批准另绑定来源与版本；fixture 模式的决定记录 `fixture_mode` 与 `fixture_approvals`，标记为 fixture 的证据在非 fixture 模式下被拒；补齐按组织的 A 可见性与同任务跨 scope 证据的守护测试；本节与 D2 记录更正。
