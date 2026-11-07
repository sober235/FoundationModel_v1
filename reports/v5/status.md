---
plan_revision: "2026-11-07"
execution_base: "GitHub sober235/FoundationModel_v1 current content (branch restart-dev); code identical to d7cbd5b"
spec_sha256: "c4657d7ba2b335db9a57e3a24c1d7a938e3054ee81e2e77ef8979a0c01e4efa7"
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
    implementation_status: TODO   # T11 import of original labels; scope fields and G-DATA (SKM-TEA, SPIDER) pending
    run_status: NOT_RUN
    scientific_conclusion: NOT_EVALUATED
  D4:
    choice: server_amax
    decided_at: "2026-11-06"
    implementation_status: PARTIAL   # read-only precheck done (reports/v5/host_precheck_amax.md); v5 tests not yet run on the host
    run_status: NOT_RUN
    scientific_conclusion: NOT_EVALUATED
blockers: ["PRIVATE_PATHS_IN_HISTORY:memory.md", "PRIVATE_PATHS_TRACKED:v41_docs", "BLOCKED_EXPERT_REVIEW",
           "EXECUTION_HOST_UNVERIFIED"]
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
    test_evidence: "tests/v5/test_source_policy.py: 9 passed; test_migrate_v41.py: 20 passed; test_geometry_v5.py: 11 passed; test_source_check.py: 9 passed"
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T08:
    code_status: PASS
    test_evidence: "tests/v5/test_exposure_v5.py: 7 passed; test_splits_v5.py: 11 passed"
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T09a:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T09b:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T09c:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T10:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T11:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
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
  - 实现状态 `TODO`：T11 原标签导入未做；scope 字段待研究者填写；SKM-TEA、SPIDER 的 G-DATA 未批准。
- **D3 原决定（2026-11-06，留作后期方案）：** 保留 SKM 联合场景；先准备 20–30 次非锁定开发扫描试点，明确主读、独立复读、仲裁责任，以及异常边界、不确定带与单层可判读规则。人员与预算未落实前 X0 不标完成，相关科学验证保持 `BLOCKED_EXPERT_REVIEW`。框、分级或伪标签不能替代原定联合验证；试点不能形成稳定体素参照时，再按实际资源提出替代场景。
- **D4 执行主机：** 选定 amax 数据服务器。2026-11-06 已完成只读预检（`reports/v5/host_precheck_amax.md`）：硬件与 ISLES/SKM 数据就位；服务器检出仍是旧 `main`，v5 测试尚未在其 Python 3.11 / NumPy 2.4 / SciPy 1.17 上运行，故仍未登记为正式执行环境（blocker `EXECUTION_HOST_UNVERIFIED`）。在服务器解释器上跑通 v5 测试后才登记为正式执行环境；解释器路径以实测为准，真实路径只进本地配置。
- **D5 长期目标定位（2026-11-07 提出，待决定）：** v5.1 是否作为“可迁移 MRI 基础模型”长期目标中的一个能力验证阶段；若是，是否在 v5.1 之后另立复用/迁移/泛化里程碑（Proposal §1.4，plan §1.5）。不改变 v5.1 的任务与完成判定。

## 阶段 A 补修（plan_v5.md §2A，外部审查后）

| 任务 | 提交 | 内容 | 测试 |
| --- | --- | --- | --- |
| FX1 | `c6427e9` | 准入证据绑定 scope_hash、来源（id、版本）与任务；测试专用批准人在非 fixture 模式下被拒 | `tests/v5/test_gates_v5.py` |
| FX2 | `f246926` | StateReference 按参照组织身份应用 A 的输入可见性（新原因位 `a_not_visible`=128） | `tests/v5/test_tissue_state_metrics.py` |
| FX3 | `fa15697` | `memory.md` 停止跟踪、加入忽略，新增 `memory.example.md` | `tests/v5/test_status_docs.py::test_memory_md_untracked_and_ignored` |
| FX4 | `4a5ae80` | D1–D4 决定登记；README、traceability、plan 同步 | `tests/v5/test_status_docs.py::test_decisions_recorded_with_separate_statuses` |

补修经独立终审（With fixes：Important 4、Minor 6）后一轮修复：G-DATA 批准另绑定来源与版本；fixture 模式的决定记录 `fixture_mode` 与 `fixture_approvals`，标记为 fixture 的证据在非 fixture 模式下被拒；补齐按组织的 A 可见性与同任务跨 scope 证据的守护测试；本节与 D2 记录更正。
