---
plan_revision: "2026-11-06"
execution_base: "GitHub sober235/FoundationModel_v1 current content (branch restart-dev); code identical to d7cbd5b"
spec_sha256: "b25b0e1beae512320b29beeba2b0eac367cf62e94304d70c28100da1fb006dbf"
scientific_conclusion: NOT_EVALUATED
core_target_validated: null
study_package_complete: false
historical_evidence:
  v41_local_tests: "144 passed, Python 3.9.6, 2026-11-06"
  v41_engineering_execution_1094_tests: "historical; code not in this repository"
  skm_a_200step_trial: "historical; code not in this repository"
open_decisions: [D1, D2, D3, D4]
blockers: ["PRIVATE_PATHS_TRACKED:memory.md", "BLOCKED_EXPERT_REVIEW"]
tasks:
  T00:
    code_status: PASS
    test_evidence: "tests/v5/test_status_docs.py: 3 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01a:
    code_status: PASS
    test_evidence: "tests/v5/test_contracts_scope.py: 16 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01b:
    code_status: PASS
    test_evidence: "tests/v5/test_contracts_outputs.py: 13 passed"
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T01c:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T02:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T03:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T04:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T05:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06a:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06b:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T06c:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_APPLICABLE
    approval_status: NOT_REQUIRED
    scientific_conclusion: NOT_EVALUATED
  T07:
    code_status: TODO
    test_evidence: null
    data_run_status: NOT_STARTED
    approval_status: PENDING
    scientific_conclusion: NOT_EVALUATED
  T08:
    code_status: TODO
    test_evidence: null
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

## 待研究者决定事项

D1 能力标准（T17）、D2 `memory.md` 私有路径被跟踪、D3 专家资源与替代场景、D4 真实数据执行主机。详见 plan_v5.md §1.3。
