from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import yaml

TASK_IDS = ["T00", "T01a", "T01b", "T01c", "T02", "T03", "T04", "T05", "T06a", "T06b", "T06c",
            "T07", "T08", "T09a", "T09b", "T09c", "T10", "T11", "T12", "T13", "T14", "T15",
            "T16", "T17", "T18a", "T18b", "T18c"]


def _front_matter() -> dict:
    text = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---\n")[1])


def test_historical_evidence_is_not_v51_pass():
    meta = _front_matter()
    assert meta["scientific_conclusion"] == "NOT_EVALUATED"
    assert meta["core_target_validated"] is None
    assert meta["study_package_complete"] is False



def test_status_and_plan_pin_the_current_spec():
    spec_sha = hashlib.sha256(Path("Proposal_v5.md").read_bytes()).hexdigest()
    assert _front_matter()["spec_sha256"] == spec_sha
    assert f"SHA256 `{spec_sha}`" in Path("plan_v5.md").read_text(encoding="utf-8")


def test_every_task_has_five_separate_statuses():
    tasks = _front_matter()["tasks"]
    assert list(tasks) == TASK_IDS
    for row in tasks.values():
        assert set(row) == {"code_status", "test_evidence", "data_run_status",
                            "approval_status", "scientific_conclusion"}
        assert row["scientific_conclusion"] == "NOT_EVALUATED"


def test_decisions_recorded_with_separate_statuses():
    meta = _front_matter()
    assert meta["open_decisions"] == ["D5"]
    assert list(meta["decisions"]) == ["D1", "D2", "D3", "D4"]
    for row in meta["decisions"].values():
        assert set(row) == {"choice", "decided_at", "implementation_status", "run_status", "scientific_conclusion"}
    assert meta["decisions"]["D1"]["choice"] == "reference_comparison_with_h3_noninferiority"
    assert meta["decisions"]["D4"]["run_status"] == "PASS"
    assert {"PRIVATE_PATHS_IN_HISTORY:memory.md", "PRIVATE_PATHS_TRACKED:v41_docs", "BLOCKED_EXPERT_REVIEW",
            "LEGACY_EXPOSURE_UNKNOWN:isles_u,skm_a"} == set(meta["blockers"])


def test_d3_early_phase_uses_original_fm_data_labels():
    meta = _front_matter()
    d3 = meta["decisions"]["D3"]
    assert d3["choice"] == "fm_data_original_labels_skm_primary_spider_secondary"
    assert d3["decided_at"] == "2026-11-07"
    assert d3["implementation_status"] == "TODO" and d3["run_status"] == "NOT_RUN"
    assert meta["core_target_validated"] is None


def test_same_case_label_check_is_redacted_and_not_an_approval():
    report = Path("reports/v5/same_case_label_check_amax.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, report) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", report)
    assert not re.search(r"MTR_\d|\b\d+_t[12]\b", report)
    assert "不是 G-DATA 批准" in report and "不是体素级" in report


def test_early_phase_research_is_redacted_and_adopted():
    report = Path("reports/v5/early_phase_scope_research_20261107.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, report) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", report)
    assert not re.search(r"MTR_\d|\b\d+_t[12]\b", report)
    assert "研究者 2026-11-07 全部采纳" in report and "待研究者批准" not in report
    assert "不是 G-DATA 批准" in report
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "scope values adopted 2026-11-07" in status


def test_memory_md_untracked_and_ignored():
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "memory.md"], capture_output=True)
    ignored = subprocess.run(["git", "check-ignore", "-q", "memory.md"])
    assert tracked.returncode != 0 and ignored.returncode == 0
    example = Path("memory.example.md").read_text(encoding="utf-8")
    assert "## 用户偏好/规则" in example
    assert _front_matter()["decisions"]["D2"]["decided_at"] in example
    assert not re.search(r"/(Users|home|data|mnt|private|srv|opt)/", example)


PRIVATE_PATH_PATTERN = r"/data[0-9]+/|/home/[a-z]"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def test_tracked_private_paths_are_recorded():
    hits = _git("grep", "-l", "-E", PRIVATE_PATH_PATTERN, "--", ".", ":!derived").stdout.split()
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    if hits:
        assert "PRIVATE_PATHS_TRACKED:v41_docs" in _front_matter()["blockers"]
        assert all(f"`{path}`" in status for path in hits)


def test_fix_pack_rows_cite_existing_commits():
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    for task in ("FX1", "FX2", "FX3", "FX4"):
        row = next(line for line in status.splitlines() if line.startswith(f"| {task} |"))
        commits = re.findall(r"`([0-9a-f]{7})`", row)
        assert commits and all(_git("cat-file", "-e", f"{c}^{{commit}}").returncode == 0 for c in commits)


def test_stage_a_report_points_to_current_status():
    head = Path("reports/v5/stage_a_report.md").read_text(encoding="utf-8").split("\n## ")[0]
    assert "reports/v5/status.md" in head and "已被" in head


def test_host_precheck_report_is_redacted_and_not_overclaimed():
    report = Path("reports/v5/host_precheck_amax.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, report) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", report)
    d4 = _front_matter()["decisions"]["D4"]
    # 2026-11-08: the v5 suite passed on the server interpreter in an independent working directory
    assert d4["implementation_status"] == "DONE" and d4["run_status"] == "PASS"
    assert "EXECUTION_HOST_UNVERIFIED" not in _front_matter()["blockers"]
    assert "2026-11-08" in report and "585 passed" in report


def test_real_t07_t08_reports_are_redacted_and_recorded():
    names = ["source_check_isles_2022", "source_check_skm_tea", "split_isles_u", "split_skm_a",
             "gdata_evidence_isles_2022", "gdata_evidence_skm_tea", "real_run_t07_t08_report_20261108"]
    for name in names:
        text = Path(f"reports/v5/{name}.md").read_text(encoding="utf-8")
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5", text)    # no case IDs or file names
    tasks = _front_matter()["tasks"]
    for task in ("T07", "T08"):
        assert tasks[task]["data_run_status"] == "PASS" and tasks[task]["approval_status"] == "PENDING"
    sources = yaml.safe_load(Path("configs/v5/sources_v5.yaml").read_text(encoding="utf-8"))
    for source_id, report in (("isles_2022", "source_check_isles_2022"), ("skm_tea", "source_check_skm_tea")):
        digest = hashlib.sha256(Path(f"reports/v5/{report}.md").read_bytes()).hexdigest()
        assert digest in sources[source_id]["index_evidence"]


def test_t10_preapproval_and_package_are_redacted_and_bound_to_the_plans():
    from anatobind.v41.contracts_v5 import validate_scope_v5
    from anatobind.v41.real_t10_v5 import smoke_plan_sha256_v5, validate_smoke_plan_v5
    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

    texts = {name: Path(f"reports/v5/{name}.md").read_text(encoding="utf-8")
             for name in ("t10_preapproval_20261109", "t10_approval_package_20261109")}
    for text in texts.values():
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5\b", text)
        assert "NOT_EVALUATED" in text
    assert "不是 smoke 证据" in texts["t10_preapproval_20261109"]
    package = texts["t10_approval_package_20261109"]
    assert "approved_by: <研究者本人填写>" in package and "synthetic-fixture" not in package
    policies = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))
    for scope_id, source_id in (("isles_u", "isles_2022"), ("skm_a", "skm_tea")):
        scope = validate_scope_v5(yaml.safe_load(Path(f"configs/v5/scope_{scope_id}.yaml").read_text()))
        source = source_binding_v5(policies[source_id])
        plan = validate_smoke_plan_v5(yaml.safe_load(Path(f"configs/v5/proposals/smoke_{scope_id}.yaml").read_text()),
                                      scope, source)
        assert f"smoke_plan_sha256: {smoke_plan_sha256_v5(plan)}" in package
        assert f"scope_hash: {scope.scope_hash}" in package and f"source_version: {source['source_version']}" in package


def test_t11_admission_is_redacted_and_records_the_scope_approval_without_gdata():
    text = Path("reports/v5/t11_skm_joint_early_admission_20261109.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    assert not re.search(r"MTR_\d|\b\d+_t[12]\b|\.h5\b", text)
    assert "NOT_EVALUATED" in text and "core_target_validated" in text
    assert "研究者 2026-11-10 批准" in text and "**提案，未批准**" not in text
    assert "G-DATA 的 data_use 尚未签署" in text                     # approving the scope opens no data gate
    assert "读取全部三个官方划分" in text and "administrative_qc" in text
    assert "real_run_t11_skm_joint_early_20261110.md" in text and "第 7 步（冻结、G-DATA 证据、data_use 签署）还没有做" in text
    assert "SPLIT_SEALED" not in text                                 # the train/dev-only rule was replaced
    assert "CORE_RELATION_REQUIRED" in text
    assert Path("configs/v5/scope_skm_joint_early.yaml").exists()
    sources = yaml.safe_load(Path("configs/v5/sources_v5.yaml").read_text(encoding="utf-8"))
    check = Path("reports/v5/source_check_skm_tea_joint_early.md")      # the joint scope's own prepare run
    assert hashlib.sha256(check.read_bytes()).hexdigest() in sources["skm_tea_joint_early"]["index_evidence"]
    assert not re.search(r"/(Users|home|data|mnt|srv|opt)/|MTR_\d|\.nii|\.h5", check.read_text(encoding="utf-8"))
    task = _front_matter()["tasks"]["T11"]
    assert task["approval_status"] == "PENDING" and task["data_run_status"] == "PARTIAL"   # import ran; no G-DATA, no SPIDER


def test_real_joint_chain_reports_are_redacted_and_bound_to_the_scope_and_policy():
    from anatobind.v41.contracts_v5 import validate_scope_v5

    names = ["real_run_t11_skm_joint_early_20261110", "box_import_skm_joint_early", "registration_skm_joint_early",
             "source_check_skm_tea_joint_early"]
    texts = {name: Path(f"reports/v5/{name}.md").read_text(encoding="utf-8") for name in names}
    for text in texts.values():
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"MTR_\d|\b\d+_t[12]\b|\.nii|\.h5\b|skmtea:", text)       # no case IDs or file names
    report = texts["real_run_t11_skm_joint_early_20261110"]
    for name in names[1:]:                                             # the report cites each summary by its hash
        assert hashlib.sha256(Path(f"reports/v5/{name}.md").read_bytes()).hexdigest() in report
    scope = validate_scope_v5(yaml.safe_load(Path("configs/v5/scope_skm_joint_early.yaml").read_text()))
    box = texts["box_import_skm_joint_early"]
    assert scope.scope_hash in box and scope.scope_hash in report
    assert "登记扫描 155，导入 155，失败 0；框总数 476" in box and "| primary | 222 |" in box
    assert "NOT_EVALUATED" in box and "NOT_EVALUATED" in report
    assert "正式 split：unassigned 155" in texts["registration_skm_joint_early"]
    assert "没有冻结正式划分，没有产生联合 G-DATA 证据，没有签署 data_use" in report
    assert "administrative_qc" in report and "CORE_RELATION_REQUIRED" in report


def test_real_smoke_reports_are_redacted_and_bound_to_their_evidence():
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    for scope_id in ("isles_u", "skm_a"):
        text = Path(f"reports/v5/gpu_measurement_{scope_id}.md").read_text(encoding="utf-8")
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5\b|skmtea:|isles2022:", text)
        assert "结果：PASS" in text and "optimizer.step" in text and "NOT_EVALUATED" in text
        digest = hashlib.sha256(Path(f"reports/v5/gpu_measurement_{scope_id}.md").read_bytes()).hexdigest()
        assert f"`gpu_measurement_{scope_id}.md`（training_smoke sha256 `{digest}`）" in status
    tasks = _front_matter()["tasks"]
    assert tasks["T10"]["data_run_status"] == "PASS"               # 2026-11-10: smokes and both real baselines done


def test_real_baseline_reports_are_redacted_and_bound_to_status():
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    report = Path("reports/v5/t10_baseline_report_20261110.md").read_text(encoding="utf-8")
    for text in (report, *(Path(f"reports/v5/module_baseline_{s}.md").read_text(encoding="utf-8")
                           for s in ("isles_u", "skm_a"))):
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5\b|skmtea:|isles2022:", text)
        assert "NOT_EVALUATED" in text and "locked_test" in text
    for scope_id in ("isles_u", "skm_a"):
        module = Path(f"reports/v5/module_baseline_{scope_id}.md").read_text(encoding="utf-8")
        assert "结果：COMPLETED" in module and "development_val" in module and "不相加" in module
        digest = hashlib.sha256(Path(f"reports/v5/module_baseline_{scope_id}.md").read_bytes()).hexdigest()
        assert f"`module_baseline_{scope_id}.md`（sha256 `{digest}`）" in status
        assert f"`module_baseline_{scope_id}.md`（sha256 `{digest}`）" in report
    assert "A100" in report and "不需要重签" in report               # the recorded device deviation for skm_a
    tasks = _front_matter()["tasks"]
    assert tasks["T10"]["data_run_status"] == "PASS" and tasks["T10"]["approval_status"] == "PASS"
    assert _front_matter()["core_target_validated"] is None


def test_t10_approval_c_package_is_redacted_and_names_the_baseline_plans():
    from anatobind.v41.contracts_v5 import validate_scope_v5
    from anatobind.v41.real_baseline_v5 import baseline_plan_sha256_v5, validate_baseline_plan_v5
    from anatobind.v41.source_policy_v5 import load_source_policies_v5, source_binding_v5

    package = Path("reports/v5/t10_approval_c_20261109.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, package) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", package)
    assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5\b|skmtea:|isles2022:", package)
    assert "approved_by: <研究者本人填写>" in package and "synthetic-fixture" not in package
    assert "NOT_EVALUATED" in package and "locked_test" in package
    policies = load_source_policies_v5(Path("configs/v5/sources_v5.yaml"))
    for scope_id, source_id in (("isles_u", "isles_2022"), ("skm_a", "skm_tea")):
        scope = validate_scope_v5(yaml.safe_load(Path(f"configs/v5/scope_{scope_id}.yaml").read_text()))
        smoke = yaml.safe_load(Path(f"configs/v5/proposals/smoke_{scope_id}.yaml").read_text())
        plan = validate_baseline_plan_v5(
            yaml.safe_load(Path(f"configs/v5/proposals/baseline_{scope_id}.yaml").read_text()), scope,
            source_binding_v5(policies[source_id]), smoke)
        report = hashlib.sha256(Path(f"reports/v5/gpu_measurement_{scope_id}.md").read_bytes()).hexdigest()
        assert plan["smoke"]["training_smoke_sha256"] == report        # the smoke the researcher saw
        assert f"baseline_plan_sha256: {baseline_plan_sha256_v5(plan)}" in package
        assert f"scope_hash: {scope.scope_hash}" in package


def test_import_registration_report_is_redacted_and_separates_code_from_real_data_checks():
    text = Path("reports/v5/import_registration_report_20261110.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5\b|skmtea:|isles2022:", text)
    assert "NOT_EVALUATED" in text and "代码验证（合成数据）" in text and "真实数据验证（amax，只读" in text
    assert "没有冻结正式划分" in text and "LEGACY_EXPOSURE_UNKNOWN" in text and "尚未实现的访问控制" in text
    assert "| test | 是 | 是 | 36 | 36 | 0 | 130 |" in text
    for name in ("test_registration_report.py", "test_real_t11.py"):
        assert Path(f"tests/v5/{name}").exists() and name in text


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


def test_unified_local_report_separates_code_checks_from_real_runs():
    text = Path("reports/v5/unified_local_report_20261110.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    for phrase in ("只做了合成数据验证", "没有在 amax 上运行", "NOT_EVALUATED", "提案，未批准",
                   "unified_knee_brain.yaml", "定位点用质心体素"):
        assert phrase in text
    for name in ("test_box_data.py", "test_box_loss.py", "test_unified_model.py", "test_unified_registration.py",
                 "test_unified_training.py", "test_findings.py", "test_findings_eval.py", "test_real_unified.py"):
        assert Path(f"tests/v5/{name}").exists() and name in text


def test_joint_split_and_gdata_evidence_are_redacted_and_bound():
    import hashlib
    split = Path("reports/v5/split_skm_joint_early.md").read_text(encoding="utf-8")
    evidence = Path("reports/v5/gdata_evidence_skm_tea_joint_early.md").read_bytes()
    for text in (split, evidence.decode("utf-8")):
        assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
        assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5", text)    # no case IDs or file names
    digest = "32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9"
    assert "v51_skm_joint_early" in split and digest in split and digest in evidence.decode("utf-8")
    for row in ("| development_val | 33 |", "| excluded | 36 |", "| locked_test | 0 |", "| train | 86 |"):
        assert row in split
    assert evidence.decode("utf-8").count("| PASS |") == 6 and "FAIL" not in evidence.decode("utf-8")
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert hashlib.sha256(evidence).hexdigest() in status and "由执行方按研究者给出的署名代为写入" in status
    assert "统一冒烟第一次运行" in status and "**FAIL**" in status and "没有产生冒烟证据" in status


def test_unified_smoke_report_is_redacted_and_recorded():
    text = Path("reports/v5/unified_smoke_20261110.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    assert not re.search(r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5", text)
    plan_hash = "50ab3e77ebc243f15583f5a6032d133ed5ca8d535383761bec65c2c2f9d1e4c1"
    assert plan_hash in text and "结果：PASS" in text and "NOT_EVALUATED" in text
    assert "参数逐位相同：是" in text and "development_val 与 locked_test 未读" in text
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "统一冒烟第二次运行" in status and plan_hash in status and "训练未开始" in status



UNIFIED_REPORT_SHA256 = "3a3d312ffbd98fc2992967ce7b79fa1489eeeeffbc5dc8d0e34ea3e1c3b536fe"
UNIFIED_PLAN_SHA256 = "2f43a97185292644b2ab2f1c91a8a9eeb7dba4dbec274200758bac910d3aa71c"
UNIFIED_CHECKPOINT_SHA256 = "5b1e8c6d14ca32f2a538bf43d04b61cdece62262fb2c34fadaa3b12466c72e35"
_CASE_OR_FILE = r"sub-[a-z]+[0-9]|[A-Z]{3}_[0-9]{3}|\.nii|\.h5"


def _unified_training_section() -> str:
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "## 统一训练结果（2026-11-11）" in status
    return status.split("## 统一训练结果（2026-11-11）", 1)[1].split("\n## ", 1)[0]


def test_unified_training_report_is_archived_byte_for_byte():
    data = Path("reports/v5/unified_training_20261111.md").read_bytes()
    assert hashlib.sha256(data).hexdigest() == UNIFIED_REPORT_SHA256           # the server's report, unchanged
    text = data.decode("utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, text) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", text)
    assert not re.search(_CASE_OR_FILE, text)
    for phrase in (UNIFIED_PLAN_SHA256, UNIFIED_CHECKPOINT_SHA256, "结果：COMPLETED",
                   "scientific_conclusion: NOT_EVALUATED", "locked_test 未读"):
        assert phrase in text


def test_unified_training_is_recorded_as_a_completed_run_not_as_validated_performance():
    section = _unified_training_section()
    for phrase in ("unified_training_20261111.md", UNIFIED_REPORT_SHA256, UNIFIED_PLAN_SHA256,
                   UNIFIED_CHECKPOINT_SHA256, "17b4656736d9e0ad2ecf0022a6e7abaf0e2370e4",
                   "运行完成，不是性能验证通过", "NOT_EVALUATED", "locked_test"):
        assert phrase in section
    for row in ("| 脑 | 梗死 Dice | 0.756 | 0.752 | +0.004 |", "| 脑 | 病灶检出率 | 0.638 | 0.641 | -0.002 |",
                "| 膝 | 半月板（外侧）Dice | 0.815 | 0.831 | -0.016 |", "| 合计 | 53 | 11 | 0.208 | 9 | 0.170 |"):
        assert row in section                                    # the original development values, unchanged
    meta = _front_matter()
    assert meta["scientific_conclusion"] == "NOT_EVALUATED" and meta["core_target_validated"] is None


def test_the_outside_count_of_the_original_report_is_recorded_with_its_real_definition():
    section = _unified_training_section()
    assert "1034" in section and "异类主分析框" in section and "不能读作" in section
    summary = Path("reports/v5/context_summary_20261111.md").read_text(encoding="utf-8")
    assert "更正（2026-11-11" in summary and "异类主分析框" in summary


def test_the_unified_training_section_carries_no_time_of_the_server():
    """Rule of 2026-11-12: times come from the writing machine's clock only; nothing of the server's clock."""
    section = _unified_training_section()
    assert "CST" not in section and "amax 时间" not in section
    assert not re.search(r"amax\s*20\d\d-\d\d-\d\d", section)


UNIFIED_EVALUATION_SERVER_SHA256 = "a14d21f814b985aa2abca2b6a374a2afcf36ea26a1421260c3b74b6c9096c49d"
UNIFIED_EVALUATION_COPY_SHA256 = "338045de5fcdc53355d0848c9485e3cce1f7882236c41489c3b577d56288c5ae"
UNIFIED_REPLAY_SERVER_SHA256 = "75cf09da0fbc4bd7ab17501d3f78a797746abafffc12b72c55a2c79a5298aa2d"
UNIFIED_REPLAY_COPY_SHA256 = "4a5ab680e0b9577131f54fc7de2e5dc2b09cf4b3e5d5046bf34c17d6cb4016ea"
UNIFIED_EVALUATION_CODE = "cfd88c6b378693337592b2e9d9318418b04e11c6"
DURATION_MARK = "用时 [服务器端计时，未取回]"
_REDACTED = (PRIVATE_PATH_PATTERN, r"/(Users|home|data|mnt|srv|opt)/", _CASE_OR_FILE)


def _unified_diagnostics_section() -> str:
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "## 统一模型诊断（2026-11-12）" in status
    return status.split("## 统一模型诊断（2026-11-12）", 1)[1].split("\n## ", 1)[0]


def test_evaluation_and_replay_reports_are_archived_with_only_the_duration_replaced():
    """The server's own measured duration is replaced before the report leaves the server (rule of 2026-11-12)."""
    for name, digest, phrases in (
            ("unified_evaluation_20261112.md", UNIFIED_EVALUATION_COPY_SHA256,
             ("读取的划分角色：train、development_val；locked_test 未读", "不能代替泛化评价",
              "| 命中的框 | 24 | 11 |", "| 有保留的连通块触及框，但定位点在框外 | 82 | 42 |")),
            ("unified_replay_20261112.md", UNIFIED_REPLAY_COPY_SHA256,
             ("逐步核对", "共 60000 项，不一致 0 项", "没有记录", "从未出现的袋 0 个"))):
        data = Path("reports/v5", name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        text = data.decode("utf-8")
        assert text.count(DURATION_MARK) == 1 and not re.search(r"用时 [0-9]", text)
        assert not any(re.search(pattern, text) for pattern in _REDACTED)
        for phrase in (UNIFIED_PLAN_SHA256, UNIFIED_CHECKPOINT_SHA256, UNIFIED_EVALUATION_CODE, "结果：COMPLETED",
                       "scientific_conclusion: NOT_EVALUATED", "权重未被修改", *phrases):
            assert phrase in text


def test_unified_diagnostics_are_recorded_as_diagnostics_of_the_fixed_checkpoint():
    section = _unified_diagnostics_section()
    for phrase in ("unified_evaluation_20261112.md", "unified_replay_20261112.md", "unified_diagnostics_20261112.md",
                   UNIFIED_EVALUATION_SERVER_SHA256, UNIFIED_EVALUATION_COPY_SHA256, UNIFIED_REPLAY_SERVER_SHA256,
                   UNIFIED_REPLAY_COPY_SHA256, UNIFIED_CHECKPOINT_SHA256, UNIFIED_EVALUATION_CODE,
                   "权重未被修改", "locked_test 未读", "NOT_EVALUATED", "不能代替泛化评价", "不是性能验证",
                   "没有重训", "计划取值没有改动", "没有推送"):
        assert phrase in section
    for row in ("| 主分析框 | 106 | 53 |", "| 命中的框 | 24 | 11 |", "| 框命中率（合并） | 0.226 | 0.208 |",
                "| 有保留的连通块触及框，但定位点在框外 | 82 | 42 |"):
        assert row in section
    meta = _front_matter()
    assert meta["scientific_conclusion"] == "NOT_EVALUATED" and meta["core_target_validated"] is None


def test_the_four_places_of_the_findings_split_the_original_counts():
    section = _unified_diagnostics_section()
    places = {label: (int(train), int(dev))
              for label, train, dev in re.findall(r"\| (定位点在[^|]+?) \| (\d+) \| (\d+) \|", section)}
    assert set(places) == {"定位点在同类主分析框内", "定位点在忽略区内", "定位点在异类主分析框内", "定位点在所有参照框之外"}
    assert tuple(map(sum, zip(*places.values()))) == (2847, 1148)                 # every finding in exactly one place
    assert places["定位点在同类主分析框内"][1] == 11 and places["定位点在忽略区内"][1] == 103       # the original counts
    assert places["定位点在异类主分析框内"][1] + places["定位点在所有参照框之外"][1] == 1034         # the original `outside`
    for phrase in ("同类主分析框 → 忽略区 → 异类主分析框 → 所有参照框之外", "不是误报"):
        assert phrase in section


def test_the_diagnostic_report_separates_facts_explanations_and_hypotheses():
    text = Path("reports/v5/unified_diagnostics_20261112.md").read_text(encoding="utf-8")
    for heading in ("已确定的事实", "证据支持的解释", "仍待检验的假设", "各个可能原因的证据强度", "最小改动"):
        assert re.search(rf"^#+ .*{heading}", text, flags=re.M), heading
    for phrase in ("unified_training_20261111.md", UNIFIED_REPORT_SHA256, UNIFIED_EVALUATION_SERVER_SHA256,
                   UNIFIED_REPLAY_SERVER_SHA256, UNIFIED_CHECKPOINT_SHA256, "NOT_EVALUATED", "不能代替泛化评价",
                   "不是误报", "缺失", "仅诊断", "未执行", "locked_test 未读"):
        assert phrase in text
    assert not any(re.search(pattern, text) for pattern in _REDACTED)
    assert "CST" not in text and "amax 时间" not in text


def test_the_supplementary_aggregates_carry_counts_only():
    import json

    folder = Path("reports/v5/unified_diagnostics_20261112")
    report = Path("reports/v5/unified_diagnostics_20261112.md").read_text(encoding="utf-8")
    for name in ("rows_aggregate.json", "rows_by_boxes.json", "rows_aggregate.py", "rows_by_boxes.py"):
        data = (folder / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() in report                      # the report names what it used
        text = data.decode("utf-8")
        assert not any(re.search(pattern, text) for pattern in _REDACTED)
        if name.endswith(".json"):
            body = json.loads(text)
            assert set(body) == {"train", "development_val"}                   # locked_test is not in it
            assert "patient_uid" not in text and "sample_uid" not in text and "finding_uid" not in text
    by_boxes = json.loads((folder / "rows_by_boxes.json").read_text(encoding="utf-8"))
    assert by_boxes["train"]["n_scans"] == 86 and by_boxes["development_val"]["n_scans"] == 33
    assert by_boxes["train"]["scans_without_any_box"]["n"] == 18


def test_status_carries_no_clock_of_the_server():
    """Rule of 2026-11-12, now for the whole file."""
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    assert "CST" not in status and "amax 时间" not in status
    assert not re.search(r"amax\s*20\d\d-\d\d-\d\d", status)


# ---------------------------------------------------------------------------
# Negative top-k pilot (researcher's option B, 2026-11-12)

PILOT_PLAN_SHA256 = "b7e9524156d61e8df3ab58c0f4e91ed429cd181ca5d914c0eb1ad3a2cc105c69"
PILOT_FOLDER = Path("reports/v5/negtopk_pilot_20261112")
PILOT_REPORT = Path("reports/v5/negtopk_pilot_20261112.md")
PILOT_CHECKPOINTS = {
    "old": ("1a2566d377b71707f0843f940a91f86e107bf347f6c0cb8fb30db0c6622b99a3",
            "c2efa91781cde65e5693ba26bec5613bf35523a682708e988b9e88da1ba52f5d",
            "f992c12d3560a2cc2d1dd119d0af75cae717ea269ea71dc4c0d8849ad436649f"),
    "new": ("d1aaae75bc8d1697ed66cc4242fb570abee1388e21fd4071b5477070f1a9a3be",
            "ada6a284656a231fd9297e2df61764362f771aaa01eedda47007ec8ddfde6c65",
            "972cfcbde182bfdfb806f5b2dee403f0d7bbce6ffee5ede0480dc4d4a8e406a3")}
PILOT_APPROVALS = ("92ee8558a91f420067448af9fe7b97eb18419ce5ffda17e079215a326861735e",
                   "7b676a8e590c5d038e67c7c8dc180c50def24412b15b25bb2895ea46fe3f7bab")
# name: (sha256 of the copy in the repository, sha256 of the untouched server file, figures of the server's timing
# replaced before the copy left the server)
PILOT_ARCHIVE = {
    "smoke.md": ("6accba44a78303d78381b33f1427b6ed03e955d76d000e1ab0fef0c8fb3460b3",
                 "3d3f3c19575a2e717e541eea93d6f14b5835e853380933e5da1f8e13503e2754", 2),
    "probe_before_training.md": ("cc67eae7fb01e2022df257b957b74c2840dc080c4a4176846e159c430b053039",
                                 "cc67eae7fb01e2022df257b957b74c2840dc080c4a4176846e159c430b053039", 0),
    "probe_new_2000.md": ("d2502d6fef599562a346e8e271ff360cb329ccbfecb762fa602d768ac4f70a6b",
                          "d2502d6fef599562a346e8e271ff360cb329ccbfecb762fa602d768ac4f70a6b", 0),
    "probe_new_4000.md": ("10a51ab0677fcf12c32c072eccb6795121867b1b1e53c113fdc7a35d4f62539d",
                          "10a51ab0677fcf12c32c072eccb6795121867b1b1e53c113fdc7a35d4f62539d", 0),
    "probe_new_6000.md": ("b29ee0468e7f79e86146ea5b26652b2f57f147f3e845bea186fc8efdc3a663c0",
                          "b29ee0468e7f79e86146ea5b26652b2f57f147f3e845bea186fc8efdc3a663c0", 0),
    "comparison.md": ("06c8df71c5a8334e8d970703ac78ab0e722e3c5b2b8cf154614eb639ba30dc04",
                      "06c8df71c5a8334e8d970703ac78ab0e722e3c5b2b8cf154614eb639ba30dc04", 0),
    "training.md": ("94e1120b9357b15152e464af14551665c5fd25c3eda113940759d2a1b534eca9",
                    "90c6f545322fa1fcb866ebb84e17c88eda15840435f3b85dccde7ebc0ef5d83d", 1),
    "evaluation_old_6000.md": ("c13731b3b98567aa22eda930b535aebbc4fa0d2cc1b15b9bfc60239963866cd1",
                               "8623568d5fc1093c751c0df02c1e1c109c662f1961d23a3a72a3bf01cd4b9110", 2),
    "evaluation_new_6000.md": ("a6b78ea094132f15f79a58704a9b0cf89b5c01629ebdbedc4add28f805922e3d",
                               "b963e0bf016a415bde14ed0eb5410c6a8f4eba88211a71e7b2454246f2b98c91", 2),
    "replay.md": ("092720bb94ad4659c135e8c02ad9797951a9e2838370cbeddcd603f4886350df",
                  "a671bed1611527332940bb2a436a718a10f0cc41c5a5c8f41619b43c47361f08", 2),
}
_SERVER_TIMING = r"用时 [0-9]|剩余 [0-9]|单步耗时：[0-9]"


def _pilot_section() -> str:
    status = Path("reports/v5/status.md").read_text(encoding="utf-8")
    heading = "## 负项短程试验（2026-11-12 至 11-13）"
    assert heading in status
    return status.split(heading, 1)[1].split("\n## ", 1)[0]


def test_pilot_reports_are_archived_with_only_the_server_timing_replaced():
    report = PILOT_REPORT.read_text(encoding="utf-8")
    for name, (copy, server, replaced) in PILOT_ARCHIVE.items():
        data = (PILOT_FOLDER / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == copy, name
        assert (copy == server) == (replaced == 0), name
        text = data.decode("utf-8")
        assert text.count("[服务器端计时，未取回]") == replaced and not re.search(_SERVER_TIMING, text), name
        assert not any(re.search(pattern, text) for pattern in _REDACTED), name
        assert "NOT_EVALUATED" in text, name
        assert copy in report and server in report, name                   # the report names both files


def test_pilot_report_is_a_record_of_a_short_pilot_not_a_performance_result():
    text = PILOT_REPORT.read_text(encoding="utf-8")
    for phrase in (PILOT_PLAN_SHA256, UNIFIED_PLAN_SHA256, UNIFIED_CHECKPOINT_SHA256, *PILOT_CHECKPOINTS["old"],
                   *PILOT_CHECKPOINTS["new"], *PILOT_APPROVALS, "NOT_EVALUATED", "core_target_validated",
                   "locked_test 未读", "不能代替泛化评价", "不是误报", "没有推送", "本机墙钟时间"):
        assert phrase in text, phrase
    for heading in ("已确定的事实", "证据支持的解释", "仍待检验的假设", "建议", "预算"):
        assert re.search(rf"^#+ .*{heading}", text, flags=re.M), heading
    assert "（待" not in text and "待填写" not in text                      # no section is left open
    assert not any(re.search(pattern, text) for pattern in _REDACTED)
    assert "CST" not in text and "amax 时间" not in text


def test_both_step_6000_evaluations_read_fresh_predictions_of_the_named_checkpoints():
    for name, plan, checkpoint in (("evaluation_old_6000.md", UNIFIED_PLAN_SHA256, PILOT_CHECKPOINTS["old"][2]),
                                   ("evaluation_new_6000.md", PILOT_PLAN_SHA256, PILOT_CHECKPOINTS["new"][2])):
        text = (PILOT_FOLDER / name).read_text(encoding="utf-8")
        for phrase in (plan, checkpoint, "（第 6000 步）", "权重未被修改", "结果：COMPLETED", "locked_test 未读",
                       "没有复用任何已保存的输出", "实例掩膜的内存上限 256 GiB", "不是评价规则", "不能代替泛化评价"):
            assert phrase in text, (name, phrase)


def test_the_step_6000_tables_are_derived_from_the_two_evaluations():
    import json

    values = json.loads((PILOT_FOLDER / "comparison_6000.json").read_text(encoding="utf-8"))
    assert values["checkpoints"]["old"]["sha256"] == PILOT_CHECKPOINTS["old"][2]
    assert values["checkpoints"]["new"]["sha256"] == PILOT_CHECKPOINTS["new"][2]
    tables = (PILOT_FOLDER / "comparison_6000_tables.md").read_text(encoding="utf-8")
    report = PILOT_REPORT.read_text(encoding="utf-8")
    for name in ("comparison_6000.json", "comparison_6000_tables.md", "compare_pilot.py", "rows_by_boxes_6000.json"):
        assert hashlib.sha256((PILOT_FOLDER / name).read_bytes()).hexdigest() in report, name
    for row in ("| 命中的框（计数） | 67 | 0 |", "| 命中的框（计数） | 28 | 0 |",
                "| 合计：有响应，但连通块都小于体素数下限 | 0 | 101 |", "| 合计：有响应，但连通块都小于体素数下限 | 0 | 49 |",
                "| 所见总数 | 54750 | 0 |", "| 所见总数 | 20391 | 0 |"):
        assert row in tables, row
    by_boxes = json.loads((PILOT_FOLDER / "rows_by_boxes_6000.json").read_text(encoding="utf-8"))
    assert set(by_boxes) == {"old_6000", "new_6000"}
    for side in by_boxes.values():
        assert set(side) == {"train", "development_val"}                       # locked_test is not in it
        assert side["train"]["n_scans"] == 86 and side["development_val"]["n_scans"] == 33
        assert side["train"]["scans_without_any_box"]["n"] == 18
    text = json.dumps(by_boxes)
    assert "uid" not in text and not any(re.search(pattern, text) for pattern in _REDACTED)


def test_the_replay_of_the_pilot_matches_its_log_step_by_step():
    text = (PILOT_FOLDER / "replay.md").read_text(encoding="utf-8")
    for phrase in (PILOT_PLAN_SHA256, PILOT_CHECKPOINTS["new"][2], "权重未被修改", "结果：COMPLETED",
                   "共 12000 项，不一致 0 项", "共 36000 项，不一致 0 项", "从未出现的袋 0 个", "locked_test 未读"):
        assert phrase in text, phrase
    report = PILOT_REPORT.read_text(encoding="utf-8")
    assert "比较 12000 项，不一致 0 项" in report and "比较 36000 项，不一致 0 项" in report
    assert "**13.004**" in report                                                  # the budget, all jobs added


def test_pilot_log_terms_carry_counts_and_means_only():
    import json

    data = (PILOT_FOLDER / "log_terms.json").read_bytes()
    report = PILOT_REPORT.read_text(encoding="utf-8")
    assert hashlib.sha256(data).hexdigest() in report
    assert hashlib.sha256((PILOT_FOLDER / "log_terms.py").read_bytes()).hexdigest() in report
    body = json.loads(data)
    assert body["steps"] == 6000 and body["window"] == 500 and len(body["windows"]) == 12
    assert body["windows"][0]["steps"] == [1, 500] and body["windows"][-1]["steps"] == [5501, 6000]
    text = data.decode("utf-8")
    assert "uid" not in text and not any(re.search(pattern, text) for pattern in _REDACTED)


def test_status_records_the_pilot_as_a_completed_run_not_as_validated_performance():
    section = _pilot_section()
    for phrase in ("negtopk_pilot_20261112.md", PILOT_PLAN_SHA256, PILOT_CHECKPOINTS["new"][2], *PILOT_APPROVALS,
                   "代码核对", "实验完成", "性能结论", "NOT_EVALUATED", "locked_test 未读", "没有推送",
                   "没有接着跑 30000 步", "INSTANCE_LIMIT_EXCEEDED"):
        assert phrase in section, phrase
    meta = _front_matter()
    assert meta["scientific_conclusion"] == "NOT_EVALUATED" and meta["core_target_validated"] is None
    assert "CST" not in section and "amax 时间" not in section
