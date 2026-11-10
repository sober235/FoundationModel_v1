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
