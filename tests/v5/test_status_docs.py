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
    assert meta["decisions"]["D4"]["run_status"] == "NOT_RUN"
    assert {"PRIVATE_PATHS_IN_HISTORY:memory.md", "PRIVATE_PATHS_TRACKED:v41_docs", "BLOCKED_EXPERT_REVIEW",
            "EXECUTION_HOST_UNVERIFIED"} == set(meta["blockers"])


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
    report = Path("reports/v5/host_precheck_148.md").read_text(encoding="utf-8")
    assert not re.search(PRIVATE_PATH_PATTERN, report) and not re.search(r"/(Users|home|data|mnt|srv|opt)/", report)
    d4 = _front_matter()["decisions"]["D4"]
    assert d4["implementation_status"] == "PARTIAL" and d4["run_status"] == "NOT_RUN"
    assert "EXECUTION_HOST_UNVERIFIED" in _front_matter()["blockers"]
