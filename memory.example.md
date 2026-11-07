# AnatoBind / FoundationModel 项目记忆（memory.example.md，脱敏示例）

本文件是 `memory.md` 的脱敏示例；真实 `memory.md` 只在本地、已被 `.gitignore` 忽略（2026-11-07 起停止 Git 跟踪，研究者决定 D2）。
其他机器上的旧克隆在 `git pull` 前先备份本机 `memory.md`，否则拉取会把它从工作区删除。
Git 历史中的旧版本未改写，另行审查（`reports/v5/status.md` blocker `PRIVATE_PATHS_IN_HISTORY:memory.md`）。

## 关键路径
- 代码仓库（本地）：`<REPO_ROOT>`
- GitHub：`sober235/FoundationModel_v1`（origin，唯一远端）
- 私有产物根（PRIVATE_ROOT）：`<PRIVATE_ROOT>`（权限 700，不在 Git 内）
- 原始数据根：`<RAW_DATA_ROOT>`
- 旧规格文件：`<LEGACY_SPEC_DIR>`
- Python：`<PYTHON_BIN>`，测试用 `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""`
- Ledger：`.superpowers/sdd/<plan>/`（被 .gitignore 排除，仅本地）

## GitHub 推送方式
- 凭据：`export GIT_ASKPASS=<ASKPASS_SCRIPT>`，或使用已登录的 `gh` CLI。
- 推送：`git -c credential.helper= -c credential.https://github.com.helper= push ...`
- 凭据脚本与 token 不进入仓库。

## 用户偏好/规则
- 中文交流，简洁直接；报告用中文。
- 对外操作（建仓库、推送）先用 AskUser 确认。
- commit 不带 AI trailer。
- 删除文件的钩子为白名单制（`~/.claude/hooks/block-delete.py`）。
- 隐私规则：仓库内不出现病例 ID/原始路径，提交前用 rg 核查。
- 签核由人填写；私有产物权限 700/600；审计产物版本化不覆盖。
