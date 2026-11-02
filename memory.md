# AnatoBind / FoundationModel 项目记忆（memory.md）

更新：2026-11-02（v2 删除、spec.md 撰写后）。来源：会话 a67fcfa3（接续 e8368e04）。
本文件含私有路径，不得提交到 Git。

## 关键路径
- 代码仓库（本地）：`/data0/congcong/code/Project_Doing/FoundationModel_v1`
- GitHub：`sober235/FoundationModel_v1`（私有，origin，唯一远端，后续以 v1 为准）。`sober235/FoundationModel_v2` 已于 2026-11-02 由用户在网页删除（API 核实 404），本地 `v2` remote 已移除。
- 私有产物根（PRIVATE_ROOT）：`/data2/congcong/code/FM_data`（权限 700，不在 Git 内）
- 原始数据根：`/data2/congcong/data/FM_data`
- 旧规格文件：`/data0/congcong/code/Project_Doing/VFM_v1/VFM_v2/`（Proposal、Contract、plan.md、FM_data_inventory.md）
- Python：`/home/congcongliu/anaconda3/envs/nvgen/bin/python`，测试用 `PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""`
- Ledger：`.superpowers/sdd/plan/progress.md`（被 .gitignore 排除，仅本地）

## GitHub 推送方式（gh token 已失效）
- `export GIT_ASKPASS=/data2/congcong/code/FM_data/.git-askpass-sober235.sh`
- `git -c credential.helper= -c credential.https://github.com.helper= push ...`
- TLS 间歇性失败（gnutls_handshake），需重试数次。
- 调 GitHub API 建仓库/改默认分支：`TOKEN=$($GIT_ASKPASS "Password for 'https://github.com':")`，然后 curl。
- 该 token 无 `delete_repo` 权限：API 删除仓库返回 403，需用户在网页操作。

## 已完成（commit b3c2d53，已推 v1 main，144 测试通过）
1. 研究者五项决定已落实：ISLES 不用 FLAIR（ROLES={dwi,adc}）；ISLES mask 按 ISLES 2022 论文（doi:10.1038/s41597-022-01875-5）注册为 `expert_voxel`/`target_complete`（3 例全零 mask 即无梗死例）；SKM 解剖参照只用 H5 seg（≡ xy 转置的 dicom 轨道），raw 轨道不再注册；SKM 几何 QC 改为 dicom-track 对 H5 seg 网格（155/155 pass）；SKM 单位按 mm（sources.yaml `xyz_units_policy: mm`）。
2. 目标异常定义：`configs/v41/target_abnormality_definition.yaml`（tad_v1）；`configs/v41/approvals.json` 的 `approved_by`/`approved_at` 为 null（Agent 不代填）。
3. manifest v3：405/405 全部准入（ISLES 250 + SKM 155）。G0-U、G0-A 均为 BLOCKED（仅缺签核）；G0-joint BLOCKED（缺签核 + 无同例专家 A+U 队列）。
4. 权威私有产物：`audits/{isles_u_v3, skm_a_v2, geometry_qc_isles_v2, geometry_qc_skm_v2, identity_v2, ontology_v2}/`、`registry/{gate_inputs_v3, manifest_v3, g0_decisions_v3}/`。报告：`reports/v41/g0_decisions_v3.md`。
5. 47 个 ISLES mask 的 1 存为浮点 0.9999999997671694，`discrete_labels`（atol 1e-5）已正确处理，无需改代码。

## spec.md（2026-11-02，未提交）
- 仓库根 `spec.md`：按现行口径合并 Proposal + Data Contract 的总规格（正文 §1–16 + 附录 A–I），已过独立 builder check；等待用户批准。批准后用 superpowers writing-plans 依据它重写 `plan.md`（覆盖完整系统，已完成的 T00–T08 作为核对任务），再用 executing-plans 执行。
- 用户在 spec 中的决定：冻结划分前先核验旧 checkpoint 训练范围（核清的病例去掉遗留曝光标记，研究者签核）；SKM 沿用官方 train/val/test，ISLES 按比例分层随机（比例/分层/种子待签核）；精标读者和工具尚无着落，先做不依赖精标的工作。
- 撰写者自定事项见 spec §15，需用户复核。
- 两份原文（Proposal、Contract）保留不动，作为历史；旧 `plan.md` 待新计划替换。

## 待用户操作
- 签核：手动填写 `configs/v41/approvals.json` 的 `approved_by`/`approved_at`，之后重跑 gate 评估，G0-U/G0-A 预期转 PASS。
- G0-joint 需等 T09 专家精标（同例 A+U 队列）。
- 未开始任务：T09–T24（含 T10 数据加载、T11 训练控制，无需等签核）。
- 阈值未冻结（thresholds.template.yaml 为 null）。

## 用户偏好/规则
- 中文交流，简洁直接；报告用中文。
- 对外操作（建仓库、推送）先用 AskUser 确认。
- commit 不带 AI trailer。
- 删除文件的钩子为白名单制（`~/.claude/hooks/block-delete.py`）。
- 隐私规则：仓库内不出现病例 ID/原始路径，提交前用 rg 核查。
- 签核由人填写；私有产物权限 700/600；审计产物版本化不覆盖。
