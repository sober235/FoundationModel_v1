# T07 — manifest v1 与 G0 门决策（匿名结论）

日期：2026-11-02。命令：`audit --scope isles_u`、`audit --scope skm_a`、`resolve-identities`、`build-manifest`、`gate --gate G0-U/G0-A/G0-joint`。
私有产物：`$PRIVATE_ROOT/audits/{isles_u_v1,skm_a_v1}/`（`annotation_provenance_report.csv`、`input_visibility_report.csv`、`eligibility_coverage.csv`、`registration_records.jsonl`）、`audits/identity_v1/resolve/`（`patient_alias_report.csv`、`split_leakage_report.json`、`groups.json`）、`registry/gate_inputs_v1/`（审计索引与门输入）、`registry/manifest_v1/`（`manifest.jsonl` + `manifest.lock.json`，不可变）、`registry/g0_decisions_v1/`（三份门决策；`registry/gates_v1/` 为修复前的过期版本，已被取代）。

## 注册记录（P0）

| 来源 | 记录 | 患者组 | 标注 | 参照角色 / 完整性 | 监督资格 |
|---|---|---|---|---|---|
| ISLES_ltr（U） | 250 | 250 | 250 个 mask | `expert_voxel`，`manual`，completeness **unknown**（发布协议未核实） | 250/250 `none`（COMPLETENESS_UNKNOWN） |
| SKM-TEA_ltr（A） | 155 | 155（按标注 subject_id） | 310 个 mask + 476 个 bbox | mask：`expert_voxel`，`target_complete`；bbox：`weak_localization`，`positive_only` | A：310 `qualified`；U(bbox)：470 `qualified`（仅定位）、6 `none`（confidence 0 → rejected） |

- 所有 405 条记录 `split=unassigned`，并带 `legacy_foundation_model_unverified` 曝光项（角色 unknown）：**没有任何病例可被称为未曝光盲测**。
- 身份解析：405 组、0 跨 split、0 曝光组落入 locked_test → `pass`；此结论的前提是尚未分 split，不等于“无泄漏风险”。
- SKM 发布 split 分布（train 86 / val 33 / test 36）只作记录，不被沿用。

## 八类审计索引（按来源作用域）

| 审计 | ISLES_ltr | SKM-TEA_ltr |
|---|---|---|
| ontology_validation_report | pass（全局） | pass（全局） |
| patient_alias_report / split_leakage_report | pass（全局） | pass（全局） |
| geometry_qc | **fail**（FLAIR 250/250 不配准） | **fail**（dicom/raw 轨道 155/155 几何不一致；raw 轨道与 H5 seg 体素不一致） |
| image_label_pairing | pass | pass |
| annotation_provenance_report / eligibility_coverage / input_visibility_report | pass | pass |

## manifest v1

405 条目，**0 admitted / 405 blocked**，每条 `reasons=["FAILED_AUDIT"]`、`failed_audits=["geometry_qc"]`。两来源状态均 `blocked`。manifest 不可变；解除需发布 v2，前提是：ISLES 记录移除或重采样 FLAIR；SKM 记录仅保留 dicom 轨道掩膜（或 H5 seg）作为 A 参照、移除 raw 轨道；重新运行 geometry_qc 并改索引状态。

## G0 决策（`scientific_conclusion` 一律 NOT_EVALUATED）

| 门 | engineering_status | 阻塞项 |
|---|---|---|
| G0-U | FAIL | failed audits: geometry_qc；缺签核 `target_abnormality_definition` |
| G0-A | FAIL | failed audits: geometry_qc；缺签核 `target_abnormality_definition` |
| G0-joint | FAIL | 同上 + 无同例专家 A 与 U 参照队列（ISLES 与 SKM 队列不相交） |

签核字段由人填写（approved_by/approved_at），Agent 不代填；本轮 `--approvals` 未提供。

## 需要研究者/临床方的决定

1. ISLES FLAIR 策略：重采样到 DWI 网格，或从 `available_modality_mask` 去除。
2. ISLES 发布 mask 的标注协议与完整性（当前 `unknown`，因此 U 任务没有任何可用监督）。
3. SKM raw 轨道掩膜的用途（当前排除）；是否认可 H5 seg ≡ 转置后的 dicom 轨道作为唯一 A 参照。
4. `target_abnormality_definition` 签核；G0-joint 所需的同例 A+U 专家队列是否存在于其他来源。
