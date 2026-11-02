# T02 — 资产清点匿名汇总（inventory_v1）

日期：2026-11-02。只读；命令：`anatobind.v41.cli inventory`（metadata 级，17 个一级入口）与 `verify-assets --level header --priority P0`。
私有产物：`$PRIVATE_ROOT/audits/inventory_v1/{assets.jsonl, asset_integrity.jsonl, inventory_summary.json, integrity_summary_header_*.json}`。

状态含义：`observed`（仅文件元数据）/ `header_only`（读头，未解码全量）/ `verified`（全量解码 + 内容哈希）/ `quarantined` / `failed` / `not_checked`。本轮**没有任何资产达到 verified**；`training_eligible` 全部为 null 或 false。

## 来源级计数（文件数，不是患者数）

| 来源 | 优先级 | 文件 | 图像 | 标签 | 归档 | 隔离 | 观察到的 case id |
|---|---|---|---|---|---|---|---|
| ISLES_ltr | P0 | 1690 | 750 | 250 | 1 | 0 | 250 |
| SKM-TEA_ltr | P0 | 625 | 155 | 310 | 156 | 0 | 155 |
| SKM-TEA（k-space） | P1 | 162 | 155 | 0 | 1 | 2 | 155 |
| UCSF-PDGM_lh | P1 | 12035 | — | — | 2 | 0 | 待配置 |
| UCSF-BMSR_cbb | P1 | 3094 | — | — | 1 | 0 | 待配置 |
| SibBMS_ms | P1 | 1428 | — | — | 1 | 0 | 待配置 |
| derived | P1 | 158446 | — | — | 0 | 0 | 不计 case；1 个疑似凭据文件仅计数未读 |
| fastMRI_lh_brain_knee | P2 | 125274 | — | — | 0 | 0 | 待配置 |
| OASIS_cbb | P2 | 19682 | — | — | 14 | 0 | 待配置 |
| TotalSegmentor_MRI_cbb | P2 | 31418 | — | — | 1 | 0 | 待配置 |
| HCP_lh_T1T2 | P2 | 2228 | — | — | 0 | 0 | 待配置 |
| KMAR-50K_ty | P2 | 3094 | — | — | 2 | 0 | 待配置 |
| AMOS_cbb | P2 | 966 | — | — | 1 | 0 | 待配置 |
| SPIDER_cbb | P2 | 898 | — | — | 2 | 0 | 待配置 |
| CMRxRecon_ty | P2 | 0 | — | — | 0 | 0 | 空目录 |
| TotalSegmentator_MRI_cbb | P2 | 0 | — | — | 0 | 0 | 空目录 |
| fastMR_lh_bran_knee_rawdata | P2 | 0 | — | — | 0 | 0 | 空目录 |

“—”：非 P0 来源本轮只记录结构，未配置图像/标签模式，不报 case 数。

## P0 头信息核验（header_only，2315 个 P0 资产中 1465 个影像/标签）

- ISLES_ltr：750 个影像、250 个标签 NIfTI 头可读；标签 shape 与同例 DWI/ADC 一致（例如 112×112×73 / 112×112×72），FLAIR 为独立网格（如 281×352×352）→ 印证“FLAIR 未配准到 mask 空间”，T05 必须按例核对 affine，不能按 shape 相同放行。
- SKM-TEA_ltr：155 个 H5 可打开，键为 `echo1 / echo2 / seg / stats`；310 个分割 NIfTI 中 302 个为 512×512×160，8 个深度不同（对应 4 例 × 2 轨道）。H5 内 `seg` 与 NIfTI 两轨道掩膜的关系、六类通道顺序待 T04/T05 实查。
- 850 个 P0 资产为 `not_checked`（归档、JSON/CSV/TSV 元数据、`.DS_Store` 等非影像），不影响影像/标签核验口径。
- 未执行 full_decode / 内容哈希：按计划在正式准入集合确定（T07）后对准入资产执行，避免对全部 155 个大 H5 重复 I/O。

## 与原盘点的差异

- 原盘点将 `CMRxRecon_ty`、`TotalSegmentator_MRI_cbb`、`fastMR_lh_bran_knee_rawdata` 记为非空；本次只读遍历（不跟随链接、不进入 raw root 之外）得到 0 个文件。需研究者确认是否存在挂载/链接差异。
- SKM-TEA k-space 目录 2 个 `.truncated` 文件已隔离（与原盘点一致）。
