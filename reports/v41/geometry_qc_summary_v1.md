# T05 — 空间与标签 QC 匿名汇总（geometry_qc_isles_v1 / geometry_qc_skm_v1）

日期：2026-11-02。只读；命令：`anatobind.v41.cli geometry-qc --source ISLES_ltr`（14 s）与 `--source SKM-TEA_ltr --skm-channel-cases 12`（约 70 min，后台）；`overlay-samples --cases 6`。
私有产物：`$PRIVATE_ROOT/audits/geometry_qc_isles_v1/`、`geometry_qc_skm_v1/`（各含 `geometry_qc.csv`、`label_value_qc.csv`、`image_label_pairing.csv`、`geometry_qc_summary_*.json`；SKM 另有 `skm_channel_check.json`）、`overlays_v2/`（18 张叠图 + `overlay_manifest.json`；`overlays_v1/` 已标记 SUPERSEDED）。

判定口径：`audit_geometry` 逐例比较 shape、spacing、affine（atol 1e-3 mm）与轴序码；**同 shape 不等于通过**。标签解码走已验证的 `discrete_labels`（tol 1e-5），非整数值拒绝，全零标签记 `needs_review` 而非阴性。

## ISLES_ltr（250 例）

| 项目 | 结果 |
|---|---|
| 配对 | 250/250 `paired`（每例 DWI、ADC、FLAIR + 1 个 mask） |
| DWI vs mask | 250/250 pass |
| ADC vs mask | 250/250 pass |
| FLAIR vs mask | 250/250 **fail**（SHAPE/SPACING/AFFINE 不匹配；其中 152 例轴序码也不同） |
| 标签值 | 247 pass（{0,1}）；3 例 ALL_ZERO → `needs_review`，不得当阴性 |

结论：FLAIR 处于独立原生网格，**未配准到 mask 空间**，本轮不上线；任何使用 FLAIR 的方案需先定重采样/配准策略并重新 QC。3 例全零 mask 需研究者确认是“确无病灶”还是“截断/缺失标注”。

## SKM-TEA_ltr（155 例，每例 1 个 H5 + 2 条 NIfTI 掩膜轨道）

| 项目 | 结果 |
|---|---|
| 配对 | 155/155 `paired`（H5 + dicom 轨道 + raw 轨道） |
| dicom 轨道 vs raw 轨道 | 155/155 **fail**（AFFINE_MISMATCH + AXIS_CODE_MISMATCH；两轨道轴序码互换前两轴，spacing 亦有 1e-4 mm 级差异） |
| 标签值 | 310/310 pass（{0..6}，无意外值，无全零） |
| H5 `seg` 通道顺序（12 例核验） | 通道 0..5 ↔ A6 标签 1..6（髌软骨 1、股骨软骨 2、胫骨内 3/外 4、半月板内 5/外 6），12/12 例六类 Dice 全为 1.000，**前提是 dicom 轨道 NIfTI 先做前两轴转置**（`transpose_xy`） |
| raw 轨道 vs H5 `seg` | 最佳为 identity，但 Dice 0.27–0.999、12/12 例 inconsistent → raw 轨道**不是** H5 `seg` 的同体素副本，属另一重建/网格 |

结论：
- T04 遗留的 `pending_verification` 已解除：H5 `seg[..., i]` 对应 A6 标签 `i+1`，与 dicom 轨道掩膜转置后体素一致。`adapters/skm.py` 固化为 `NIFTI_TO_H5_TRANSFORM = "transpose_xy"` 并有测试。
- H5 `echo1/echo2` 与 `seg` 同网格；因此 H5 图像配 dicom 轨道掩膜必须转置，配 raw 轨道掩膜则**不可用**（几何不一致），raw 轨道本轮不作为 A 任务参照。
- 教训：首版叠图对 SKM 用了恒等方向，肉眼误判为“对齐”；定量 channel_check（Dice 0 vs 1）才揭示转置。叠图只作人审证据，**不能替代**定量几何核对。

## 对下游的约束

- 仅 ISLES DWI/ADC + mask、SKM H5(echo1/echo2) + 转置后 dicom 轨道掩膜 满足体素级对齐；其余组合（ISLES FLAIR、SKM raw 轨道）进入 manifest 时应判 `fail`/排除。
- 3 例 ISLES 全零 mask、SKM 8 个非 160 层掩膜（4 例 × 2 轨道）需在准入前人工复核。
- 没有任何 QC 结果构成临床意义上的“配准合格”；`atol` 只是数值一致性阈值。
