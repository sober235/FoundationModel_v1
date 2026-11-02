# manifest v3 与 G0 门决策（匿名结论）

日期：2026-11-02。依据研究者当日的五项决定重跑审计。权威产物：`audits/{isles_u_v3,skm_a_v2,geometry_qc_isles_v2,geometry_qc_skm_v2,identity_v2,ontology_v2}/`、`registry/gate_inputs_v3/`、`registry/manifest_v3/`、`registry/g0_decisions_v3/`（均在 `$PRIVATE_ROOT`）。v1/v2 产物保留为历史。

## 研究者决定

1. ISLES 不使用 FLAIR，模型输入只有 DWI 与 ADC。
2. ISLES 发布 mask 已对照 ISLES 2022 论文（Hernandez Petzsche 等，Sci Data 2022）核实：U-Net 预分割，经受训标注员修改，由主治神经放射科医生审批；只标急性到亚急性梗死。3 例全零 mask 即论文所述 3 例无梗死。中心 2（伯尔尼）的流程论文只在技术验证一节概述。
3. 目标异常定义（`configs/v41/target_abnormality_definition.yaml`，tad_v1）：ISLES 只含急性/亚急性缺血性梗死（白质高信号和陈旧胶质增生不在范围内）；SKM 含四类（软骨损伤、半月板撕裂、韧带撕裂、积液）。**尚待研究者在 `configs/v41/approvals.json` 填写 approved_by/approved_at。**
4. SKM 解剖参照只用 H5 seg（等于 xy 转置后的 dicom 轨道 NIfTI）；raw 轨道不再注册。
5. SKM NIfTI 头单位为 unknown，按研究者确认视为 mm。

## 审计结果

| 审计 | ISLES_ltr | SKM-TEA_ltr |
|---|---|---|
| 注册记录 | 250 条，250 个 `expert_voxel`/`target_complete` mask | 155 条，155 个 A `expert_voxel` + 476 个框（`weak_localization`） |
| 监督资格 | U qualified 250/250（正负均可监督） | A qualified 155/155；U 框 470 个只作定位，6 个 confidence 为 0 被拒 |
| geometry_qc | pass：DWI 250/250、ADC 250/250 与 mask 同网格 | pass：dicom 轨道转置后与 H5 seg 形状一致 155/155（H5 无 affine，只比较形状，另有 12/12 Dice=1.0 通道核验） |
| 标签值 | 247 pass，3 例全零（真阴性） | 155/155 pass |
| 配对 | 250/250 | 155/155 |
| 身份/泄漏 | 405 组，0 跨划分，0 泄漏（尚未分 split） | 同左 |
| 本体 | pass；H5 通道顺序改为 verified | 同左 |

浮点 mask：47 例 ISLES mask 的 1 存为 `0.9999999997671694`，`discrete_labels` 在 1e-5 容差内取整，已有测试覆盖，无需修改代码。

## manifest v3

405 条，**405 admitted / 0 blocked**（ISLES 250，SKM 155）。所有记录仍带“遗留曝光未核实”标记，不得称为未曝光盲测集。

## G0 决策（`scientific_conclusion` 一律 NOT_EVALUATED）

| 门 | engineering_status | 阻塞项 |
|---|---|---|
| G0-U（ISLES） | BLOCKED | 缺签核 `target_abnormality_definition` |
| G0-A（SKM） | BLOCKED | 缺签核 `target_abnormality_definition` |
| G0-joint | BLOCKED | 缺签核；无同例专家 A+U 队列（ISLES 与 SKM 不相交，SKM 的 U 只有框） |

研究者填写签核后，G0-U 与 G0-A 预期转为 PASS；G0-joint 需等 T09 专家精标。
