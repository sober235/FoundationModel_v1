# 框导入摘要：skm_tea_joint_early（skm_tea_annotations_v1.0.0）→ skm_joint_early

只含计数与哈希；不含病例 ID、路径或文件名。框保持框级，不转成体素真值；无框记 annotation_unknown，不作阴性。本摘要不是评价结果，`scientific_conclusion: NOT_EVALUATED`。

- scope_hash：`f6e4b5f6e3c95a49f5f8cb45591b2a2c437b8db2d432551e1bd1e9e81935ec9f`
- 框策略 `skm_box_reference_v1`，policy_sha256：`09b1003e63fb5e8517be7626ba7e522dba9a7397a53dcd4ef910226bc734f2a9`
- 登记记录集 records_sha256：`b3b893d29bfb6b89573afc643b084300698c4e891019432866148a40f1d37ade`
- 官方划分文件：train 存在、val 存在、test 存在
- 登记扫描按来源划分：train 86、val 33、test 36
- 登记扫描 155，导入 155，失败 0；框总数 476
- 来源划分只记录数据出处，不是正式划分；导入不分配正式划分。训练、模型选择和评价只能按冻结划分的角色读取这些参照。
- 每个导入扫描的接触另记为 `administrative_qc`（155 条），与训练、模型选择和查看测试结果的曝光分开。官方 test 的计数已在本摘要中可见：此后据此修改框策略或规则属于研究性曝光，须另行登记。

## 目标框的逐级计数

| 阶段 | 框数 |
| --- | --- |
| target_supercategory_consistent | 318 |
| confidence_nonzero | 314 |
| nonzero_size | 307 |
| integer_coordinates | 307 |
| in_bounds | 307 |
| host_tissue_in_box | 297 |
| primary | 222 |
| secondary | 57 |
| out_of_scope_extrusion | 18 |

## 排除与范围外

| 原因或类别 | 框数 |
| --- | --- |
| 排除：category_tissue_mismatch | 2 |
| 排除：confidence_zero | 4 |
| 排除：degenerate_box | 7 |
| 排除：no_host_tissue_in_box | 10 |
| 范围外：Effusion | 117 |
| 范围外：Ligament Tear (Full Thickness/Complete Tear) | 15 |
| 范围外：Ligament Tear (Low-Grade Sprain) | 10 |
| 范围外：Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration) | 14 |
| 范围外：Meniscal Tear (Extrusion) | 18 |

## 主分析阳性扫描数（按来源划分）

| 来源划分 | 异常类别:组织组 | 扫描数 |
| --- | --- | --- |
| test | cartilage_lesion:femoral_cartilage | 15 |
| test | cartilage_lesion:patellar_cartilage | 11 |
| test | cartilage_lesion:tibial_cartilage | 7 |
| test | meniscal_tear:meniscus | 19 |
| train | cartilage_lesion:femoral_cartilage | 31 |
| train | cartilage_lesion:patellar_cartilage | 11 |
| train | cartilage_lesion:tibial_cartilage | 10 |
| train | meniscal_tear:meniscus | 26 |
| val | cartilage_lesion:femoral_cartilage | 15 |
| val | cartilage_lesion:patellar_cartilage | 5 |
| val | cartilage_lesion:tibial_cartilage | 5 |
| val | meniscal_tear:meniscus | 17 |

无主分析框的扫描数（annotation_unknown，不是阴性）：test 10、train 40、val 10
