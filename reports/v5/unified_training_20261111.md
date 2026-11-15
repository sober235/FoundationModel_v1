# 统一模型运行：unified_knee_brain_v1

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `2f43a97185292644b2ab2f1c91a8a9eeb7dba4dbec274200758bac910d3aa71c`；run_sha256 `9019fb4f9d2f1450e85d455acfd2d144f50a3924f723a623cc40cd5d47b3e2e9`
- 结果：COMPLETED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 10.140 h（预算 20.00 h）
- scientific_conclusion: NOT_EVALUATED；只是开发集描述，不计算区间，不形成阈值或临床判定。

| 协议 | scope | 成员 run_sha256 | split_sha256 |
| --- | --- | --- | --- |
| brain_dwi_adc | isles_u | `7bb1d9781d4b855187e7ae15be5e027320b76b862bfdcd9a6e383e998d1b2d62` | `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b` |
| knee_qdess | skm_joint_early | `84e310b4f40c036df7842b4e7c3be1543e239767c24249d9f9fea4ec897ff194` | `32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9` |

## 训练

- 同一个 checkpoint；随机初始化；每步各协议一个批次；完成步数 30000；状态 COMPLETED；选择规则 last
- train 样本：brain_dwi_adc 175，knee_qdess 86
- 最终检查点 sha256 `5b1e8c6d14ca32f2a538bf43d04b61cdece62262fb2c34fadaa3b12466c72e35`

## 与模块基线的对照（development_val，患者等权）

| 协议 | 指标 | 统一模型 | 基线 | 差值 |
| --- | --- | --- | --- | --- |
| brain_dwi_adc | U ischemic_infarct dice | 0.756 | 0.752 | +0.004 |
| brain_dwi_adc | U ischemic_infarct hd95_mm | 13.753 | 15.795 | -2.042 |
| brain_dwi_adc | U ischemic_infarct nsd | 0.882 | 0.881 | +0.001 |
| brain_dwi_adc | 病灶 pooled_sensitivity | 0.638 | 0.641 | -0.002 |
| brain_dwi_adc | 病灶 fp_per_scan | 2.632 | 2.342 | +0.289 |
| knee_qdess | A patellar_cartilage dice | 0.874 | 0.887 | -0.012 |
| knee_qdess | A femoral_cartilage dice | 0.862 | 0.869 | -0.006 |
| knee_qdess | A tibial_cartilage_medial dice | 0.852 | 0.855 | -0.003 |
| knee_qdess | A tibial_cartilage_lateral dice | 0.852 | 0.861 | -0.009 |
| knee_qdess | A meniscus_medial dice | 0.825 | 0.831 | -0.006 |
| knee_qdess | A meniscus_lateral dice | 0.815 | 0.831 | -0.016 |

- 基线一列来自各协议单独训练的模块基线汇总；没有给出时写“未提供”。差值只作并列对照，不设通过界限。

## brain_dwi_adc：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 疑似缺血性梗死 | 328 |

## knee_qdess：所见评价（对照医生的主分析框）

- scientific_conclusion: NOT_EVALUATED
- 患者 33，扫描 33，有主分析框的患者 23，主分析框 53
- 分母固定为医生的全部主分析框；未命中的框计入每一项的分母。

| 范围 | 框数 | 框命中率（患者等权） | 框命中率（合并） | 归属正确率（患者等权） | 归属正确率（合并） | 组织级阳性召回 |
| --- | --- | --- | --- | --- | --- | --- |
| 合计 | 53 | 0.166 | 0.208 | 0.139 | 0.170 | 0.881 |
| cartilage_lesion | 30 | 0.069 | 0.133 | 0.030 | 0.067 | 0.800 |
| meniscal_tear | 23 | 0.265 | 0.304 | 0.265 | 0.304 | 1.000 |

| IoBB 阈值 | 达标的主分析框比例 |
| --- | --- |
| 0.1 | 0.170 |
| 0.25 | 0.132 |
| 0.5 | 0.075 |

- 所见总数 1148：定位点在同类主分析框内 11，在非主分析框（忽略区）内 103，在所有框外 1034
- 医生框外的所见，每次扫描平均 31.333（描述性计数，不是误报：医生没有标注不等于正常）

## knee_qdess：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 在半月板上，疑似半月板撕裂 | 140 |
| 在股骨软骨上，疑似软骨病变 | 106 |
| 在胫骨软骨上，疑似软骨病变 | 32 |
| 在髌软骨上，疑似软骨病变 | 18 |
| 疑似半月板撕裂，所在结构未能确定 | 285 |
| 疑似软骨病变，所在结构未能确定 | 567 |

- 读 train 与 development_val；locked_test 未读。医生框外的所见是描述性计数，不是误报；医生没有标注不等于正常。
