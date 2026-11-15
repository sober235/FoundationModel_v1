# 统一模型运行：unified_knee_brain_negtopk_pilot_v1

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `b7e9524156d61e8df3ab58c0f4e91ed429cd181ca5d914c0eb1ad3a2cc105c69`；run_sha256 `1573701efe0cfa6e613adcaf6702053074453e35eff9200c8634c135321f194b`
- 结果：COMPLETED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（预算 16.00 h）
- scientific_conclusion: NOT_EVALUATED；只是开发集描述，不计算区间，不形成阈值或临床判定。

| 协议 | scope | 成员 run_sha256 | split_sha256 |
| --- | --- | --- | --- |
| brain_dwi_adc | isles_u | `f00ced1bc4ee5628dd62b2c6bacd057dc2f81b157785ed2acc3c68ef1ea0dd9d` | `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b` |
| knee_qdess | skm_joint_early | `fa6afca8eff3ff5c6f0d36b4a9d9655d4b5101fbe26975f3ba389d62e96f0937` | `32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9` |

## 训练

- 同一个 checkpoint；随机初始化；每步各协议一个批次；完成步数 6000；状态 COMPLETED；选择规则 last
- train 样本：brain_dwi_adc 175，knee_qdess 86
- 最终检查点 sha256 `972cfcbde182bfdfb806f5b2dee403f0d7bbce6ffee5ede0480dc4d4a8e406a3`
- 受控停止：按计划在第 6000 步结束；学习率仍按 30000 步的衰减期限计算，所以已训练的各步与完整训练的前同样多步学习率相同。这个检查点是本次运行的最后一个检查点，不是完整期限训练的终点。

## 与模块基线的对照（development_val，患者等权）

| 协议 | 指标 | 统一模型 | 基线 | 差值 |
| --- | --- | --- | --- | --- |
| brain_dwi_adc | U ischemic_infarct dice | 0.681 | 0.752 | -0.071 |
| brain_dwi_adc | U ischemic_infarct hd95_mm | 32.745 | 15.795 | +16.950 |
| brain_dwi_adc | U ischemic_infarct nsd | 0.794 | 0.881 | -0.087 |
| brain_dwi_adc | 病灶 pooled_sensitivity | 0.519 | 0.641 | -0.122 |
| brain_dwi_adc | 病灶 fp_per_scan | 4.026 | 2.342 | +1.684 |
| knee_qdess | A patellar_cartilage dice | 0.747 | 0.887 | -0.140 |
| knee_qdess | A femoral_cartilage dice | 0.793 | 0.869 | -0.076 |
| knee_qdess | A tibial_cartilage_medial dice | 0.765 | 0.855 | -0.089 |
| knee_qdess | A tibial_cartilage_lateral dice | 0.732 | 0.861 | -0.129 |
| knee_qdess | A meniscus_medial dice | 0.697 | 0.831 | -0.134 |
| knee_qdess | A meniscus_lateral dice | 0.710 | 0.831 | -0.121 |

- 基线一列来自各协议单独训练的模块基线汇总；没有给出时写“未提供”。差值只作并列对照，不设通过界限。

## brain_dwi_adc：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 疑似缺血性梗死 | 327 |

## knee_qdess：所见评价（对照医生的主分析框）

- scientific_conclusion: NOT_EVALUATED
- 患者 33，扫描 33，有主分析框的患者 23，主分析框 53
- 分母固定为医生的全部主分析框；未命中的框计入每一项的分母。

| 范围 | 框数 | 框命中率（患者等权） | 框命中率（合并） | 归属正确率（患者等权） | 归属正确率（合并） | 组织级阳性召回 |
| --- | --- | --- | --- | --- | --- | --- |
| 合计 | 53 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cartilage_lesion | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| meniscal_tear | 23 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| IoBB 阈值 | 达标的主分析框比例 |
| --- | --- |
| 0.1 | 0.000 |
| 0.25 | 0.000 |
| 0.5 | 0.000 |

- 所见总数 0：定位点在同类主分析框内 0，在非主分析框（忽略区）内 0，其余 0（含定位点落在异类主分析框内的所见）
- 其余所见，每次扫描平均 0.000（描述性计数，不是误报：医生没有标注不等于正常）

## knee_qdess：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 未见所列异常（仅限：软骨病变、半月板撕裂）。医生没有标注不等于正常。 | 33 |

- 读 train 与 development_val；locked_test 未读。医生框外的所见是描述性计数，不是误报；医生没有标注不等于正常。
