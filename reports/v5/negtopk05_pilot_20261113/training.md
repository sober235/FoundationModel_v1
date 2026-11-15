# 统一模型运行：unified_knee_brain_negtopk05_pilot_v1

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `b4c3813a56bedd1548c0d4c8dd839cabf38cca30e694f195e0bcd2e86c6f627e`；run_sha256 `e7ca6105e8396e1012e86db4fe17418a226732c88dc7986efa71f3abbeb0ff42`
- 结果：COMPLETED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（预算 16.00 h）
- scientific_conclusion: NOT_EVALUATED；只是开发集描述，不计算区间，不形成阈值或临床判定。

| 协议 | scope | 成员 run_sha256 | split_sha256 |
| --- | --- | --- | --- |
| brain_dwi_adc | isles_u | `5cbfae3c7e73c4565532ba8bca8875020e4744feef7f5e908164a12fbfca023b` | `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b` |
| knee_qdess | skm_joint_early | `c44942131aff79bc0c5d5252708a4afd0328c4ff2d7b9c05baaab13dd3ac6581` | `32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9` |

## 训练

- 同一个 checkpoint；随机初始化；每步各协议一个批次；完成步数 6000；状态 COMPLETED；选择规则 last
- train 样本：brain_dwi_adc 175，knee_qdess 86
- 最终检查点 sha256 `d0ea7813227d300f9b038e281a26c84dada72811beb39fa323f9b629a8d2b0b3`
- 受控停止：按计划在第 6000 步结束；学习率仍按 30000 步的衰减期限计算，所以已训练的各步与完整训练的前同样多步学习率相同。这个检查点是本次运行的最后一个检查点，不是完整期限训练的终点。

## 与模块基线的对照（development_val，患者等权）

| 协议 | 指标 | 统一模型 | 基线 | 差值 |
| --- | --- | --- | --- | --- |
| brain_dwi_adc | U ischemic_infarct dice | 0.700 | 0.752 | -0.052 |
| brain_dwi_adc | U ischemic_infarct hd95_mm | 19.824 | 15.795 | +4.029 |
| brain_dwi_adc | U ischemic_infarct nsd | 0.829 | 0.881 | -0.052 |
| brain_dwi_adc | 病灶 pooled_sensitivity | 0.516 | 0.641 | -0.124 |
| brain_dwi_adc | 病灶 fp_per_scan | 1.474 | 2.342 | -0.868 |
| knee_qdess | A patellar_cartilage dice | 0.729 | 0.887 | -0.157 |
| knee_qdess | A femoral_cartilage dice | 0.790 | 0.869 | -0.079 |
| knee_qdess | A tibial_cartilage_medial dice | 0.736 | 0.855 | -0.119 |
| knee_qdess | A tibial_cartilage_lateral dice | 0.703 | 0.861 | -0.158 |
| knee_qdess | A meniscus_medial dice | 0.688 | 0.831 | -0.143 |
| knee_qdess | A meniscus_lateral dice | 0.654 | 0.831 | -0.177 |

- 基线一列来自各协议单独训练的模块基线汇总；没有给出时写“未提供”。差值只作并列对照，不设通过界限。

## brain_dwi_adc：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 疑似缺血性梗死 | 226 |

## knee_qdess：所见评价（对照医生的主分析框）

- scientific_conclusion: NOT_EVALUATED
- 患者 33，扫描 33，有主分析框的患者 23，主分析框 53
- 分母固定为医生的全部主分析框；未命中的框计入每一项的分母。

| 范围 | 框数 | 框命中率（患者等权） | 框命中率（合并） | 归属正确率（患者等权） | 归属正确率（合并） | 组织级阳性召回 |
| --- | --- | --- | --- | --- | --- | --- |
| 合计 | 53 | 0.205 | 0.264 | 0.162 | 0.189 | 1.000 |
| cartilage_lesion | 30 | 0.200 | 0.300 | 0.104 | 0.167 | 1.000 |
| meniscal_tear | 23 | 0.206 | 0.217 | 0.206 | 0.217 | 1.000 |

| IoBB 阈值 | 达标的主分析框比例 |
| --- | --- |
| 0.1 | 0.245 |
| 0.25 | 0.226 |
| 0.5 | 0.151 |

- 所见总数 2427：定位点在同类主分析框内 23，在非主分析框（忽略区）内 199，其余 2205（含定位点落在异类主分析框内的所见）
- 其余所见，每次扫描平均 66.818（描述性计数，不是误报：医生没有标注不等于正常）

## knee_qdess：所见句式计数

| 句式 | 条数 |
| --- | --- |
| 在半月板上，疑似半月板撕裂 | 240 |
| 在股骨软骨上，疑似软骨病变 | 395 |
| 在胫骨软骨上，疑似软骨病变 | 291 |
| 在髌软骨上，疑似软骨病变 | 228 |
| 疑似半月板撕裂，所在结构未能确定 | 402 |
| 疑似软骨病变，所在结构未能确定 | 871 |

- 读 train 与 development_val；locked_test 未读。医生框外的所见是描述性计数，不是误报；医生没有标注不等于正常。
