# 统一模型运行：unified_knee_brain_negzone_pilot_v1

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `c5a0ba7dcbc58cef4a3abd166be9511523728784b1ee9fd59f41e95ed7e81efb`；run_sha256 `0bca0d58ef6474fbeeca0999476752cafc345cc17c43ff6a108b67eae6d6412e`
- 结果：FAILED（停止：INSTANCE_LIMIT_EXCEEDED，阶段 evaluation）
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（预算 14.00 h）
- scientific_conclusion: NOT_EVALUATED；只是开发集描述，不计算区间，不形成阈值或临床判定。

| 协议 | scope | 成员 run_sha256 | split_sha256 |
| --- | --- | --- | --- |
| brain_dwi_adc | isles_u | `2a8db96481d72281f68246316711a1c60c5a605de27c8e414e9eabbb52a57e99` | `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b` |
| knee_qdess | skm_joint_early | `9ea499f9610320a53b22dcb3e344db5efcecc7ef7540d157a5b68d0325223592` | `32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9` |

## 训练

- 同一个 checkpoint；随机初始化；每步各协议一个批次；完成步数 6000；状态 COMPLETED；选择规则 last
- train 样本：brain_dwi_adc 175，knee_qdess 86
- 最终检查点 sha256 `6517022bcd650b97d26c8ffb01a0c7518f95385aa8ac904bb2de869c02f57e00`
- 受控停止：按计划在第 6000 步结束；学习率仍按 30000 步的衰减期限计算，所以已训练的各步与完整训练的前同样多步学习率相同。这个检查点是本次运行的最后一个检查点，不是完整期限训练的终点。

- 读 train 与 development_val；locked_test 未读。医生框外的所见是描述性计数，不是误报；医生没有标注不等于正常。
