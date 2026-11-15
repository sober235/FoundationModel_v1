# 统一冒烟：unified_knee_brain_negtopk_pilot_v1

本报告只含配置、计数、耗时与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `b7e9524156d61e8df3ab58c0f4e91ed429cd181ca5d914c0eb1ad3a2cc105c69`
- 结果：PASS；scientific_conclusion: NOT_EVALUATED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（预算 2.00 h）
- 连续执行的单步耗时：[服务器端计时，未取回]（每步各协议一个批次）
- 显存峰值：23.42 GiB

| 执行 | 状态 | 步数 | 失败码 |
| --- | --- | --- | --- |
| continuous | COMPLETED | 40 | 无 |
| resumed | COMPLETED | 40 | 无 |

- 续训比较：通过；暂停于第 20 步；比较步数 20；参数逐位相同：是；参数最大相对差 0；损失最大绝对差 0

| 协议 | scope | 训练样本 | 整卷推理 | 所见数 |
| --- | --- | --- | --- | --- |
| brain_dwi_adc | isles_u | 2 | module | 4 |
| knee_qdess | skm_joint_early | 2 | joint | 124 |

- 只读各协议登记的 train 子集；development_val 与 locked_test 未读。
