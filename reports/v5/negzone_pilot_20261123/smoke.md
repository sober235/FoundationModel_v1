# 统一冒烟：unified_knee_brain_negzone_pilot_v1

本报告只含配置、计数、耗时与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `c5a0ba7dcbc58cef4a3abd166be9511523728784b1ee9fd59f41e95ed7e81efb`
- 结果：PASS；scientific_conclusion: NOT_EVALUATED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（预算 2.00 h）
- 实例掩膜的内存上限 256 GiB（本次冒烟显式指定；默认 16 GiB）。这是实现上的内存保护：阈值、体素数下限与通过条件都不依赖它。
- 连续执行的单步耗时：1.582 s（每步各协议一个批次）
- 显存峰值：23.42 GiB

| 执行 | 状态 | 步数 | 失败码 |
| --- | --- | --- | --- |
| continuous | COMPLETED | 40 | 无 |
| resumed | COMPLETED | 40 | 无 |

- 续训比较：通过；暂停于第 20 步；比较步数 20；参数逐位相同：是；参数最大相对差 0；损失最大绝对差 0

| 协议 | scope | 训练样本 | 整卷推理 | 所见数 |
| --- | --- | --- | --- | --- |
| brain_dwi_adc | isles_u | 2 | module | 3 |
| knee_qdess | skm_joint_early | 2 | joint | 3316 |

- 只读各协议登记的 train 子集；development_val 与 locked_test 未读。
