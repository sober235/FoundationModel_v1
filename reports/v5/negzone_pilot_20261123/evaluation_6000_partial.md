# 统一模型只评价运行：unified_knee_brain_negzone_pilot_v1

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划 sha256 `c5a0ba7dcbc58cef4a3abd166be9511523728784b1ee9fd59f41e95ed7e81efb`；被评价的运行 run_sha256 `0bca0d58ef6474fbeeca0999476752cafc345cc17c43ff6a108b67eae6d6412e`
- 检查点 sha256 `6517022bcd650b97d26c8ffb01a0c7518f95385aa8ac904bb2de869c02f57e00`（第 6000 步）；评价前后的权重哈希未完成核对（state sha256 `5c04cc02c181c4209673a16ca4bfaaeda827f637dcc7bb7d760b59b52f31028d`）
- 检查点是按步数与哈希指定的：第 6000 步。该运行的最后一个检查点是第 6000 步（sha256 `6517022bcd650b97d26c8ffb01a0c7518f95385aa8ac904bb2de869c02f57e00`），就是本次读取的这一个。
- 实例掩膜的内存上限 256 GiB（本次评价显式指定；默认 16 GiB）。这是实现上的内存保护，不是评价规则：阈值、体素数下限、定位点、所在组织与匹配规则都不变；默认上限容得下的扫描，结果与默认设置相同。
- 训练代码 `e7f02b845b57772f8bde6e53e4695b528b52b1f5`；本次评价代码 `e7f02b845b57772f8bde6e53e4695b528b52b1f5`
- 结果：FAILED（停止：EVAL_BUDGET_EXCEEDED，阶段 train）
- 设备：NVIDIA A800 80GB PCIe × 1；用时 [服务器端计时，未取回]（本次上限 4.90 h；计划整体预算在本次之前剩余 [服务器端计时，未取回]）
- scientific_conclusion: NOT_EVALUATED；只做推理与统计，不更新权重。train 上的数值只用于诊断，不能代替泛化评价。
- 读取的划分角色：train、development_val；locked_test 未读。
- 该运行的训练已完成，但它自己在训练结束时的开发集描述因实例掩膜的内存上限（工程保护，INSTANCE_LIMIT_EXCEEDED）而停止，没有写出；本次只评价直接读它的检查点，不复用它保存的任何输出。
- 全部预测都是本次用上述检查点重新推理得到的，没有复用任何已保存的输出。
