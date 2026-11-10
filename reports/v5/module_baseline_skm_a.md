# 基线运行：skm_a

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划：`baseline_skm_a_v1`，sha256 `3809a1cea78312f94358df1de9581511d56f1add34bd2554a6b76421b67d53bf`
- 所依据的 smoke：计划 sha256 `93ce6daaa21654e7e729b2e6796095ccccb075a5e246bc77bbc34062a1d725c0`；training_smoke sha256 `0fc50ac8ae0b2af99bba498c340214c5ad2518487c944577a617f5f5d0e64038`
- 结果：COMPLETED
- 设备：NVIDIA A100 80GB PCIe × 1；用时 4.331 h（预算 8.00 h）
- 配置（smoke 按规则冻结）：patch 192×192×48，batch 2，fp32
- run_sha256 `cbd9d32cdae69da7ebde6e8496a5057e89ba8c15f3ba0b2384cf078ff3fe54e3`；recipe_sha256 `8ba70a7c151f6ad5ebec282024813f89dbafc47ee5560a07bd772c3c2ed0ab33`；supervision_sha256 `ac3cfc76f5db68e065c0c1197fa194c96008fb75fb2fca38528620cfb18a91d5`
- data_manifest_sha256 `0d5368baea5f8b4b3937501c1479e148a49ffe1feaf66fb60a93aafd9fb62dd9`；split_sha256 `6a2c955c9764472c55311513766ea9791ee3cbdc9a12178982151990b461eb61`

## 训练

- 随机初始化；train 样本 86；完成步数 20000；状态 COMPLETED；选择规则 last
- 最终检查点 sha256 `420ae273abdd5b8b90c5bad9e8f0f543a4af6e3171ab907776a4bbfd33be2c0c`

## 开发集评估

- split: development_val
- scientific_conclusion: NOT_EVALUATED
- 只作开发集描述：逐例指标先在患者内平均，再患者等权；不计算区间（区间属于 T14 的 H1 比较）。
- 各模块结果分开报告，不相加为同例联合成功（Proposal §1.1）。
- 患者数：33；扫描数：33
- acceptance_policy_id: skm_a_engineering_v1
- split_sha256: 6a2c955c9764472c55311513766ea9791ee3cbdc9a12178982151990b461eb61
- run_sha256: cbd9d32cdae69da7ebde6e8496a5057e89ba8c15f3ba0b2384cf078ff3fe54e3
- checkpoint_sha256: 420ae273abdd5b8b90c5bad9e8f0f543a4af6e3171ab907776a4bbfd33be2c0c
- reference_set_hash: d9ca82fbef4d7fe11d328cbc699e282cafbffadb3b35ca95d27bdbaa02e4075f
- metric_policy_sha256: 1e6a4a9d6163f721e01e03aada3c4f883ca676be6d1cf997e36433a88738f660

## A（解剖）分割

| 类别 | 指标 | 患者等权均值 | 有值患者数 | 有值扫描数 | None 扫描数 |
| --- | --- | --- | --- | --- | --- |
| patellar_cartilage | dice | 0.887 | 33 | 33 | 0 |
| femoral_cartilage | dice | 0.869 | 33 | 33 | 0 |
| tibial_cartilage_medial | dice | 0.855 | 33 | 33 | 0 |
| tibial_cartilage_lateral | dice | 0.861 | 33 | 33 | 0 |
| meniscus_medial | dice | 0.831 | 33 | 33 | 0 |
| meniscus_lateral | dice | 0.831 | 33 | 33 | 0 |

- 读 train 与 development_val；locked_test 未读；`scientific_conclusion: NOT_EVALUATED`。
