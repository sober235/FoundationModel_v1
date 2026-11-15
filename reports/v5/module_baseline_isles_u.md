# 基线运行：isles_u

本报告只含配置、计数、指标与哈希；不含样本或患者 ID、路径或文件名。

- 计划：`baseline_isles_u_v1`，sha256 `96040004dd1afce9a10c79e30b308fa49d47654b67205151412bf0479b96b8de`
- 所依据的 smoke：计划 sha256 `02b5c2603ac78bb55ddc22d7d71a3851c517a6bd13dcdb9e60e0f6455b0933d8`；training_smoke sha256 `bf4ebff6cad2d0991327ca9252a2a9af53dc05de76208d27634b93b4cc507f73`
- 结果：COMPLETED
- 设备：NVIDIA A800 80GB PCIe × 1；用时 1.477 h（预算 5.00 h）
- 配置（smoke 按规则冻结）：patch 112×112×64，batch 2，fp32
- run_sha256 `80c484cbe6ddd415877e51cfb488c07d7b098f8b3266b65a8e24f999c1ff95fe`；recipe_sha256 `52b525c768a84a080e0d20f68e3c46e079a3a250ef4e471229c954ba05bba35a`；supervision_sha256 `9ab81b055839c61e1b6c3230550553644cec24f039abc3cd2cea69fbec00867f`
- data_manifest_sha256 `6a7e1b6757faed3d8523a3ac5dfdc9402ab97bf51f3550ca597229f9130c5617`；split_sha256 `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b`

## 训练

- 随机初始化；train 样本 175；完成步数 30000；状态 COMPLETED；选择规则 last
- 最终检查点 sha256 `2c86983adb512f357ad070897efdba056bfcde7ec3642320746e2c87c934f989`

## 开发集评估

- split: development_val
- scientific_conclusion: NOT_EVALUATED
- 只作开发集描述：逐例指标先在患者内平均，再患者等权；不计算区间（区间属于 T14 的 H1 比较）。
- 各模块结果分开报告，不相加为同例联合成功（Proposal §1.1）。
- 患者数：38；扫描数：38
- acceptance_policy_id: isles_u_engineering_v1
- split_sha256: 02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b
- run_sha256: 80c484cbe6ddd415877e51cfb488c07d7b098f8b3266b65a8e24f999c1ff95fe
- checkpoint_sha256: 2c86983adb512f357ad070897efdba056bfcde7ec3642320746e2c87c934f989
- reference_set_hash: 3d7977cfea1956621fef9ee7507ede49ece1c238c5d9bb5ce9745cdd1e64dcd2
- metric_policy_sha256: a415a422a2ae2c98f7424a0b561a0fc955993c7a273959756512d72721ac9118

## U（异常）分割

| 类别 | 指标 | 患者等权均值 | 有值患者数 | 有值扫描数 | None 扫描数 |
| --- | --- | --- | --- | --- | --- |
| ischemic_infarct | dice | 0.752 | 38 | 38 | 0 |
| ischemic_infarct | hd95_mm | 15.795 | 37 | 37 | 1 |
| ischemic_infarct | nsd | 0.881 | 37 | 37 | 1 |

## U 病灶检测（患者内按 T06c 汇总，再患者等权）

| 指标 | 患者等权均值 | 有值患者数 | 有值扫描数 | None 扫描数 |
| --- | --- | --- | --- | --- |
| pooled_sensitivity | 0.641 | 37 | 37 | 1 |
| fp_per_scan | 2.342 | 38 | 38 | 0 |

- 读 train 与 development_val；locked_test 未读；`scientific_conclusion: NOT_EVALUATED`。
