# GPU 实测：isles_u

- 设备：NVIDIA A800 80GB PCIe × 1
- 每个候选分两段、各用新建模型测量，每段先预热一次再计时一次：
  1. 前向+反向；
  2. 完整训练步，与 train_v5 相同：autocast 前向、logits 有限性检查、损失、（scaler）反向、optimizer.step、scaler.update、scheduler.step；优化器与调度器新建，峰值显存包含 AdamW 状态的建立。
- 只有完整训练步成功才记 OK；仅前向+反向成功不能作为锁定配置的依据。OOM 或非有限值记录失败阶段后进入下一项，同一配置不重试、不重复列出。
- 本报告只含设备、显存、吞吐、配置与通过与否。

| patch_size | batch_size | precision | 状态 | 前向+反向峰值 (GiB) | 前向+反向秒数 | 完整训练步峰值 (GiB) | 完整训练步秒数 | 样本/秒 | 含优化器状态 | GradScaler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 112×112×64 | 2 | fp32 | OK | 5.99 | 0.292 | 6.19 | 0.276 | 7.24 | 是 | 关 |
| 112×112×64 | 2 | amp_bf16 | OK | 3.12 | 0.189 | 3.32 | 0.201 | 9.96 | 是 | 关 |
| 128×128×64 | 2 | amp_bf16 | OK | 4.03 | 0.248 | 4.23 | 0.268 | 7.46 | 是 | 关 |
| 112×112×64 | 4 | amp_bf16 | OK | 6.08 | 0.342 | 6.29 | 0.338 | 11.85 | 是 | 关 |

## 小样本过拟合检查

- 步数：500（完成 500）
- 损失：初始 1.0900 → 末 0.0617；下降比例 0.943（判据 ≥ 0.500；首末各 20 步平均）
- Dice（样本与类别均值）：U 0.929（判据 ≥ 0.500）
- 结果：PASS

## 冒烟运行

- 计划：`smoke_isles_u_v1`，sha256 `02b5c2603ac78bb55ddc22d7d71a3851c517a6bd13dcdb9e60e0f6455b0933d8`
- 结果：PASS
- 用时 0.057 h（预算 2.00 h）
- 选定候选：patch 112×112×64，batch 2，fp32
- 小样本：规则 `hash_rank_v1`，盐 `v51-smoke-isles_u-20261109`，2 例，层：volume_t2（只读 train）

### 真实 batch 核对

- 通道按 scope 顺序：是（dwi、adc）
- 模型网格即原网格：是；FOV 占比均值 1.000
- 标签映射：isles_binary_v1 2
- 输入可见性：U unknown 2
- 监督支持（小样本合计）：U 阳性 2431，e_pos 1831424，e_neg 1831424
- 一个 batch：形状 2×2×112×112×64，有限值 是，FOV 占比 0.662；U 阳性 2431 / 受监督 1062986

### 登记与训练

- run_sha256 `e06d4726c20428beb4768bd905d760498e41934d15ce1892f6c060dd82e57e28`；recipe_sha256 `8bb845c3bb65ae25f82e9616f4bd98d9208e0d44e7f522b1600131b32ecbb5cf`
- supervision_sha256 `9ab81b055839c61e1b6c3230550553644cec24f039abc3cd2cea69fbec00867f`；data_manifest_sha256 `bf51616b70ad95fd2a7b75dc44ffb0d7085152d9cf1e63b93a5206fc5ed97a18`；split_sha256 `02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b`
- 连续训练：COMPLETED，步数 500
- 暂停与恢复：暂停 PAUSED 于第 300 步；恢复后 COMPLETED

### 恢复核对

- 精确一致：step 是；sampler 是；scheduler 是；scaler 是；rng 是；data_sha256 是；optimizer_counters 是；steps_after_pause 是
- 比较步数 200；最大损失差 0.00000000；参数最大相对差 0.00000000；AdamW 矩最大相对差 0.00000000；参数逐位一致 是
- 容差：loss_atol 0.00010000；loss_rtol 0.00100000；param_rtol 0.00100000
- 结果：PASS

### 全体积推理

- 经 `infer_and_save_v5` 落盘并通过 `validate_module_output_v5`；按哈希重新读取 是；原网格形状一致 是
- 接受占比 1.000；与发布标签的 Dice 均值 0.900（1 类，训练样本，只作冒烟核对，不是开发集结果）
- 输出记录 sha256 `f2573f3b85000764f8c6214f3eadb3ad3c3d1156d25588e9391ea529e0e44abf`

- 只读 train；`scientific_conclusion: NOT_EVALUATED`。
