# GPU 实测：skm_a

- 设备：NVIDIA A800 80GB PCIe × 1
- 每个候选分两段、各用新建模型测量，每段先预热一次再计时一次：
  1. 前向+反向；
  2. 完整训练步，与 train_v5 相同：autocast 前向、logits 有限性检查、损失、（scaler）反向、optimizer.step、scaler.update、scheduler.step；优化器与调度器新建，峰值显存包含 AdamW 状态的建立。
- 只有完整训练步成功才记 OK；仅前向+反向成功不能作为锁定配置的依据。OOM 或非有限值记录失败阶段后进入下一项，同一配置不重试、不重复列出。
- 本报告只含设备、显存、吞吐、配置与通过与否。

| patch_size | batch_size | precision | 状态 | 前向+反向峰值 (GiB) | 前向+反向秒数 | 完整训练步峰值 (GiB) | 完整训练步秒数 | 样本/秒 | 含优化器状态 | GradScaler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192×192×48 | 2 | fp32 | OK | 15.98 | 0.452 | 16.18 | 0.453 | 4.41 | 是 | 关 |
| 192×192×48 | 2 | amp_bf16 | OK | 8.18 | 0.338 | 8.38 | 0.340 | 5.88 | 是 | 关 |
| 256×256×32 | 2 | amp_bf16 | OK | 9.66 | 0.400 | 9.86 | 0.401 | 4.99 | 是 | 关 |
| 256×256×48 | 2 | amp_bf16 | OK | 14.40 | 0.593 | 14.60 | 0.596 | 3.36 | 是 | 关 |

## 小样本过拟合检查

- 步数：500（完成 500）
- 损失：初始 1.7630 → 末 0.7335；下降比例 0.584（判据 ≥ 0.500；首末各 20 步平均）
- Dice（样本与类别均值）：A 0.607（判据 ≥ 0.600）
- 结果：PASS

## 冒烟运行

- 计划：`smoke_skm_a_v1`，sha256 `93ce6daaa21654e7e729b2e6796095ccccb075a5e246bc77bbc34062a1d725c0`
- 结果：PASS
- 用时 0.332 h（预算 2.00 h）
- 选定候选：patch 192×192×48，batch 2，fp32
- 小样本：规则 `hash_rank_v1`，盐 `v51-smoke-skm_a-20261109`，2 例，层：全部（只读 train）

### 真实 batch 核对

- 通道按 scope 顺序：是（qdess_e1、qdess_e2）
- 模型网格即原网格：是；FOV 占比均值 1.000
- 标签映射：skm_a6_h5_channels_v1 2
- 输入可见性：A visible 2
- 监督支持（小样本合计）：A 阳性 788972，e_pos 503316480，e_neg 503316480
- 一个 batch：形状 2×2×192×192×48，有限值 是，FOV 占比 0.822；A 阳性 118283 / 受监督 17452800

### 登记与训练

- run_sha256 `372cd5c2dcffcbd2d945196bef3cf0c37db1939e2896c52b4bbf393fcc16075c`；recipe_sha256 `0a48db770e4e7307803969d0e561499dcb1e46862f7de39cbf6ae4265101cb9e`
- supervision_sha256 `ac3cfc76f5db68e065c0c1197fa194c96008fb75fb2fca38528620cfb18a91d5`；data_manifest_sha256 `ebb7c4c8bdc663b2f95df7b837a9767044925be959f1a89c34e5ca88d4093b1e`；split_sha256 `6a2c955c9764472c55311513766ea9791ee3cbdc9a12178982151990b461eb61`
- 连续训练：COMPLETED，步数 500
- 暂停与恢复：暂停 PAUSED 于第 300 步；恢复后 COMPLETED

### 恢复核对

- 精确一致：step 是；sampler 是；scheduler 是；scaler 是；rng 是；data_sha256 是；optimizer_counters 是；steps_after_pause 是
- 比较步数 200；最大损失差 0.00000000；参数最大相对差 0.00000000；AdamW 矩最大相对差 0.00000000；参数逐位一致 是
- 容差：loss_atol 0.00010000；loss_rtol 0.00100000；param_rtol 0.00100000
- 结果：PASS

### 全体积推理

- 经 `infer_and_save_v5` 落盘并通过 `validate_module_output_v5`；按哈希重新读取 是；原网格形状一致 是
- 接受占比 1.000；与发布标签的 Dice 均值 0.588（6 类，训练样本，只作冒烟核对，不是开发集结果）
- 输出记录 sha256 `e0c244678cfd18610b5c4cbd3145f15f868cef575ca77724bd2ae7ca4c141f5b`

- 只读 train；`scientific_conclusion: NOT_EVALUATED`。
