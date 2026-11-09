# T10 待批准材料（2026-11-09，一次性提交）

这份材料供研究者审阅签署。签署前，所有不依赖批准的工作都已完成，见 `t10_preapproval_20261109.md`。

- 批准文件由研究者本人写入服务器的私有根下的 `approvals/` 子目录（目录 700，文件 600）。代理不代填 `approved_by`，也不代写任何批准文件。
- 下文用 `$V` 表示私有根、`$R` 表示原始数据根。两者的实际值只写在服务器本地，不进仓库。
- 科学结论：NOT_EVALUATED。仅完成 smoke 时，T10 的真实基线不标完成。

## 1. 已有并可复用的授权（2026-11-08）

| 授权 | 范围 | 本材料如何使用 |
| --- | --- | --- |
| 写服务器 | 独立工作目录，加上私有根下新建的子目录（不覆盖） | smoke 和门控的输出都写到新建的 `$V/t10/…` 目录 |
| GPU | 同一时间最多 2 张空闲卡 | 每个 smoke 用 1 张卡。这只是数量上限，**不等于**任何训练配方已获批准 |
| 数据使用（口头） | 只用 ISLES-2022 和 SKM-TEA，不含 SPIDER | G-DATA 关卡仍需要下面 A 项的书面文件 |
| L8 划分规则 | 已按规则冻结 | smoke 和基线都直接复用 T08 的冻结划分与曝光记录，不改动、不删除遗留的 exposure unknown |

## 2. 实际缺少的批准（3 项）

### A. G-DATA 数据使用：现在可签

- 文件：`$V/approvals/gdata_data_use_20261109.yaml`。日期可以改，执行命令时会照实际文件名填写。
- 签署的对象：每个 scope、每个来源的六项 G-DATA 证据。这些证据已在本轮复核，字节与已提交的报告相同：
  - `gdata_evidence_isles_2022.md`，sha256 前缀 `5292f5cf`；
  - `gdata_evidence_skm_tea.md`，sha256 前缀 `51deb748`。

```yaml
isles_u:
  G-DATA:
    data_use:
      approved_by: <研究者本人填写>
      approved_at: <YYYY-MM-DD>
      scope_hash: 0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb
      source_id: isles_2022
      source_version: isles2022_bids1.6.0
skm_a:
  G-DATA:
    data_use:
      approved_by: <研究者本人填写>
      approved_at: <YYYY-MM-DD>
      scope_hash: c70f7dca4c94ea4454ce0a41794d931a2a81fb27db1e605b18ca24e98f85bfd7
      source_id: skm_tea
      source_version: skm_tea_annotations_v1.0.0
```

### B. smoke 资源：现在可签，每个 scope 一份

- 文件：`$V/approvals/t10_smoke_resources_20261109.yaml`。
- 签署的对象是对应计划文件的全部内容，由 sha256 绑定。计划文件改动任何一处，sha256 就会变，需要重新批准；注释不计入哈希。
  - isles_u：`configs/v5/proposals/smoke_isles_u.yaml`
  - skm_a：`configs/v5/proposals/smoke_skm_a.yaml`
- smoke 不会打开 G-BASE 关卡：G-BASE 只读取 C 项。

```yaml
isles_u:
  resources:
    approved_by: <研究者本人填写>
    approved_at: <YYYY-MM-DD>
    scope_hash: 0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb
    smoke_plan_sha256: 02b5c2603ac78bb55ddc22d7d71a3851c517a6bd13dcdb9e60e0f6455b0933d8
skm_a:
  resources:
    approved_by: <研究者本人填写>
    approved_at: <YYYY-MM-DD>
    scope_hash: c70f7dca4c94ea4454ce0a41794d931a2a81fb27db1e605b18ca24e98f85bfd7
    smoke_plan_sha256: 93ce6daaa21654e7e729b2e6796095ccccb075a5e246bc77bbc34062a1d725c0
```

两个计划中需要审阅的值如下，全部是提案：

| 项 | isles_u | skm_a |
| --- | --- | --- |
| 来源 / 任务 | ISLES-2022 DWI+ADC，只做 U（梗死） | SKM-TEA qDESS E1+E2，只做 A（6 类组织） |
| 模型 | resunet3d，宽度 32–320，每级 2 个块，随机初始化 | 同左；第一级下采样步长为 [2,2,1] |
| 种子 | 初始化 20261109，采样 20261110 | 同左 |
| 优化 | AdamW，lr 1e-3，weight decay 1e-4；poly 调度，指数 0.9 | 同左 |
| 监督 | released_reference：U 用 sigmoid，CE 1 + Dice 1；`allow_unknown_visibility: true`，input_visibility 仍为 unknown | released_reference：A 用 softmax，CE 1 + Dice 1；`allow_unknown_visibility: false` |
| 输入 | 原始网格，downsample [1,1,1]；每个体积单独 z-score | 同左。物理几何为 unknown，mm 指标为 NA |
| 正样本采样比例 | 0.67 | 0.5 |
| 候选（按顺序） | [112,112,64] b2 fp32；[112,112,64] b2 bf16；[128,128,64] b2 bf16；[112,112,64] b4 bf16 | [192,192,48] b2 fp32；[192,192,48] b2 bf16；[256,256,32] b2 bf16；[256,256,48] b2 bf16 |
| 候选规则 | 第一个 OK 且完整训练步峰值 ≤ 0.5 × 单卡显存 | 同左 |
| 子集 | `hash_rank_v1`，盐为 `v51-smoke-isles_u-20261109`，2 个 train 样本，只取 `volume_t2` 层 | 盐为 `v51-smoke-skm_a-20261109`，2 个 train 样本，不分层 |
| 步数与检查点 | 500 步，每 100 步保存一次 | 同左 |
| 过拟合标准 | 损失下降 ≥ 0.5（前 20 步与后 20 步的均值比较），子集整卷 Dice ≥ 0.5 | 损失下降 ≥ 0.5，Dice ≥ 0.6 |
| 续训对比 | 第 300 步检查点写完后受控暂停，再用新模型续训到 500 步。计数器必须完全一致；损失 atol 1e-4 / rtol 1e-3，参数和 AdamW 矩 rtol 1e-3；开启确定性模式 | 同左 |
| 推理 | `infer_and_save_v5` 对子集第一个样本做整卷推理，窗口 [112,112,64]、步长 [56,56,32]，恢复到原始网格，然后校验并按哈希重新加载 | 窗口 [192,192,48]，步长 [96,96,24] |
| 接受策略 | 仅作工程判定（不是正式阈值）：阈值 0.5，26 连通，最小 1 体素 | 仅作工程判定：取 argmax |
| 单次训练墙钟上限 | 0.5 h | 0.75 h |
| 整个 smoke 墙钟上限 | 2.0 h | 2.0 h |
| 预计实际训练时间（由预测量推算，不是承诺） | 2 × 500 步 × 约 0.12 s，约 2 分钟 | 2 × 500 步 × 约 0.48 s，约 8 分钟 |
| 读取的数据 | 只读 2 个 train 样本，登记为 run_kind `smoke` 的曝光；development_val 和 locked_test 都不读 | 同左 |

停止规则：
- 任何阶段出现 OOM、非有限值或超时，就停在该阶段，写入 failure 记录，不重试；
- 没有可用候选时报 `NO_USABLE_CANDIDATE` 并停止；
- 每个阶段开始前检查剩余预算，不够时报 `SMOKE_BUDGET_EXCEEDED` 并停止，不会把墙钟用尽；
- 以下情况判为 FAIL 并如实记录，不为了 PASS 修改标准：过拟合未达标、续训超出容差、推理校验失败。

### C. G-BASE 资源与基线预算：smoke PASS 后再签，现在不可签

- 文件：`$V/approvals/gbase_resources_<日期>.yaml`。按关卡分键，与 `configs/v5/approvals.template.yaml` 一致。
- 待签内容：smoke PASS 后，代理提交基线计划提案，内容包括：
  - 冻结的 patch、batch 和精度；
  - 随机初始化，用全部 train 训练；
  - 步数和预算；
  - 在全部 development_val 上按患者等权做开发评估；
  - ISLES 和 SKM 分开报告。
- 研究者签署时写入该提案的 sha256。

```yaml
isles_u:
  G-BASE-U:
    resources: {approved_by: <研究者>, approved_at: <YYYY-MM-DD>,
                scope_hash: 0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb,
                baseline_plan_sha256: <基线提案的 sha256>}
skm_a:
  G-BASE-A:
    resources: {approved_by: <研究者>, approved_at: <YYYY-MM-DD>,
                scope_hash: c70f7dca4c94ea4454ce0a41794d931a2a81fb27db1e605b18ca24e98f85bfd7,
                baseline_plan_sha256: <基线提案的 sha256>}
```

## 3. 批准后的执行顺序

A 和 B 签署后即可执行第 1–5 步；第 6–8 步还需要 C。

服务器上每一步都使用以下环境：

```sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
umask 077
```

只有 GPU 步骤再设置 `CUDA_VISIBLE_DEVICES=<确认空闲的 1 张卡>`。

1. **同步代码。** 用 bundle 把代码同步到包含本材料的提交，在服务器上用 CPU 跑全套测试，再确认两个计划的 sha256 没有变：

   ```sh
   python -m anatobind.v41.real_t10_v5 plan-hash --plan configs/v5/proposals/smoke_isles_u.yaml --scope configs/v5/scope_isles_u.yaml --policies configs/v5/sources_v5.yaml --source-id isles_2022
   ```

   skm_a 同样执行一次，`--source-id` 用 `skm_tea`。

2. **G-DATA 关卡**，两个 scope 各执行一次：

   ```sh
   python -m anatobind.v41.real_t10_v5 gdata-gate --evidence $V/gdata/isles_2022_r3/gdata_evidence.json \
     --approvals $V/approvals/gdata_data_use_20261109.yaml --scope configs/v5/scope_isles_u.yaml \
     --policies configs/v5/sources_v5.yaml --source-id isles_2022 --out $V/t10/gdata_gate_isles_u
   ```

   skm_a 的参数：`--evidence $V/gdata/skm_tea_r2/gdata_evidence.json`、`--scope configs/v5/scope_skm_a.yaml`、`--source-id skm_tea`、`--out $V/t10/gdata_gate_skm_a`。

   如果结果不是 PASS，就在这一步停下并报告阻断项。

3. **smoke isles_u**，先做这个 scope，用 1 张卡：

   ```sh
   python -m anatobind.v41.real_t10_v5 smoke --plan configs/v5/proposals/smoke_isles_u.yaml \
     --scope configs/v5/scope_isles_u.yaml --policies configs/v5/sources_v5.yaml --source-id isles_2022 \
     --raw $R --records $V/t07/isles_migrate/records_v5.jsonl --split-registry $V/t08/isles_u/split_registry.json \
     --gdata-decision $V/t10/gdata_gate_isles_u/gdata_decision.json \
     --approvals $V/approvals/t10_smoke_resources_20261109.yaml --run-id smoke_isles_u_v1 --device cuda:0 \
     --out $V/t10/smoke_isles_u
   ```

   - 如果 FAIL，诊断原因并保留全部记录，然后报告。不修改标准，不自动进入下一步。

4. **smoke skm_a**：
   - 只改这些参数，命令其余部分与第 3 步相同：

     | 参数 | 值 |
     | --- | --- |
     | `--plan` | `configs/v5/proposals/smoke_skm_a.yaml` |
     | `--scope` | `configs/v5/scope_skm_a.yaml` |
     | `--source-id` | `skm_tea` |
     | `--records` | `$V/t07/skm_migrate/records_v5.jsonl` |
     | `--split-registry` | `$V/t08/skm_a/split_registry.json` |
     | `--gdata-decision` | `$V/t10/gdata_gate_skm_a/gdata_decision.json` |
     | `--run-id` | `smoke_skm_a_v1` |
     | `--out` | `$V/t10/smoke_skm_a` |

   - 失败时的处理同第 3 步。

5. **整理 smoke 结果入库。**
   - 每个 smoke 的私有输出目录里会生成 `gpu_measurement.md`。先人工复核它是否脱敏，再复制为 `reports/v5/gpu_measurement_<scope>.md`，并加入脱敏测试。
   - `training_smoke_evidence.json` 留在私有目录里。
   - 按规则选中的 patch、batch 和精度，记为基线提案的冻结值。

6. **提交基线计划提案**，用于 C 项。提案入库后，研究者签署 C。

7. **G-BASE 关卡**，两个 scope 各执行一次：

   ```sh
   python -m anatobind.v41.real_t10_v5 gbase-gate --plan configs/v5/proposals/smoke_isles_u.yaml \
     --scope configs/v5/scope_isles_u.yaml --policies configs/v5/sources_v5.yaml --source-id isles_2022 \
     --gdata-decision $V/t10/gdata_gate_isles_u/gdata_decision.json \
     --split-summary $V/t08/isles_u/split_summary.json \
     --smoke-evidence $V/t10/smoke_isles_u/training_smoke_evidence.json \
     --approvals $V/approvals/gbase_resources_<日期>.yaml --out $V/t10/gbase_gate_isles_u
   ```

   skm_a 对应 G-BASE-A，参数替换方式同第 4 步。

8. **运行基线。** 只有在 G-BASE 为 PASS、C 已签署之后才执行。两个 scope 分开训练、分开评估、分开报告。

## 4. 还没写的代码

- 基线驱动尚未编写。它要做的事：
  - 注册 run_kind `baseline` 的运行；
  - 在全部 train 上训练；
  - 用 `module_eval_v5` 在全部 development_val 上做患者等权评估。
- 它依赖第 5 步冻结的 patch、batch 和精度，会与基线计划提案一起提交。
- smoke 到 G-BASE 这一段的驱动已经完成并有测试。
