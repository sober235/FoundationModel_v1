# v5.1 真实 T07/T08 运行与 G-DATA 证据报告（2026-11-08，amax）

- 计划：`plan_v5.md` §2B（T07–T08）与 §2C T10 的真实运行前置；规格：`Proposal_v5.md`。
- 分支：`docs/v51-stage-b1-report-20261107`，对应 PR #2（目标 `restart-dev`，未合并）。`main` 和 `restart-dev` 都没有改动。
- 本轮提交：`dfbef2d..90065d4`，共 11 个，已推送到 origin。
- 数据运行状态：T07 和 T08 已在真实数据上完成，G-DATA 6 项证据 PASS。G-DATA 的批准还没签署，所以关卡还没评估。
- 科学结论：NOT_EVALUATED
- 逐任务状态见 `reports/v5/status.md` 的“真实 T07/T08 运行”一节。本报告汇总这一轮的上下文、裁定和待办。

本报告只含计数、哈希和规则，不含病例 ID、私有路径或数据文件名。真实路径只写在服务器本地配置里。

## 研究者授权（2026-11-08）

| 授权项 | 内容 |
| --- | --- |
| 写服务器 | 同意在 amax 上新建一个独立工作目录，代码以 git bundle 克隆进去；私有产物写在私有根下新建的子目录（目录 700，文件 600，不覆盖）。旧检出不动。 |
| GPU | 同一时间最多用 2 张空闲 GPU。本轮只跑 CPU，还没用 GPU。 |
| 数据使用（G-DATA） | 只批准 ISLES-2022 和 SKM-TEA，SPIDER 不批准。批准文件由研究者本人写入，**尚未写入**。 |
| L8 划分规则 | SKM-TEA 用官方划分。ISLES-2022 按患者病灶体积的三分位分层，比例 70/15/15，种子 `20261108`，每层至少 5 组，不足的层合并。 |

不读的内容：服务器上的派生目录（derived）和凭据脚本。

## 执行环境

- amax，服务器解释器为 Python 3.11 / PyTorch 2.5.1+cu121。
- 代码同步方式：本地生成 git bundle → 复制到服务器私有传输目录 → `git fetch <bundle>` 再 `git merge --ff-only`。服务器不连 GitHub。
- 服务器上 CPU 跑全套测试：585 passed，运行后检出无改动。据此 D4 登记为 `DONE` / `PASS`，`EXECUTION_HOST_UNVERIFIED` 撤销。详见 `host_precheck_amax.md`。
- 运行约定：
  - `CUDA_VISIBLE_DEVICES=` 置空，同时设 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`，`umask 077`；
  - 每个输出目录新建，已存在即失败（不覆盖）；
  - 标准输出只打印计数。

## 流程

1. **清单**：复用 v41 已有的资产清单，只筛出这两个来源，不重新扫描数据树。
2. **T07 prepare**（`python -m anatobind.v41.real_t07_v5 prepare`）完成以下工作：
   - 几何 QC 与标签取值 QC，可用 `--reuse-qc` 复用上一轮结果；
   - SKM 通道与变换核验；
   - 降采样损失报告；
   - 写出准备记录和脱敏的来源核验摘要。
3. **T07 migrate**（`... migrate`）：v41 记录 → v5 记录。迁移时绑定来源 release、scope 的模态和几何。
4. **T08 冻结划分**（`python -m anatobind.v41.real_t08_v5`）：按登记的规则冻结划分；私有注册表和脱敏摘要分开写。
5. **G-DATA 证据**（`python -m anatobind.v41.real_gdata_v5`）：根据以上计数逐项判定 6 项证据，并生成一份脱敏 md；每项证据都带这份 md 的 sha256。

## 代码与配置交付

| 提交 | 内容 | 测试 |
| --- | --- | --- |
| `dfbef2d` | 来源政策新增 `source_release` 字段。迁移时三类不一致都会拒绝：记录的 release 与政策不符（`SOURCE_VERSION_MISMATCH`）；政策模态与 scope 输入不符（`MODALITY_SCOPE_MISMATCH`）；几何标为 pass 却没有 `image_geometry`（`GEOMETRY_MISSING`）。关闭 §1.6 的 L1–L3。 | `test_source_policy.py`（11）、`test_migrate_v41.py`（23） |
| `7a76512` | 真实 T07 驱动 `real_t07_v5.py`：prepare 与 migrate 两步。 | `test_real_t07.py` |
| `88c25d9` | 真实 T08 驱动 `real_t08_v5.py`：官方划分取自 release；分层按病灶体积三分位，体积单位为 mL。 | `test_real_t08.py`（5） |
| `034340b` | G-DATA 证据驱动 `real_gdata_v5.py`：6 项规则（readability、semantics、identity、pairing、task_geometry、provenance）。 | `test_real_gdata.py` |
| `a7be53a` | 真实来源政策 `configs/v5/sources_v5.yaml` 中的 ISLES 部分，以及 L8 划分规则 `configs/v5/split_rule_isles_u.yaml` 和 `split_rule_skm_a.yaml`。 | — |
| `031a280` | 标签 QC 键里的竖线会把 markdown 表格拆坏，改为显示成斜线。 | `test_real_gdata.py`（共 10） |
| `47373b9` | 接受 SKM 的 z 反转 DICOM 轨道（`transpose_xy_flip_z`），并配上对应的仿射矩阵；通道核验只统计已登记的 DICOM 轨道。 | `test_real_t07.py`（共 21） |
| `1290b50` | 真实来源政策中的 SKM-TEA 部分，引用其来源核验摘要的 sha256。 | — |
| `99c30b8`、`4f369ac` | 记录 ISLES 和 SKM 的划分摘要与 G-DATA 证据 md，内容与私有产物逐字节一致。 | `test_status_docs.py` |
| `90065d4` | status、host 预检、plan §1.6 同步更新；新增报告脱敏守护测试。 | `test_status_docs.py` |

全套测试 621 passed、1 skipped。测试命令：`env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python3 -m pytest -q -p no:cacheprovider tests`。

## 结果

| 项 | ISLES-2022 → `isles_u`（任务 U） | SKM-TEA → `skm_a`（任务 A） |
| --- | --- | --- |
| 来源版本 | `isles2022_bids1.6.0`，release `ISLES-2022` | `skm_tea_annotations_v1.0.0`，release `SKM-TEA v1.0.0` |
| 配对 | 250/250 | 155/155 |
| 几何 | 图像与 mask 共 500/500 一致 | 155/155 pass。H5 与 NIfTI 的关系：137 例为 x/y 互换，18 例在此基础上 z 反转 |
| 通道顺序 | — | 155/155 一致（通道 c 对应标签 c+1） |
| 标签取值 | 247 例通过；3 例 mask 全零，保留为完整阴性 | 155 例通过 |
| 迁移 | 250/250；索引对齐和物理几何均为 verified | 155/155；索引对齐 verified，物理几何 unknown |
| 降采样损失 | 2×2×2：147 例中共有 846 个病灶连通域消失 | 2×2×1：薄软骨和半月板共有 151 个连通域消失 |
| 冻结划分 | train 175 / development_val 38 / excluded 37 / locked_test 0 | train 86 / development_val 33 / excluded 36 / locked_test 0 |
| 分层 | 三分位阈值 3.368 / 14.904 mL；各层 84/83/83 组 | 官方划分，不分层 |
| `split_sha256` | `02bc40f0…a790a1b` | `6a2c955c…b461eb61` |
| G-DATA 证据 | 6 项 PASS | 6 项 PASS |

相关脱敏报告：`source_check_isles_2022.md`、`source_check_skm_tea.md`、`split_isles_u.md`、`split_skm_a.md`、`gdata_evidence_isles_2022.md`、`gdata_evidence_skm_tea.md`。

## 运行中遇到的问题与处理

| 问题 | 处理 |
| --- | --- |
| SKM 第 1 轮有 18/155 例被判为几何失败：当时只接受 x/y 互换，而这 18 例的 DICOM 轨道还有 z 反转。 | 新增 `transpose_xy_flip_z` 变换，配上相应的仿射矩阵（z 行为 `[0,0,-1,Z-1]`），用 `--reuse-qc` 重跑后 155/155 通过。第 1 轮输出保留，不覆盖。 |
| SKM 第 1 轮的通道核验显示 0 例一致，原因是把未登记的原始轨道也计算在内。 | 改为只统计已登记的 DICOM 轨道。 |
| ISLES 的 G-DATA md 表格被标签 QC 键里的竖线拆坏。 | 先写失败测试再修复，修复后在新目录重新生成证据，旧目录保留。 |
| 服务器的 zsh 不会对 `$VAR` 做分词，`PY="env ... python"` 这种写法失效。 | 环境变量改用 export，`PY` 只放解释器路径。 |
| 等待循环用的 `pgrep -f` 匹配到循环自身，永远不结束。 | 改用 `[r]eal_t07_v5` 这种方括号写法，避免匹配自身。 |

## 代研究者做出的裁定（可推翻）

| 裁定 | 理由 | 若错的代价 |
| --- | --- | --- |
| ISLES 的 `input_visibility` 记为 `unknown`，不默认填 visible。 | Proposal §2.3 不允许伪填 visible。 | 只影响可见性相关的统计；改回需要重新迁移。 |
| SKM 物理几何记为 unknown，毫米指标记为 NA。 | 头文件没有单位，政策设为 `physical_units_policy: null`。 | 若能证实单位，需要补登政策并重新迁移。 |
| SKM 的标注模态登记为 `qdess_e1` 和 `qdess_e2`。 | 数据集文档没写分割是在哪个回波上画的。 | 若只对应其中一个回波，需要改政策并重新迁移。 |
| `skm_a` 不纳入 U 框。 | B2 的真实基线只有 A-only（plan §2C）。 | 无。框级 U 留给 T11/T12。 |
| 被遗留曝光阻断的组移到 `excluded`，不并入 train。 | 保守处理，不让曝光状态不明的数据进入训练。 | train 少 37 组（ISLES）和 36 组（SKM）。 |
| 复用 v41 资产清单，不重新扫描数据树。 | 避免读取派生目录。 | 若清单不完整会漏样本；配对结果显示两源均为全数配对。 |
| 降采样损失报告用的候选因子：ISLES 2×2×2、SKM 2×2×1。 | 用于看损失量级，不是配方。 | 无。配方建议两源都用 1×1×1，待批准。 |
| G-DATA semantics 把全零 mask 计为通过。 | 3 例保留为完整阴性，并在报告中列出。 | 若研究者要求排除，需要重新冻结划分。 |

## 阻断项与待办

1. **签署 G-DATA 数据使用批准**（研究者本人写入，agent 不代填）。在服务器的私有批准目录写一个 YAML 文件（目录 700，文件 600），结构如下：

   ```yaml
   isles_u:
     G-DATA:
       data_use:
         approved_by: <研究者>
         approved_at: "<日期>"
         scope_hash: 0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb
         source_id: isles_2022
         source_version: isles2022_bids1.6.0
   skm_a:
     G-DATA:
       data_use:
         approved_by: <研究者>
         approved_at: "<日期>"
         scope_hash: c70f7dca4c94ea4454ce0a41794d931a2a81fb27db1e605b18ca24e98f85bfd7
         source_id: skm_tea
         source_version: skm_tea_annotations_v1.0.0
   ```

2. **评估 G-DATA**：批准写好后，用 `evaluate_gate_v5` 生成私有的判定 JSON 和脱敏 md。
3. **`LEGACY_EXPOSURE_UNKNOWN:isles_u,skm_a`**：所有 v41 记录都带 `legacy_foundation_model_unverified`（角色 unknown），按 Proposal §3.3 会阻断 locked_test，因此两个 scope 的 locked_test 都是 0。
   - B2 只用开发集，不受影响。
   - 到 G-LOCKED 之前，研究者需要核实旧模型的实际曝光，然后按同一规则和种子重新冻结。
4. **T10 smoke 驱动**（本地 TDD，尚未开始），内容：
   - 登记一次 smoke 运行（可读角色只有 train）；
   - 在 GPU 上测量候选配置；
   - 小样本过拟合检查；
   - 一次全体积推理，并用 `validate_module_output_v5` 校验；
   - 从检查点恢复后与连续训练的结果对比；
   - 生成 `reports/v5/gpu_measurement_<scope>.md`。
5. **请研究者审批的配方提案**：
   - 两个来源 `downsample_factor` 都用 `[1,1,1]`；
   - ISLES 的监督使用发布的参考标签，`allow_unknown_visibility: true`；
   - 验收策略、候选配置、过拟合判据。
   - GPU 资源批准同样由研究者签署。
6. 之后的顺序：锁定 patch、batch 和精度 → G-BASE-A（`skm_a`）和 G-BASE-U（`isles_u`）→ 基线运行 → `module_baseline` 报告。
