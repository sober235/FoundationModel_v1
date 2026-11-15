# T10 真实基线运行报告（2026-11-10，amax）

本报告汇总批准 C 签署、G-BASE 关卡和两个模块基线的真实运行。只含配置、计数、指标、哈希与时长；不含样本或患者 ID、路径或文件名。`scientific_conclusion: NOT_EVALUATED`；`core_target_validated` 保持 `null`。两个基线分开训练、分开评估、分开报告，结果不相加为同例联合成功（Proposal §1.1）。

## 1. 批准 C

- 文件：私有根 `approvals/` 下的 `gbase_resources_20261109.yaml`（目录 700，文件 600）。
- 签名 CONGCONG，日期 2026-11-09，按关卡分键（`isles_u.G-BASE-U.resources`、`skm_a.G-BASE-A.resources`），各写明 `scope_hash` 与 `baseline_plan_sha256`。
- 写入前用 `real_baseline_v5 plan-hash` 重新计算两份计划的哈希，与 `t10_approval_c_20261109.md` 一致：isles_u `96040004dd1afce9a10c79e30b308fa49d47654b67205151412bf0479b96b8de`，skm_a `3809a1cea78312f94358df1de9581511d56f1add34bd2554a6b76421b67d53bf`。
- 记录偏差：该文件由执行代理在研究者本次会话内的明确授权下写入，签名为研究者本人。`t10_approval_c_20261109.md` 原规定由研究者本人写入；本次为研究者明确授权的例外，不作为后续惯例。
- 代码：`c7cd0c1`（与 GitHub 前沿一致，工作树干净）。

## 2. G-BASE 关卡

| scope | 关卡 | 结果 | 阻断 |
| --- | --- | --- | --- |
| isles_u | G-BASE-U | PASS | 无 |
| skm_a | G-BASE-A | PASS | 无 |

五项证据（g_data、supervision_policy、input_policy、frozen_dev_split、training_smoke）沿用 2026-11-09 预评估的输入；本次只是加上批准 C 后重新评估并输出到新目录。

## 3. 基线运行

| 项 | isles_u（U） | skm_a（A） |
| --- | --- | --- |
| 计划 | `baseline_isles_u_v1`，sha256 `96040004…` | `baseline_skm_a_v1`，sha256 `3809a1ce…` |
| 结果 | COMPLETED | COMPLETED |
| 设备 | NVIDIA A800 80GB PCIe × 1 | NVIDIA A100 80GB PCIe × 1 |
| 用时（整体） | 1.477 h（预算 5.00 h） | 4.331 h（预算 8.00 h） |
| 配置（smoke 冻结） | patch 112×112×64，batch 2，fp32 | patch 192×192×48，batch 2，fp32 |
| 训练 | 随机初始化；train 175；30000 步完成；选择规则 last | 随机初始化；train 86；20000 步完成；选择规则 last |
| 开发评估 | development_val 38 患者 / 38 扫描，整卷推理、原网格、患者等权 | development_val 33 患者 / 33 扫描，同左 |
| 报告 | `module_baseline_isles_u.md`（sha256 `93467cb664be1eb44666f900d77bdea8993c5f5ae38d3dff26a9f906bf1d7409`） | `module_baseline_skm_a.md`（sha256 `6171e1381e78f42fc4725e9ff87f2400a217ce07bba1f1f47c86dd137a27aee5`） |

- 启动方式：isles_u 先启动；skm_a 在 isles_u 写出第一个检查点（第 2000 步，约 7 分钟后）后启动，两卡同时占用，在"同时最多 2 张空闲 GPU"的授权内。
- 读取的数据：train 登记为 `training` 曝光，development_val 登记为 `model_selection` 曝光；locked_test 未读。
- 没有失败、没有重试，计划在运行期间没有改动。
- 两份 `module_baseline.md` 由驱动在私有输出目录生成，入库前做了隐私正则核查，再逐字节复制到 `reports/v5/`。

### 偏差：skm_a 使用 A100

计划 `baseline_skm_a.yaml` 的资源描述写的是 A800。运行时空闲的第二张卡为 A100 80GB，研究者在启动前明确同意改用。两者显存相同；计划文件、哈希、配方、步数、预算均未改动，所以不需要重签。该偏差只影响时长估计，不影响指标定义。

## 4. 开发集结果（患者等权，只作描述）

isles_u，类别 `ischemic_infarct`：

| 指标 | 均值 | 有值患者 | None 扫描 |
| --- | --- | --- | --- |
| dice | 0.752 | 38 | 0 |
| hd95_mm | 15.795 | 37 | 1 |
| nsd（2.0 mm） | 0.881 | 37 | 1 |
| 病灶 pooled_sensitivity | 0.641 | 37 | 1 |
| 病灶 fp_per_scan | 2.342 | 38 | 0 |

skm_a，Dice（物理几何 unknown，mm 指标按规则 NA，未请求）：

| 类别 | dice | 有值患者 |
| --- | --- | --- |
| patellar_cartilage | 0.887 | 33 |
| femoral_cartilage | 0.869 | 33 |
| tibial_cartilage_medial | 0.855 | 33 |
| tibial_cartilage_lateral | 0.861 | 33 |
| meniscus_medial | 0.831 | 33 |
| meniscus_lateral | 0.831 | 33 |

- 这些数值只是开发集描述，不计算区间；区间与比较属于 T14。
- isles_u 有 1 例扫描的 hd95/nsd 为 None（预测或参照为空时表面指标无定义），病灶灵敏度同样 1 例无值；它们按 T06a/T06c 的空值规则登记，不删除病例。
- 不形成正式阈值、容差或临床判定，不做 locked_test 评价。

## 5. 状态变化

- T10 `data_run_status`：PARTIAL → PASS（smoke 与两个真实基线均完成，报告已入库）。
- T10 `approval_status`：批准 A、B、C 均已由研究者签署。
- `scientific_conclusion` 保持 NOT_EVALUATED；`core_target_validated` 保持 `null`；blockers 不变。

## 6. 下一步

- B2 完成判定要求对整段 B2 做一次独立审查（plan_v5 §2C）。
- T11：SKM-TEA 真实导入与 SPIDER；`skm_joint_early` scope 与来源绑定仍为提案，需单独批准与新的 G-DATA。
- 核实旧模型实际曝光，才能按同一规则与种子重新冻结并恢复 `locked_test`（blocker `LEGACY_EXPOSURE_UNKNOWN`）。
- D5 待研究者决定。
