# AnatoBind-MRI / FoundationModel_v1

## 当前版本：v5.1（2026-11-03）

研究目标：在同一MRI观察上交付并独立验证解剖A、异常范围U、组织状态S和核心组织归属R；最终同一checkpoint在严格单切片与三维模式下分别通过A/U/S/R八项能力验收。S可由A/U和接受策略组合，不强制增加网络头。

| 入口 | 用途 |
|---|---|
| [Proposal_v5.md](Proposal_v5.md) | 当前科学规格、数据/监督条件、H1/H2/H3、评价和完成定义 |
| [plan_v5.md](plan_v5.md) | 当前执行计划；T00–T18的依赖、接口、反例、验证与交付 |
| [v5.1补齐说明](docs/v5.1_core_target_alignment.md) | 核心目标—输出—直接评价—任务—完成状态的追踪 |
| [v4.1归档](docs/archive/v4.1/README.md) | 原spec/plan逐字节保留及历史证据入口 |

**文档优先级：** v5后续实施以Proposal_v5为科学依据、plan_v5为执行依据；根目录spec.md和plan.md仅作导航。旧文档中的唯一规格/初始TODO/旧启动命令仅适用于旧版本，不能覆盖v5。

## 当前边界

执行基线：GitHub `restart-dev`（自 `d7cbd5b` 起），见 plan_v5.md 2026-11-06 修订；`9bef283`/`02a2447` 不再作为执行依据。v5.1 阶段 A（T00–T06c：契约、签名资格、评价器，至 `14c8a9d`）与 §2A 补修（FX1–FX4，至 `4a5ae80`，其后为终审修复）已在合成数据上实现并通过测试，详见 [阶段 A 完成报告](reports/v5/stage_a_report.md) 与 [任务状态](reports/v5/status.md)；尚未接触真实数据，全部科学结论仍为 `NOT_EVALUATED`，`core_target_validated` 为 `null`。D1–D4 已由研究者决定，但能力容差未冻结、148 主机未核验、专家资源未落实。

历史G0报告登记ISLES 250条U和SKM 155条A，SKM异常仍主要是框；405 admitted不等于405个同例精标病例。历史144项测试通过不等于v5全部验收通过。实际状态见 [G0 v3历史报告](reports/v41/g0_decisions_v3.md) 与v5任务核对结果。

先核对已有实现，再关闭监督资格、最大基数匹配、固定参照分母和患者等权统计缺口；并行推进输入/划分和真实专家试点。H2与双模式工程可按依赖并行，不代替同例独立验证。不得因本次文档提交而宣称上述任务已经完成。

## 核心验收

联合推理必须输出并保存 `PerceptionOutput`，包括 `tissue_state_map`；A不确定时保留异常U和实例，不能抹去异常或判正常。模块基线显式使用ModuleOutput，不伪装完整结果。S用同例独立StateReference直接评价；J提高不能替代正常—异常组织分离验收。辅助near/nearest不能代替核心归属。

最终分别报告 `capability_status[2d/3d][A/U/S/R]`、`core_target_validated`、`hypothesis_supported`、`study_package_complete`。软融合H1未获额外增益，不自动否定合格模型的核心能力；只完成H2或文档/单元测试，也不能认证核心目标。本次没有新增真实能力验证结果。

## 给执行者

先读Proposal_v5和plan_v5，执行T00，再按依赖推进。临床定义、资源预算、外部下载、测试解封需要各自实际批准；文档发布授权不替代这些许可。原始数据与旧结果只读，医学数组、患者级清单、权重及预测不得进入Git。

代码仍在 `anatobind/v41/`，不要仅因文档升级而复制整个代码包。新schema/config及CLI是计划中的v5合同，只有实现并测试后才能使用。所有状态分别报告代码、测试、真实数据运行、批准和科学结论。
