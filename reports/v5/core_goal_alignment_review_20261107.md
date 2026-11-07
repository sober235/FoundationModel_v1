# v5.1 核心目标与执行计划对齐评估

- **审查日期：** 2026-11-07（Asia/Shanghai）
- **目标分支：** `docs/v51-stage-b1-report-20261107`
- **审查基线：** `aad3e5e02620ea477f4c9f33c486bfa0299d4e85`
- **审查范围：** 当前科学规格、执行计划、`reports/v5` 报告、相关配置及代码目录；对照 v4.1 目标与范围的演变。
- **证据边界：** 本报告基于指定提交中的文件与实现。测试数量、服务器预检和既往执行结果均引用仓库报告；本次未在 amax 服务器重跑测试、训练模型或重新认证医学性能。
- **文档性质：** 目标对齐评估与执行建议；不替代 `Proposal_v5.md`、`plan_v5.md` 或 `status.md`，不改变现有任务状态、批准记录或科学结论。

## 结论

**相对于当前仓库明确规定的 v5.1 目标，执行计划没有根本偏离；但执行重心需要尽快从工程前置工作转向真实模型与同例联合验证。同时，v5.1 的完成条件只覆盖一个受限场景的联合感知系统，尚不能代表“通用 MRI 基础模型”这一更长期目标已经完成。**

建议保留已完成的 A/B1，优先推进 B2 的真实训练与推理闭环，并同步落实 SKM 专家标注试点；修正 H2 与核心能力验收之间的依赖关系。对长期基础模型愿景，应明确当前阶段的位置，以及后续迁移与泛化需要补充的证据。

## 一、项目的核心目标

按照当前科学规格，项目的核心目标是：

> 在明确的 MRI 输入条件和异常类别范围内，让同一个模型对同一份影像同时回答：这是什么组织、哪里存在目标异常、各组织处于什么状态，以及异常属于哪些组织；最终用同一份权重分别处理严格单切片和三维输入，并通过独立验证。

A/U/S/R 是一个整体任务的技术分解：

| 能力 | 真正需要回答的问题 | 不能替代它的结果 |
| --- | --- | --- |
| A：解剖身份 | 这个位置属于什么组织？ | 当目标要求区分组织时，仅有全器官轮廓不足；未经来源核验的自动标签也不能直接充当独立参照 |
| U：异常范围 | 目标异常实际占据哪些位置？ | 一个检测框，或把框内部全部填成异常 |
| S：组织状态 | 在已定义的异常范围内，该组织是目标阳性、目标阴性，还是不可评价？ | 只给 A/U 两个分数，或把未标注区域视为正常 |
| R：组织归属 | 这个异常属于哪些组织，或符合哪一种明确的累及关系？ | 仅报告“距离哪个组织最近” |

最终必须满足**同例独立参照**，以及**同一 checkpoint 的二维、三维各四项能力验收**。这些要求在当前规格中已明确规定。[依据：研究目标与输出定义][spec-goal]

### 具体例子：U 正确不代表联合感知成功

假设模型在膝关节 MRI 上正确检出一处软骨异常，但把它所在的股骨软骨误认成胫骨软骨：

- U 的异常范围可能仍然正确。
- A 的组织身份错误。
- S 的“组织身份—异常状态”联合结果应受到惩罚。
- R 的异常归属也可能错误。

如果只报告病灶 Dice，这个病例可能看起来成功；但按整体目标，它的联合感知并没有成功。v5.1 增加 S 的直接评价、强调核心 R，正是在补这个缺口。[依据：组织状态直接评价与八项验收][spec-state]

这里的“正常”仅指**在指定输入、可评价范围和已定义目标类别内未见目标异常**，不能解释为组织病理学正常，或排除了所有疾病。[依据：正常状态的范围][spec-normal]

## 二、当前执行进展与证据层级

当前进展可以概括为：**可信的数据与评价基础已经有了软件实现，模型能力本身尚未进入本分支的真实验证。**

| 工作层面 | 当前可核实状态 | 对整体目标的意义 |
| --- | --- | --- |
| 阶段 A：契约、监督资格、匹配、统计、组织状态评价 | 已交付，报告记录合成测试通过 | 建立正确训练与评价的基础 |
| 阶段 B1：来源政策、迁移、几何、曝光与划分 | 已交付，B1 报告记录全套 414 passed | 数据处理代码通过合成验证 |
| amax 服务器预检 | 报告已核对数据计数、部分头信息与环境 | 说明资源存在，尚未完成正式执行环境核验 |
| T07/T08 的真实来源核验与划分 | `NOT_STARTED` | 真实数据层证据尚未形成 |
| T09/T10：训练、推理与模块基线 | `TODO` | 当前分支尚未交付完整训练闭环 |
| T11–T14：同例参照、联合模型、H1 | `TODO` | 核心联合感知尚未实证 |
| T16/T17：共享权重双模式、八项能力 | `TODO` | 最终目标尚未验证 |

上述状态与 `status.md` 中的三个字段一致：

- `scientific_conclusion: NOT_EVALUATED`
- `core_target_validated: null`
- `study_package_complete: false`

**“B1 完成”应理解为 B1 的软件交付完成，不能解释为真实数据接入完成，更不能解释为模型能力已经完成了一大半。**报告本身对此作了明确区分，没有把合成测试包装成医学性能结果。[依据：阶段 A 报告][report-a]、[B1 完成报告][report-b1]、[逐任务状态][status-tasks]、[amax 预检][host-precheck]

这一判断仅适用于当前分支纳入验收的实现与证据，不用于否定其他服务器检出或历史分支可能存在的实现。历史试训也不能自动作为当前版本完成的证据。[依据：执行基线修订][plan-baseline]

## 三、与核心目标一致、应当保留的工作

不能仅因为当前代码主要是契约、评价器和数据处理，就认定这些工作都属于过度工程。仓库记录过会直接制造虚假科研结论的评价缺陷。

例如，一个患者有两个具有参照的病灶，第二个漏检后，因为 `relation_ok=None`，它被从分母中删除，最后得到 `J=1.0`。在这个例子中，模型只成功处理了一个病灶，评价却给出了全部成功的结果。修正固定分母问题，直接关系到核心目标能否被正确验证。[依据：已复现的评价缺陷][status-defects]

以下安排均与整体目标一致：

1. **明确区分真实体素标注、框和伪标签。**这决定 U 的边界精度是否具有可信参照。
2. **将 S 保存为正式输出，并直接评价。**这避免只完成分割而没有回答组织是否存在目标异常。
3. **A 不确定时保留 U。**解剖错误不能把已发现的异常抹掉。
4. **分别验收二维与三维。**三维结果无法自动证明严格单切片条件下的能力。
5. **区分核心能力、方法增益和研究包完成。**普通共享模型可能完成任务，而软解剖融合未获得额外收益；两种结论应分别报告。

这些是 v5.1 对核心目标的强化。[依据：v5.1 对齐补齐记录][alignment-record]

**现阶段没有理由推倒 A/B1 重来。需要调整的是后续工作的优先级和验收方式。**

## 四、最需要处理的四个问题

### 4.1 同例联合参照仍未落实，直接决定主线能否成立

当前路线是：ISLES 提供真实 U 模块基线，SKM 提供真实 A 模块基线，再通过 SKM 定向精标建立同例联合主线。[依据：路线与范围][spec-route]

但 SKM 联合配置中的目标异常类别、状态组织、关系语义和参照政策仍然是待填写项；D3 的实际读者、仲裁和预算也尚未落实。[依据：联合范围模板][joint-scope]、[D3 执行状态][status-d3]

核心问题是：**尚未通过实际专家试点确认，在选定的 E1/E2 输入下，能否稳定定义和复核目标异常的体素范围，并形成可用的 S/R 参照。**

如果这个条件最终不成立，后续即使训练、保存、统计都做得完善，也无法完成原定联合感知验证。

尤其需要避免以下进度推断：

> 膝关节解剖分割做好了，脑卒中病灶分割也做好了，因此联合感知已经具备基础能力。

前两项可以证明模块和流水线可行，但不能回答“某一膝关节影像里的异常属于哪块软骨”。不同来源各完成一项任务，不能相加为同例联合成功。

**计划已经允许 X0 专家资源线立即并行，现在需要落实这一安排。**建议把原定的 20–30 次非锁定开发扫描试点，与 B2 放在同一级优先级，尽早得到边界可定义性、单层可判读性、复读一致性和标注成本的实际结果。[依据：X0 并行安排][plan-x0]

### 4.2 依赖图可能把 H2 错误地设为核心能力验收前置

这是本次审查发现的最具体、可以直接修改的计划问题。

当前路线图中，C 是联合参照、联合模型与 H1，D 是 H2 缺标实验，E 是双模式；C、D、E 全部进入 F，而 F 同时包含 T17 核心能力验收和 T18 最终证据交付。[依据：阶段依赖与任务表][plan-roadmap]

但科学规格明确允许：**H2 尚未完成，可以不阻止已有核心能力结论，只使 `study_package_complete=false`。**[依据：能力与研究包分开判定][spec-completion]

如果执行 Agent 严格按依赖图调度，可能出现：同例联合模型与双模式已经具备验收条件，却因为 ISLES 缺标实验未完成，迟迟不能进入核心能力评价。

建议最小修正为拆开两个终点：

| 终点 | 必须等待的内容 |
| --- | --- |
| 核心能力验证 | 所需同例参照、联合模型、共享权重双模式、八项指标及对应冻结条件 |
| 完整研究包完成 | 所有预定必需实验，包括 H1/H2/H3，以及完整分析与证据 |

H2 应继续保留，但不应成为与其无关的核心能力结论的额外阻塞。目前这是**计划表达中的过强依赖**，尚无证据表明它已经造成实际延期。

### 4.3 B1 的部分遗留问题应在真实训练接入时解决

B1 报告主动披露了尚待接入的事项，其中以下几项直接影响科研有效性：

| 遗留事项 | 对核心目标的影响 | 应在哪一步关闭 |
| --- | --- | --- |
| 来源版本、模型输入模态与实际 scope 的绑定 | 训练和评价可能不对应同一观察条件 | 真实 T07 与 B2 数据加载接入 |
| 降采样丢失阳性尚未接入阴性资格剔除 | 小病灶、薄软骨可能被错误当作正常背景 | T09 采样与损失接入 |
| 曝光判断只精确匹配 lineage ID，尚不推断后代谱系 | 继承旧权重的新模型可能错误获得独立测试资格 | T10 前确定谱系规则并接入 |
| 划分冻结与测试独立性证明尚未完全连接 | 有固定划分文件，仍不能证明它是独立测试 | 正式测试准入前 |

这些问题可以在现有框架上补齐，但不宜一直按普通低优先级事项延期。[依据：B1 执行裁定与遗留项][report-b1-deferred]

例如，模型 B 从模型 A 的 checkpoint 继续训练，即使 B 换了新的 lineage ID，A 曾经见过的测试患者也不会因此变成对 B 未曝光。需要让数据曝光随权重继承关系传播。

建议以**是否会污染下一次真实运行**为标准选择 B2 前必须完成的收尾项；格式、提示和汇总细节按实际影响处理，不必全部升级成全项目阻塞。

### 4.4 应区分尚未实施与计划遗漏

T09/T10 已安排训练、推理和模块基线，T13 安排联合模型与核心归属，T16/T17 安排共享权重双模式及八项验收。因此，当前问题是这些任务尚未实施，不能说计划已经删除了模型路线。[依据：B–F 阶段任务][plan-stages]

先完成三维 H1，再迁入共享二维/三维骨干，也是科学规格允许的路线。合理的改进是在 B2 就固定公共输入、模式、网格和输出接口，降低后续迁移返工；不要求 B2 一次性实现所有联合网络和双模式网络。[依据：同 checkpoint 双模式路线][spec-dual-mode]

**执行存在继续停留在前置工程的风险，但仅凭当前 A/B1 的交付内容，还不足以认定已经发生长期、实质性的工程过度。接下来一轮是否开始产生真实数据与模型产物，将是更有区分力的判断依据。**

## 五、v5 完成与基础模型长期目标之间的关系

**v5.1 完成不等于通用 MRI 基础模型目标完成。**

当前 `core_target_validated=true` 的条件，是至少一个预注册独立同例场景，通过同 checkpoint 的二维、三维八项能力。跨来源、跨部位泛化没有被列为这一核心完成条件；预训练也只是可选扩展。[依据：扩展范围][spec-extension]、[核心完成定义][spec-completion]、[数据角色][spec-data]

一个符合当前规格的例子是：

> 一个只处理 SKM qDESS 膝关节软骨的共享 U-Net，预测 A/U、组合 S，再生成具有合格参照的几何归属 R；只要同一 checkpoint 在二维和三维达到预注册标准，就可能通过当前核心目标。

但这个模型仍然可能无法处理脑 MRI、新的扫描协议或未纳入范围的异常，也没有证明可复用表征或少样本迁移价值。

所以，如果“整体目标”仍然包括**可迁移的 MRI 基础模型**，当前 v5.1 只能算局部能力验证阶段。它缺少从这一阶段通向通用表征、迁移和泛化验证的明确后续里程碑。

不能把问题简化成“没有 SSL，所以不是基础模型”。**真正没有被当前验收覆盖的是复用、迁移和泛化能力。**

### 这一范围收窄并非 v5/B1 才发生

v4.1 已经选择“脑部 U 模块 + SKM 同例联合主线”，并把全身全病种基础模型及一期 SSL 列为非目标。因此，不能仅依据更早的脑部路线，直接认定这次执行者突然把项目改成膝关节项目。仓库可以证明文档范围的演变，不能单独证明每一次历史范围变更的授权过程。[依据：v4.1 目标、非目标和路线选择][v41-scope]

当前 A–F 是工程与验证阶段，不能直接等同于“表征预训练—结构化感知—解剖绑定”的模型培养阶段。若保留后一条宏观路线，应说明当前阶段覆盖其中哪部分，以及尚待完成哪些迁移和泛化验证。

## 六、建议采用的后续执行主线

保留现有目标和已完成的软件基础，把后续进展落实到以下可检查产物：

| 顺序 | 重点工作 | 应交付的实际证据 |
| --- | --- | --- |
| 立即并行：B2 | 核验 amax 执行环境，完成真实来源检查和开发划分，接入训练与推理 | 真实 batch、小样本过拟合、恢复训练、原网格预测，以及 A-only/U-only 模块结果 |
| 立即并行：X0 / T11–T12 | 完成原定 SKM 20–30 次开发扫描专家试点 | 明确异常定义、可判读范围、读者一致性、标注成本，以及能否形成同例 A/U/S/R 参照 |
| 随后：T13–T14 | 在合格同例场景上训练和比较联合模型 | 普通共享 AU、简单组合系统与 AnatoBind 的公平比较；直接报告 S 和核心 R 的错误 |
| 最终核心验收：T16–T17 | 共享权重二维/三维，按独立观察条件分别评价 | 同一 checkpoint、八项能力表、参照支持、性能界限与不确定性 |
| 并行研究线：T15 | H2 缺标实验 | 支持或否定监督机制的证据；进入完整研究包，不额外阻塞已具备条件的核心能力结论 |

B2 的模块结果只能按模块报告；X0 试点只回答参照是否可建立，不能提前当成最终盲测。下一步最有价值的进展，是让这两条线汇合成**同一病例的完整联合输出及其独立评价**。

### 已经作出的决定应进入落实阶段

D1 已经选择参照比较并保留 H3 非劣要求，D3 已选择 SKM 试点，D4 已选择 amax。现在应推进这些决定的落实。D1 的正式容差尚未冻结，可以继续保持待定，它不应阻塞 B2 工程开发。[依据：当前 D1–D4 决定及执行边界][status-decisions]

当前最重要的纠偏是把**同例参照和真实模型闭环**放到执行中心，并拆开**核心能力验收**与**完整研究包验收**。若长期仍以 MRI 基础模型为目标，应明确将 v5.1 定位为其中一个能力验证阶段，保留后续迁移与泛化的独立里程碑。

## 证据索引

以下链接均固定到本次审查基线 `aad3e5e02620ea477f4c9f33c486bfa0299d4e85`，避免后续分支更新改变本报告所依据的内容。

| 文件 | 主要用途 |
| --- | --- |
| [Proposal_v5.md][spec-goal] | 当前目标、输出、独立评价、H1/H2/H3、完成定义 |
| [plan_v5.md][plan-roadmap] | A–F 阶段、X0、任务依赖与执行边界 |
| [docs/v5.1_core_target_alignment.md][alignment-record] | S、核心 R、八项能力和三类完成状态的补齐 |
| [reports/v5/status.md][status-decisions] | 逐任务实际状态及 D1–D4 决定 |
| [reports/v5/stage_a_report.md][report-a] | 阶段 A 的软件交付与测试记录 |
| [reports/v5/stage_b1_report.md][report-b1] | B1 交付、执行裁定与遗留事项 |
| [reports/v5/host_precheck_amax.md][host-precheck] | amax 只读预检与尚未完成项 |
| [configs/v5/scope_skm_joint.template.yaml][joint-scope] | 同例联合范围尚待落实的字段 |
| [docs/archive/v4.1/spec.md][v41-scope] | 历史目标、非目标及路线演变 |

[spec-goal]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L42
[spec-normal]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L46
[spec-route]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L62
[spec-data]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L139
[spec-state]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L340
[spec-dual-mode]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L253
[spec-extension]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L293
[spec-completion]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/Proposal_v5.md#L385
[plan-baseline]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/plan_v5.md#L13
[plan-roadmap]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/plan_v5.md#L70
[plan-stages]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/plan_v5.md#L100
[plan-x0]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/plan_v5.md#L127
[alignment-record]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/docs/v5.1_core_target_alignment.md#L7
[report-a]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/stage_a_report.md#L1
[report-b1]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/stage_b1_report.md#L1
[report-b1-deferred]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/stage_b1_report.md#L85
[status-tasks]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/status.md#L107
[status-defects]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/status.md#L209
[status-decisions]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/status.md#L220
[status-d3]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/status.md#L229
[host-precheck]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/reports/v5/host_precheck_148.md#L1
[joint-scope]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/configs/v5/scope_skm_joint.template.yaml#L1
[v41-scope]: https://github.com/sober235/FoundationModel_v1/blob/aad3e5e02620ea477f4c9f33c486bfa0299d4e85/docs/archive/v4.1/spec.md#L10
