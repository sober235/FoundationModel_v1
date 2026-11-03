# AnatoBind-MRI v5 Implementation Plan

> **For agentic workers:** 依据 [Proposal_v5.md](Proposal_v5.md)，使用 `superpowers:executing-plans` 逐任务实施；仅在存在真实调度能力且已获执行授权时使用独立subagent。不得虚称独立评审。`- [ ]`仅在相应步骤实际完成后勾选。

**Goal:** 实现并独立验证同例体素级A、目标U、明确R及有范围的未见异常状态，最终同一checkpoint支持严格二维与三维。

**Architecture:** 复用并修正现有数据治理与评价模块；先完成正确评价、输入/划分与强基线，并行落实SKM专家试点。精标合格后做普通联合与软解剖融合H1，H2在ISLES上可并行；双模式工程可提前，正式六项能力后续独立验收。

**Tech Stack:** 现有Python/PyTorch/NumPy/SciPy/nibabel/h5py/pytest；沿用经核验环境，不自动升级或安装。nnU-Net与表面度量参考实现仅在版本、许可和适配核验后使用。

**Spec:** [Proposal_v5.md](Proposal_v5.md)，v5.0，2026-11-03。  
**Baseline:** `9bef2834ff72b8200465ff44c197bfcd38471305`。  
**本文件性质：** 新版待执行计划；发布文档不表示代码已修、数据已重新认证、专家已到位、审批已完成或训练已启动。

## 0. 执行范围与状态

### 0.1 只做有证据的增量

旧T00–T08不是必须重写的空白工程。先核对已存在的模块、测试与审计，再补缺。新任务号是v5工作包，不沿用旧checkbox推断完成度；旧训练、失败、指标、schema和批准记录保留原版本。

| 工作包 | 起点 | 本计划初始状态 |
|---|---|---|
| T00 | 已有v4.1代码、两轮复核、v5文档 | 待核对 |
| T01–T06 | 已有契约、资格、评价基础，有明确缺口 | 待修复/重验 |
| T07–T08 | 有匿名审计报告，私有实时证据尚待读取 | 待数据核验与冻结 |
| T09–T10 | 部分来源适配器已存在，完整训练管线未认证 | 待实现/实测 |
| T11–T12 | 尚未落实专家资源与同例精标 | 工程可做；临床执行BLOCKED_EXPERT_REVIEW |
| T13–T14 | 依赖同例队列及基线 | 待依赖 |
| T15 | 可基于ISLES完整参照并行 | 待运行基础与资格 |
| T16–T17 | 双模式设计尚未验证 | 工程可先；科学验收待参照 |
| T18 | 按主张范围分期调用 | 待相应协议冻结与解封授权 |

每个任务记录 `task_id, code_status, test_evidence, data_run_status, approval_status, scientific_conclusion, input_hashes, output_hashes, command, exit_code, blockers, next_task`。代码状态使用TODO/IN_PROGRESS/PASS/FAIL/BLOCKED/DEFERRED；不将 `BLOCKED_EXPERT_REVIEW` 混作schema未定义的枚举，作为blocker code记录。科学状态为NOT_EVALUATED/SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE。

### 0.2 Global Constraints

- 原始数据和旧派生只读；医学数据、逐例记录、权重与输出仅写获准仓库外 `$PRIVATE_ROOT`，目录700、文件600。实际路径不写公共配置。
- 同例联合目标不变；框填充、自动A、几何伪R不能升级专家体素/临床参照。未复核、不可观察、padding不当正常。
- 训练可以不完整；正式评价至少一个同例合格队列。每run绑定ActiveScope，不以总本体替代启用范围。
- 推理只用影像、有效FOV及合法几何/模式，不读GT、复核图、真实tissue_id、来源标签或隐藏参照。
- 同患者及派生组不得跨划分；曝光与初始化谱系有证据，unknown不得改名清除。
- 统计独立单位为患者；主J患者等权，病灶池化另报。漏检/拒判不改变冻结参照分母；无分母返回NA。
- 只在开发角色选阈值；锁定测试不调参。仅按本任务必需的条件放行，不等待无关阶段的null参数。
- 现有代码保留在 `anatobind/v41/`；新schema/config使用v5版本。禁止复制整个包、跨仓库动态import旧代码或静默覆盖旧实验。
- 无训练、外部下载或测试解封授权时，只做获准只读检查、合成测试和登记dry-run；Agent不填写人类签名。
- NaN、访问越权、OOM、预算耗尽停止并存证，不无限重试。未知GPU/耗时不编造。
- 测试—证据—提交逐项完成；不使用 `git add .`，不自动push后续代码或重写Git历史。本次文档push授权不扩展为未来所有外部写操作授权。

### 0.3 Review Focus

| 最易漏掉的情形 | 必须行为 | 责任任务 |
|---|---|---|
| unknown可见性与公开基准合法使用冲突 | 两轨道显式登记，不伪填visible；不可观察不进入主观察监督 | T01、T02、T07 |
| 6个低IoU配对对5个高IoU配对 | 先最大基数，必须选6个 | T03 |
| 漏检时预测关系为空 | 有参照病灶仍留在J分母 | T04 |
| 一患者100病灶、另一患者1病灶 | 主终点患者等权，不随病灶重复加权 | T05 |
| partial review裁边/二维偷看邻层 | 裁边不冒充真实表面；单层访问边界可检测 | T06、T16 |

## 1. 依赖与文件布局

```text
T00 → T01 → T02
          ├→ T03 → T04 → T05
          └→ T06
T01+T02 → T07 → T08
T01+T02+T07+T08 → T09 → T10
T01+T07+T08 → T11(标注包) → 专家实际试点 → T12
T02–T06+T09+T10+T12 → T13 → T14(H1_READY)
T02+T03+T05+T06+T09+T10(ISLES) → T15(H2_READY)
T09 → T16工程；T13+T16+T12的二维参照 → T17(H3_READY)
相应READY+独立性+授权 → T18(分期或统一锁定评价)
```

T07/T08的只读审计可与指标修复并行；T11读者协调不应等所有模型代码。T16合成工程不等精标，真实模块开发须合法划分与许可。T13不依赖H2或外部扩展，T18不要求无关阶段都READY。

### 1.1 复用与新增位置

| 路径 | 策略 |
|---|---|
| `anatobind/v41/{schema,paths,inventory,identity,splits,ontology,labels,geometry,geometry_qc,overlays,audits,registry,cli}.py` | 先核对原实现；必要时小范围修改 |
| `anatobind/v41/{eligibility,losses}.py`、`evaluation/*.py` | 修正语义与接口，添加v5严格入口；历史结果不变 |
| `anatobind/v41/contracts_v5.py` | 新增本文各TypedDict/数据类型及公共枚举 |
| `anatobind/v41/{annotation,observations,datasets}.py` | 缺失部分逐任务新增 |
| `anatobind/v41/models/{core,anatomy_fusion,relation,observation_adapter}.py` | 经审计后最小实现，不先构建大模型 |
| `anatobind/v41/engine/{registration,trainer,inference,access}.py` | 登记、训练/恢复、推理与读取边界 |
| `anatobind/v41/experiments/{h1,missingness,h2,h3}.py` | 冻结比较矩阵与缺标/双模式实验 |
| `schemas/v5/*.schema.json`、`configs/v5/*.yaml` | 显式新版本；真实批准在私有根 |
| `tests/v5/test_*.py` | 合成/匿名fixture；保留并重跑受影响的tests/v41 |
| `reports/v5/{status,traceability,CLAIMS,EVIDENCE_INDEX}.md` | 脱敏、实际证据；不预填PASS |

### 1.2 接口约定

以下接口是**待实现合同**。不宣称当前仓库已有相应函数/CLI。

在 `contracts_v5.py` 定义并经schema验证：`SampleV5`、`AnnotationV5`、`ActiveScope`、`VisibilityPolicy`、`ReferenceLesion`、`LesionOutcome`、`PatientContrast`、`RunV5`，均为字段明确的TypedDict。数组留在内存对象，manifest只记录私有引用和哈希。

- `VisibilityPolicy`: `track`为observable_target/released_reference；`unknown_action`为exclude/approved_reference；`approval_ref`可空但采用例外必须非空；不可将unobservable自动例外。
- `ReferenceLesion`: `patient_uid, instance_uid, u_reference_available, r_reference_available, visibility_qualified, eligible, exclusion_reason, reference_set_hash`，以及私有GT引用。
- `LesionOutcome`: `reference_instance_uid, detected, extent_ok, relation_correct, accepted, prediction_status`；不能决定参照可用性。
- `PatientContrast`: `patient_uid, method_value, comparator_value, delta, seed_policy_id`；主统计每患者恰一行，delta必须等于两者差。
- 训练Batch采用 `[B,C,Z,Y,X]`，字段为 `image, valid_fov, a_target, u_target, e_a_pos, e_a_neg, e_u_pos, e_u_neg, metadata`。二维外部输入只接受平面数组，内部适配为一个有效层。
- `GateDecision`保留工程/科学双状态；新增 `scope_id, required_evidence, blockers`，审批绑定内容哈希与角色。

## 2. 通用执行与验证规则

T00确定 `$PY`、仓库工作目录与获准私有根。不自动安装依赖；环境缺失先报告，不把依赖错误当目标红测。

每个代码任务使用：先写目标反例 → 确认正确失败原因 → 最小修复 → 本任务和受影响旧测试通过 → 记录命令/退出码/输出 → 提交指定文件。已有正确功能不故意破坏以制造红测。需要实际临床/训练工作时，软件完成和实际工作分别登记。

示例测试命令（仅在解释器核验后）：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  "$PY" -m pytest tests/v5 -q -p no:cacheprovider
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
  "$PY" -m pytest tests/v41 -q -p no:cacheprovider
```

以上仅用于CPU测试，不得用覆盖CUDA变量的方式绕过实际训练资源约束。无独立审查工具时，完成自检并如实标self-review。

## 3. 分任务实施

### T00 — 接手、版本与既有资产核对

**依赖：** 无。**Files:** 读取v5文档、历史报告、当前源码；新增 `reports/v5/status.md`、`reports/v5/traceability.md`。  
**接口：** 私有 `task_record.json` 与脱敏状态表，不新增训练命令。

- [ ] 读取Proposal/plan并记录哈希，查询当前HEAD、工作区和用户未提交修改；若非基线提交，只核对相关增量，不覆盖新工作。
- [ ] 核对现有模块与复用来源，按“已存在/需修/缺失/未核实”记录；不重跑旧训练、旧推理来恢复上下文。
- [ ] 核验解释器、依赖、存储和权限；只读取资源约束，不申请GPU作业，不读取凭据文件内容。
- [ ] 审查memory与提交规则冲突，报告敏感字段类别而非值；历史重写/删除/凭据处置另行授权。
- [ ] 生成v5任务与已有旧任务对照，分别列代码、测试、真实运行、批准和科学状态。
- [ ] 自检脱敏与链接；获准工作区中仅提交两份状态文件。

**验收：** 无已完成代码被误判成全新任务；旧144项测试仅记“历史报告”；服务器不可读则阻塞真实核验，不编造通过。

### T01 — v5契约迁移、范围配置与任务级gate

**依赖：** T00。  
**Files:** Create `contracts_v5.py`、`schemas/v5/{sample,annotation,scope,reference,outcome,run,approval}.schema.json`、`configs/v5/{scope_isles_u,scope_skm_a,scope_skm_joint,policy_observable,policy_released_reference,approvals.template}.yaml`；Modify `schema.py,registry.py`；Test `tests/v5/test_contracts_gates.py`。本文未带目录前缀的Python文件均位于 `anatobind/v41/`。

**接口：** `validate_sample_v5(record: dict) -> SampleV5`；`migrate_v41(record: dict, decisions: dict) -> SampleV5`；`evaluate_gate_v5(gate_id: str, scope: ActiveScope, evidence: dict, approvals: dict) -> dict`。

- [ ] 写 `test_scope_is_not_master_ontology`：SKM软骨scope不能自动启用韧带/积液阴性；写 `test_quality_and_adjudication_are_distinct`：合法adjudicated+accepted通过，仲裁字段accepted拒绝。
- [ ] 写gate反例：A-only缺U签核仍可按A必需条件放行；H1缺匹配/分母验证必须BLOCKED；H3缺容差不得科学放行；无关near_mm不阻塞overlap或A-only。
- [ ] 运行 `tests/v5/test_contracts_gates.py`，记录预期缺功能/错误语义的FAIL。
- [ ] 实现封闭字段校验、版本迁移与按任务required keys；旧missing可见性变unknown，不自动继承visible；审批模板签名保持null。
- [ ] 运行新测试及 `tests/v41/test_schema.py tests/v41/test_registry.py`；验证旧记录未被覆盖、unknown字段不回显敏感值。
- [ ] 保存迁移计数与gate规则版本，提交本任务明确文件。

**验收：** 规格/批准/数据状态分别可追溯；新字段不是自由字典绕过验证；正式运行只消费已验证v5契约。

### T02 — 正负资格、可见性与实际损失路径

**依赖：** T01。  
**Files:** Modify `eligibility.py,losses.py`；Test `tests/v5/test_eligibility_gradients.py`。

**接口：** `build_eligibility_v5(record: SampleV5, task: str, classes: list[str], targets: dict, review_regions: dict, visibility_regions: dict, valid_fov: np.ndarray, policy: VisibilityPolicy, channel: str) -> tuple[np.ndarray,np.ndarray]`，返回按类正/负资格；`masked_objective_v5(logits, target, e_pos, e_neg, task_spec: dict) -> torch.Tensor`。

- [ ] 写8体素测试：observable轨道unobservable/unknown全零资格与零梯度；limited仅独立可见区域有效；padding即使target_complete也无资格。
- [ ] 写两类positive-only：类图[1,2]经显式映射后c1支持[1,0]、c2支持[0,1]，负资格全零；无映射的多类输入拒绝。写pseudo_only在显式pseudo训练可用、专家评价拒绝。
- [ ] 写released_reference例外：unknown可见性只有绑定批准的基准政策才允许相应专家参照监督；字段仍是unknown，不能生成observable正常主张。
- [ ] 运行上述测试确认红测；修正资格组合、FOV交集、按类目标和角色路由，不改变来源真实性。
- [ ] 验证masked BCE/Dice与互斥A策略：ignore位置logit改变不改loss；无支持不稀释；全ignore有限零；零总权重/非有限输入明确拒绝；伪/人工分组归一化。
- [ ] 重跑 `tests/v41/test_eligibility_losses.py`；记录每类正/负/pseudo支持及版本，提交。

**验收：** 资格函数与实际反传均正确，不能只靠上游口头承诺。旧宽松入口只保留legacy用途，正式v5运行禁止绕过严格入口。

### T03 — 最大基数优先的一对一实例匹配

**依赖：** T01。  
**Files:** Modify `evaluation/instances.py`；Test `tests/v5/test_matching_cardinality.py`。

**接口：** `match_instances_v5(pred: list[dict], truth: list[dict], policy: dict) -> list[tuple[int,int]]`；返回索引对，稳定ID用于解平局，允许边须先满足类别/IoU政策。

- [ ] 写下面的确定性反例，原实现在阈值0.1时只返回5，正确结果应为6：

```python
truth, pred = [], []
for i in range(6):
    t = np.zeros((680, 1, 1), dtype=bool)
    p = np.zeros_like(t)
    t[100*i:100*i+100] = True
    p[100*i+80:100*i+180] = True
    truth.append({"id": f"t{i}", "mask": t})
    pred.append({"id": f"p{i}", "mask": p})
pairs = match_instances_v5(pred, truth, {"iou_match_threshold": 0.1})
assert len(pairs) == 6
```

- [ ] 运行 `tests/v5/test_matching_cardinality.py`确认正确红测；增加空集、分裂/合并、重复预测、类别不符、阈值相等、null/NaN/0/大于1阈值、形状不一致测试。
- [ ] 实现先最大基数、再最优IoU的两阶段目标；若使用M权重，取 `min(n_pred,n_ref)+1` 并在注释给出基数优先证明，平局不能改变前两级最优性。
- [ ] 对小规模合成图穷举匹配作oracle验证；输入顺序改变只改变索引表示，不改变配对ID集合或充分统计量。
- [ ] 重跑新测试及 `tests/v41/test_metrics.py`，确认反例TP=6/FP=0/FN=0，不以调真实阈值绕开反例。
- [ ] 保存匹配版本与验证输出，提交实现与测试。

**验收：** 配对数、IoU、稳定次序优先级有证据。保留旧指标历史，不重写既有报告。

### T04 — 固定参照分母、关系集合与联合评价

**依赖：** T01、T03。  
**Files:** Modify `evaluation/joint.py,evaluation/relations.py`；Test `tests/v5/test_joint_reference_denominator.py`。

**接口：** `freeze_reference_set(refs: list[ReferenceLesion], policy: dict) -> dict`；`joint_score_v5(refs: list[ReferenceLesion], outcomes: list[LesionOutcome], reference_hash: str) -> dict`。

- [ ] 写固定两病灶均有R参照、仅一例成功测试：漏检/关系未输出/拒判变化均保持分母2、J=0.5或更低；不得借None降为分母1。
- [ ] 写无R参照、真实空关系集合、无病灶患者分别处理；空分母 `None`，包含原因、覆盖和计数，不返回0/1。
- [ ] 写预测结果不能改变reference_hash；未知/重复reference ID拒绝；没有outcome的合格病灶自动计未成功而非丢弃。
- [ ] 运行红测，分离参照冻结与预测判定；患者J和池化J分别输出，阴性扫描不进入阳性J分母。
- [ ] 同步关系评价：有参照但集合为空仍计参照覆盖；多组织集合正确性明确；GT/预测角色不混用。
- [ ] 重跑metrics旧测试与新测试，发布充分统计量和参照覆盖摘要，提交。

**验收：** 任意预测退化不改变固定G_i；没有参照与预测错误不是同一状态。函数接口不能再以relation_ok是否None推断参照存在。

### T05 — 患者等权、配对统计与训练种子

**依赖：** T01、T04。  
**Files:** Modify `evaluation/statistics.py`；Test `tests/v5/test_patient_estimand.py`。

**接口：** `make_patient_contrasts(method_rows: list[dict], baseline_rows: list[dict], policy: dict) -> list[PatientContrast]`；`paired_patient_bootstrap_v5(rows: list[PatientContrast], seed: int, n_boot: int, ci_level: float) -> dict`。池化指标另用 `pooled_cluster_bootstrap(...)`，不可共用未标明的estimate。

- [ ] 写患者A100病灶值1、患者B1病灶值0，患者均值0.5、池化100/101；复制A内等值记录不得改变患者均值。
- [ ] 写两方法必须按patient UID配对，缺配对/重复患者行拒绝或按预注册缺失政策明确标缺；不得静默删难例。
- [ ] 写重复种子不增加n_patients，n_boot<=0、ci_level不在(0,1)拒绝；空输入返回NA且不计算百分位。
- [ ] 运行红测；先按冻结规则汇总同患者扫描/病灶/种子，再计算配对delta，bootstrap整患者；点估计与区间针对同一estimand。
- [ ] 用已知小例验证置信区间、确定性种子、0.5结果；训练种子变异独立报告，不能把seed当患者。
- [ ] 重跑旧metrics与新测试，记录统计版本、单位、区间与患者数，提交。

**验收：** 主J点估计与区间均患者等权；现有按行池化行为不被误称主终点正确。

### T06 — 分项指标、正常状态与按任务评价政策

**依赖：** T01；联合汇总另依赖T04/T05。  
**Files:** Modify `evaluation/{segmentation,selective}.py`；Create `evaluation/policy.py`、`configs/v5/metrics.template.yaml`；Test `tests/v5/test_metric_policy.py`。

**接口：** `validate_metric_policy(task: str, stage: str, policy: dict) -> dict`；`segmentation_metrics_v5(pred, truth, eligible, geometry: dict, policy: dict) -> dict`；`normal_state_metrics(rows: list[dict], policy: dict) -> dict`。

- [ ] 写A-only Dice不因near_mm/H3容差缺失而阻塞，正式HD95缺物理几何返回NA，正式J缺extent规则阻塞；诊断参数与锁定参数版本分开。
- [ ] 写空GT、单边空、全ignore、只有padding、贴边、部分复核裁边、不等距spacing反例；无参照不误计真阴性。
- [ ] 写按冻结逐类阈值生成组织状态图（任一目标阳性/全部目标阴性/拒判）的测试；写false-normal两个分母的手算例、全拒判NA+覆盖率、阴性扫描不足时仅报告实际n；risk-coverage正确性未知另报覆盖而非静默删。
- [ ] 运行红测；固定表面提取/面积权重/HD95合并与插值政策；优先使用经锁定的参考实现或可验证适配，不无授权安装。[Proposal R6]
- [ ] 对partial review采用显式surface eligibility：不确定/未复核界面不作已知表面，外部最近边界无法判定时该距离为NA；保留区域Dice及表面可评价覆盖。写裁切mask产生人工平面的反例确保不会虚高。
- [ ] 与参考实现作合成差分，重跑旧metrics测试；保存metric_version、分母/失败/NA计数，提交。

**验收：** 不将无权边界体素比例无说明称标准NSD；每指标只检查其必要参数，生产结果无隐含默认阈值。

### T07 — 来源、输入可见性与真实空间链

**依赖：** T01、T02；可并行准备只读审计。  
**Files:** Modify `adapters/{isles,skm}.py,geometry_qc.py,audits.py,ontology.py`；Create `configs/v5/sources.yaml`；Test `tests/v5/test_source_visibility_geometry.py`。

**接口：** `audit_source_v5(records: list[SampleV5], source_policy: dict) -> dict`；`resolve_geometry_chain(record: SampleV5) -> dict`。

- [ ] 写相同shape但翻轴/方向错误不通过；缺affine仅可标索引对齐，不能获得physical pass；unknown单位必须有批准来源政策。
- [ ] 写ISLES标注模态DWI/ADC/FLAIR与模型DWI/ADC分列；专家来源不自动把visibility设visible；released_reference与observable轨道各自计数。
- [ ] 红测后实现显式E1/E2—H5 seg—参考轨道—框映射、NIfTI scaling与类别值QC；近整数容差1e-5，概率图拒绝，历史全零截断从原源新导出。
- [ ] 在允许只读真实数据时，核对本地版本、hash、配对、标签值、空间链；导出私有叠加图供人审，覆盖小病变、边缘和变形，不只比较两份mask。
- [ ] 若服务器不可读，保留synthetic PASS/data NOT_VERIFIED；不把历史155/155形状pass替代新物理证据。发布按来源/任务的匿名结果与例外。
- [ ] 重跑新测试和 `tests/v41/test_geometry_labels.py tests/v41/test_geometry_qc.py tests/v41/test_adapters_pairing.py`，提交代码与匿名汇总。

**验收：** 每条训练/评价空间路径有对应层级证据；不要求所有P2资产先通过才放行P0。

### T08 — 曝光审计、患者分组与正式划分

**依赖：** T01、T07。  
**Files:** Modify `identity.py,splits.py`；Create `configs/v5/split_policy.template.yaml`；Test `tests/v5/test_exposure_split.py`。

**接口：** `audit_exposure(records: list[SampleV5], evidence: list[dict]) -> list[dict]`；`freeze_splits_v5(records: list[SampleV5], policy: dict, approvals: dict, out: Path) -> dict`。

- [ ] 写回波/轨道/随访/裁剪跨划分拒绝；SibBMS跨队列同号不盲合并；已知曝光不能因新随机种子变unexposed。
- [ ] 写administrative_qc与训练/模型选择接触区分；unknown保持未知，锁定角色需相应独立性证据。历史队列属性不是未经定义的新split枚举。
- [ ] 运行红测；实现来源内患者分组、明确别名和模型谱系曝光；读取旧运行manifest/初始化/结果使用记录，不重跑旧训练。
- [ ] SKM核对官方划分；ISLES在管理员角色按已批准比例和可得分层变量抽样；稀疏层的合并规则先定，缺中心信息不猜测，不凭测试表现换种子。
- [ ] 人工确认曝光与划分政策后冻结；发布患者支持数、实际阴性数和哈希，重新检查零已知跨split组。无法独立者进入历史/内部验证队列。
- [ ] 重跑 `tests/v41/test_identity_splits.py`及新测试，提交实现；病例清单只存私有根。

**验收：** 真正的划分冻结及泄漏检查，不把未分split时0泄漏当就绪。读者试点从非锁定病例抽取。

### T09 — 输入管线、运行登记、资源与访问控制

**依赖：** T01、T02、T07、T08；CPU合成开发可早于真实划分。  
**Files:** Create/complete `observations.py,datasets.py,engine/{registration,trainer,inference,access}.py`、`configs/v5/run.template.yaml`；Modify `cli.py`；Test `tests/v5/test_run_pipeline.py`。

**接口：** `build_observation_v5(record: SampleV5, scope: ActiveScope, roles: set[str]) -> dict`；`register_run_v5(config: dict, evidence: dict, approvals: dict) -> RunV5`；`run_training_v5(run: RunV5, resume: Path | None) -> dict`；`infer_v5(model, observation: dict, policy: dict) -> dict`。 公共 `freeze_protocol_v5(config: dict, evidence: dict, approvals: dict, out: Path) -> dict` 在 `engine/registration.py` 实现，供H1/H2/H3复用，不依赖任何实验已完成。

- [ ] 写无GT推理输入、缺模态拒绝、相同增广同时变换图像/标签/资格、有效FOV不当背景、缓存键含模式/源/变换/本体测试。
- [ ] 写无预算/资源许可拒绝真实run；哈希不匹配拒绝resume；锁定/H2隐藏角色不可读；第二控制器不能覆盖第一run。
- [ ] 红测后实现最小数据批次与控制器；记录参数/优化器/RNG/sampler等恢复状态。将数据读取经允许角色包装；能使用权限/最小挂载时加系统层隔离。
- [ ] 做直接文件打开的负向访问测试，区分流程隔离与实际强制隔离；若不能证明OS拒绝读取，不声称强隔离。受限评价逐例输出不回流调参。
- [ ] CPU合成单batch及确定性中断恢复测试；获得实际资源许可后才做GPU单batch和小样本过拟合，记录显存、吞吐和容差。OOM/NaN按策略退出，不擅自延长预算。
- [ ] 补全并验证已实现的CLI帮助，保存匿名工程报告和run记录，提交指定文件。

**验收：** 输入—损失—优化—保存—恢复—全体积推理闭环有实测。预算字段非空且批准；不继承历史卡号或耗时。

### T10 — ISLES U与SKM A强基线

**依赖：** T02、T06–T09，相应G-BASE；实例评价另需T03。  
**Files:** Create/complete `models/core.py`、`configs/v5/{isles_u,skm_a}.yaml`；Test `tests/v5/test_baselines.py`。

**接口：** `build_model_v5(config: dict) -> torch.nn.Module`；`forward(image, valid_fov, mode: str) -> dict`，按任务输出a_logits/u_logits/image_features，不接受GT参数。

- [ ] 写U-only无需A标签、A-only不读框、SKM输出7通道、ISLES输入2通道测试；训练统计不读取validation/test来拟合。
- [ ] 红测后包装经核验的残差U-Net/nnU-Net；只支持相同资格/输入条件的对照，输出回原生网格。
- [ ] 完成合成与获准真实小样本过拟合，检查loss、采样、量纲和反变换；性能异常先查管线，不以更大模型掩盖。
- [ ] 获准后登记开发基线；候选seed17，仅在批准后使用。保存曲线、全体积开发结果、支持量、失败与实际资源使用。
- [ ] 对ISLES标注released_reference轨道与可见性证据，不将U成绩称完整联合能力；对SKM未核物理几何项明确NA。
- [ ] 重跑软件测试，发布匿名基线证据并提交配置，不提交权重/逐例医学输出。

**验收：** 实际A/U基线可运行且评价正确；不预设Dice阈值、不保证增益。

### T11 — 标注工具接口与真实专家试点

**依赖：** T01、T07、T08；读者协调可从T00开始。  
**Files:** Create `annotation.py`、`configs/v5/annotation_protocol.template.yaml`；Test `tests/v5/test_annotation_roundtrip.py`。

**接口：** `export_review_package(records: list[SampleV5], protocol: dict, out: Path) -> dict`；`import_review_package(package: Path, protocol: dict) -> list[AnnotationV5]`。

- [ ] 写试点/锁定互斥、默认不带模型/几何预测、读者版本不可覆盖、纯框不能作为voxel GT导入、额外序列可见性单列测试。
- [ ] 写正确仲裁字段用例：quality=accepted且adjudication=adjudicated在相应批准政策下可导入；无读者/质量依据不能自动升级。不使用旧T09的accepted仲裁枚举。
- [ ] 红测后实现NIfTI+JSON往返、U/不确定带/复核区域/可见区域/A修订/关系记录的字段校验；标注工具转换与契约分离。
- [ ] 落实负责人、读者、仲裁者和预算；未落实则临床BLOCKED，软件测试可PASS。由真实读者完成20–30次扫描规划起点，不由Agent/伪标签冒充。
- [ ] 记录读者时间、一致性、不可判读比例、组织支持、阴性范围和实例拆并分歧；先导范围定义在扩展前书面化。
- [ ] 验证标注导入及脱敏汇总，提交软件；实际标注与签核仅存私有根。

**验收：** 软件完成与真实标注完成独立记录；体积不可稳定定义者单列表面支线，不宣布体素目标已达成。

### T12 — 联合队列发布、参照冻结与精度设计

**依赖：** T03–T08、T11专家试点。  
**Files:** Modify `annotation.py,registry.py`；Create `configs/v5/joint_cohort.template.yaml`；Test `tests/v5/test_joint_cohort_release.py`。

**接口：** `release_joint_cohort_v5(records: list[SampleV5], protocol: dict, approvals: dict, out: Path) -> dict`；消费T04的参照冻结接口。

- [ ] 写缺同例A/U不发布联合资格；临床R无独立依据不能由几何补齐；独立合格A/U可按批准谓词生成R_geo参照。
- [ ] 写参照可用性与不确定/排除原因预测前固定；试点病例和历史曝光不能进入独立盲测；空组织集合不等于缺R。
- [ ] 红测后实现版本化发布与原始读者版本保留；质量/可见性/完整性按任务和区域，不以文件存在认证。
- [ ] 根据试点可判读率和患者配对差异做精度情景计算，提交规模依据；60–100扫描与20–30测试患者仍是规划，不当统计学结论。若目标阴性不足，缩窄特异性主张。
- [ ] 临床负责人批准ActiveScope、实例规则、R语义、不确定带和正式参照政策；管理员保管测试，不向训练者开放。
- [ ] 重跑新测试，发布G-ANNOTATION及匿名患者/病灶/参照覆盖数，提交软件与脱敏说明。

**验收：** 至少一个真正可检验的同例队列，或明确阻塞；不依赖“所有训练样本必须完整精标”的不必要条件。

### T13 — 普通联合、软解剖融合与关系模块

**依赖：** T02–T06、T09–T12，G-JOINT。  
**Files:** Modify `models/core.py,engine/inference.py`；Create `models/{anatomy_fusion,relation}.py`、`configs/v5/joint_models.yaml`；Test `tests/v5/test_joint_models.py`。

**接口：** `AnatomyResidualFusion.forward(features, anatomy_prob, uncertainty, enabled: bool)`；`extract_instances_v5(u_prob, geometry: dict, policy: dict) -> list[dict]`；`bind_geometry_v5(instances: list[dict], a_prob, policy: dict) -> list[dict]`；`RelationHead.forward(roi_features, anatomy_features, spatial_features)`。

- [ ] 写shared/independent/AnatoBind均仅影像forward；缺某任务标签不产生其梯度；关闭融合为恒等，A漏分不硬删除U影像通路。
- [ ] 写默认stop-gradient、显式联合反传、零残差初始化仍有可学习参数梯度；不要同时零初始化乘积两支导致永久零梯度。
- [ ] 写多组织集合、无交叠不推出临床未累及、GT-A仅oracle、真实tissue_id不输入；GT ROI只属诊断轨道。
- [ ] 红测后实现最小架构与几何R；轻量学习头只在数据支持时启用，不默认大模型或全库预训练。
- [ ] 获批后比较普通联合与融合；全体积/预测ROI计入漏检、重复和分裂合并；固定候选与端到端结果分开。
- [ ] 重跑软件测试，记录参数/计算/调参/监督曝光量及开发结果；不宣称必然优于简单基线，提交。

**验收：** 可部署路径无GT泄漏；关系结果语义匹配；缺少收益不构成增加复杂性的自动理由。

### T14 — H1比较、阈值校准与主协议冻结

**依赖：** T05、T06、T12、T13。  
**Files:** Create `experiments/h1.py`、`configs/v5/h1.template.yaml`；Modify `engine/registration.py,evaluation/selective.py`；Test `tests/v5/test_h1_protocol.py`。

**接口：** `build_h1_matrix(config: dict) -> list[dict]`；`calibrate_working_point(dev_results: list[dict], policy: dict) -> dict`；复用T09的 `freeze_protocol_v5` 公共接口，不在H1模块另写一份。

- [ ] 写非oracle数据/输入/预算可比、候选哈希一致、oracle不可primary、测试角色不能进入校准测试；改变冻结阈值必须使协议哈希失效。
- [ ] 写FP约束无可行开发工作点时按预注册规则返回无可行/指定退化点并报告，不秘密使用测试找阈值。
- [ ] 红测后登记Proposal§7矩阵与唯一主对照选择规则；分清需要重训的消融和推理扰动，控制预算。
- [ ] 获批运行开发比较；选1–2个主终点，冻结患者J、A保持性、FP判据、接受策略、extent与表面容差、关系语义及多重比较范围。
- [ ] 分析A纠错/引错与病灶大小/组织/边界破坏；不把置换下降当充分因果证明，不按测试结果选分层。
- [ ] 保存H1_READY模型/协议/评价哈希，测试仍锁定；提交实现和脱敏开发证据。

**验收：** 有可执行且可证伪的联合主比较，支持与否留待T18；临床必要阈值不能由Agent随意赋值。

### T15 — H2现实缺标与公平比较

**依赖：** T02、T03、T05、T06、T09、T10(ISLES)；协议冻结复用T09公共接口，不依赖H1实验完成。  
**Files:** Create `experiments/{missingness,h2}.py`、`configs/v5/h2.template.yaml`；Modify `losses.py`；Test `tests/v5/test_h2_information_boundary.py`。

**接口：** `create_missingness_release_v5(records, policy: dict, visible_out: Path, restricted_out: Path) -> dict`；`build_h2_arms(config: dict, visible_release: dict) -> list[dict]`；`box_presence_loss_v5(logits, boxes, policy: dict)`。

- [ ] 写两个隐藏完整参照不同、公开上下文相同的整管线测试：公开资格相同，manifest无完整mask路径/删去实例ID/精确ignore洞；不能仅重复调用同函数充当验证。
- [ ] 写所有非oracle组可见信息hash相同；完整标签/oracle资格标privileged；框MIL不把全框阳性或框外阴性。
- [ ] 红测后实现区域复核、随机实例遗漏、大小/可见性偏置遗漏；候选25/50/75%明确为遗漏率，开发前登记。
- [ ] 加入独立抽样的可信阴性区域或完整子集；隔离隐藏参照并负向测试直接读取，不根据删除mask反推阴性。类别先验不能来自隐藏或测试标签。
- [ ] 必做缺标当背景、masked同资格、完整监督上限；新算法臂仅在机制不同且批准后启用。评价固定开发工作点，报告完整灵敏度—FP曲线、false-normal、接受覆盖与大小分层。
- [ ] 使用公共freeze_protocol生成H2_READY，记录协议贡献与算法增益分别是否支持；测试留给T18，提交。

**验收：** 更少负标签不能通过无限误报制造“改善”；没有新机制不另起新算法名称。

### T16 — 双模式骨干与严格单切片边界

**依赖：** T09；可先CPU合成。迁入完整联合结构需T13。  
**Files:** Create `models/observation_adapter.py`；Modify `observations.py,datasets.py,engine/{trainer,inference}.py`；Test `tests/v5/test_dual_mode_boundary.py`。

**接口：** `build_single_slice_v5(image_2d: np.ndarray, geometry: dict, policy: dict) -> dict`；`ObservationAdaptiveModel.forward(image, valid_fov, mode: str) -> dict`，mode仅2d/3d。

- [ ] 写单层接口不接受volume/z索引；同一切片在不同邻层背景中产生完全相同合法输入；归一化不访问整卷/三维缓存。
- [ ] 写文件访问拦截而非自报标志：禁止邻层、3D A/ROI与同病例缓存；补零/重复层observed_slices仍1；无可靠几何不输出伪毫米/左右结论。
- [ ] 红测后实现共享平面编码/解码和带有效性mask的深度聚合；公布共享参数范围，不能只是把两个模型放进一个checkpoint。
- [ ] 将A/U融合与R迁入；训练时两模式分别产生有效监督。3D教师如使用仅来自获准训练谱系，二维推理不读教师输出。
- [ ] 同一checkpoint hash在两模式加载，完成合成forward/backward、原网格输出和资源smoke；合法模块开发可在SKM A/ISLES U先做。
- [ ] 重跑输入和训练控制回归，保存工程状态，提交；不提前宣称六项科学能力完成。

**验收：** 真实单层信息边界、同权重与共同模型结构有证据；无需等待全部精标才做工程测试。

### T17 — 双模式同例训练、专用对照与六项验收准备

**依赖：** T12、T13、T16、T05/T06，且单层可见性参照合格；H1开发稳定后运行正式比较。  
**Files:** Create `experiments/h3.py`、`configs/v5/h3.template.yaml`；Test `tests/v5/test_h3_protocol.py`。

**接口：** `build_h3_matrix(config: dict) -> list[dict]`；`summarize_mode_coverage(results: dict, protocol: dict) -> dict`；复用freeze_protocol与患者统计。

- [ ] 写2模式×3任务六项独立状态；3D不能填二维缺项；3D切片数学参照不能自动升级单层可见性；fastMRI框不能验收像素U。
- [ ] 写独立2D、独立3D、共享双模式的来源/序列/患者split可比；预算与曝光量单列，不能将两checkpoint包装为一个。
- [ ] 红测后实现预注册抽片（含阴性、非质心层）、模式参照与指标；二维只评价本层，跨来源原生二维单列。
- [ ] 经临床/统计批准冻结模式和任务特异非劣界、区间方向及总体六项判定策略；不以无显著差异当非劣。
- [ ] 获批训练完整双模式模型及专用对照；开发校准后生成H3_READY。某项无参照或不达标如实保留，不删除目标。
- [ ] 保存同权重哈希、六项支持/覆盖、未验证扩展；提交。正式结论留给相应T18评价批次。

**验收：** 至少同一场景两模式均具合法A/U/R验证条件；原生二维泛化未完成不伪称已完成。

### T18 — 范围化锁定评价、证据与发布

**依赖：** 该批次所需H1/H2/H3_READY、T01–T09相关验证、独立性、一次性授权；不依赖无关阶段。  
**Files:** Modify `registry.py,cli.py`；Create `reports/v5/{CLAIMS,EVIDENCE_INDEX}.md`；Test `tests/v5/test_locked_release.py`。

**接口：** `evaluate_locked_v5(protocol_lock: Path, authorization: Path) -> dict`；`release_evidence_v5(protocol: dict, runs: list[dict], results: dict, out: Path) -> dict`。

- [ ] 写只有工程日志不能科学SUPPORTED、未授权不读测试、授权内容hash不匹配拒绝；一次性授权不可重放。
- [ ] 写按主张scope解封：允许H1先交付，不要求无关H3；首次结果曝光后，相同队列不得继续作为未曝光测试指导后续选择。
- [ ] 红测后实现评价者角色、模型/阈值/参照/代码锁和一次预登记批次；多方法可在同批比较。逐例输出留受限根，不回流调参。
- [ ] 实际评价后按患者配对汇总，报告A/U/R、FP、拒判、false-normal、参照覆盖、阴性n、种子变异和协议偏离；测试超FP预算不重新调阈值。
- [ ] 区分完成实验、假设支持、能力达成；无R/无二维参照对应状态NOT_EVALUATED。评价代码确有bug时新版本重算并保留旧输出、理由和曝光账本，不借机改模型。
- [ ] 审核公共证据无患者ID/路径/权重；获得本次对外发布授权后提交/推送指定文件。生成commit、环境、协议、schema、数据与checkpoint哈希索引。

**验收：** 证据可重放、失败保留、主张对应实际范围。H3未完成只能标三维里程碑；假设不支持可以完成实验但不等于能力目标达成。

## 4. CLI合同、批准清单与交接

### 4.1 CLI按实现逐步出现

现有v41命令继续仅按当前帮助使用。新增命令建议采用明确的v5协议参数，只有其任务通过测试后才加入 `python -m anatobind.v41.cli --help`：

| 命令/能力 | 责任任务 | 必需输入 |
|---|---|---|
| validate-v5 / gate-v5 | T01 | scope、数据/软件证据、审批 |
| audit-v5 / freeze-splits-v5 | T07/T08 | 来源与曝光证据、划分政策 |
| register-run-v5 / smoke-v5 / train-v5 / infer-v5 | T09/T10 | 已批准run登记、角色/预算 |
| annotation-pack-v5 / annotation-import-v5 | T11/T12 | 非锁定试点、读者协议 |
| freeze-protocol-v5 | T09公共接口；T14/H2/H3使用 | 开发证据、相关阈值、人工批准 |
| prepare-missingness-v5 / run-matrix-v5 | T15及实验任务 | 信息条件与隔离政策 |
| evaluate-locked-v5 / release-evidence-v5 | T18 | 锁定协议、一次性授权 |

不提供一个绕过审批的一键全训练脚本。没有实现的命令不能在交接里写成已经可用。

### 4.2 需实际负责人决定的事项

临床：软骨异常具体边界、可判读范围、关系语义、实例规则、读者仲裁和正常范围。统计/研究：测试独立性、样本精度、主要终点、FP约束、表面容差、非劣界、切片抽样与分期测试安排。资源：GPU、预算、存储、外部授权与下载。数据管理：隐藏参照和测试角色隔离。

未批准时模板null并返回对应blocker；可继续不依赖它的任务。工程默认如Python模块名和测试fixture由计划规定，不虚称临床负责人已认可。

### 4.3 可直接交给Coding Agent的首轮指令

```text
读取README.md、Proposal_v5.md、plan_v5.md，先执行T00，不从旧plan.md的TODO重新做一遍。
本轮先核对/修复T01–T06并保存合成测试证据；在已有授权内并行准备T07–T08只读审计。
不自动训练、不填人工签名、不解封测试、不下载外部数据、不改临床范围。
修复实例匹配、固定参照分母、患者等权和监督资格必须先重现对应反例。
已有正确模块复用；真实资产不可读就记录未核实，不以旧报告替代。
每任务分开报告代码、测试、实际数据运行、批准和科学结论；仅提交指定文件。
```

## 5. 需求覆盖与交付检查

| Proposal要求 | 计划落实 |
|---|---|
| §0版本、状态和历史证据 | T00、T01、T18 |
| §1–2目标、ActiveScope、未知与双轨道 | T01、T02、T07 |
| §3数据、空间、身份/曝光 | T07、T08 |
| §4契约、资格、损失 | T01、T02、T09 |
| §5专家资源与同例队列 | T11、T12 |
| §6模型、双模式与运行 | T09、T10、T13、T16 |
| §7H1/H2/H3与扩展声明 | T14、T15、T17、T18 |
| §8正确评价、阈值与统计 | T03–T06、T14、T17、T18 |
| §9按任务gate、分期解封 | T01、T09、T18 |
| §10隐私、版本与资源 | T00、T01、T09、T18 |
| §11完成定义与风险 | T12、T14、T15、T17、T18 |

交付前检查：接口/字段同名同义；所有反例有责任测试；临床阈值没有伪造默认；无病灶/无参照/无几何各自明确；没有把planned写成implemented；不编造独立reviewer；所有实际结果来自对应运行证据。本文与Proposal发布完成只代表**v5文档就绪**，不代表上述任务已执行。
