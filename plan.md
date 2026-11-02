# AnatoBind-MRI v4.1 Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. When an actual subagent tool is available and execution has been authorized, `superpowers:subagent-driven-development` may be used for independent tasks. Steps use `- [ ]` for tracking. Do not report an independent review unless a separate reviewer actually performed it.

**Goal:** 在同一 MRI 观察上实现并独立验证体素级解剖身份 A、目标异常范围 U、病灶—解剖关系 R，以及有明确目标范围的“未见异常”状态；最终保留同一 checkpoint 支持独立二维输入与三维输入的目标。

**Architecture:** 以已有脑部专家病灶掩膜建立真实 U 基线与缺标实验，同时在 SKM 开展同例 A/U/R 精标。先比较独立模型、普通共享编码器双头与软解剖引导模型，再验证关系绑定和监督资格；二维/三维共享作为后续完整任务，不用三维结果替代二维验收。

**Tech Stack:** Python、PyTorch、NumPy、SciPy、nibabel、h5py、pytest；需要 DICOM 时使用 pydicom。nnU-Net 作为独立强基线后端，配置与版本由执行前检查锁定；复用仓库已验证的组件，不自动升级环境、不假定旧命令兼容当前版本。

**Spec:**
- [研究方案 v4.1](AnatoBind_MRI_v4.1_Feasibility_Proposal.md)。
- [监督与数据契约 v4.1](AnatoBind_MRI_v4.1_Supervision_Data_Contract.md)。
- [2026-11-01 数据盘点（脱敏版）](FM_data_inventory_redacted.md)。

**日期：** 2026-11-01。  
**文件性质：** 待执行计划，不是已完成实施、实验或服务器就绪报告。  
**本次交付范围：** 只编写 `plan.md`；未修改远端仓库、未向148写入数据、未启动训练。  
**任务状态：** 本文所有任务初始均为 TODO；既有工程资产先审计再复用，不因本文件而重新训练。

配套文档可在私有规划目录相邻保存。原盘点和历史说明可能带病例文件名，不能直接整体提交Git；仓库只保留经审核脱敏的版本及原文私有哈希索引。

> **执行备注（2026-11-02，仓库 `sober235/FoundationModel_v1`）：** 本计划原在 `sober235/Foundation_Model` 语境下编写；在本新仓库执行时以下四点按研究者指示调整，裁决细节见执行 ledger。
> 1. 代码仓库为 `sober235/FoundationModel_v1`（新建、私有、空起点）；旧仓库 `sober235/Foundation_Model` 提交 `4deeead` 仅作只读审计对象。
> 2. 私有输出根 `PRIVATE_ROOT=/data2/congcong/code/FM_data`（研究者批准；不是 `$HOME/outputs/anatobind_v41`，也不是原始数据根 `/data2/congcong/data/FM_data`）。
> 3. 文中 `FM_data_inventory_20261101.md` 对应本仓库脱敏版 `FM_data_inventory_redacted.md`；原文仅存私有根并以 sha256 索引。
> 4. 新仓库不含旧 `anatobind/aur/`。文中“复用/包装”旧组件均指**带 blob 哈希来源记录的复制**进入 `anatobind/v41/`，不跨仓库 import。

---

## 0. 执行优先级、已有工程与本计划的关系

### 0.1 以哪个目标为准

v4.1 的科学定义与监督契约是本计划的新研究依据。仓库历史“跑通”“伪参照通过”“同网格一致”属于工程证据，保留但不升级为独立专家参照上的能力证明。工程线可以复用；旧失败、旧阈值和旧完成证书不能追溯性改写。

本次只读核对的仓库为 `sober235/Foundation_Model`，`main` 指向提交 `4deeead762e89e7facf0f9d754578fb5be8e4877`。已阅读 `AGENTS.md`、`PROJECT_CONTEXT.md`、`STATUS.md` 顶部、`CLAUDE.md` 顶部和 `anatobind/aur/labels.py`。项目交接文件记录正式 run_v5 训练已完成、最终质量评价尚未执行；这是仓库记录，不是本轮对私有完成产物的重新认证。[E4–E6]

执行者必须先读最新交接，只核对 `main` 与登记工作分支 `work/aur-formal-supervised` 的相关增量。禁止为恢复上下文重跑旧训练、遍历所有历史分支或覆盖冻结 runtime。旧 A32/伪R/双模式模型可以进入兼容性对照；新 SKM A6 和精标 U 不与其标签编码直接互换。

### 0.2 输出路径与隐私约束

- 原始数据根：`/data2/congcong/data/FM_data`，只读。`derived/` 中旧实验也只读。
- 仓库当前隐私契约要求新医学数组、逐例清单、标注、权重及逐例预测存于仓库外的私有 home 根。因此本计划默认 `PRIVATE_ROOT=$HOME/outputs/anatobind_v41`，使用权限受限的新版本子目录。
- Proposal 中 `derived/anatobind_v41/` 是示例，不是必须路径。仅在研究者明确批准另一私有存储根且路径检查通过后替换；不自动把共享数据根设为新医学输出目录。
- 公共 Git 只放代码、合成测试、无真实ID的配置模板和经过白名单筛选的匿名汇总。真实文件名、manifest 的键、读者逐例记录同样可能识别病例，不能输出到终端、对话或 Git。
- 校验真实解析路径后写入同一个规范化路径；拒绝 `..`、符号链接越界、解析后进入 Git 的目标。错误消息也要脱敏。

### 0.3 允许立即推进与需要签核的事项

本次请求只授权编写计划，不构成执行授权。之后获准实施时，默认先推进 T00–T08 的代码与合成测试，以及经过批准的只读数据审计；按结果准备 T09–T13，不自动开正式 GPU 训练。

研究者/医学专家必须签核：异常语义、目标正常范围、标注可判读性、临床 R 定义、实例拆并规则、缺损定义、盲测抽样与评价容差。计算资源负责人/用户必须签核：GPU 资源、训练预算、外部数据下载、正式测试解封。

没有签核时生成 `BLOCKED` 和具体证据需求；继续不依赖该事项的任务。不能通过改目标、放宽阈值、删除难例、伪造标注或循环重试绕过阻塞。

## Global Constraints

1. 最终目标是**同例** A/U/R，不把分别在不同数据集上的成功相加为联合成功；最终二维/三维均须独立评价。
2. `Dataset902` 旧框填充标签不得注册为精确 U 真值；`numTraining=1860` 不得解释为患者数。[E1–E3]
3. SKM A6 为髌骨软骨、股骨软骨、内/外侧胫骨软骨、内/外侧半月板；ACL/PCL 不在已确认 A6 范围。原始通道顺序必须实查，不能照整数猜测。
4. `inspected=true`、Norm 队列名、空 mask、被删除的低置信框都不自动提供完整目标阴性资格。
5. 来源不明、自动或几何生成标签只在相应辅助/伪监督角色使用；独立评价参照必须来源合格。
6. 所有回波、随访、裁剪、降质、缓存和派生副本继承父患者分组；明确跨来源别名与既有暴露历史。
7. 标签读入应保留 NIfTI proxy scaling；复用并测试 `anatobind.aur.labels.discrete_labels`，其容差 `1e-5` 只修复接近整数的浮点误差，不用于把概率变成标签。已截断为零的旧派生标签须从原始源重新导出。[E6]
8. 同 shape 不代表同空间；禁止通过静默改 header 假装配准。薄软骨和小病灶不得未经评估统一降到 1 mm。
9. 训练时允许使用复核资格遮蔽损失；部署输入不得包含复核图、GT A/U/R、GT框、真实组织ID或标注来源。
10. 所有主要结果在患者级统计；种子、病灶、切片、序列不得当作额外独立患者。
11. 不承诺性能增益；使用预注册主要终点、误报预算、非劣容差和置信区间。技术流程跑通与科学假设成立分开判定。
12. 禁止全数据默认混训、默认 SSL-first、未经允许用测试影像自监督、自动获取外部数据或启动无预算训练。
13. 不继承旧计划的 GPU 编号、预算或训练状态。执行时解析调度器/CUDA 约束到物理 GPU UUID；不能通过覆盖环境变量突破原有资源限制。
14. 原始数据、旧结果、旧失败、旧协议与历史哈希不可覆盖；新版本输出使用创建时排他检查，续跑必须匹配登记身份。

## Review Focus

|最容易出错的条件|必须具备的行为|负责测试|
|---|---|---|
|缩放后的类别值略小于1、真实概率图、历史全零导出|近整数保真；概率图拒绝；损坏导出从源重建，不把空值当阴性|T05 `test_scaled_categorical_labels`|
|同患者多轨道/随访、跨来源重叠、已有测试暴露|父谱系连通分组；已暴露集不能改名为独立盲测|T03 `test_lineage_split_lock`|
|框/伪mask/未复核区域被赋予阴性资格，或H2泄漏被删病灶位置|参照角色拒绝、无资格梯度为零、H2隔离隐藏参照|T06、T18|
|单切片路径暗中用邻层、整卷统计、三维A或缓存|二维数据接口物理隔离；访问日志中越界读为零|T22 `test_single_slice_information_boundary`|
|全拒判、空mask、边界截断、无有效关系参照|不抬高主指标；明确NA与覆盖率；病灶总分母预先冻结|T08、T16、T17|

---

## 1. 路线、依赖与验收门

### 1.1 执行顺序

```text
T00 接手/复用审计
  └─ T01 数据契约与状态记录
      ├─ T02 文件资产/完整性 → T03 患者谱系/划分
      └─ T04 本体 → T05 标签/空间QC → T06 监督资格与损失
             └─ T07 发布审计清单与G0
                 ├─ T08 评价器
                 ├─ T09 精标试点（专家） → T13 精标版本/G1
                 └─ T10 输入管线 → T11 训练控制 → T12 ISLES U / SKM A
T06+T08+T10+T11+T13 → T14 同例普通联合基线/G2
T14 → T15 软解剖引导 → T16 关系与端到端 → T17 H1/G3_READY
T12+T06+T08+T11 → T18 缺标生成器 → T19 公平方法组 → T20 H2/G4_READY
T17 → T21 独立联合扩展
T17+T10+T11 → T22 同checkpoint双模式 → T23 双模式验收/G5_READY
已批准且冻结的阶段证据 → T24 锁定测试、正式科学裁决及最终交付
```

T09 标注与 T10–T12 工程可并行。H2 可在合格脑部 U 子集上开展，不必等待所有 SKM 精标；但不得用 H2 替代同例联合研究。T21 与 T22 可在依赖满足后并行。没有 subagent 能力时串行执行，不伪称并行代理。

### 1.2 Gate 是版本化裁决，不是自动“全通过”

|门|放行内容|硬性条件|不通过后的行动|
|---|---|---|---|
|G0-U / G0-A|各来源的基线|该来源本体、谱系、配对、空间、可读性、监督来源合格|只隔离不合格样本/来源；不必等待P2全库|
|G0-joint|同例队列准备|同例身份与A/U/R参照资格明确|记录缺标，不使用代理参照顶替|
|G1|精标扩展及正式联合定义|试点可判读性、双读者流程、目标阴性范围获专家签核|收窄病变表现或换合格场景，保留最终目标|
|G2|方法比较|独立/共享基线和评价器可复现；数据冻结|修正工程或监督，不扩大模型掩盖问题|
|G3|H1联合科学结论|同例独立参照＋固定候选/端到端公平比较＋冻结主要终点|允许假设不支持；记录效应及区间|
|G4|H2科学结论|同信息缺标比较、来源内阴性、误报约束|无增益时仅报告资格治理，不硬称算法创新|
|G5|完整双模式/扩展声明|同checkpoint二维和三维分别A/U/R验收，独立扩展范围透明|只声明实际已验证范围；双模式未完成明确保留|

一项 `GateDecision` 必须同时记录 `engineering_status` 与 `scientific_conclusion`；前者为 PASS/BLOCKED/FAIL，后者为 SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE/NOT_EVALUATED。训练结束只改变运行状态，不自动改变任一科学结论。

## 2. 文件布局与接口契约

以下除已明确标为 Existing 的文件外，均为**计划新增**。任务的Files中，未以 `anatobind/`、`configs/`、`schemas/`、`tests/` 或 `reports/` 开头的Python相对路径，统一相对于 `anatobind/v41/`；根部既有文档和私有产物另行标明。先查重，已有等价实现优先通过包装复用；出现重名不能覆盖。所有新代码限定 `anatobind/v41/`，不重写旧 `anatobind/aur/`。

```text
plan.md
AnatoBind_MRI_v4.1_Feasibility_Proposal.md
AnatoBind_MRI_v4.1_Supervision_Data_Contract.md
FM_data_inventory_20261101.md
anatobind/v41/
  schema.py             # SampleRecord/AnnotationRecord/Prediction/GateDecision
  paths.py              # 私有写路径/脱敏/原子发布
  registry.py           # 资产manifest版本/审计发布
  inventory.py          # 受控遍历/增量完整性
  identity.py           # 别名、患者/父扫描谱系
  splits.py             # 冻结划分和接触历史
  ontology.py           # 显式语义映射
  labels.py             # 包装已验证类别解码；不改旧标签空间
  geometry.py           # 物理空间与变换链
  eligibility.py        # 逐任务/类别/区域资格
  annotation.py         # 标注包/导入/双读者记录
  adapters/{isles,skm,pdgm,bmsr}.py
  observations.py       # 3D/strict-2D输入构造
  datasets.py           # observation→Batch
  losses.py             # masked目标、来源/任务归一化
  models/{core,anatomy_fusion,relation,observation_adapter}.py
  engine/{registration,trainer,inference}.py
  experiments/{h1,missingness,h2,h3}.py
  evaluation/{segmentation,instances,relations,selective,statistics,joint}.py
  cli.py                # 本文约定的统一入口
configs/v41/
  paths.example.yaml
  sources.yaml
  ontology.yaml
  annotation_protocol.yaml
  isles_u.yaml
  skm_a.yaml
  joint_baselines.yaml
  h1.yaml
  h2.yaml
  h3.yaml
  thresholds.template.yaml
  run_registration.template.yaml
schemas/v41/{sample,annotation,gate,experiment}.schema.json
tests/v41/{conftest.py,test_*.py}     # 仅合成/匿名fixture
reports/v41/                        # 仅审核后的匿名汇总
```

私有数据布局：`$PRIVATE_ROOT/{audits,registry,annotations,derived,runs,evaluation,restricted_reference}/<version-or-run-id>/`。锁定测试与 H2 隐藏完整参照需要独立角色/作业访问；不同目录或Python对象本身不是安全隔离。至少按作业只提供白名单 manifest 与挂载范围，并用读访问审计验证；可用独立账户/ACL时优先使用。

### 2.1 T01 负责的共同类型

- `SampleRecord` / `AnnotationRecord`：逐字段落实监督契约第3节，不允许自由字典跳过验证。增加 `leakage_group_uid`、`exposure_history`、`native_acquisition_dim`、`observation_mode`、`geometry_status`、`label_value_qc`；AnnotationRecord另加 `reference_role`（expert_voxel/weak_localization/pseudo_only/unverified）、`adjudication_status`、`reader_ids`、`protocol_hash` 和 `visibility_evidence`；注册阶段允许 `split=unassigned`，训练阶段不允许。
- `Prediction`：`sample_uid, observation_uid, mode, geometry, a_prob, u_prob, instances, relations, acceptance, lineage`。与参照独立，不能包含GT字段。
- `GateDecision`：`gate_id, scope_hash, engineering_status, scientific_conclusion, evidence_hashes, blockers, approved_by, approved_at`。
- `RunRegistration`：源码/配置/数据/划分/本体/监督/初始化哈希、可读角色、预算、硬件、种子、数值设置、评价规则与审批。
- 其余接口用 `Path`、`list[SampleRecord]`、`dict[str, Any]`、NumPy数组或PyTorch张量；配置字典必须经对应 schema 校验。示例断言的 fixture 由各任务在 `tests/v41/conftest.py` 或本任务测试文件补齐，全部用合成数据。

### 2.2 Tensor、坐标与结果约定

训练张量采用 `[B,C,Z,Y,X]`；原始 NIfTI voxel index 采用其 header 对应的 xyz，经显式变换转换。坐标数组顺序与物理轴不混用。A6 主线为7通道互斥分类含背景；U按批准的类别集合为多标签logits，范围/资格按通道对齐。不能把病灶实例ID当语义类别。

`Batch` 是受检字典：`image, valid_fov, a_target, u_target, e_a, e_u, metadata`。训练 metadata 中的读者/来源/GT组织属性不传给模型；推理仅允许影像、有效FOV和必要几何/模式信息。使用任务已知的固定协议不等于推断病种。

模型输出A/U概率、病灶实例和关系；缺损表面仅在有批准标签时输出，缺损体积D不属于一期必需代码。`A_prob * U_prob` 只能作为交叠特征，不称校准联合概率。

### 2.3 命令和任务完成记录

本文 `python -m anatobind.v41.cli ...` 是**待实现的CLI合同**，不是对当前仓库已存在命令的断言。任务只增加其负责的子命令。执行前令 `PY` 指向经核验的虚拟环境解释器；示例中的 `$PRIVATE_ROOT` 必须先经过私有路径检查。

所有代码任务遵循：先写本任务失败测试→运行确认正确失败原因→实现最小功能→运行本任务及受影响回归测试→保存匿名证据→在获准的工作区提交指定代码文件。禁止 `git add .`、自动push、自动合并或提交医学输出。

每项写私有 `task_record.json`：任务ID、状态、输入哈希、命令、退出码、输出哈希、测试实际结果、阻塞项、人工签核及下一项；公共报告仅从白名单字段生成。任务未经执行不得预填“测试通过”。

---

## 3. 分任务实施

### T00 — 接手审计与复用裁决

**依赖：** 无。**负责人：** Coding Agent＋研究者。**性质：** 只读审计，不启动模型。

**Files:** Existing：`AGENTS.md`、`PROJECT_CONTEXT.md`、`STATUS.md`、`CLAUDE.md` 及它们明确引用的当前证据；Create：`reports/v41/reuse_decisions.md`（匿名）。私有详细审计写 `audits/<audit-id>/`。

**产出：** `reuse_decisions.md` 对每项候选给出 REUSE / WRAP / REIMPLEMENT / LEGACY_ONLY，以及输入本体、初始化暴露、数值解码、评价角色、可验证证据。

- [ ] 读取本计划和两份spec，记录哈希；核对当前repo HEAD、相关工作分支差异及工作区修改状态，只输出匿名数量和允许的代码路径。
- [ ] 只读核验既有完成证书、登记和哈希；不为“确认完成”重复运行训练、影像推理或旧诊断。
- [ ] 核对Python解释器、依赖锁、输出路径空间与权限；只读取GPU约束，不修改可见性、不申请计算任务。
- [ ] 将现有标签解码、坐标工具、checkpoint恢复、训练控制器、双模式模型分别评为可复用或需包装。旧checkpoint暴露史不明时，默认不得进入严格inductive主结果。
- [ ] 记录新科研验收与旧工程门的差异；不能将旧A32和SKM A6统一成同一ID表。未核验私有记录的条目标 `UNVERIFIED`。
- [ ] 输出接手裁决；只在获准实施时创建隔离工作区。不得变更登记分支指针或自动推送。

**验收：** 研究目标、当前工程状态、计划新增资产、未核验事项四者分栏；零远端数据改写、零重复GPU作业。本任务是审计，不伪造红/绿单元测试记录。

### T01 — 统一schema、私有路径与任务状态

**依赖：** T00。**Files:** Create：`anatobind/v41/{__init__,schema,paths,cli}.py`、`schemas/v41/{sample,annotation,gate,experiment}.schema.json`、`configs/v41/paths.example.yaml`、`tests/v41/{conftest,test_schema,test_paths}.py`。

**Interfaces:** `validate_sample(record: dict) -> SampleRecord`；`validate_annotation(record: dict) -> AnnotationRecord`；`resolve_private_output(path: Path, root: Path, repo: Path) -> Path`；`write_task_record(path: Path, record: dict) -> None`。schema涵盖§2.1全部类型，CLI先仅提供`--help`。

- [ ] 写失败测试：完整合成sample可验证；缺`provenance_type`拒绝；Git内输出、符号链接越界拒绝；错误文本不包含传入的私有路径或病例ID。
```python
def test_missing_provenance_is_rejected(annotation_dict):
    annotation_dict.pop("provenance_type")
    with pytest.raises(ValueError):
        validate_annotation(annotation_dict)
```
- [ ] 运行 `$PY -m pytest tests/v41/test_schema.py tests/v41/test_paths.py -q`，确认因缺少新实现失败，不把依赖错误当目标红测。
- [ ] 实现版本化schema、受限输出路径、原子新建/续写、匿名错误码；`source_release`未知可登记但不得伪填。所有状态初值TODO。
- [ ] 同命令重跑，并测试部分写入/中断不会留下可误认为完成的证书。
- [ ] 提交上述代码、模板和合成测试；私有输出不入Git。

**验收：** schema拒绝隐式“缺字段即默认正常”；公共模板不含真实患者信息；真实私有路径只存在受保护登记。

### T02 — 资产发现、来源溯源与增量完整性

**依赖：** T01。**Files:** Create：`anatobind/v41/inventory.py`、`configs/v41/sources.yaml`、`tests/v41/test_inventory.py`；Modify：`cli.py` 增加 `inventory`、`verify-assets`。

**Interfaces:** `discover_assets(root: Path, source_config: dict) -> list[dict]`；`verify_asset(asset: dict, level: str) -> dict`。level限定 `metadata/header/full_decode`，与内容哈希状态分别记录。

- [ ] 写测试：空目录不计为数据集病例；归档与解压资产不能双计病例；`.truncated`隔离；不越界跟随链接；历史token/凭据文件仅记录排除数量，不读取内容。
```python
def test_truncated_asset_is_quarantined(asset_fixture):
    report = verify_asset(asset_fixture("damaged.h5.truncated"), "header")
    assert report["training_eligible"] is False
    assert report["reason"] == "TRUNCATED_ASSET"
```
- [ ] 运行 `$PY -m pytest tests/v41/test_inventory.py -q` 确认红测。
- [ ] 实现17个一级入口的源级清单，P0为ISLES/SKM；其他来源先记录结构及待验证状态。派生条目必须有父源，不以条目数量报患者数。仅经批准读取医学影像头及内容。
- [ ] 重跑测试；对正式准入的图像/标签执行完整可读性与内容哈希检查，按首次读取记录避免重复I/O；只读头的资产保持`header_only`，不能标“全量验证通过”。
- [ ] 保存私有 `assets.jsonl`、`asset_integrity.jsonl` 及匿名计数，提交代码。

**验收：** 所有结果可区分 `observed / verified / quarantined / not_checked`；TB级k-space不列为一期必须解码对象。P0来源可先独立放行。

### T03 — 患者身份、派生谱系与冻结划分

**依赖：** T01、T02。**Files:** Create：`anatobind/v41/{identity,splits}.py`、`tests/v41/test_identity_splits.py`；Modify：`cli.py` 增加 `resolve-identities`、`freeze-splits`。

**Interfaces:** `resolve_groups(records: list[SampleRecord], aliases: list[dict]) -> dict[str, str]`；`validate_splits(records: list[SampleRecord], groups: dict[str, str]) -> dict`；`freeze_splits(records: list[SampleRecord], policy: dict, out: Path) -> dict`。

- [ ] 写测试：同父扫描不同回波、两个轨道、裁剪、随访别名及跨源已知映射都归同组；SibBMS的MS/Norm同号不误合并；已有测试暴露不能改称unexposed。
```python
def test_lineage_split_lock(related_records, alias_edges):
    groups = resolve_groups(related_records, alias_edges)
    report = validate_splits(related_records, groups)
    assert report["cross_split_groups"] == 1
    assert report["pass"] is False
```
- [ ] 运行 `$PY -m pytest tests/v41/test_identity_splits.py -q` 确认红测。
- [ ] 实现别名/父源连通分组；只接受有证据的别名边。PDGM随访映射、BMSR病例表待审核，不能靠裁掉数字/字母猜测。文件哈希去重只是同图检测，不证明不同图来自不同人。
- [ ] 重跑；冻结train/development_val/locked_test角色、抽样规则、哈希、历史接触记录；未解析的身份歧义隔离或保守同组并披露，不能随机跨分割。
- [ ] 保存私有 `patient_alias_report.csv`、`split_leakage_report.json`、`exposure_ledger.jsonl`；提交代码与匿名汇总。

**验收：** 正式子集零已知跨split谱系泄漏；未知跨源重叠明确列出。已经被旧实验反复使用的病例只能称historical/内部验证，不以新随机split宣称独立盲测。

### T04 — A/U/R本体与参照资格映射

**依赖：** T01、T02。**Files:** Create：`anatobind/v41/ontology.py`、`configs/v41/ontology.yaml`、`tests/v41/test_ontology.py`。

**Interfaces:** `map_skm_a_to_host(label: int) -> str`；`resolve_relation_scope(tissue_id: int) -> dict`；`validate_ontology(config: dict) -> dict`。

- [ ] 写测试：A1→髌骨、A2→股骨、A3/4→胫骨、A5/6→半月板；旧tissue ACL/PCL返回范围外，而不是背景；不得凭旧组织属性补侧别。
```python
def test_skm_semantics_are_not_integer_identity():
    assert map_skm_a_to_host(1) == "patellar_cartilage"
    assert map_skm_a_to_host(5) == "meniscus"
    assert resolve_relation_scope(2)["status"] == "out_of_anatomy_scope"
```
- [ ] 运行 `$PY -m pytest tests/v41/test_ontology.py -q` 确认红测。
- [ ] 实现源标签→统一语义的显式映射；原H5通道另登记。U病变语义、R_geo/R_clin语义、A32与A6配置分别版本化；`no_host / unknown / out_of_scope`不合并。
- [ ] 重跑；对PDGM/BMSR先记录待核验值域，不猜标签号。缺损表面与缺损体积使用不同任务名。
- [ ] 保存 `ontology_validation_report.json`，提交代码与配置。

**验收：** 无未知整数被静默映射为背景；任何临床定义尚未签核的任务保持不可正式运行。

### T05 — 标签解码、空间QC与可逆变换

**依赖：** T01、T02、T04。**Files:** Create：`anatobind/v41/{labels,geometry}.py`、`tests/v41/test_geometry_labels.py`；Existing reference：`anatobind/aur/labels.py`；Modify：`cli.py` 增加 `geometry-qc`。

**Interfaces:** `read_categorical_nifti(path: Path) -> tuple[np.ndarray, dict]`；`audit_geometry(image: dict, label: dict) -> dict`；`transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray`。

- [ ] 写测试：NIfTI proxy scaling后近整数正确解码；概率值拒绝；相同shape但错affine不通过；标签变换保留类别；有效FOV与padding分开。
```python
def test_scaled_categorical_labels():
    assert discrete_labels(np.array([0.0, 0.999999]))[1] == 1
    with pytest.raises(ValueError):
        discrete_labels(np.array([0.25, 0.75]))
```
- [ ] 运行 `$PY -m pytest tests/v41/test_geometry_labels.py -q`；确认新增读入/空间测试先失败，不需要让已存在的解码函数故意失败。
- [ ] 包装已验证解码，记录shape/spacing/affine/qform/sform/单位/轴序/物理FOV/变换链。ISLES DWI/ADC-mask实查，FLAIR未配准则不上线；SKM轨道与bbox轴序按源协议检查。
- [ ] 重跑；合成可逆仿射往返误差 `atol=1e-4` voxel，仅为数值测试，不是临床配准容差。每来源/转换分支导出私有叠图供人审，不能以同shape自动通过。
- [ ] 输出 `geometry_qc.csv`、`label_value_qc.csv`、`image_label_pairing.csv`；提交代码。旧全零截断派生标签只从原源新版本重建，不就地“修复”。

**验收：** 正式数据全部完成空间检查；人工叠图覆盖边缘、小病灶和变形情况。真实空病灶mask另有明确阴性来源，否则隔离。

### T06 — 监督资格与masked loss

**依赖：** T01、T04、T05。**Files:** Create：`anatobind/v41/{eligibility,losses}.py`、`tests/v41/test_eligibility_losses.py`。

**Interfaces:** `build_eligibility(record: SampleRecord, task: str, class_scope: list[str]) -> np.ndarray`；`masked_bce(logits: Tensor, target: Tensor, eligible: Tensor) -> Tensor`；`masked_dice_loss(logits: Tensor, target: Tensor, eligible: Tensor) -> Tensor`；`aggregate_task_losses(losses: dict, supports: dict, weights: dict) -> Tensor`。

- [ ] 写测试：未复核体素梯度为零；Dice分子/分母都排除；某类完整标注不产生另一病种阴性；confidence=0框排除不转背景；全ignore返回有限零损失且不稀释其他任务均值。
```python
def test_unreviewed_has_zero_gradient():
    logits = torch.zeros(2, requires_grad=True)
    loss = masked_bce(logits, torch.tensor([1., 0.]), torch.tensor([True, False]))
    loss.backward()
    assert logits.grad[1].item() == 0.0
```
- [ ] 运行 `$PY -m pytest tests/v41/test_eligibility_losses.py -q` 确认红测。
- [ ] 实现按任务/类别/区域的正负资格与权重分离；人工、伪标签来源分组归一化；框仅生成定位目标，不生成U逐体素真值。
- [ ] 重跑并验证改变全部ignore位置的logit不改变masked Dice；训练接口不得将资格图送入预测模型。
- [ ] 输出 `eligibility_coverage.csv` 与按来源任务的有效正/负监督计数，提交代码。

**验收：** 已确认目标阴性是显式记录；缺标不是unknown图像类别；无负监督支撑的实验不进入正常状态/FP科学验收。

### T07 — 审计manifest发布与G0签核

**依赖：** T02–T06。**Files:** Create：`anatobind/v41/registry.py`、`tests/v41/test_registry.py`；Modify：`cli.py` 增加 `build-manifest`、`audit`、`gate`。

**Interfaces:** `publish_manifest(records: list[SampleRecord], audit_reports: list[dict], out: Path) -> dict`；`evaluate_gate(gate_id: str, reports: list[dict], approvals: dict) -> GateDecision`。

- [ ] 写测试：box_fill拒绝注册为expert_voxel_test；缺空间/别名审计阻塞相应子集；自动A不能因路径名含`labels`就成为人工参照。
```python
def test_box_fill_cannot_enter_expert_test(box_fill_record, good_audits, tmp_path):
    box_fill_record["annotation_records"][0]["intended_role"] = "test"
    box_fill_record["annotation_records"][0]["reference_role"] = "expert_voxel"
    with pytest.raises(ValueError, match="UNQUALIFIED_REFERENCE"):
        publish_manifest([validate_sample(box_fill_record)], good_audits, tmp_path)
```
- [ ] 运行 `$PY -m pytest tests/v41/test_registry.py -q` 确认红测。
- [ ] 汇总schema、身份、本体、空间、来源、可见性与完整性，发布不可变 `manifest.jsonl`、`manifest.lock.json`；每个来源独立G0，不因P2未审计阻塞合格P0。
- [ ] 重跑；发布必须包含 `ontology_validation_report.json`、`patient_alias_report.csv`、`split_leakage_report.json`、`geometry_qc.csv`、`annotation_provenance_report.csv`、`eligibility_coverage.csv`、`image_label_pairing.csv` 和 `input_visibility_report.csv` 八类报告。尚未临床批准的completeness保持unknown，机器不代填。
- [ ] 输出 `G0-U.json`、`G0-A.json` 与 `G0-joint.json`；joint可为BLOCKED且不影响合格基线。提交实现及匿名结论。

**验收：** manifest每一正式目标可追溯；准入/排除原因可审计；临床参照尚缺时明示，不伪称全部数据就绪。

### T08 — 先实现评价器及极端样本测试

**依赖：** T01、T04、T06。**Files:** Create：`anatobind/v41/evaluation/{__init__,segmentation,instances,relations,selective,statistics,joint}.py`、`configs/v41/thresholds.template.yaml`、`tests/v41/test_metrics.py`。

**Interfaces:** `segmentation_metrics(pred: np.ndarray, truth: np.ndarray, eligible: np.ndarray, spacing: tuple, policy: dict) -> dict`；`match_instances(pred: list[dict], truth: list[dict], policy: dict) -> list[tuple[int,int]]`；`joint_score(rows: list[dict]) -> dict`；`paired_patient_bootstrap(rows: list[dict], seed: int, n_boot: int) -> dict`。

- [ ] 写测试：单体素/贴边/合并/分裂一对一匹配；同病灶重复预测计FP；全拒判不能提高全覆盖J；无R参照计覆盖率而非自动正确。
```python
def test_all_rejected_is_not_joint_success(eligible_lesion_rows):
    for row in eligible_lesion_rows:
        row["accepted"] = False
    scores = joint_score(eligible_lesion_rows)
    assert scores["j_all_coverage"] == 0.0
    assert scores["acceptance_coverage"] == 0.0
```
- [ ] 运行 `$PY -m pytest tests/v41/test_metrics.py -q` 确认红测。
- [ ] 实现Dice/NSD/HD95、实例检出/FP、关系集合/每类F1、false-normal和风险—覆盖率；输出原始充分统计量。匹配先最大化合格配对数，再以IoU作确定性次序；匹配阈值属于冻结协议。
- [ ] 重跑：空GT病例的病灶Dice不并入阳性病例均值；另报真阴性/FP。单边空mask的表面距离为未定义并列失败计数，不能静默删掉；不同实现需同口径。
- [ ] 提交代码；临床距离容差、FP预算、J范围阈值、A非劣界初值null，生产评价遇null必须BLOCKED，不拿测试数据补值。

**验收：** 主J报告患者等权平均（有合格目标病灶的患者），同时报告病灶池化J；阴性患者进入FP/特异性，不能混入阳性J分母。建议配对患者bootstrap 2,000次、95%区间、统计seed=20261101，作为本计划提出的默认值在登记前确认。训练种子另列，不能增加患者n。

### T09 — SKM精标试点包与专家SOP

**依赖：** T03–T07；可与T08/T10并行。**Files:** Create：`anatobind/v41/annotation.py`、`configs/v41/annotation_protocol.yaml`、`tests/v41/test_annotation.py`；Modify：`cli.py` 增加 `annotation-pack`、`annotation-import`。

**Interfaces:** `export_review_package(records: list[SampleRecord], protocol: dict, out: Path) -> dict`；`import_annotations(package: Path, protocol: dict) -> list[AnnotationRecord]`。

- [ ] 写测试：试点患者与最终盲测互斥；专家包默认没有模型/几何预测；读者版本不覆盖；额外参考序列使用必须显式记录；纯框无法冒充精标导入。
```python
def test_unadjudicated_mask_stays_unqualified(review_package, protocol):
    rows = import_annotations(review_package, protocol)
    assert all(r.quality_status != "accepted" for r in rows if r.adjudication_status != "accepted")
```
- [ ] 运行 `$PY -m pytest tests/v41/test_annotation.py -q` 确认红测。
- [ ] 实现20–30扫描的开发试点选择与私有导出：固定E1/E2观察，覆盖软骨位置、微小/模糊、疑难和目标阴性；它不是患病率抽样。分组规则与实际患者数记录。
- [ ] 重跑；由两位读者独立评价预先指定子集并仲裁。protocol必须记录实际双读者数量、抽样和理由；Agent不能把无人标注状态改为已完成。
- [ ] 保存标注耗时、可判读率、边界/实例一致性、目标阴性范围与R依据。临床签核前任务保持 `BLOCKED_EXPERT_REVIEW`，代码可单独完成。

**验收：** U边界及不确定带、A必要修订、实例规则、review_regions、visibility、R_geo或R_clin来源齐全。无法稳定划界的裂隙/缺损转独立表面或等级任务，不制作假体积GT。

### T10 — 数据适配器与无GT依赖的输入管线

**依赖：** T06、T07。**Files:** Create：`anatobind/v41/adapters/{__init__,isles,skm,pdgm,bmsr}.py`、`anatobind/v41/{observations,datasets}.py`、`tests/v41/test_observations.py`。

**Interfaces:** `build_observation(record: SampleRecord, spec: dict) -> dict`；`load_training_batch(observations: list[dict]) -> dict`；`build_inference_input(record: SampleRecord, spec: dict) -> dict`。adapters的共同接口为 `load_record(record: SampleRecord, allowed_roles: set[str]) -> dict`。

- [ ] 写测试：ISLES DWI/ADC与目标正确配对；SKM E1/E2显式读取已核验H5键；padding不产生正常监督；推理不访问GT、review_region、标注类别或真实tissue_id。
```python
def test_inference_does_not_require_annotations(record_without_labels, observation_spec):
    batch = build_inference_input(record_without_labels, observation_spec)
    assert "image" in batch and "valid_fov" in batch
    assert not {"a_target", "u_target", "e_a", "e_u", "tissue_id"} & batch.keys()
```
- [ ] 运行 `$PY -m pytest tests/v41/test_observations.py -q` 确认红测。
- [ ] 实现固定协议：ISLES DWI/ADC；SKM E1/E2。PDGM/BMSR先提供经过审计的独立适配器，不自动纳入一期训练。读不到所需模态时阻塞该样本，不能默认补零后当完整协议。
- [ ] 重跑，新增确定性抽样、可逆输出空间和输入访问记录。训练patch允许从合格训练标签抽样；验证/测试必须全体积滑窗或预测ROI，禁止GT框裁剪。预测ROI漏掉目标仍计漏检。
- [ ] 保存输入快照的匿名shape、通道、FOV、协议哈希，提交代码。

**验收：** 正式推理只依赖部署可获得输入；缓存键含源哈希、变换、模式、本体版本。二维接口仅预留合同，严格实现与访问测试由T22负责。

### T11 — 实验登记、训练控制与可恢复性

**依赖：** T01、T08、T10。**Files:** Create：`anatobind/v41/engine/{__init__,registration,trainer,inference}.py`、`configs/v41/run_registration.template.yaml`、`tests/v41/test_run_control.py`；Modify：`cli.py` 增加 `register-run`、`smoke`、`train`、`infer`。

**Interfaces:** `register_run(config: dict, gates: list[GateDecision], out: Path) -> RunRegistration`；`run_training(registration: RunRegistration, resume: Path | None = None) -> dict`；`infer_observation(model: nn.Module, observation: dict, policy: dict) -> Prediction`；通用 `freeze_protocol(config: dict, dev_summary: dict, approvals: dict, out: Path) -> dict` 实现在 `engine/registration.py`，H1/H2/H3均复用，不互相形成代码依赖。

- [ ] 写测试：缺预算/审批拒绝；locked_test角色不可读取；错误数据/源码哈希拒绝续跑；第二控制器不覆盖第一控制器状态；父进程退出后子worker终止。
```python
def test_registration_requires_explicit_budget(config, passed_gates, tmp_path):
    config["max_steps"] = None
    with pytest.raises(ValueError):
        register_run(config, passed_gates, tmp_path)
```
- [ ] 运行 `$PY -m pytest tests/v41/test_run_control.py -q` 确认红测。
- [ ] 优先包装T00审计合格的控制器；记录model/optimizer/scheduler/scaler/RNG/sampler状态、数值backend、硬件UUID和实际读入角色。先CPU合成单batch，再获准GPU单batch和小样本过拟合；显存实测后确定patch/batch。
- [ ] 重跑并注入中断：确定性CPU路径验证连续运行与恢复一致；CUDA仅按登记容差报告实际恢复检查，不把RNG恢复称未来bitwise可重复。
- [ ] 提交实现。训练不得无上限重试；NaN、容量溢出、未批准资源、私有路径违规立即停止并保留失败证据。更换配方新建run，不修改旧登记。

**验收：** `max_steps / max_wall_seconds / checkpoint_every / evaluation_every / resource_assignment / initialization_lineage` 均为明确非空的已批准值；历史4h/2h或旧卡号不自动继承。无签核只可执行合成测试和注册dry-run。

### T12 — ISLES真实U与SKM真实A强基线

**依赖：** 相应来源G0、T08、T10、T11。**Files:** Create：`configs/v41/{isles_u,skm_a}.yaml`、`anatobind/v41/models/{__init__,core}.py`、`tests/v41/test_baseline_wiring.py`；Existing backend：经T00确认的nnU-Net/分割组件，必要时仅加包装。

**Interfaces:** `build_model(config: dict) -> nn.Module`；模型 `forward(image: Tensor, valid_fov: Tensor) -> dict[str, Tensor]` 输出启用任务的 `a_logits/u_logits/image_features`。A-only/U-only不能隐式要求另一个GT头。

- [ ] 写测试：U-only在无A标签时训练；A-only不读取病灶框；SKM A输出7通道；ISLES输入2通道；测试集不影响归一化/采样权重。
```python
def test_skm_anatomy_channels(skm_model_config, synthetic_volume):
    model = build_model(skm_model_config)
    output = model(synthetic_volume, torch.ones_like(synthetic_volume[:, :1], dtype=torch.bool))
    assert output["a_logits"].shape[1] == 7
```
- [ ] 运行 `$PY -m pytest tests/v41/test_baseline_wiring.py -q` 确认红测。
- [ ] 建立独立强基线配置；先开发集检查输入、loss、原网格输出和读者认可的标签范围。nnU-Net不支持相同资格时只使用合格完整子集或明确实现masked适配，禁止把ignore转背景凑兼容。
- [ ] 重跑代码测试；在实际训练获批后登记并运行基线，保存学习曲线、非有限值、有效监督覆盖、体积级开发指标。先开发seed=17，正式比较建议seed=17/29/43；这是拟定设置，需进入签核登记。
- [ ] 归档匿名基线报告和可重放评估数据；提交配置与实现，不提交权重/逐例结果。

**验收：** 能从全体积图像产生实际A或U输出；ISLES U不被称作完整联合能力。旧checkpoint仅在暴露、语义和初始化可核验时纳入独立对照；“已有代码跑通”不能替代这两套新目标的验证。

### T13 — 精标队列发布、G1与最终参照预注册

**依赖：** T09专家工作、T03–T08。**Files:** Modify：`annotation.py`、`registry.py`；Create：`tests/v41/test_joint_reference_release.py`；私有输出：`annotations/<release>/`、`registry/<release>/`。

**Interfaces:** `release_joint_cohort(records: list[SampleRecord], protocol: dict, approvals: dict, out: Path) -> dict`；复用 `evaluate_gate` 和 `publish_manifest`。

- [ ] 写测试：无读者签核不能发布expert参照；缺R不能伪补；试点病例不进锁定测试；A或U由模型生成时不自动具备独立R_geo测试资格。
```python
def test_unsigned_cohort_cannot_be_released(joint_records, protocol, tmp_path):
    with pytest.raises(ValueError):
        release_joint_cohort(joint_records, protocol, approvals={}, out=tmp_path)
```
- [ ] 运行 `$PY -m pytest tests/v41/test_joint_reference_release.py -q` 确认红测。
- [ ] 实现双读者/仲裁/复核范围导入，保留原版本。专家根据试点冻结异常表现、可判读范围、实例规则、R类型、表面容差和不确定边界处理，再决定60–100扫描扩展预算。
- [ ] 重跑；目标20–30名锁定测试患者只是初始规划。依据每类病例支持和患者级区间精度签核实际规模；历史已接触患者单列，不能作为新独立验证充数。
- [ ] 发布精标manifest、预先可评价参照集合、G1裁决和coverage表；拒绝未签核/不可靠类进入主要终点。代码提交与临床签核分别记录。

**验收：** 同例A/U/R至少一个可测试场景；有明确复核的目标阴性区域。无真实目标阴性扫描时，只准备区域指标，不声称病例阴性特异性。正式测试影像/标签由受限角色保管，不交训练者。

### T14 — 同例普通联合模型与G2

**依赖：** T06、T08、T10–T13。**Files:** Modify：`models/core.py`、`losses.py`；Create：`configs/v41/joint_baselines.yaml`、`tests/v41/test_joint_baselines.py`。

**Interfaces:** `build_model` 新增 `mode=a_only/u_only/shared_au/independent_au`；shared_au共享编码器、独立A/U解码器，不含显式A→U融合。训练接口统一复用T11。

- [ ] 写测试：同一影像可仅监督A、仅监督U或同时监督；不存在的标签无梯度；独立模型和共享模型输出同一评价schema。
```python
def test_joint_forward_is_independent_of_gt(model, image, fov):
    output = model(image, fov)
    assert {"a_logits", "u_logits", "image_features"} <= output.keys()
    assert "annotation" not in inspect.signature(model.forward).parameters
```
- [ ] 运行 `$PY -m pytest tests/v41/test_joint_baselines.py -q` 确认新增功能红测。
- [ ] 实现普通双头基线，独立A+U系统与几何R对照；同例精标子集、输入、增广、采样和初始化路线保持可比。伪标签辅助默认关闭。
- [ ] 重跑；获批后在开发集训练A-only、U-only、shared_au、independent_au。分别记录参数、训练曝光量、计算量和调参预算，不能只称“相同epoch”。
- [ ] 输出G2证据：可复现数据流、可恢复运行、开发性能与失败病例类别；提交代码/配置。

**验收：** G2首先是基线可复现与监督正确，不预设联合模型一定胜出；测试数据尚未解封。

### T15 — 软解剖引导与错误传播消融

**依赖：** T14。**Files:** Create：`anatobind/v41/models/anatomy_fusion.py`、`tests/v41/test_anatomy_fusion.py`；Modify：`models/core.py`、`configs/v41/h1.yaml`（在此创建）。

**Interfaces:** `AnatomyResidualFusion.forward(image_features: Tensor, anatomy_prob: Tensor, anatomy_uncertainty: Tensor, enabled: bool = True) -> Tensor`。生产A来自模型自身；GT/扰动A仅由隔离实验驱动器提供。

- [ ] 写测试：关闭融合退化为原影像通路；错误A不硬置零U；默认stop-gradient不向A概率回传；联合反传作为显式消融。
```python
def test_disabled_fusion_is_identity(fusion, features, anatomy_prob, uncertainty):
    output = fusion(features, anatomy_prob, uncertainty, enabled=False)
    assert torch.equal(output, features)
```
- [ ] 运行 `$PY -m pytest tests/v41/test_anatomy_fusion.py -q` 确认红测。
- [ ] 实现 `F_u + gate * projection(A_prob)`，门控以零残差初始化；保留完整影像通路。先分离梯度，再比较联合反传；不能将硬A裁剪作为唯一U输入。
- [ ] 重跑；加入容量匹配影像支路、冻结A/联合A、空间扰动A和跨病例置换A配置，扰动来源只能来自相同允许角色。GT-A标`oracle_only`且不能注册生产checkpoint。
- [ ] 提交模型和配置，记录干预A后U改变、A纠错/引错计数；所有真实效应等待实际实验。

**验收：** 不只比较“加了一个头”；同输入和相近预算的对照齐全。置换A引起下降不能单独当作因果证明。

### T16 — 病灶实例、R_geo/R_clin及全体积推理

**依赖：** T08、T13–T15。**Files:** Create：`anatobind/v41/models/relation.py`、`tests/v41/test_relation_pipeline.py`；Modify：`engine/inference.py`、`evaluation/{instances,relations,joint}.py`。

**Interfaces:** `extract_instances(u_prob: np.ndarray, geometry: dict, policy: dict) -> list[dict]`；`geometric_binding(instances: list[dict], anatomy_prob: np.ndarray, policy: dict) -> list[dict]`；`RelationHead.forward(roi_features: Tensor, anatomy_features: Tensor, geometry_features: Tensor) -> Tensor`。

- [ ] 写测试：一个病灶跨两组织输出集合；无交叠不自动当临床未累及；R_geo参照拒绝使用模型A/U；R_clin须独立来源；候选集合哈希不一致时拒绝公平比较。
```python
def test_cross_tissue_binding_is_multilabel(two_tissue_case, geometric_policy):
    result = geometric_binding(two_tissue_case["instances"], two_tissue_case["a_prob"], geometric_policy)
    assert set(result[0]["hosts"]) == {"femoral_cartilage", "tibial_cartilage"}
```
- [ ] 运行 `$PY -m pytest tests/v41/test_relation_pipeline.py -q` 确认红测。
- [ ] 实现按批准物理规则的实例提取和几何R；交叠与nearest是不同语义，只有显式配置才能启用后者。多标签关系分类器只用预测A、预测U和影像/几何特征，不读取真实tissue_id。
- [ ] 重跑；固定人工实例或固定预测候选进行class-only/class+position/image-only/geometry-only/learned比较，人工类别只放在标明oracle的诊断组；再运行开发集端到端，包括漏检、合并、分裂和重复FP。
- [ ] 提交实现，保存候选版本和匿名比较。学习型R无增益时保留几何R；不改变R语义来制造提升。

**验收：** 全体积输出可恢复原观察网格；可见残存组织与病灶范围分别输出。临床R缺标就明确缺口，不用几何自证填充。

### T17 — H1实验矩阵、开发比较与G3_READY

**依赖：** T14–T16。**Files:** Create：`anatobind/v41/experiments/{__init__,h1}.py`、`tests/v41/test_h1_protocol.py`；Modify：`configs/v41/h1.yaml`、`cli.py` 增加 `run-matrix`、`evaluate-dev`、`freeze-protocol`。

**Interfaces:** `build_h1_matrix(config: dict) -> list[dict]`；复用T11的 `freeze_protocol(config: dict, dev_summary: dict, approvals: dict, out: Path) -> dict`。矩阵绑定训练数据、输入、种子、预算、候选及评价版本。

- [ ] 写测试：oracle组不能成为主结果；真值组织ID不可输入非oracle R；固定候选必须相同；冻结后阈值或测试角色改变使哈希失效。
```python
def test_oracle_is_never_primary(h1_config):
    experiments = build_h1_matrix(h1_config)
    assert all(not e["primary"] for e in experiments if e["oracle"])
```
- [ ] 运行 `$PY -m pytest tests/v41/test_h1_protocol.py -q` 确认红测。
- [ ] 登记A-only/U-only/shared_au/independent_au/AnatoBind/容量匹配/扰动A/GT-A，R比较固定与端到端分层；同病种多个组织作为归属机制重点。按病灶大小、边界破坏、A质量记录纠错与引错。
- [ ] 重跑；获批后只在开发集选工作点和非oracle主要基线。冻结主要J、A保持性、FP预算、拒判策略、表面容差、阈值选择算法及多重比较范围。
- [ ] 发布 `G3_READY` 与封存的模型/配置/评价哈希；正式G3和假设结论由T24执行锁定测试后裁决，不提前解封。

**验收：** 至少一项同例联合主比较可运行；未通过预设A保持性或误报要求时如实记录，不能用平均Dice上升掩盖。没有合格R时只保留模块结果，不能生成联合主结论。

### T18 — H2现实缺标生成器与隐藏参照隔离

**依赖：** 合格ISLES或BMSR目标完整训练参照、T06/T08/T11/T12。**Files:** Create：`anatobind/v41/experiments/missingness.py`、`configs/v41/h2.yaml`、`tests/v41/test_missingness.py`；Modify：`cli.py` 增加 `prepare-missingness`。

**Interfaces:** `create_missingness_release(records: list[SampleRecord], policy: dict, visible_out: Path, restricted_out: Path) -> dict`；`visible_eligibility(review_region: np.ndarray, retained_positives: np.ndarray, completeness: str) -> np.ndarray`。生成器只能在参照管理角色运行；trainer无权访问其隐藏输出。

- [ ] 写测试：只看可见复核区域与保留阳性时，资格图不随隐藏遗漏病灶的位置变化；训练manifest不包含完整mask路径、被删实例ID或精确ignore洞。
```python
def test_hidden_lesions_cannot_change_visible_eligibility(visible_context):
    e1 = visible_eligibility(**visible_context)
    e2 = visible_eligibility(**visible_context)
    assert np.array_equal(e1, e2)
    assert "hidden_mask" not in inspect.signature(visible_eligibility).parameters
```
- [ ] 运行 `$PY -m pytest tests/v41/test_missingness.py -q`；另加两个隐藏参照不同而公开上下文完全相同的整管线对照，不能只靠上面同输入重复调用完成隔离验收。
- [ ] 实现三种机制：预先选区域完成复核；随机保留部分实例；按大小/可见性偏置遗漏。建议比例25/50/75%，实际seed和比例预注册；region选择不能暗中按隐藏病灶形状精确抠洞。
- [ ] 重跑，验证训练角色受限目录与访问日志；公开产物只包含现实标注者会提供的信息。完整参照上限和oracle-eligibility必须使用不同实验角色并明确privileged。
- [ ] 保存visible/restricted双版本哈希及匿名机制报告；提交代码，不公开隐藏逐例映射。

**验收：** 资格由真实已知信息决定，不泄漏被删病灶位置。始终保留独立抽样并复核的目标阴性支撑，不能由每个被删病灶位置反推阴性范围。

### T19 — H2公平基线与框/伪标签弱监督桥接

**依赖：** T06、T11、T18；框桥接另需SKM G0。**Files:** Create：`anatobind/v41/experiments/h2.py`、`tests/v41/test_h2_arms.py`；Modify：`losses.py`、`configs/v41/h2.yaml`。

**Interfaces:** `build_h2_arms(config: dict, visible_release: dict) -> list[dict]`；`box_presence_loss(u_logits: Tensor, boxes: list[dict], eligible: Tensor) -> Tensor`；其余复用masked损失和训练控制。

- [ ] 写测试：公平组的可见数据/辅助信息哈希相同；完整标签/精确ignore组标privileged；框内存在性损失不把框内全部体素设为正、框外不产生未经复核负梯度。
```python
def test_non_oracle_arms_receive_same_information(h2_config, visible_release):
    arms = build_h2_arms(h2_config, visible_release)
    hashes = {a["visible_information_hash"] for a in arms if not a["privileged"]}
    assert len(hashes) == 1
```
- [ ] 运行 `$PY -m pytest tests/v41/test_h2_arms.py -q` 确认红测。
- [ ] 必须实现三组：缺标当背景、同资格masked BCE/Dice、完整监督上限。适用的PU/一致性方法先登记假设和附加信息；类别先验不得由隐藏完整标签或测试集获得。资格控制若就是masked loss，不复制同一实现另命名“新方法”。
- [ ] 重跑；弱框默认只对合格保留阳性做MIL存在性目标，精确U子集不变。自动A/伪U仅在TRAIN生成并带teacher谱系；真值、伪监督、旧box-fill检测基线分别统计，默认不进入H1主比较。
- [ ] 提交实现与消融登记；真实训练获批后才运行。新增半监督方法只有在签核配方和适用前提后启用，不自动搜索大量方法。

**验收：** 不以更多标签或隐藏先验制造“资格控制”优势；弱监督带来的收益须在独立精标参照上评价，不能在框填充GT上自证。

### T20 — H2开发评价与G4_READY

**依赖：** T18、T19、T08。**Files:** Modify：`experiments/h2.py`、`evaluation/selective.py`；Create：`tests/v41/test_h2_evaluation.py`。

**Interfaces:** `summarize_h2(rows: list[dict], protocol: dict) -> dict`；复用 `freeze_protocol` 和患者级bootstrap。

- [ ] 写测试：同FP预算才能比较目标灵敏度；只有阳性病例无病灶区域时不得生成病例级阴性特异性；目标外病种无negative资格。
```python
def test_scan_specificity_requires_true_negative_scans(positive_scan_rows, protocol):
    result = summarize_h2(positive_scan_rows, protocol)
    assert result["scan_specificity"] is None
    assert result["scan_specificity_reason"] == "NO_REVIEWED_NEGATIVE_SCANS"
```
- [ ] 运行 `$PY -m pytest tests/v41/test_h2_evaluation.py -q` 确认红测。
- [ ] 对每个遗漏机制/比例报告病灶Dice、灵敏度—FP曲线、positive→negative错误率、拒判/覆盖率和分层大小；以同一完整开发参照评估，完整测试仍锁定。
- [ ] 重跑；冻结H2主要比较和阈值选择规则。接受集上的false-normal同时报告接受覆盖率；全覆盖漏检及拒判计数不得消失。
- [ ] 保存 `G4_READY`；若不优于普通masked强基线，登记“资格协议必要、算法增益未支持”，不修改遗漏机制迎合结果。

**验收：** 有效监督比例、阴性参照、误报及拒判共同披露；正式G4科学结论留到T24。

### T21 — 一个独立同例扩展及P2资产接入门

**依赖：** T17开发就绪、T07/T09/T13同类流程。**Files:** Modify：相应adapter、`configs/v41/sources.yaml`、`configs/v41/ontology.yaml`；Create：`configs/v41/external_joint.yaml`、`tests/v41/test_external_release.py`。

**Interfaces:** `validate_external_release(records: list[SampleRecord], training_lineage: dict, policy: dict) -> dict`，实现于`registry.py`。

- [ ] 写测试：与训练已知同患者/同图的外部候选被识别；T2Synth不伪装实采T2；临床R缺标不由几何补齐；外部调参集不混作盲测。
```python
def test_external_overlap_is_not_independent(external_records, training_lineage, policy):
    report = validate_external_release(external_records, training_lineage, policy)
    assert report["known_overlap_count"] > 0
    assert report["independent_test_allowed"] is False
```
- [ ] 运行 `$PY -m pytest tests/v41/test_external_release.py -q` 确认红测。
- [ ] 研究者从ISLES/BMSR选一个有可判读A/R的补标场景，沿用精标SOP；不要求低分辨率DWI完成无法可靠划界的精细皮层任务。零样本、适配后、重新训练三种实验角色分开。
- [ ] 重跑；OAI只在影像/标签实际取得、授权、空间和病例重叠核验后接入，不是硬依赖。SPIDER/AMOS-MRI/TotalSegmentator只先扩A；SibBMS检查标注覆盖；KMAR/k-space研究退化；HCP/OASIS SSL均为单独批准的扩展。
- [ ] 发布独立扩展协议和患者/来源覆盖，提交实现。资源不满足则记录明确缺口，不能改称已有来源内验证为外部验证。

**验收：** 至少一个扩展场景具有同例独立参照，或明确报告未完成。多个部位各自训练的模型不能称同一通用checkpoint。

### T22 — 同checkpoint双模式与严格二维信息边界

**依赖：** T17开发就绪、T10/T11；本任务是完整目标的必需后续项，不是可静默删除的可选项。**Files:** Create：`anatobind/v41/models/observation_adapter.py`、`configs/v41/h3.yaml`、`tests/v41/test_h3_observation_boundary.py`；Modify：`observations.py`、`engine/{trainer,inference}.py`。

**Interfaces:** `build_single_slice_input(image_2d: np.ndarray, geometry_2d: dict, policy: dict) -> dict`；`ObservationAdaptiveModel.forward(image: Tensor, valid_fov: Tensor, mode: str) -> dict[str, Tensor]`，`mode`仅`2d/3d`。

- [ ] 写测试：单切片构造器不接受volume或z索引；同slice内容在不同邻层体积中构造出的合法2D输入完全相同；不读取3D A/整卷统计/同病例特征缓存；两模式加载同一个checkpoint哈希。
```python
def test_single_slice_information_boundary(slice_2d, slice_geometry, policy):
    sample = build_single_slice_input(slice_2d, slice_geometry, policy)
    assert sample["observed_slices"] == 1
    assert sample["used_volume_statistics"] is False
    assert not {"volume", "neighbor_slices", "anatomy_3d"} & sample.keys()
```
- [ ] 运行 `$PY -m pytest tests/v41/test_h3_observation_boundary.py -q` 确认新增边界测试红测；另以禁止邻层/缓存访问的mock文件系统测试，不只依赖输出自报布尔字段。
- [ ] 优先审计并包装已有双模式组件。若不满足v4.1，则实现共享平面编码/解码＋带有效性mask的深度聚合；2D仅一有效slice，3D完整体积滑窗。不强行共享不兼容的A32/A6头或加载不匹配权重。
- [ ] 重跑；训练每一模式都产生A/U/R监督曝光记录。2D归一化只用当前slice或预冻结训练集统计；人工抽片服务可读取源体积，但不得把邻层信息带入2D输入。重复/补零切片不能计额外观测。
- [ ] 提交代码；登记双模式初始化、参数共享范围与两模式预算，获批后训练。若复用query结构，审计真实实例容量；不丢弃超过query数的GT以通过运行。

**验收：** 一份权重文件两种mode，分别可输出A/U/R；2D只声称当前slice范围。缺失DICOM患者几何时标`geometry=assumed`且不输出患者坐标/患者左右；无真实物理间距不报告伪mm指标。

### T23 — H3观察适配、原生二维与G5_READY

**依赖：** T22、T08；原生二维像素分割还依赖其独立精标子集。**Files:** Create：`anatobind/v41/experiments/h3.py`、`tests/v41/test_h3_evaluation.py`；Modify：`configs/v41/h3.yaml`。

**Interfaces:** `build_h3_matrix(config: dict) -> list[dict]`；`summarize_mode_coverage(rows: list[dict]) -> dict`。

- [ ] 写测试：输出2mode×3任务六项独立状态；3D分数不能填补2D缺项；框标注不能通过2D像素分割门；模式输入缺额外序列时不能偷偷读取完整3D协议。
```python
def test_missing_2d_u_is_reported(three_d_only_rows):
    report = summarize_mode_coverage(three_d_only_rows)
    assert report["2d"]["U"]["status"] == "NOT_EVALUATED"
    assert report["complete_dual_mode"] is False
```
- [ ] 运行 `$PY -m pytest tests/v41/test_h3_evaluation.py -q` 确认红测。
- [ ] 在同来源、同输入序列和同患者划分比较独立2D、独立3D与共享checkpoint双模式；切片抽样规则先定，不能只取病灶质心层。三维mask切片的数学评价与单slice可见性评审分别记录。
- [ ] 重跑；原生二维fastMRI/SPIDER测试单列来源/协议变化。fastMRI现有框只评定位；补人工U mask和合格A/R后再验收二维联合能力，不沿用三切片R缓存当单slice参照。
- [ ] 冻结双模式六项指标与覆盖率，保存 `G5_READY`。共享模型未达到某一模式门时保留专用模型作对照，不把两份checkpoint打包冒充同一个模型。

**验收：** 同来源观察适配与跨来源泛化分开；未完成H3时可交付三维里程碑，但整个2D/3D项目状态仍是未完成。

### T24 — 受限测试解封、统计汇总与最终证据包

**依赖：** 所需场景的T17/T20/T21/T23已冻结及人工测试授权。未完成阶段不阻塞已批准里程碑交付，但必须在最终范围表标明。**Files:** Create：`tests/v41/test_evidence_release.py`、`reports/v41/{EVIDENCE_INDEX.md,CLAIMS.md}`；Modify：`cli.py` 增加 `evaluate-locked`、`release-evidence`，`registry.py`增加证据发布逻辑。

**Interfaces:** `release_evidence(protocol: dict, run_records: list[dict], results: dict, out: Path) -> dict`。

- [ ] 写测试：只有日志没有完成证书不能算完成；只有代码测试没有实际结果不能算科学通过；oracle/伪GT被用于主结果必须拒绝；患者ID不能进入公共证据。
```python
def test_engineering_pass_is_not_scientific_success(protocol, engineering_only_run, tmp_path):
    report = release_evidence(protocol, [engineering_only_run], results={}, out=tmp_path)
    assert report["scientific_conclusion"] == "NOT_EVALUATED"
    assert report["project_complete"] is False
```
- [ ] 运行 `$PY -m pytest tests/v41/test_evidence_release.py -q` 确认红测。
- [ ] 先冻结所有本轮主张的模型、阈值、比较臂、种子和分析，再由授权评价者执行锁定队列的一次预登记评价批次；多个预注册方法可在同批比较。测试期间训练者不接触逐例标签/结果以继续调参。
- [ ] 重跑软件测试；实际评价后按患者配对汇总，跨种子变异另报；联合指标同时呈现A/U/R分项、FP、拒判及参照覆盖率。空/稀有类别保留n，不凑宏均值成功。
- [ ] 发布工程/科学双裁决、运行证据、失败记录、协议偏离、同例证明、二维/三维六项状态与所有未完成项；提交脱敏报告，保持医学产物私有。

**验收：** 原始失败不删；假设不支持可以完成一次实验但不是“目标已实现”。测试解封后新增H2/H3调优须使用新的独立测试或明确标为已暴露的探索性分析。评价代码确有bug时可版本化重算，但保留旧输出、修复原因与接触历史，不借机改变方法或阈值。

---

## 4. 实验登记、医学审批与评价口径

### 4.1 不能由Agent代填的签核字段

|字段|负责决定的人|最迟冻结点|缺失时处理|
|---|---|---|---|
|`target_abnormality_definition`、目标阴性检查范围|临床负责人|G1；脑U按发布协议先核验|相关训练/评价BLOCKED|
|A/U边界可判读性、额外序列可见性|标注负责人|G1|不将该标签用于主要监督|
|实例拆并、R语义、多组织判断|临床＋研究负责人|G1|关系主结果BLOCKED|
|表面容差、J范围门、误报预算、A非劣界|临床＋统计/研究负责人|T17冻结前|不能生产盲测成绩|
|正式样本规模和盲测独立性|研究负责人/数据管理员|T13|只做开发研究|
|GPU、max_steps/max_wall_seconds、预算|用户/资源负责人|每次正式register-run前|仅dry-run|
|外部授权与下载|用户/数据管理员|T21外部资产导入前|不下载、不注册正式数据|
|测试解封与对外发布|研究负责人|T24|保留READY，不运行锁定评价|

这些不是随意留空的开发细节。模板中保留null，软件必须拒绝未经签核的正式执行；专家签核需要实际人名/角色、时间、版本与依据，不能由Agent伪造批准。

### 4.2 最小H1矩阵与预算控制

|ID|输入与监督|目的|结果角色|
|---|---|---|---|
|H1-A|图像→A|解剖强基线|正式分项|
|H1-U|图像→U|异常强基线|正式分项/候选主对照|
|H1-MT|共享编码器A+U，无显式融合|普通多任务收益|候选主对照|
|H1-SEP|独立A/U＋几何R|简单系统是否足够|候选主对照|
|H1-AB|预测A软引导U＋预注册R|主要方法|正式|
|H1-CAP|匹配参数的影像支路|排除增参收益|机制|
|H1-PERT|扰动/置换预测A|解剖依赖与脆弱性|机制，不独立等同因果证明|
|H1-ORACLE|GT-A辅助|上限诊断|非部署、非主要|

正式比较建议3个预注册种子；资源不足改为更小矩阵须在看测试结果前签核，主对照不能事后选择。不能把旧父权重接触过测试影像的模型与从未见测试的模型混称严格公平；初始化谱系作为比较维度说明。

### 4.3 正常、拒判、联合终点的固定分母

- `false_normal_on_accepted`：在所有“参照合格且被接受为未见目标异常”的体素/区域中，实际存在目标异常的比例。分母是已接受的正常判定，不是所有阳性或全图体素；另报 `accepted_positive_fnr`，其分母是参照合格且被接受的真实异常体素/区域，分子是其中被判未见异常者。两者分母为零时均为NA，并报告接受覆盖率和总拒判数；全拒判不是零风险成功。
- `target_sensitivity_all`：合格目标病灶中被检出者占比；漏检与拒判不可从真实病灶总数移除。
- `specificity_scan`：仅在经协议复核的目标阴性扫描中计算；没有此类扫描则NA，并说明缺口。
- `j_all_coverage`：预先可评价真实病灶中，检出、范围合格、关系正确且未拒判的比例。关系缺参照的病灶不参与该J，但必须报被排除数、理由、来源和完整U结果。
- 患者等权主J与病灶池化J并列；多扫描患者先按冻结规则形成患者汇总。统计时成组抽样患者的全部扫描/病灶/种子结果。

### 4.4 测试集不会被阶段性开发“消耗掉”

T17/T20/T23默认只做开发比较与READY登记，正式科学结论由T24解封后给出。确需先交付H1独立论文结果时，可按批准范围先解封H1队列；其后不能再用同一队列指导H2/H3选择并仍称独立测试。后续使用新锁定队列，或如实注明历史暴露。

### 4.5 训练效率与信息保真

先P0两条基线，再同例联合，不先扫描/复制TB级k-space、不先全库SSL。精标薄软骨保留原始几何；patch和混合精度由实测锁定。降采样比较同时报告薄组织/小病灶保留情况；不能为省显存把消失目标当负样本。

临床补序列能帮助标注，但模型不能“看不到却被要求必然预测”；这类标签必须另记目标输入可见性。视野外、padding、缺失解剖和真正背景分别处理。

## 5. CLI合同与建议的第一轮交接

以下命令仅在对应任务实现并测试后可用。运行时 `PY`、代码工作目录和私有root必须来自已验证环境，不假定某个历史conda环境或固定GPU仍可用。CLI敏感错误需脱敏。

```bash
# 对应T02：只读资产清点；源码和私有路径经T00/T01确认。
"$PY" -m anatobind.v41.cli inventory \
  --config configs/v41/sources.yaml --out "$PRIVATE_ROOT/audits/inventory_v1"

# 对应T03–T07：命令按各自子任务执行，不应变成一键绕过审批。
"$PY" -m anatobind.v41.cli audit \
  --scope isles_u --out "$PRIVATE_ROOT/audits/isles_u_v1"
"$PY" -m anatobind.v41.cli audit \
  --scope skm_a --out "$PRIVATE_ROOT/audits/skm_a_v1"

# 对应T11：只验证登记，不启动GPU；未签核预算预期BLOCKED。
"$PY" -m anatobind.v41.cli register-run \
  --config configs/v41/isles_u.yaml \
  --out "$PRIVATE_ROOT/runs/isles_u_dev_v1" --dry-run

# 所有新软件任务都能先在合成输入上验证，不接触锁定测试。
"$PY" -m pytest tests/v41/ -q
```

正式子命令合同：

|命令|必须消费的批准输入|不得自行做的事|
|---|---|---|
|`audit`|scope、source配置、私有输出根|把未知completeness判完整|
|`annotation-pack`|冻结试点名单、标注协议|暴露模型预测诱导读者|
|`register-run`|manifest/gate/预算/初始化/角色哈希|直接开训或自动补GPU配置|
|`train`|已批准registration文件|修改split、延长预算、静默换数据|
|`infer`|checkpoint与观察协议|读取GT裁剪/组织类别|
|`evaluate-dev`|冻结开发角色、合格参照|读locked_test|
|`freeze-protocol`|开发摘要、专家阈值、比较矩阵|依据测试成绩选阈值|
|`evaluate-locked`|协议锁、评价授权、受限参考角色|回传逐例信息给调参循环|
|`release-evidence`|实际产物哈希与运行证书|把计划值当实测成绩|

### 可直接交给Coding Agent的启动指令

```text
阅读plan.md、v4.1 Proposal、监督契约，以及仓库AGENTS/PROJECT_CONTEXT/STATUS当前入口。
先执行T00接手审计，随后仅推进T01–T08代码、合成测试和经授权的只读数据审计。
必须区分旧工程资产、框/伪标签与专家真值；优先包装可验证组件，不重复已完成训练。
新增私有产物默认放经验证的 $HOME/outputs/anatobind_v41，禁止覆盖原始数据及旧实验。
每项记录实际测试/审计证据，未运行不得勾选完成；敏感路径和病例记录不出现在公共日志。
遇医学语义/标签可见性/预算/测试解封缺审批时写BLOCKED及所需证据，继续独立可做任务。
本轮不得启动正式GPU训练、外部数据下载、锁定测试、Git推送或改变历史验收口径。
交付reuse_decisions、schema/路径检查、P0审计报告、监督与评价单元测试和G0逐来源裁决。
```

该指令仅供用户在后续授权执行时使用，本次生成文件不触发其动作。

## 6. 每个里程碑的产物与完成定义

|里程碑|必须存在的产物|不等价的替代品|
|---|---|---|
|M0 数据可信|八类审计报告、manifest锁、患者/暴露谱系、逐来源G0|只列目录或看到mask文件|
|M1 标注可用|试点protocol、两读者记录、仲裁、目标阴性范围、G1|框填充、模型预测、自报“已标注”|
|M2 基线可运行|登记、实际训练/恢复证据、原网格开发预测与指标、G2|历史测试数、loss下降或代码存在|
|M3 同例联合|固定候选及端到端A/U/R，独立参照，H1统计与误报|不同数据集上各做一项|
|M4 部分监督|同信息缺标实验、真实U测试、FP/false-normal与覆盖率|精确被删mask生成ignore的oracle结果|
|M5 双模式/扩展|同checkpoint 2D/3D分别A/U/R，原生二维精标与外部范围说明|三切片模型、两个checkpoint、仅3D分数|
|M6 可审计交付|版本化代码/协议、匿名汇总、私有证据索引、失败和限制清单|只给最好分数或模型截图|

建议每任务状态限定 `TODO / IN_PROGRESS / BLOCKED / PASS / FAIL / DEFERRED`。PASS仅用于该任务明确验收；DEFERRED不代表完成。科研假设另使用SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE，不用软件PASS替代。

最终证据包公共部分至少包含：代码commit、环境锁、schema与本体版本、匿名样本/患者支持数、实验和评价协议哈希、所用参照层级、患者级区间、随机种子变异、FP/拒判/覆盖率、协议偏离和完整二维/三维状态表。私有部分保存实际manifest、读者版本、逐例预测、checkpoint及完整运行日志，只有权限合格的审核者访问。

## 7. Proposal覆盖映射

|v4.1要求|对应任务|最终可声明的范围|
|---|---|---|
|原始/框填充/自动/未知来源隔离|T01/T02/T07/T06|监督等级透明|
|SKM A6与tissue_id区分|T04/T05|显式语义与坐标|
|PDGM/BMSR/SibBMS身份及跨源重叠|T03/T21|来源内/外独立性按证据声明|
|可观察A、真实U、正常范围、unknown三分|T06/T09/T13/T14|目标范围内的状态，不是全病种健康|
|R_geo与R_clin不混同|T04/T13/T16|几何或临床关系分开|
|缺损表面与反事实体积不混同|T04/T09/T13|一期表面/等级，D体积仍未完成|
|H1强对照、固定候选及端到端|T14–T17/T24|同例解剖引导作用及错误传播|
|H2无遗漏位置泄漏、公平信息和FP|T18–T20/T24|现实部分监督，不虚构算法创新|
|同checkpoint独立2D/3D与原生二维|T22/T23/T24|两模式各自合格能力|
|OAI不作前置、SSL和k-space后置|T12/T21|按实际接入范围声明|
|患者级统计、拒判和联合评价|T08/T17/T20/T23/T24|估计效应及不确定性|
|算力与审批、数据不覆盖、Agent边界|T00/T01/T11/T24|有限、可恢复、可审计执行|

**项目完成不能只检查代码任务勾选数。** 至少一个同例场景的A/U/R具有独立参照，并完成相应实验；完整双模式目标还要求T22–T24六项验收。缺损体积、全身/全病种、未做的外部泛化必须继续明确标为未验证。

## 8. 文档与证据来源

- **E1**：`AnatoBind_MRI_v4.1_Feasibility_Proposal.md`，本计划的任务、数据角色和证据要求来源。
- **E2**：`AnatoBind_MRI_v4.1_Supervision_Data_Contract.md`，字段、资格、临床SOP与空间要求来源。
- **E3**：`FM_data_inventory_20261101.md`，服务器目录盘点；不代表全文件解码/患者去重完成。
- **E4**：仓库 `AGENTS.md`，本次读取blob `8a621176f7f0463a09815f4549c2d673a2f8a439`；私有输出、进展回顾和资源约束。
- **E5**：仓库 `PROJECT_CONTEXT.md`，本次读取blob `e3ebb8aea669f14ddf713083e252c7dc57b9c633`，并交叉读取 `STATUS.md` / `CLAUDE.md` 顶部；旧工程状态只按该快照引用。
- **E6**：仓库 `anatobind/aur/labels.py`，本次读取blob `303c510eb507ca0ebf35520e3cdf7cf13eeded84`；类别解码接口与数值容差。

本计划的临床依据沿用E1中的发布方来源，不重新声称核验全部外部数据。执行前若文件/版本变化，先做差异审计，不能无记录覆盖本计划的证据基础。

**建议的第一交付点：M0＋可执行合成测试＋精标试点包。不是直接交付“大一统模型”，也不是用缺少参照的代理分数宣称研究目标已经达成。**
