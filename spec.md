# AnatoBind-MRI v4.1 总规格（spec.md）

**日期：** 2026-11-02
**性质：** 项目唯一的规格依据，写的是项目现在的口径。新的 `plan.md` 依据本文撰写，并用 superpowers executing-plans 执行。
**读法：** 正文写目标、定义、规则和验收门；附录放可以逐项核对的清单。正文用「见附录 X」引用附录。
**路径约定：** 本文不出现服务器上的实际路径。`$RAW_ROOT` 指原始数据根（只读），`$PRIVATE_ROOT` 指私有产物根（仓库外，权限 700）。实际取值只保存在私有配置里。`$PY` 指经过核验的 nvgen conda 环境解释器。

---

## 1. 目标与原因

### 1.1 要做成什么

在同一份 MRI 观察上，给出并独立验证以下四项：

- **A**：体素级的可观察解剖身份。
- **U**：目标异常的可观察范围，可以是多标签。
- **R**：病灶与解剖结构之间预先定义的关系。
- **目标范围内未见异常**：一个有明确类别范围和观察条件的状态。

最终要求同一个 checkpoint 既能接受单张二维切片，也能接受三维体积。两种模式各自分别验收 A、U、R。

### 1.2 为什么这样做

服务器上虽然有大量分割文件，但其中混着几类东西：框填充的伪 mask、自动生成的解剖、几何推出来的关系，以及来源不明的派生标签。直接拿来混训，得到的分数证明不了任何临床能力。本项目的核心贡献是：

> 在异构、不完整、观察条件受限的 MRI 监督下，建立可以验证的同例解剖—异常联合感知，并说明预测出的解剖在什么情况下改善、在什么情况下损害异常边界和组织归属。

### 1.3 目标

1. 至少一个场景在**同一批测试病例**上有独立的合格 A、U、R 参照，并完成联合评价。
2. 每一级监督都能追溯来源，不跨级宣称。
3. 在现实的缺标条件下控制误报和 false-normal。
4. 用同一个 checkpoint 完成二维、三维两种模式，各自独立验收。

### 1.4 非目标

- 全身、全病种的基础模型。
- 一期就做缺损体积 D（反事实的缺失组织体积）。
- 一期就做 k-space 重建、SSL 预训练或外部数据接入。
- 承诺性能一定提升。结论按预注册的终点和置信区间来定。

### 1.5 路线选择

| 路线 | 好处 | 主要缺口 | 结论 |
|---|---|---|---|
| 立即全数据联合训练 | 表面规模大 | mask 语义不一致、框填充标签、重复患者、伪监督混杂 | 不采用 |
| 等 OAI 等外部膝关节精标数据到位再启动 | 可能更快得到同例骨与病变监督 | 本地没有资产，授权、配对、可见性和病例重叠都待核验 | 只作可选增益 |
| 脑部 U 先行，SKM 定向补标建立联合主线 | 用上已有的专家病灶 mask，同时保留联合目标 | 需要专家精标和冻结的评价子集 | **采用** |

脑部 U 实验只是模块证据。只有在同一批测试病例上同时验证 A、真实 U 和合格的 R，才构成联合感知证据。

### 1.6 反目标（技术上能跑通，但仍算失败）

- 在三个数据集上各做成一项，然后相加，称为联合能力。
- 把框填充 mask、SynthSeg 伪 A 或几何 R 当作专家真值去评价。
- 把未复核的区域当作背景或"正常"来训练。
- 事后改阈值、删难例、换测试集，以求达标。
- 用三维的分数顶替二维的验收。
- 训练之后修改冻结的划分或协议，却不留记录。

---

## 2. 问题定义

### 2.1 输出

给定输入 X、序列集合 S、物理几何 G 和可评价视野 Ω，模型输出：

| 符号 | 含义 |
|---|---|
| A_k(v) | 可观察的解剖身份。同一本体内互斥类别可以用 softmax；层级或重叠结构用 sigmoid。 |
| U_c(v) | 第 c 类目标异常的范围，多标签 sigmoid。 |
| M_j | 按冻结规则从 U 中提取的病灶实例。 |
| R_jk | 病灶 j 与组织 k 的关系，可以是多标签集合。 |
| C(v)、Q_jk | 校准后的接受或拒判信息及原因。 |

A 和 U 分头预测。不使用"正常组织 A / 异常组织 A / ……"这种笛卡尔积单一标签表。`A_prob × U_prob` 只能当作交叠特征，不能称为校准过的联合概率。

### 2.2 三种 unknown 必须分开

| 状态 | 含义 | 处理 |
|---|---|---|
| annotation_unknown | 该位置或类别没有人复核过 | 训练时 ignore（梯度为 0），不能当作图像语义类别 |
| not_observable | 当前输入或视野不足以评价 | 记录可见性，不要求模型推断看不到的组织 |
| predictive_uncertain | 模型对可观察的目标不确定 | 在验证集上校准拒判，报告风险—覆盖率 |

推理时没有复核地图可用。拒判只能来自输入和模型输出。

### 2.3 "目标范围内未见异常"的操作定义

对目标类别集合 C，要同时满足两条：该位置对 C 中所有类别都完成了复核，并且输入可以评价。满足后，如果 C 中每一类都没有异常，才能称为"C 范围内未见异常"。

- 只复核了一个病种，就只能在这个病种的范围内下结论。
- 不能把多个来源各自的单病种背景拼成一个全病种阴性集。
- 这个状态只是模型预测，误判风险要用独立参照来验证。

现行的目标异常定义采用 tad_v1，见附录 B.3。

### 2.4 R 分成两类

- **R_geo**（几何关系）：谓词为 inside / overlaps / abuts / near / disjoint。预测值由预测出的 A 和 U 计算；参照值只能由独立的合格人工 A 和 U 计算。这不算自证，因为两边的分割来源不同。`near` 的距离 `near_mm` 冻结之前，R_geo 不能正式运行。
- **R_clin**（临床关系）：谓词为 involves / originates_from / compresses，必须有独立的专家判读。邻接、压迫和浸润不能合并成一个"受累"。
- 一个病灶跨两个组织时，输出组织集合。没有交叠，也不能自动推出"临床未累及"。
- 第一期 R 只选一个清晰的语义。nearest 和 overlap 是不同的语义，只有显式配置时才启用 nearest。

### 2.5 缺损

现存组织的异常 U、缺损表面足迹或等级（`surface_defect`）、缺损体积 D（`defect_volume`）是三个不同的任务。一期只做前两项。D 推迟到有明确的期望组织包络参照之后再做，任何时候都不能伪造体积真值。

---

## 3. 数据与来源角色

完整的来源表见附录 C。下面是现行的硬性口径。

### 3.1 P0：ISLES_ltr（ISLES-2022，250 例）

- **模型输入只用 DWI 和 ADC。** FLAIR 不是模型输入，原因是 250 例中 FLAIR 全部不在 mask 网格上。
- **mask 已对照 ISLES 2022 论文核实**（Hernandez Petzsche 等，Sci Data 2022，doi:10.1038/s41597-022-01875-5）。流程是 U-Net 预分割，受训标注员修改，再由主治神经放射科医生审批。只标急性到亚急性缺血性梗死。登记口径为 `expert_voxel`、`target_complete`、`visible`，类别是 `ischemic_infarct`。中心 2 的流程论文只在技术验证一节做了概述，这一点记在 `visibility_evidence` 里。
- 3 例全零 mask 就是论文里的 3 例无梗死病例，算真阴性扫描。
- 47 例 mask 把 1 存成了 `0.9999999997671694`。`discrete_labels` 在 1e-5 容差内取整，处理是正确的。
- 在本项目中的角色：U 强基线、H2 缺标实验的完整参照、脑部联合扩展的候选场景。它本身不提供 A 或 R 参照。

### 3.2 P0：SKM-TEA_ltr（155 例 qDESS H5）

- **A 参照只用 H5 seg。** 通道 i 对应 A6 标签 i+1。dicom 轨道的 NIfTI 在前两轴转置（`transpose_xy`）后与 H5 seg 一致（12/12 例，6 类 Dice 都是 1.0）。raw 轨道 mask 和 H5 seg 在体素上不一致，不登记。
- **坐标单位按 mm 处理。** NIfTI 头里写的是 unknown，研究者已确认为 mm（见 `sources.yaml` 的 `xyz_units_policy: mm`）。
- H5 没有 affine，所以几何检查只比较 dicom 轨道和 H5 seg 网格的形状，另有通道核验作为补充证据。
- **U 目前只有框。** v1.0.0 标注里共 476 个三维框：470 个登记为 `weak_localization`、`positive_only`，只用于定位；6 个 confidence 为 0，被拒。体素级 U 参照还不存在，要等精标（§7）。
- 患者分组用标注里的 subject_id（`skmtea:subject:<id>`）。
- 模型输入为 echo1 + echo2（E1/E2）。旧的单回波和降采样导出只作对照。
- 在本项目中的角色：A 强基线，以及同例 A/U/R 联合主线（精标之后）。

### 3.3 P1、P2 与可选来源

- UCSF-PDGM、UCSF-BMSR：作为 U 扩展。标签值域、随访映射和 mask 版本都还没核验，不能凭整数猜。
- SibBMS：MS 组和 Norm 组共用 `sub-###` 编号，`patient_uid` 必须带上组别前缀。
- SPIDER、TotalSegmentator MRI、AMOS MRI 子集：只用于扩展 A。
- fastMRI：框弱监督，以及原生二维观察。
- HCP、OASIS：可选的预训练或伪 A。
- KMAR：退化鲁棒性研究。
- SKM-TEA k-space：P2，一期不解码。
- OAI 派生标签：可选的外部同例数据。必须在授权、配对、空间检查和病例重叠核验都完成后才能进入 manifest，不是前置依赖。
- 空目录和 `derived/` 副本不计入样本规模。`derived/` 的旧导出一律只读；Dataset902 的框填充永远不能当作 U 参照。

### 3.4 遗留曝光与核验

旧仓库 checkpoint 的训练数据范围还没有核验，所以目前所有 P0 记录都带有一条 `legacy_foundation_model_unverified` 曝光记录。有曝光记录的病例组不能进入 `locked_test`（`splits.py` 的现行规则）。

**冻结划分之前必须先核验遗留曝光：**

1. 证据来源：旧仓库（只读）的私有运行登记、训练和验证 manifest、划分文件和完成证书，按病例确认每个旧 checkpoint 实际读过哪些影像和标签，以及各自的角色（训练、验证、测试，或仅无标签影像）。
2. 产出：私有的 `legacy_exposure_audit_<version>`，逐例给出 `verified_unexposed`、`verified_exposed`（附角色）或 `unverified` 三种结论之一，再加上匿名汇总。
3. 签核：研究者签核这份核验结论（附录 G.2）。Agent 不代填。
4. 更新：签核后，用新版本的 identity 和 manifest 更新 `exposure_history`，不覆盖旧版本。`verified_unexposed` 的病例移除遗留曝光记录；`verified_exposed` 的病例改记实际角色；仍然 `unverified` 的病例保留原记录。
5. 只有 `verified_unexposed` 的病例组可以进入 `locked_test`。如果核验后可用于锁定测试的病例不够，就停下来，由研究者决定下一步，Agent 不自行放宽规则。

用旧权重初始化的模型只能作为兼容性对照，不进入严格 inductive 的主结果。

---

## 4. 监督资格

### 4.1 资格是一个向量

对每个任务、每个类别、每个区域，分别记录六个维度：

| 维度 | 取值 |
|---|---|
| 表示粒度 | voxel / surface / bbox / instance / study |
| 来源 | manual / expert_corrected / automated / box_fill / geometry / unknown |
| 完整性 | target_complete / region_complete / positive_only / unreviewed / unknown |
| 输入可见性 | visible / limited / unobservable / unknown |
| 质量 | accepted / needs_review / rejected |
| 用途 | train / val / test / pseudo_only / excluded |

此外记录参照角色 `reference_role`（expert_voxel / weak_localization / pseudo_only / unverified）和仲裁状态 `adjudication_status`。

### 4.2 什么才算 expert_voxel

以下四条要同时满足：

1. 来源是 `manual` 或 `expert_corrected`。
2. 表示粒度是 `voxel` 或 `surface`。
3. 质量是 `accepted`。
4. 完整性是 `target_complete` 或 `region_complete`。

任何一条不满足，就不能进入专家测试角色。违反时报错，错误码为 `UNQUALIFIED_REFERENCE`。

### 4.3 资格规则

| 已知的监督 | 可以监督 | 不能监督 |
|---|---|---|
| 某一类的完整人工 mask，检查范围明确 | 该类在检查范围内的阳性和阴性 | 其他没检查的病种的阴性 |
| 局部精标，周围没复核 | 精标阳性，以及明确复核过的局部阴性 | 整卷正常 |
| 三维框 | 定位、目标存在（MIL） | 框内全部阳性、框外全部阴性 |
| 自动 A | 单独加权的伪 A 目标 | 专家级评价参照 |
| 专家 A 和专家 U 的交叠 | 有明确几何语义的 R_geo 参照 | 临床 R |
| 人工 tissue_id | 定义范围内的实例归属 | 体素边界、侧别 |
| 队列名为 Norm | 队列信息 | 体素阴性 |
| 被排除的阳性框（confidence 为 0 或被删） | ignore 或需复核 | 背景 |
| 输入质量不足 | 记为可见性受限，必要时拒判 | 判为正常 |
| completeness 为 unknown 或 unreviewed | 什么都不监督（原因码分别为 `COMPLETENESS_UNKNOWN` 和 `COMPLETENESS_UNREVIEWED`） | — |

`inspected=true` 只说明至少有人看过，不代表完整阴性。

### 4.4 损失

```
L = Σ_t λ_t · [Σ E·w·ℓ_t] / [Σ E·w + ε]
```

- 没有资格的体素梯度为 0。masked Dice 的分子和分母都要排除这些体素。
- 任务、来源分组（人工 vs 伪标签）分别归一化。全部 ignore 时返回有限的 0，并且不稀释其他任务的均值。
- `build_eligibility` 默认 `roles=("qualified",)`；伪标签资格用 `roles=("pseudo",)` 单独获取。
- 资格图只用于训练损失，不能送进模型。
- 每个病种都至少要保留来源内已复核的阴性区域。只有正监督，就约束不了误报。

---

## 5. 本体与空间

### 5.1 本体

完整映射见附录 B，配置以 `configs/v41/ontology.yaml` 为准。

- 整数本身没有语义。每一种源编码都要命名、版本化、显式映射。
- SKM A6 本体（`skm_a6_v1`）和 SKM 标注 tissue_id 本体（`skm_tissue_table_v1`）是两个不同的命名空间。ACL 和 PCL 不在 A6 范围内，记为 `out_of_anatomy_scope`，不能映射成背景。
- 旧的 SynthSeg A32 只作遗留用途，不和 A6 合并。
- `no_host`、`unknown`、`out_of_scope` 是三个不同的状态，不能合并。

### 5.2 空间

- 每对正式的图像和标签，都要记录 shape、spacing、affine、qform/sform、单位、轴码和物理视野。
- shape 相同不能证明在同一空间。禁止静默修改 header 来"对齐"。
- 标签变换必须保持离散值，并记录变换链。合成的可逆仿射往返误差要求 `atol=1e-4` voxel（这只是数值测试，不是临床配准容差）。
- 有效视野、padding 和真实背景分开标记。
- 薄软骨和小病灶不能不经评估就统一降到 1 mm。重采样只生成派生版本，原始几何要保留。
- `geometry_from_nifti` 保留头里声明的单位：unknown 就保持 unknown。转换成 schema 几何时，如果单位不是 mm，就报 `UNITS_NOT_MM`，除非来源层面已经批准了单位策略（SKM 已批准为 mm）。
- 叠加图只给人审，不能替代定量检查。
- 截断成全零的旧派生标签，只能从原始源重新导出成新版本，不能就地"修复"。

---

## 6. 身份、谱系、划分与曝光

- 同一父扫描的回波、轨道、裁剪、降质和缓存，以及同一患者的随访，都归到同一个泄漏组（`leakage_group_uid`），用连通分组。
- 只接受有证据的别名边。PDGM 按官方随访映射合并；BMSR 按病例表合并；跨来源的重叠（例如 BraTS 与 PDGM/BMSR）按公开映射排除。文件哈希相同只能说明是同一张图，不能证明不同图来自不同人。
- 身份有歧义时，要么隔离，要么保守地并入同组并披露，不能随机跨划分。
- 划分角色为 `train`、`development_val`、`locked_test`。冻结时记录抽样规则、哈希和接触历史。检测到泄漏时 `freeze_splits` 报 `SPLIT_LEAKAGE`，并且只写泄漏报告。
- 登记阶段允许 `split=unassigned`，训练阶段不允许。
- **划分规则：**
  - **SKM-TEA_ltr** 沿用官方发布划分（标注 v1.0.0 的 train/val/test），映射为 `train` → `train`、`val` → `development_val`、`test` → `locked_test`。官方 test 中没有核验为 `verified_unexposed` 的病例组，不进入 `locked_test`，改记为 historical，只用于开发或内部验证。
  - **ISLES_ltr** 按比例分层随机划分，以患者组为单位。比例、分层变量和种子由研究者签核后才冻结（附录 G.2）。**撰写者建议的初值**：`train/development_val/locked_test = 0.6/0.2/0.2`；分层变量为采集中心（元数据中有时）、是否有梗死、病灶体积三分位；种子为 20261101。H2 的锁定测试也从 ISLES 的 `locked_test` 中出。
  - 冻结顺序为：遗留曝光核验并签核 → 划分规则签核 → `freeze_splits` → 从非 `locked_test` 的病例中抽取精标试点。
- 已经被旧实验使用过的病例只能称为 historical 或内部验证。没有新中心的独立来源时，只能称来源内验证，不能称外部泛化。

---

## 7. 精标与联合队列

### 7.1 SKM 精标试点

- 从非 `locked_test` 的病例（`train` 或 `development_val`）中抽 20–30 次扫描，按患者去重。分层要覆盖：典型阳性、微小或模糊的边界、多组织候选、完整复核过的目标阴性，以及质量受限的扫描。这个分层是为了探索可行性，不代表患病率。
- 输入固定为 E1/E2。如果用了额外序列来定位，要标明该标签在模型输入中是否可见。
- 读者流程：
  1. 固定异常语义。
  2. 标体素边界，不确定的地方用不确定带或 ignore。
  3. 记录已复核范围（完整、局部或仅阳性）。
  4. 独立评定 R 及其可判断性。
  5. 双读者子集独立完成，由第三人仲裁，两版原始标注都保留。
  6. 统计耗时、不可判读比例、一致性和支持数。
- 默认不给读者看模型预测或几何预测。如果使用预分割辅助，要明确记录，并保留一个盲态的独立复核子集。
- 试点病例不进入最终盲测。
- 体素精标优先做在 E1/E2 上能可靠划界的软骨病变表现。半月板病变可以单独试点。无法稳定划界的裂隙或缺损，改做表面或等级任务。
- 第一期 R 的主分析在三类软骨组织（股骨、髌骨、胫骨）之间进行。
- 实例拆分与合并的规则必须写成文字，阈值由临床语义和试点结果决定，不能由模型性能反推。
- 临床签核前，任务状态保持 `BLOCKED_EXPERT_REVIEW`。
- **读者现状：** 读者和标注工具都还没有着落。在此之前，标注包导出和导入的代码按与工具无关的格式实现：每个读者、每个病例分别交回一组 NIfTI，包括 U mask、不确定带 mask、已复核范围 mask 和必要的 A 修订 mask，外加一份 JSON 标注记录，字段同附录 A.2，再加读者 ID、耗时和 R 判读。ITK-SNAP 和 3D Slicer 都能直接读写这种格式。等工具确定后，只再加一层转换，不改变记录契约。所有需要真人标注的步骤都保持 `BLOCKED_EXPERT_REVIEW`，不能用模型输出代替。

### 7.2 扩展与锁定测试

试点通过 G1 之后，规划 60–100 次扫描的精标队列，其中 20–30 名患者作为锁定测试的起点。这些锁定测试患者必须来自 SKM-TEA 官方 test 中 `verified_unexposed` 的病例组（§3.4、§6）。实际规模根据可判读比例、每组织的病灶数和患者级区间精度重新确定，并由人签核。正式测试的影像和标签由受限角色保管，不交给训练者。

### 7.3 脑部联合补充

从 ISLES 或 BMSR 中选一个场景，补一个有限的独立 A/R 测试子集，选在目标输入中确实可以判读的粗结构。开发集和盲测集分别补标。如果最终只有一条联合主线达标，就只按那个场景报告。

---

## 8. 模型与训练

### 8.1 强基线

- 独立的 A-only 和 U-only（nnU-Net 或残差 U-Net），普通共享编码器双头（`shared_au`），独立 A+U 加后处理几何 R（`independent_au`）。
- 输入协议固定：ISLES 用 DWI/ADC 两通道；SKM 用 E1/E2，A 输出 7 通道（含背景）。PDGM 和 BMSR 是单独的实验，不自动纳入一期。
- 推理时不能输入真实病种、tissue_id、标注来源、复核图、GT 框或 GT A/U/R。
- 如果 nnU-Net 不支持同样的资格规则，就只用合格的完整子集，或者显式实现 masked 适配。禁止把 ignore 转成背景来凑兼容。

### 8.2 AnatoBind 最小结构

共享编码器，A 和 U 两个解码器并行。U 保留完整的影像通路，再加上软解剖残差融合：

```
F'_U = F_U + g(F_U, A_hat, H_A) · φ(A_hat)
```

- H_A 是解剖不确定性。门控 g 初始化为零残差，可以关闭；关闭时这一层退化为恒等映射。
- 默认对 A 概率做 stop-gradient。联合反传作为一项显式消融。
- 不用硬组织 mask 裁剪 U。
- 训练时加入可控的 A 扰动。

### 8.3 实例与 R

- 实例从 U 中按预注册规则提取。连通域是基线做法；合并和分裂都要记录。
- R 先用几何方法实现，再加入轻量关系分类器（ROI 影像特征、A 概率、距离特征），不读取真实 tissue_id。
- 如果学习型 R 超不过同条件下的 geometry-only，就保留几何 R。

### 8.4 二维/三维双模式（同一 checkpoint）

- 结构：共享平面编码和解码，加上带有效性 mask 的深度聚合。二维模式只有一个有效切片；三维模式用完整体积滑窗。
- 单切片输入不能读取邻层、全体积统计、三维 A、三维框裁剪或同病例缓存。归一化只能用当前切片的统计，或者预先冻结的训练集统计。
- 补零或复制切片不算多层观测。
- 二维只对当前切片的范围下结论。缺少 DICOM 患者几何时，标 `geometry=assumed`，不输出患者坐标和左右侧，也不报告伪 mm 指标。
- 优先审计并包装旧仓库的双模式组件。不兼容的头（A32 vs A6）不强行共享。
- **哪个模型：** H1 先在三维上完成，H1-AB 用三维骨干即可，不要求一开始就按双模式设计。双模式 checkpoint 是把 H1-AB（AnatoBind）的结构，包括 A/U 解码器、软解剖融合和 R，放进 §8.4 的双模式骨干，再在同一场景下重新训练出来的一份权重。
- **哪个场景：** 至少是 SKM 的同例联合场景。脑部场景是可选扩展。
- **验收标准：** 每种模式的 A、U、R 都在合格参照上完成评价，并且相对同场景、同输入、同划分的独立 2D 或独立 3D 模型，不劣于 `dual_mode_noninferiority_margin`（附录 G.1，由统计负责人冻结）。这两条都满足，才算这一项通过。
- 双模式的工程部分（输入边界、访问测试、合成数据训练、SKM A 和 ISLES U 上的开发运行）不需要等精标，可以提前做；正式的 A/U/R 验收要等同例精标参照。

### 8.5 训练控制

- 每次 run 都要登记（RunRegistration，见附录 A.5）。`max_steps`、`max_wall_seconds`、`checkpoint_every`、`evaluation_every`、`resource_assignment`、`initialization_lineage` 都必须是非空且已批准的值。没有批准时只能 dry-run。
- 要保存 model、optimizer、scheduler、scaler、RNG 和 sampler 的状态，以及数值后端和硬件 UUID。续跑时如果源码或数据哈希不一致，就拒绝续跑。
- 顺序：CPU 合成单 batch → 获批后 GPU 单 batch 和小样本过拟合 → 实测显存后再定 patch 和 batch。
- 遇到 NaN、容量溢出、未批准的资源或私有路径违规，立即停止并保留证据。不能无上限重试。换配方就新建 run。
- 不沿用旧的 GPU 编号、预算或训练状态。不能靠覆盖环境变量绕过资源限制。
- 种子：开发阶段用 17，正式比较建议 17/29/43（拟定值，要进入签核登记）。

### 8.6 训练顺序

1. 资产索引、坐标验证、冻结划分。
2. ISLES U 和 SKM A 基线，同时做 SKM 精标试点。
3. 在同例精标队列上做 U-only、A+U 和 AnatoBind。
4. 固定候选的 R 实验，以及端到端的 A/U/R 实验。
5. H2 缺标实验。
6. 新版本 manifest 引入框和伪标签，强、弱结果分开报告。
7. H1 在开发集上就绪（G3_READY）之后，做双模式和原生二维的 H3。H3 不依赖脑部扩展（§9.4），两者可以并行。

**读者到位之前的排序：** 读者和标注工具没有着落时，先做不依赖精标的工作，包括：遗留曝光核验、划分冻结、数据适配器和输入管线、训练控制、ISLES U 和 SKM A 基线、H2（基于 ISLES）、双模式的工程部分、标注包代码（与工具无关），以及所有 BLOCKED 项的证据清单。SKM 联合主线（精标、G1、H1）保持 `BLOCKED_EXPERT_REVIEW`。

---

## 9. 科学假设与实验

### 9.1 H1：预测出的解剖是否真正改善 U 和 R

实验矩阵见附录 F.4。规则：

- 比较时要求同数据、同输入、同划分，容量和预算相近。
- oracle 组（GT-A 辅助）永远不能成为主结果，也不能注册为生产 checkpoint。
- R 的比较先固定同一组候选（人工实例或预测候选，候选集合哈希必须一致），再做端到端实验。
- 按病灶大小、组织、边界破坏程度和 A 的质量分层，报告 A 纠正了哪些错误、引入了哪些错误。重点是同一病种在多个可能组织之间的判别。
- 置换 A 引起的下降不能单独作为因果证明。

### 9.2 H2：资格控制能否在不放大误报的前提下降低假阴性

- 在 ISLES（或 BMSR）的完整参照上模拟三种缺标机制：预选区域完成复核、随机保留部分实例、按大小或可见性偏置遗漏。比例建议 25/50/75%，种子和比例都要预注册。
- 资格只能来自真实采集流程会提供的信息。被删病灶的位置不能影响可见资格。训练 manifest 里不能出现完整 mask 路径、被删实例 ID 或精确的 ignore 洞。缺标生成器只在参照管理角色下运行，隐藏参照要做访问隔离和读访问审计（落地方式见 §12.4）。
- 必须有三组：缺标当背景、同资格的 masked BCE/Dice、完整监督上限。PU 或半监督基线需要先登记前提；类别先验不能来自隐藏标签或测试集。
- 非 oracle 组拿到的可见信息哈希必须相同。
- 框监督只对合格的保留阳性做 MIL 存在性目标，框外不产生负梯度。
- 如果新方法和普通 masked loss 形式相同，就只算数据协议，不算算法创新。

### 9.3 H3：同一 checkpoint 的二维/三维能力

- 同来源、同序列、同划分下，比较独立 2D、独立 3D 和共享双模式。
- 切片抽样规则先定好，不能只取病灶质心所在的层。
- 原生二维数据（fastMRI、SPIDER）单列来源和协议变化。fastMRI 的框只评定位，像素级验收需要补精标。
- 2 种模式 × 3 项任务，共 6 项状态，各自独立报告。三维的分数不能填补二维的缺项。
- 两个 checkpoint 打包在一起，不能冒充同一个模型。

### 9.4 扩展

至少一个扩展场景（ISLES 或 BMSR 补标，或经过核验的 OAI）要有同例独立参照，否则明确报告未完成。零样本、适配后、重新训练三种角色分开报告。T2Synth 不能当作实采 T2。外部调参集不能混作盲测。

---

## 10. 评价

指标定义和分母见附录 F，冻结字段见附录 G。

- 每项指标都要报告合格参照的覆盖率，并固定参照来源。
- 患者是独立抽样单位。同一患者的病灶、回波、随访和种子成组处理。采用患者级配对 bootstrap，默认 2000 次、种子 20261101、95% 区间（拟定值，登记前确认）。训练种子单独报告，不增加患者数 n。
- 预注册一到两个主要终点，其余分析标明为探索性。
- 推荐的主比较：在预注册的 FP 预算下，AnatoBind 相对最强的非 oracle 基线，患者级联合成功率 J 的改善；同时 A 不劣于预设容差。
- 生产评价（开发或锁定）遇到附录 G 中任何一个 null 字段时，返回 BLOCKED，不能用测试数据补值。
- 锁定测试只在 §11 的 G3/G4/G5_READY 都冻结、并获得解封授权之后，由授权评价者跑一次预登记批次。解封之后再调优，就必须使用新的独立测试，或者明确标注为已暴露的探索性分析。如果评价代码确实有 bug，可以版本化重算，但旧输出和修复原因都要保留。

---

## 11. 验收门

每一项 `GateDecision` 同时记录两种状态：`engineering_status`（PASS / BLOCKED / FAIL）和 `scientific_conclusion`（SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_EVALUATED）。训练结束只改变运行状态，不改变科学结论。没有签核就是 BLOCKED，并写出缺少什么证据，然后继续做不依赖这项签核的任务。

| 门 | 放行内容 | 硬性条件 | 不通过时 |
|---|---|---|---|
| G0-U / G0-A | 各来源的基线 | 该来源的本体、谱系、配对、空间、可读性和来源都合格，八类审计报告齐全（附录 D），目标异常定义已签核 | 只隔离不合格的样本或来源 |
| G0-joint | 同例队列准备 | 同例身份明确，A、U、R 参照资格明确 | 记录缺口，不用代理参照顶替 |
| G1 | 精标扩展、正式联合定义 | 试点可判读性、双读者流程、目标阴性范围都获专家签核 | 收窄病变表现或换场景，最终目标保留 |
| G2 | 方法比较 | 独立和共享基线、评价器可复现，数据冻结 | 修工程或修监督，不靠扩大模型掩盖问题 |
| G3_READY → G3 | H1 结论 | 同例独立参照，固定候选和端到端公平比较，主要终点冻结；正式结论在锁定测试后给出 | 允许"假设不支持"，记录效应和区间 |
| G4_READY → G4 | H2 结论 | 同信息的缺标比较，来源内阴性，误报约束 | 没有增益时只报告资格治理 |
| G5_READY → G5 | 双模式和扩展声明 | 同一 checkpoint 的 2D 和 3D 分别完成 A/U/R 验收，扩展范围透明 | 只声明实际验证过的范围 |

**现状（2026-11-02，`registry/g0_decisions_v3`）：**
- G0-U（ISLES）：BLOCKED，只缺目标异常定义的签核。
- G0-A（SKM）：BLOCKED，只缺目标异常定义的签核。
- G0-joint：BLOCKED，既缺签核，也缺同例专家 A+U 队列（ISLES 和 SKM 没有重叠，SKM 的 U 只有框）。
- 三者的科学结论都是 NOT_EVALUATED。

审计状态只允许 `pass`、`fail`、`missing` 三个值，并且必须写明 scope。拼错状态时报 `INVALID_AUDIT_STATUS`。审计按 `source_dataset` 分 scope，锁文件以 `name@scope` 为键；重复的 sample_uid 报 `DUPLICATE_SAMPLE_UID`。

---

## 12. 治理与执行边界

### 12.1 隐私与存储

- 原始数据根和 `derived/` 都只读。
- 医学数组、逐例清单、标注、权重和逐例预测，只存放在 `$PRIVATE_ROOT/{audits,registry,annotations,derived,runs,evaluation,restricted_reference}/<version-or-run-id>/` 下。目录权限 700，文件权限 600。
- 产物按版本新建，创建时做排他检查，不覆盖旧版本。旧版本标 SUPERSEDED，但保留。删除受白名单钩子限制。
- 写路径要先解析真实路径。拒绝 `..`、越界的符号链接和落进 Git 的目标路径。错误信息要脱敏，只给匿名错误码。
- Git 里只放代码、合成测试、不含真实 ID 的配置模板，以及经过审核的匿名汇总。病例 ID、真实文件名、manifest 的键、读者逐例记录和服务器路径，都不能进入 Git、终端摘要或对话。提交前用 `rg` 检查。
- 凭据类文件只统计排除数量，不读内容。

### 12.2 人工签核

签核清单见附录 G.2。Agent 永远不填 `approved_by` 和 `approved_at`，即使内容来自研究者在对话里的口述。软件遇到未签核的正式执行时必须拒绝。

### 12.3 Agent 可以做的和不能做的

- **可以自动做：** 文件可读性、格式、空间一致性、ID 映射、单元测试、合成数据实验、只读审计，以及注册 dry-run。
- **禁止：**
  - 覆盖原始数据或旧结果；
  - 把 box_fill 注册为 expert_voxel；
  - 把 unreviewed 当作背景；
  - 用 SynthSeg 派生的 R 当金标准；
  - 改冻结的划分或阈值；
  - 用测试影像做 SSL；
  - 擅自改变临床本体、正常定义或缺损定义；
  - 自动下载外部数据；
  - 启动没有预算的 GPU 任务；
  - 读取锁定测试。
- **对外操作**（建仓库、推送、改远端设置）都要先征得用户确认。

### 12.4 受限角色的落地方式

目前所有私有产物都属于同一个操作系统用户，没有单独的账号或 ACL。

- **最低要求：** 隔离靠作业白名单来实现。每个作业（训练、推理、开发评价、缺标生成、锁定评价）都在登记里声明可读的角色，只拿到对应白名单 manifest 中的路径。数据读取统一经过一个记录访问的读取层，访问日志写入私有 run 目录。`restricted_reference/` 的路径只出现在参照管理和锁定评价作业的白名单里。验收时检查访问日志，训练作业对受限路径的读取次数必须为 0。
- **能开独立账号或 ACL 时**，优先使用，作为附加的隔离层。开通由用户或管理员负责，不作为前置阻塞。
- **谁能执行：**
  - `prepare-missingness` 可以由 Agent 以参照管理角色执行，并留访问日志。
  - `evaluate-locked` 每次执行前，需要研究者在 approvals 里写入一次性的解封授权。Agent 只在授权存在时执行，执行后把逐例结果留在受限目录，不回传给调参流程。

### 12.5 工程约定

- 环境不自动升级。`jsonschema` 没有安装，schema 由仓库内的最小校验器验证（支持 type/required/enum/const/items/properties/additionalProperties），遇到未知关键字直接拒绝。
- 测试命令：`PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. $PY -m pytest tests/v41 -q -p no:cacheprovider`。
- 开发流程：先写失败测试 → 确认失败原因正确 → 写最小实现 → 跑本任务和回归测试 → 保存匿名证据 → 提交指定文件。偏离 TDD 时必须披露。
- 提交规则：禁止 `git add .`；提交信息用英文祈使句，不加 AI 署名；不自动推送或合并；医学输出不提交。
- 每个任务都写一份私有的 `task_record.json`，内容包括：任务 ID、状态、输入哈希、命令、退出码、输出哈希、实际测试结果、阻塞项、签核和下一项。没有执行过的任务不能预填"通过"。
- 任务状态只允许 TODO / IN_PROGRESS / BLOCKED / PASS / FAIL / DEFERRED。DEFERRED 不算完成。

---

## 13. 系统组成

代码限定在 `anatobind/v41/` 下，不跨仓库 import 旧代码。复用旧代码时按原文复制，并在源码中注明来源 blob 哈希。

| 模块 | 职责 |
|---|---|
| `schema.py` | SampleRecord / AnnotationRecord / Prediction / GateDecision / RunRegistration 的类型与校验 |
| `paths.py` | 私有写路径、脱敏、原子发布、task record |
| `inventory.py` | 受控遍历，分 metadata / header / full_decode 三级做完整性检查 |
| `identity.py`、`splits.py` | 别名与谱系分组、冻结划分、曝光账本 |
| `ontology.py` | 显式语义映射 |
| `labels.py`、`geometry.py`、`geometry_qc.py` | 类别解码、物理空间、变换、QC |
| `overlays.py` | 私有叠加图（只给人审） |
| `eligibility.py`、`losses.py` | 逐任务、类别、区域的资格，以及 masked 损失 |
| `audits.py`、`registry.py` | 审计汇总、manifest 发布、验收门评估、证据发布。接口：`evaluate_gate(gate_id, reports, approvals, joint_evidence=None, source=None) -> GateDecision`，其中 `reports` 是审计索引列表（这时必须给 `source`），或者扁平的状态字典（G0-joint 用） |
| `adapters/{isles,skm,pdgm,bmsr}.py` | 来源适配器：`load_record(record, allowed_roles)` |
| `annotation.py` | 标注包导出与导入、双读者记录 |
| `observations.py`、`datasets.py` | 3D 和严格 2D 的输入构造、Batch |
| `models/{core,anatomy_fusion,relation,observation_adapter}.py` | 基线、软解剖融合、关系头、双模式适配 |
| `engine/{registration,trainer,inference}.py` | 登记、训练控制与恢复、推理、`freeze_protocol` |
| `experiments/{h1,missingness,h2,h3}.py` | 实验矩阵与缺标生成 |
| `evaluation/{segmentation,instances,relations,selective,statistics,joint}.py` | 评价器 |
| `cli.py` | 统一入口 |

其他目录：`configs/v41/`（配置与模板）、`schemas/v41/`（JSON Schema）、`tests/v41/`（只用合成或匿名 fixture）、`reports/v41/`（只放匿名汇总）。

**CLI 合同**（子命令在对应功能实现并测试之后才出现）：
- 现有：inventory、verify-assets、resolve-identities、freeze-splits、geometry-qc、overlay-samples、audit、build-manifest、gate。
- 待增加：annotation-pack、annotation-import、register-run、smoke、train、infer、prepare-missingness、run-matrix、evaluate-dev、freeze-protocol、evaluate-locked、release-evidence。
- 每个子命令只消费已批准的输入，不能自行修改划分、阈值、预算或数据范围。

**张量与坐标约定：**
- 训练张量的维度顺序是 `[B,C,Z,Y,X]`；NIfTI 体素索引按 header 的 xyz，通过显式变换转换。
- `Batch` 包含 `image, valid_fov, a_target, u_target, e_a, e_u, metadata`。推理输入只含 `image`、`valid_fov`，以及必要的几何和模式信息。
- 验证和测试用全体积滑窗或预测 ROI，禁止用 GT 框裁剪。预测 ROI 漏掉的目标仍计为漏检。
- 缓存键要包含源哈希、变换、模式和本体版本。

---

## 14. 完成定义

**项目完成**要满足：至少一个同例场景的 A/U/R 有独立参照，并完成 H1；同一 checkpoint 的二维和三维共 6 项分别验收；最终证据包已发布。

最终证据包的公共部分包括：代码 commit、环境锁、schema 和本体版本、匿名支持数、协议哈希、参照层级、患者级区间、种子变异、FP/拒判/覆盖率、协议偏离，以及完整的 2D/3D 状态表。私有部分包括：manifest、读者版本、逐例预测、checkpoint 和完整日志。

缺损体积、全病种覆盖，以及没有做过的外部泛化，继续明确标为未验证。"假设不支持"也算完成了一次实验，但不等于目标已经实现。

里程碑：
- **M0 数据可信**：八类审计、manifest 锁、谱系、逐来源 G0。
- **M1 标注可用**：G1。
- **M2 基线可运行**：G2。
- **M3 同例联合**：H1。
- **M4 部分监督**：H2。
- **M5 双模式与扩展**：H3，以及 §9.4 的扩展。
- **M6 可审计交付**：证据包。

## 15. 撰写时确定的事项

以下几项是撰写本文时定下的，没有经过研究者逐项确认。读的时候请重点检查：

1. ISLES 划分的初值：0.6/0.2/0.2，分层变量为中心、是否有梗死、体积三分位，种子 20261101。冻结前仍需签核。
2. 精标试点从非 `locked_test` 的病例中抽取。
3. 标注包采用与工具无关的 NIfTI + JSON 格式。
4. 双模式 checkpoint 等于 H1-AB 的结构在双模式骨干上重新训练；验收标准为"评价完成，且不劣于独立 2D/3D 模型"；场景至少是 SKM。
5. H3 不依赖脑部扩展。
6. 受限角色以作业白名单加访问日志为最低要求，独立账号作为附加层；`evaluate-locked` 每次执行都需要一次性授权。
7. `evaluate_gate` 的现有接口作为规格。

## 16. 留给计划撰写者的事项

- 任务如何拆分、如何编号、依赖顺序怎么排（要符合 §6 的冻结顺序、§8.6 和 §11）。已有实现（附录 H）应安排为"核对任务"：对照附录 A、D、E 逐项检查，发现缺口再补，不重复做已经完成并有证据的工作。
- 具体的网络结构、patch 和 batch、优化器（实测后决定）。
- 文件内部组织、辅助函数命名，前提是不改变 §13 的模块职责和 CLI 合同。
- 每个任务的测试用例细节，前提是覆盖附录 E 中对应的不变量。

---

# 附录

## 附录 A：记录类型字段

以 `schemas/v41/*.schema.json` 为准，下面列出字段及其含义。

### A.1 SampleRecord（一行对应一个可唯一追踪的影像观察）

`schema_version, sample_uid, source_dataset, source_release, cohort, patient_uid, study_uid, session_uid, parent_scan_uid, derivative_uid, leakage_group_uid, exposure_history, split, native_acquisition_dim, observation_mode, image_paths, modality_names, modality_is_synthetic, available_modality_mask, image_geometry, original_geometry, transform_chain, geometry_status, label_value_qc, anatomy_ontology_version, abnormality_ontology_version, relation_semantics_version, annotation_records, review_regions, visibility_records, quality_records, provenance`

- 以上字段全部必填。`patient_uid` 要带来源命名空间，必要时还要带队列命名空间。
- `image_geometry` 在 `geometry_status == not_checked` 时允许为 null，下游使用前要先检查 `geometry_status`。
- `source_release` 未知时可以登记，但不能伪填。
- `review_regions`、`visibility_records`、`quality_records`、`transform_chain`、`provenance` 目前是开放子对象，待收紧（附录 H.3）。

### A.2 AnnotationRecord（嵌在 SampleRecord 中）

| 字段 | 取值 |
|---|---|
| task | A / U / R_geo / R_clin / surface_defect / localization |
| class_scope | 本次检查的目标类别 |
| representation | voxel / surface / bbox / instance / study |
| provenance_type | manual / expert_corrected / automated / box_fill / geometry / unknown |
| completeness | target_complete / region_complete / positive_only / unreviewed / unknown |
| input_visibility | visible / limited / unobservable / unknown |
| quality_status | accepted / needs_review / rejected |
| intended_role | train / val / test / pseudo_only / excluded |
| reference_role | expert_voxel / weak_localization / pseudo_only / unverified |
| adjudication_status | single_reader / dual_reader / adjudicated / consensus / not_applicable |

其余必填字段：`annotation_uid, path, label_map_version, creator_or_model, creator_version, parent_annotation_uid, review_region_path, reader_ids, protocol_hash, visibility_evidence`。缺少 `provenance_type` 时直接拒绝，不能默认。

### A.3 Prediction

`sample_uid, observation_uid, mode, geometry, a_prob, u_prob, instances, relations, acceptance, lineage`。`additionalProperties=false`，不能携带任何 GT 字段。

### A.4 GateDecision

`gate_id, scope_hash, engineering_status, scientific_conclusion, evidence_hashes, blockers, approved_by, approved_at`。

### A.5 RunRegistration

源码、配置、数据、划分、本体、监督、初始化的哈希；可读角色；预算（`max_steps, max_wall_seconds, checkpoint_every, evaluation_every`）；`resource_assignment`（物理 GPU UUID）；`initialization_lineage`；种子；数值设置；评价规则；审批。

## 附录 B：本体

### B.1 SKM A6（`skm_a6_v1`，背景为 0）

| 标签 | 名称 | host | 侧别 | H5 seg 通道 |
|---|---|---|---|---|
| 1 | patellar_cartilage | patellar_cartilage | none | 0 |
| 2 | femoral_cartilage | femoral_cartilage | none | 1 |
| 3 | tibial_cartilage_medial | tibial_cartilage | medial | 2 |
| 4 | tibial_cartilage_lateral | tibial_cartilage | lateral | 3 |
| 5 | meniscus_medial | meniscus | medial | 4 |
| 6 | meniscus_lateral | meniscus | lateral | 5 |

### B.2 SKM 标注 tissue_id（`skm_tissue_table_v1`，与 A6 是两个命名空间）

| tissue_id | 名称 | 对应 A6 | 状态 |
|---|---|---|---|
| 1 | Meniscus | 5, 6 | in_scope_laterality_unknown |
| 2 | ACL | — | out_of_anatomy_scope |
| 3 | PCL | — | out_of_anatomy_scope |
| 4 | Femoral Cartilage | 2 | in_scope |
| 5 | Patellar Cartilage | 1 | in_scope |
| 6 | Tibial Cartilage | 3, 4 | in_scope_laterality_unknown |
| -1 | none | — | no_host |

A6 可以向上合并成 tissue_id 语义，用来和旧的 R 比较；反向细分到侧别，需要新的独立标注。

### B.3 U 类别与目标异常定义 tad_v1

| 类别 | 来源 | host tissue_id | 范围说明 | 当前参照 |
|---|---|---|---|---|
| ischemic_infarct | ISLES_ltr | — | 急性到亚急性缺血性梗死；白质高信号和陈旧胶质增生不在范围内 | expert_voxel，target_complete |
| cartilage_lesion | SKM-TEA_ltr | 4, 5, 6 | — | 只有框 |
| meniscal_tear | SKM-TEA_ltr | 1 | — | 只有框 |
| ligament_tear | SKM-TEA_ltr | 2, 3 | out_of_anatomy_scope | 只有框 |
| effusion | SKM-TEA_ltr | -1 | no_host | 只有框 |

- ISLES 的正常结论范围："未见急性或亚急性梗死"，不涉及其他所见。
- SKM 的正常结论范围：只针对已复核的类别和可见输入。
- tad_v1 要等 `configs/v41/approvals.json` 签核之后才生效。
- 每个 U 类别的 `clinical_definition_signoff` 目前都是 null。正式运行 U 需要这项签核。

### B.4 关系

- `r_geo_v1`：inside / overlaps / abuts / near / disjoint，`near_mm` 为 null，状态为 engineering。
- `r_clin_v1`：involves / originates_from / compresses，`clinical_definition_signoff` 为 null。

### B.5 其他

- `synthseg_a32_v1` 只作遗留用途。
- PDGM 和 BMSR 的值域状态为 pending_verification。

## 附录 C：数据来源角色

| 来源 | 优先级 | 角色 | 现状与限制 |
|---|---|---|---|
| ISLES_ltr | P0 | U | 250 例，DWI/ADC，专家 mask，3 例真阴性 |
| SKM-TEA_ltr | P0 | A；框弱 U；精标后做联合主线 | 155 例，H5 seg 为 A 参照，476 个框 |
| SKM-TEA（k-space） | P2 | 重建退化 | TB 级，一期不解码，`.truncated` 隔离 |
| UCSF-PDGM_lh | P1 | U（四序列肿瘤） | 501 个扫描目录不等于 501 名患者，随访映射待核验 |
| UCSF-BMSR_cbb | P1 | U（增强转移瘤） | 原 seg 与 BraTS-seg 是两个版本，T2Synth 为合成通道 |
| SibBMS_ms | P1 | 队列 / 少量病灶 | MS 和 Norm 编号重叠，Norm 不等于完整阴性 |
| derived | P1 | 遗留派生（只读） | Dataset902 框填充永远不是 U 参照，SynthSeg R 只是几何或伪 R |
| SPIDER_cbb | P2 | 脊柱 A，原生二维 | 分级不等于体素边界 |
| TotalSegmentator_MRI_cbb / TotalSegmentor_MRI_cbb | P2 | 自动 A | 后者可能是重复，确认前不计数 |
| AMOS_cbb | P2 | 腹部 A（MRI 子集） | CT 和 MRI 混合 |
| fastMRI_lh_brain_knee | P2 | 框弱监督，原生二维 | 框不能验收像素分割 |
| HCP_lh_T1T2、OASIS_cbb | P2 | 可选预训练或伪 A | 不是全病种健康真值 |
| KMAR-50K_ty | P2 | 退化鲁棒性 | GroundTruthData 不是病灶 mask |
| CMRxRecon_ty | P2 | 范围外（心脏） | 目录为空 |
| fastMR_lh_bran_knee_rawdata | P2 | raw k-space | 目录为空 |
| OAI 派生 | 可选 | 外部同例 | 未接入，满足授权、配对、空间和重叠核验后才能进入 |

`sources.yaml` 中每个来源的初始状态都是 `pending_verification`，状态由审计结果更新。

## 附录 D：必备审计报告（每个正式来源的 G0 都需要）

1. `ontology_validation_report.json`
2. `patient_alias_report.csv`
3. `split_leakage_report.json`
4. `geometry_qc.csv`
5. `annotation_provenance_report.csv`
6. `eligibility_coverage.csv`
7. `image_label_pairing.csv`
8. `input_visibility_report.csv`

另外还要有：`label_value_qc.csv`、`assets.jsonl`、`asset_integrity.jsonl`、`exposure_ledger.jsonl`、`manifest.jsonl`、`manifest.lock.json`、`G0-*.json`。以上全部存放在 `$PRIVATE_ROOT`，仓库里只放匿名汇总。

## 附录 E：必备自动测试（不变量）

**数据与身份**
- 空目录不计为病例；归档和解压资产不重复计数；`.truncated` 被隔离；不越界跟随链接；凭据文件不读取。
- 同一患者的回波、序列、随访、视图、裁剪、降质和缓存，跨划分的数量为 0；已暴露的病例不能改称 unexposed；SibBMS 的 MS 和 Norm 同号时不合并。

**本体与空间**
- 本体：A1 对应髌骨，A5/A6 对应半月板；tissue_id 2 返回 out_of_anatomy_scope；未知整数不会静默变成背景。
- 标签：近整数可以解码，概率图被拒绝；shape 相同但 affine 错误时不通过；标签变换保持类别；可逆往返误差在 1e-4 voxel 以内。

**监督资格与参照**
- box_fill 不能进入 expert_voxel 测试；自动 A 不能因为路径名里有 `labels` 就变成人工参照；confidence 为 0 或被删的框不生成负标签。
- 未复核体素的梯度为 0；masked Dice 的分母也排除这些体素；某个病种的背景不给其他病种提供负监督；全部 ignore 时损失有限且不稀释其他项。
- R_geo 参照只接受合格的 A 和 U；R_clin 参照必须有独立依据；一个病灶跨两个组织时输出多标签。
- 审计状态拼错时报错；重复的 sample_uid 被拒绝；泄漏时 `freeze_splits` 拒绝冻结。

**评价**
- 单体素、贴边、合并、分裂的情况都按一对一匹配；重复预测计为 FP。
- 全部拒判不能提高全覆盖 J；没有 R 参照时计入覆盖率，不自动算正确。
- 空 GT 病例不并入阳性病例的 Dice 均值；单边为空时表面距离为未定义，并计入失败数。

**输入与模型**
- 推理输入不含 `a_target, u_target, e_a, e_u, tissue_id`；padding 不产生正常监督；资格图不送进模型。
- 训练读不到锁定测试；登记时如果缺预算或缺审批，就拒绝；哈希不一致时拒绝续跑。
- 融合关闭时为恒等映射；oracle 组不能成为主结果。

**H2 与 H3**
- 隐藏参照不同、公开上下文相同的两条完整流程，生成的可见资格必须相同；非 oracle 组的可见信息哈希相同。
- 二维输入构造不接受体积或 z 索引，不读取邻层、三维 A、整卷统计或缓存（用 mock 文件系统验证）；两种模式加载同一个 checkpoint 哈希；三维结果不能填补二维缺项。

**证据发布**
- 只有工程 PASS、没有科学结果时，结论保持 NOT_EVALUATED；公共证据中不出现患者 ID。

## 附录 F：评价指标与分母

### F.1 分层指标

| 层级 | 指标 | 约束 |
|---|---|---|
| A | 每组织 Dice、NSD、HD95 | 只在合格区域上计算，单位为物理 mm |
| U | 每病例 Dice、NSD、病灶灵敏度、FP/scan、体积误差 | 小病灶分层，负例和空 mask 单列 |
| 正常状态 | 已复核范围内的 false-normal、FP 体积、扫描或区域特异性 | 病例特异性需要真正的目标阴性扫描 |
| R_geo | 受累结构集合准确率、每组织 F1 | 参照来自独立的 A 和 U |
| R_clin | 多标签 macro-F1、集合完全匹配率、拒判率 | 稀有类注明 n |
| 选择性预测 | 风险—覆盖率 | 拒判不能从总分母里消失 |
| 联合 | J（同时报告 FP 和各分项） | 不能只用乘积总分 |

### F.2 实现约定

- Dice、HD95、NSD 都先用 eligible 遮蔽预测和真值。
- HD95 使用 `numpy.percentile` 的默认线性插值，由测试固定。
- 匹配时先最大化合格配对数，再按 IoU 确定次序；匹配阈值属于冻结协议。
- 输出原始的充分统计量。
- `risk_coverage` 的风险分母排除 `correct=None`，另报 `unknown_correctness`。
- relation 的 `reference_coverage` 按病灶数计算，不按元组计算。

### F.3 固定分母

| 指标 | 定义 | 分母为 0 时 |
|---|---|---|
| `false_normal_on_accepted` | 被接受为未见异常、并且参照合格的体素或区域中，实际有目标异常的比例 | NA，同时报告接受覆盖率和拒判总数 |
| `accepted_positive_fnr` | 参照合格、被接受的真实异常体素或区域中，被判为未见异常的比例 | NA |
| `target_sensitivity_all` | 合格的目标病灶中被检出的比例；漏检和拒判都留在分母里 | — |
| `specificity_scan` | 只在协议复核过的目标阴性扫描中计算 | NA，并说明缺口 |
| `j_all_coverage` | 预先确定可评价的真实病灶中，同时满足检出、范围合格、关系正确且未拒判的比例 | 没有 R 参照的病灶不进入分母，但要报告排除数、理由和来源 |

主 J 按患者等权平均（只算有合格目标病灶的患者），同时报告病灶池化的 J。阴性患者进入 FP 和特异性，不进入阳性 J 的分母。

### F.4 H1 最小矩阵

| ID | 内容 | 角色 |
|---|---|---|
| H1-A | 图像 → A | 正式分项 |
| H1-U | 图像 → U | 正式分项，候选主对照 |
| H1-MT | 共享编码器 A+U，无融合 | 候选主对照 |
| H1-SEP | 独立 A/U + 几何 R | 候选主对照 |
| H1-AB | 预测 A 软引导 U + 预注册 R | 主要方法 |
| H1-CAP | 参数匹配的影像支路 | 机制 |
| H1-PERT | 扰动或置换预测 A | 机制 |
| H1-ORACLE | GT-A 辅助 | 只作诊断 |
| R 对照 | class-only / class+position / image-only / geometry-only / learned | 固定候选 |

## 附录 G：冻结字段与人工签核

### G.1 评价冻结字段（`configs/v41/thresholds.template.yaml`）

| 字段 | 当前值 | 负责人 | 最迟冻结点 |
|---|---|---|---|
| surface_tolerance_mm | null | 临床负责人 | H1 协议冻结前 |
| iou_match_threshold | null | 研究负责人 | 匹配协议冻结时 |
| fp_budget_per_scan | null | 临床与统计负责人 | H1 和 H2 冻结前 |
| j_target_range | null | 临床与研究负责人 | H1 冻结前 |
| a_noninferiority_margin | null | 统计负责人 | H1 冻结前 |
| bootstrap_n / seed / ci_level | 2000 / 20261101 / 0.95（拟定） | 统计负责人 | 登记前确认 |
| near_mm（ontology） | null | 研究负责人 | R_geo 正式运行前 |
| dual_mode_noninferiority_margin | null | 统计负责人 | H3 协议冻结前 |

### G.2 只能由人填写的签核

| 事项 | 负责人 | 缺失时 |
|---|---|---|
| target_abnormality_definition（tad_v1，`approvals.json`） | 研究者 | G0-U 和 G0-A 保持 BLOCKED |
| 遗留曝光核验结论（§3.4） | 研究者 | 不能冻结划分，锁定测试集为空 |
| 划分规则（ISLES 的比例、分层、种子；SKM 沿用官方划分）（§6） | 研究者 | 不能冻结划分 |
| 每个 U 类别的 clinical_definition_signoff | 临床负责人 | 该类别不能正式运行 |
| 标注可判读性、额外序列可见性 | 标注负责人 | 不作为主要监督 |
| 实例拆并规则、R 语义、多组织判定 | 临床与研究负责人 | 关系主结果 BLOCKED |
| 精标规模、盲测独立性 | 研究负责人或数据管理员 | 只做开发研究 |
| GPU、预算（每次正式登记前） | 用户或资源负责人 | 只能 dry-run |
| 外部授权与下载 | 用户或数据管理员 | 不下载 |
| 锁定测试解封与对外发布 | 研究负责人 | 保留 READY 状态 |

## 附录 H：现有实现与证据快照（截至 2026-11-02，commit `b3c2d53`）

### H.1 已实现（供核对）

- 模块：`schema, paths, cli, inventory, identity, splits, ontology, labels, geometry, geometry_qc, overlays, eligibility, losses, audits, registry, adapters/{isles,skm}, evaluation/{segmentation,instances,relations,selective,statistics,joint}`。
- Schema：sample、annotation、gate、prediction、experiment。
- 配置：sources、ontology、target_abnormality_definition、approvals、thresholds.template、paths.example。
- 测试：`tests/v41` 共 144 个，全部通过（2026-11-02 实测）。
- 报告：`reports/v41/{reuse_decisions.md, inventory_summary_v1.md, geometry_qc_summary_v1.md, g0_decisions_v1.md, g0_decisions_v3.md, ontology_validation_report.json}`。

### H.2 权威私有产物（`$PRIVATE_ROOT`）

- audits：`isles_u_v3`、`skm_a_v2`、`geometry_qc_isles_v2`（DWI/ADC 500/500 通过）、`geometry_qc_skm_v2`（155/155 通过，标签值 155/155 通过）、`identity_v2`（405 组，0 泄漏，尚未分划分）、`ontology_v2`、`overlays_v2`。
- registry：`gate_inputs_v3`、`manifest_v3`（405/405 准入，其中 ISLES 250、SKM 155）、`g0_decisions_v3`。
- 更早的 v1 和 v2 产物保留为历史，并标为 SUPERSEDED。

### H.3 已知待处理的小问题（核对时一并处理）

- `write_private_json` 会对已存在的父目录执行 chmod。
- 父目录是悬空符号链接时，原始 OSError 会泄漏路径。
- `UNKNOWN_FIELD` 错误会回显键名。
- inventory 对符号链接取的是 lstat 大小；根目录内被符号链接的子目录不会被遍历；增量校验忽略 mtime。
- full_decode 设置的是 `training_eligible`，应改名为 `integrity_verified`。
- instances 在 `float(None)` 时抛出 TypeError，而不是 MetricPolicyError。
- `joint_score` 在空分母时返回 0.0，应返回 None。
- bootstrap 的点估计是按行池化的，并且允许 `n_boot=0`。
- HD95 的约定与 MedPy 不同（已由测试固定，需要在评价协议里写明）。
- manifest 锁写入失败时，会留下发布了一半的目录。
- 重复资产采用"后写覆盖"。
- `--help` 会 import PIL。
- 损失权重全为 0 时出现 ZeroDivisionError。
- sample schema 中的开放子对象需要收紧。
- PDGM 和 BMSR 适配器尚未实现。
- 划分尚未冻结。遗留曝光核验还没做，所以目前所有病例组都会被挡在 `locked_test` 之外。
- `freeze_splits` 目前只支持按比例哈希分桶，以及预先指定的划分。SKM 沿用官方划分，可以通过预先指定 `split` 实现；ISLES 的分层抽样还需要补实现。

## 附录 I：参考

- ISLES 2022：https://www.nature.com/articles/s41597-022-01875-5 ；https://zenodo.org/records/7153326
- SKM-TEA：https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md
- UCSF-PDGM：https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- UCSF-BMSR：https://pubs.rsna.org/doi/10.1148/ryai.230126
- OAI 派生标签：https://github.com/wooboyeong/Anomaly-Aware-3D-Segmentation
- SCEMA：https://doi.org/10.1002/mrm.70365
- nnU-Net：https://github.com/MIC-DKFZ/nnUNet
- Metrics Reloaded：https://www.nature.com/articles/s41592-023-02151-z
- Marginal / exclusion loss：https://arxiv.org/abs/2007.03868
- DoDNet：https://arxiv.org/abs/2011.10217
- 旧仓库只读审计对象：`sober235/Foundation_Model` @ `4deeead`（复用裁决见 `reports/v41/reuse_decisions.md`）
- 脱敏数据盘点：`FM_data_inventory_redacted.md`
