# AnatoBind-MRI v4.1：监督需求、数据契约与验收清单

日期：2026-11-01。用途：交给研究者、标注团队及Coding Agent的规范建议。**本文是待实施规范，未在服务器生成这些新资产或运行测试。**

## 1. 实验必须回答什么

主结论必须在同一批合格测试病例上回答：组织是什么（A）、目标异常在哪些体素（U）、它与什么组织形成何种关系（R）。在三个不同数据集上分别成功，不等于同例联合能力。

单病种强监督U、框定位、自动解剖蒸馏、几何关联、临床组织累及、单切片观察及完整三维观察分别登记，不跨级宣称。

## 2. 数据入口与旧资产隔离

根目录：`/data2/congcong/data/FM_data`。

|原入口|新登记角色|必须核查|
|---|---|---|
|SKM-TEA_ltr/image_files|qDESS图像|H5内部键与维度、回波、真实patient ID|
|SKM-TEA_ltr/segmentation_masks|组织A参照|raw/dicom轨道、六类编码、空间映射|
|SKM-TEA_ltr/annotations/v1.0.0|框弱U、有限R|inspected、confidence、tissue_id、本体对应|
|derived/nnunet/raw/Dataset902_SKMTEAlesion|旧框填充定位基线|导出commit和父扫描；禁止作精确U GT|
|derived/nnunet/raw/Dataset901_SKMTEAm1r|旧A训练导出|视图/降质版本与原患者关系|
|ISLES_ltr/ISLES-2022|目标梗死U|DWI/ADC-mask空间，FLAIR注册|
|UCSF-PDGM_lh|肿瘤U、多序列|随访患者映射、mask版本、bias版本|
|UCSF-BMSR_cbb|增强转移瘤U|原seg/BraTS-seg标签差异、纵向扫描、T2Synth|
|derived/relation/v1|几何/伪R基线|不能升级人工R；patch_slices=3|
|derived/synthseg、FSL_SEG|自动A/辅助资产|teacher版本、来源、失败QC；不能作独立人工评价|
|新精标目录|联合A/U/R正式参照|标注协议、读者、仲裁、输入可见性、复核范围|

新产物使用新版本路径，示例 `derived/anatobind_v41/`。路径示例不表示已创建。旧结果保持只读，不覆盖、不删除。

## 3. 最小manifest字段

主表每行对应一个可唯一追踪的影像观察，不按切片或增强数冒充患者数。

```text
sample_uid
source_dataset
source_release
cohort
patient_uid
study_uid
session_uid
parent_scan_uid
derivative_uid
split
image_paths
modality_names
modality_is_synthetic
available_modality_mask
image_geometry
original_geometry
transform_chain
anatomy_ontology_version
abnormality_ontology_version
relation_semantics_version
annotation_records
review_regions
visibility_records
quality_records
provenance
```

`patient_uid`必须带来源及必要的队列命名空间，但“来源内编码相同”不自动说明跨来源同人；跨来源已知映射要另存。SibBMS的MS/Norm须保留组别。PDGM按官方FU映射而非简单去掉字符串前导零完成患者合并。BMSR按病例表而非盲目移除字母后缀推断。

每条annotation_record：

```text
annotation_uid
task                    # A / U / R_geo / R_clin / surface_defect
class_scope              # 此次检查的目标类别，不是数据集全部类别
representation           # voxel / surface / bbox / instance / study
path
label_map_version
provenance_type          # manual / expert_corrected / automated / box_fill / geometry / unknown
creator_or_model
creator_version
parent_annotation_uid
completeness             # target_complete / region_complete / positive_only / unreviewed
review_region_path
input_visibility         # visible / limited / unobservable / unknown
quality_status           # accepted / needs_review / rejected
intended_role            # train / val / test / pseudo_only / excluded
```

`inspected=true`只代表至少有人看过，不能自行转换为所有病种全视野完整阴性；confidence低的阳性框被移除后，所涉区域不能自动变正常。

## 4. 本体与空间契约

### 4.1 SKM A与R映射

|A标签|A含义|可直接映射的旧组织语义|
|---|---|---|
|1|patellar_cartilage|Patellar Cartilage|
|2|femoral_cartilage|Femoral Cartilage|
|3|tibial_cartilage_medial|Tibial Cartilage（不带侧别）|
|4|tibial_cartilage_lateral|Tibial Cartilage（不带侧别）|
|5|meniscus_medial|Meniscus（不带侧别）|
|6|meniscus_lateral|Meniscus（不带侧别）|

该编号来自本轮Dataset901，原始H5为通道顺序，转整数需显式记录。旧tissue_id中的ACL/PCL没有对应已确认A类，不能硬映射到背景；应标为out_of_anatomy_scope或补A标签。

### 4.2 空间QC要求

对所有正式图像/标签对读取shape、spacing、affine、axis codes、qform/sform、单位与物理视野。数组shape相同不证明同空间。数组不同但空间可变换，也不代表配准已经正确。标签重采样用相应离散/概率插值策略，记录变换链，禁止静默改header以“对齐”。

首轮至少对每个来源、每个空间转换分支进行覆盖边缘/小病灶/异常形变样本的叠加检查，并抽查坐标来回变换。SKM bbox遵守原始轴序和[x,y,z,dx,dy,dz]约定；mm与voxel、行列与xyz须显式转换。

保留原始物理空间，重采样只生成派生版本。薄软骨和小病灶不能统一1mm后不评估信息损失。缺失视野、padding与真实背景分别标记。

## 5. 监督资格规则

|已知监督|可以监督|不得监督|
|---|---|---|
|某类完整人工病灶mask，且检查范围明确|该类范围内的阳性与阴性|其他未检查病种的阴性|
|局部精标病灶mask，周围未复核|精标阳性；已明确复核的局部阴性|整卷正常|
|三维框|定位、弱监督目标存在|框内全部异常；框外全部正常|
|自动A mask|单独加权的伪A目标|专家级解剖评价参照|
|专家A和专家U的交叠|明确几何语义的R_geo参照|浸润/压迫/组织起源等临床R|
|人工tissue_id|其定义内的实例归属|病灶体素边界；未提供的内外侧或多组织关系|
|正常组队列名称|队列信息|全部异常类别的体素阴性|
|正标签被判不可信而排除|ignore/需复核|自动背景|
|输入质量不足|可见性受限、必要时拒判|确定正常|

缺标签所产生的ignore不能作为“图像中的unknown类”训练。训练时可以用复核资格遮蔽损失；部署时没有这些资格图，模型的预测拒判需另行校准。

## 6. 精标SOP

### 6.1 试点样本

开发集取20–30次扫描作为预算起点，患者去重并分层选择：典型阳性、微小/模糊边界、多组织候选、经完整复核的目标阴性、质量受限。该分层用于探索可行性，不用于估计自然患病率。正式测试需要另行定义抽样分布。

### 6.2 读者流程

1. 固定输入：是否只看E1/E2或是否允许临床补充序列。用额外序列定位的标签必须注明在目标模型输入中是否可见。
2. 固定异常语义：标残存组织异常、裂隙、积液、表面缺损，或其他明确目标；不要混成任意异常blob。
3. 标注可复核的体素边界；对不确定边界存band/soft consensus或ignore记录。
4. 完成预定义组织/视野的目标异常检查，记录完整/局部/仅阳性。
5. 独立评定R关系和可判断性；不展示几何基线结果诱导结论。
6. 双读者子集独立完成；第三人/共识仲裁，保留原始两版而非只留最终mask。
7. 统计每例时间、不可判读比例、一致性、各组织/病种支持数，决定是否扩展。

实例分裂/合并原则必须书面化，例如间隔超过多少物理距离可计为不同实例；具体阈值由临床语义与试点决定，不由模型性能反推。

### 6.3 缺损标注

观察到组织缺失与“缺失组织应该占多大体积”不是同一问题。缺损初期可记录表面足迹及等级，不用镜像或统计形状估计出来的体积作为专家观察真值。后续定义反事实包络任务时另设参照、误差与不确定性。

## 7. 缺标模拟协议

完整参照只由评估器保存。训练版本可以按以下机制减少已知信息：
- 指定区域只完成局部复核（region级资格已知）；
- 某类异常不标（class级资格已知）；
- 保留部分阳性，但其他区域全部记为未复核；
- 按大小/可见性倾向遗漏，训练者不知道具体被遗漏位置。

禁止根据被删病灶的真实mask给方法生成精确ignore区后，与“不知道遗漏位置”的基线比较。可另列oracle-eligibility诊断上限，但不能作为主要结果。

全部方法得到相同复核范围、保留阳性、来源与辅助标签。需要有已复核阴性区域，否则全图预测阳性可能获得低正损失。完整测试目标阴性参照用于比较FP和false-normal；测试阈值不单独调优。

## 8. 数据划分和评价测试

必须具有以下测试/审计输出，不能仅在文档写“无泄漏”：

```text
ontology_validation_report.json
patient_alias_report.csv
split_leakage_report.json
geometry_qc.csv
annotation_provenance_report.csv
eligibility_coverage.csv
image_label_pairing.csv
input_visibility_report.csv
```

建议自动测试：
- 同patient的回波/序列/FU/视图/裁剪/降质/缓存跨split数为0。
- `box_fill`不能被选入expert_voxel_test。
- confidence=0或删除框不生成负标签。
- unreviewed体素对监督损失梯度为0；masked Dice分母也排除。
- 某病种mask的背景不自动给其他病种提供负监督。
- R_geo参考只接受合格参考A/U；R_clin参考必须有独立依据。
- 2D输入构建不读取相邻层、3D候选、整卷统计或病例缓存。
- 训练数据不能读锁定测试的A/U/R；生成伪标签/SSL时也检查接触记录。
- 空mask、单体素、贴边病灶、多实例合并/分裂、拒判全图等情况有明确指标输出。

## 9. 停止与继续条件

**可以立即做：**索引、患者映射、标签来源审计、空间QC、ISLES U及SKM A基线、SKM标注可行性试点。

**满足同例精标后做：**主模型A/U/R、H1、临床关系评估、严格正常状态指标。

**满足主线后做：**弱标签联合、外部OAI、全区域扩展、H3、重建鲁棒性。

**不能仅靠代码决定：**临床标签语义、是否可见、复核范围、缺损包络定义、什么叫多组织受累、读者分歧是否可接受。

若没有精标预算或外部合格同例标签，当前数据仍能完成真实U与弱定位实验，但不得据此宣称最终联合体素目标已经实现。此时应明确保留项目缺口，不用代理指标改变目标。
