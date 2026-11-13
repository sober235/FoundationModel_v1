<!-- 脱敏版本。原文（可能含病例级文件名）仅存私有根 $PRIVATE_ROOT/docs/FM_data_inventory.md
原文 sha256: 42cb8d18891a992283f46b54cdabf5921c0186e08e93d535d6f59e144266f7f3
脱敏规则：原始数据根路径、历史下载路径、SKM-TEA/ISLES/OASIS/fastMRI/HCP/TotalSegmentator/SibBMS 示例病例或文件编号均替换为占位符。
本文件不代表全文件解码或患者去重完成。 -->

# FM_data 全目录内容盘点

盘点日期：2026-11-01

根目录：`<RAW_ROOT>`
## 范围与结论

本次通过 amax 服务器只读列目录、读取本地说明/元数据、抽查代表文件，覆盖全部 **17 个一级目录**及 `derived` 的 **18 个一级分支**。其中 **14 个一级目录非空、3 个为空**。这不是17套互不重复、完整可训练的数据集。

**本报告是内容结构盘点，不是全量递归文件清单或完整性认证。** 未逐个解码全部HDF5/NIfTI/DICOM/MHA，未计算全库文件数、物理占用、哈希或跨数据集唯一患者数。目录分页计数不是文件系统快照；历史日志/README/既有训练清单数字均与本次目录观察分开标注。未修改服务器数据。

空目录：`CMRxRecon_ty`、`TotalSegmentator_MRI_cbb`、`fastMR_lh_bran_knee_rawdata`。实际TotalSegmentator数据在 `TotalSegmentor_MRI_cbb`，fastMRI原始H5已在 `fastMRI_lh_brain_knee/kspace` 发现。

## 一、17个一级目录总表

|编号|目录|状态|内容类别|已核实/声明的数量|
|---:|---|---|---|---|
|1|`AMOS_cbb`|非空|腹部器官分割；CT+MRI混合|影像240/120/240（train/val/test）；训练目录另含1个.DS_Store。标签240/120/0。|
|2|`CMRxRecon_ty`|空目录|空占位目录|0个一级条目。|
|3|`HCP_lh_T1T2`|非空|脑部配对T1w/T2w|根目录1115条目；下载日志记录1113名受试者、2226个T1w/T2w文件。|
|4|`ISLES_ltr`|非空|脑卒中病灶分割|本地README声明250例、每例3种MRI及1份病灶mask；本次未逐例重计所有序列。|
|5|`KMAR-50K_ty`|非空|伪影影像/参考影像配对|2个数据分包、6个数据子目录、2个cohort CSV；总影像/患者数量未完成计数。|
|6|`OASIS_cbb`|非空|OASIS-1及OASIS-2脑MRI|OASIS-1有12个解压disc；OASIS-2有2个解压part；病例、访视及图像总数未递归统计。|
|7|`SKM-TEA`|非空|SKM-TEA重建/原始数据包|files_recon_calib-24共157条目：155个.h5及2个.h5.truncated。|
|8|`SKM-TEA_ltr`|非空|SKM-TEA图像/组织分割/病灶框|image_files共155条目；JSON定义16类病变、4个上位病变类、6类组织名称；各split病灶数未统计。|
|9|`SPIDER_cbb`|非空|脊柱MRI/结构掩膜/影像分级|images目录447条目；masks目录447条目；不等于447名患者。|
|10|`SibBMS_ms`|非空|MS/Norm脑MRI及部分分割标注|MS 93个受试者目录；Norm 100个受试者目录；Annotation 10个目录（非额外10名患者）。|
|11|`TotalSegmentator_MRI_cbb`|空目录|空占位/近似命名|0个一级条目。|
|12|`TotalSegmentor_MRI_cbb`|非空|MRI多结构分割|根目录617条目：meta.csv+616个样本目录；样本s0001有50个mask文件。|
|13|`UCSF-BMSR_cbb`|非空|脑转移瘤MRI/两版分割|训练目录461个扫描条目；变更日志声明461例添加合成T2、324例添加BraTS分割（后者未逐例复核）。|
|14|`UCSF-PDGM_lh`|非空|胶质瘤多参数MRI/临床分子信息|501个病例/扫描目录条目；样本0004有24个文件；未跨目录/临床ID去重。|
|15|`derived`|非空|派生数据/模型处理/训练与评估|18个分支均已查看一级内容；nnunet/raw有8个Dataset901–908任务；既有SSL清单6130条序列记录。|
|16|`fastMRI_lh_brain_knee`|非空|脑/膝k-space、DICOM及病灶框|历史日志处理5641个体积，记录1个失败；当前全量H5、DICOM和独立患者总数未重计。|
|17|`fastMR_lh_bran_knee_rawdata`|空目录|空占位/近似命名|0个一级条目。|

## 二、逐数据集详细内容

### 01. AMOS_cbb

**定位路径：** `<RAW_ROOT>/AMOS_cbb/AMOS22/amos22`

**影像与格式：** imagesTr / imagesVa / imagesTs；NIfTI .nii.gz；本地说明为500 CT+100 MRI；ID<500为CT、>=500为MRI。

**标注与监督：** 15类腹部器官体素标签；labelsTr 240、labelsVa 120；labelsTs为空。

**元数据和辅助文件：** dataset.json、readme.md；上级目录有labeled_data_meta_0000_0599.csv及amos22.zip。

**数量口径：** 影像240/120/240（train/val/test）；训练目录另含1个.DS_Store。标签240/120/0。

**核验依据：** 目录分页末尾、labelsTs空目录；dataset.json头部；readme.md全文。

**注意事项：** 不能全部按MRI计数；README的test=120与dataset.json和实测240不一致，应以实际目录及清单核对；尚未逐文件解码。

**补充：** 器官标签：脾、左右肾、胆囊、食管、肝、胃、主动脉、下腔静脉、胰腺、左右肾上腺、十二指肠、膀胱、前列腺/子宫。

### 02. CMRxRecon_ty

**定位路径：** `<RAW_ROOT>/CMRxRecon_ty`

**影像与格式：** 未发现条目。

**标注与监督：** 未发现。

**元数据和辅助文件：** 未发现。

**数量口径：** 0个一级条目。

**核验依据：** list_files_page返回eof=true、entries=[]。

**注意事项：** 不能因为目录名存在，就计入已有心脏MRI数据。

**补充：** 本次只确认该路径为空，不推断服务器其他位置是否有同名数据。

### 03. HCP_lh_T1T2

**定位路径：** `<RAW_ROOT>/HCP_lh_T1T2`

**影像与格式：** <subject>/T1w/T1w_acpc_dc_restore_brain.nii 与 T2w_acpc_dc_restore_brain.nii。

**标注与监督：** 原始目录抽查未见人工病灶标签；derived/synthseg/hcp另有派生处理目录。

**元数据和辅助文件：** manifest.tsv（subject/modality/path/compressed_bytes）、download.log。

**数量口径：** 根目录1115条目；下载日志记录1113名受试者、2226个T1w/T2w文件。

**核验依据：** 根目录首尾分页、<CASE_ID>/T1w完整列表、manifest表头和下载日志结尾。

**注意事项：** 2226为历史下载日志记录，不是本次逐文件完整性认证；compressed_bytes不能当作当前解压后磁盘体积。

**补充：** 同一受试者两种对比应绑定划分；原始数据与SynthSeg输出分开管理。

### 04. ISLES_ltr

**定位路径：** `<RAW_ROOT>/ISLES_ltr/ISLES-2022`

**影像与格式：** BIDS结构sub-strokecase*/ses-0001；README记载FLAIR、DWI(b=1000)、ADC；抽查到DWI/ADC NIfTI及JSON。

**标注与监督：** derivatives/sub-*/ses-0001/*_msk.nii.gz；抽查另有*_snp.png预览。

**元数据和辅助文件：** README、dataset_description.json、participants.tsv、LICENSE；上级有center_ids.xlsx及ISLES-2022.zip。

**数量口径：** 本地README声明250例、每例3种MRI及1份病灶mask；本次未逐例重计所有序列。

**核验依据：** README全文、sub-<CASE_ID>的dwi和derivatives目录。

**注意事项：** README明确为原生空间、未预配准；不能默认FLAIR与DWI/ADC逐体素对齐。已有SSL子集仅200名患者，不是原始总量。

**补充：** 病灶分割与解剖分割不同；本次未验证mask的取值、非空性及空间一致性。

### 05. KMAR-50K_ty

**定位路径：** `<RAW_ROOT>/KMAR-50K_ty`

**影像与格式：** KMAR-50K/ArtifactData_part1、GroundTruthData_part1、Testing _GroundTruthData；KMAR-50K-part2/ArtifactData_part2、GroundTruthData_part2、Testing_ArtifactData；抽查为.nii.gz。

**标注与监督：** 当前核实到的是Artifact/GroundTruth同名图像配对，不是病灶分割标签。

**元数据和辅助文件：** TrainingCohort.csv、TestingCohort.csv；含序列描述、场强、TR/TE、翻转角、层厚、像素间距、厂商；两个zip及无扩展名文件6。

**数量口径：** 2个数据分包、6个数据子目录、2个cohort CSV；总影像/患者数量未完成计数。

**核验依据：** 两分包目录完整列表；part1两类文件前5项；TrainingCohort.csv表头和样本。

**注意事项：** 不能把名称50K直接当作50000名患者或50000个3D体积；CSV有前导引号/空格，文件名还存在_0与_0.0形式差异。

**补充：** Testing _GroundTruthData名称含内部空格；两个归档为95w9f5tzz8-6.zip和xw7mrg7ntg-6.zip。

### 06. OASIS_cbb

**定位路径：** `<RAW_ROOT>/OASIS_cbb`

**影像与格式：** OASIS-1/raw：disc1–disc12及12个tar.gz；样本含RAW、PROCESSED。OASIS-2/raw：PART1/PART2及2个tar.gz；已见同一编号MR1/MR2。

**标注与监督：** OASIS-1样本FSL_SEG包含fseg.hdr/.img及说明/预览；OASIS-1/freesurfer为空。

**元数据和辅助文件：** OASIS-1病例级.txt/.xml；RAW影像为Analyze .hdr/.img，并有.gif预览。

**数量口径：** OASIS-1有12个解压disc；OASIS-2有2个解压part；病例、访视及图像总数未递归统计。

**核验依据：** 两个版本raw列表、OASIS-1样本RAW/FSL_SEG列表、OASIS-2样本列表。

**注意事项：** FSL_SEG不是专家病灶真值；RAW/PROCESSED/FSL_SEG不得作为不同患者累加；纵向MR1/MR2需要患者级划分。

**补充：** OASIS-2抽查<CASE_ID>仅见RAW入口；未验证其全部影像内容及临床表覆盖。

### 07. SKM-TEA

**定位路径：** `<RAW_ROOT>/SKM-TEA`

**影像与格式：** files_recon_calib-24/*.h5；保留files_recon_calib-24.tar.gz。

**标注与监督：** 该分支未见独立标注入口；图像/标注分支在SKM-TEA_ltr。

**元数据和辅助文件：** download_skmtea_kspace.py、download.log、download.status、extract_20260903.log。

**数量口径：** files_recon_calib-24共157条目：155个.h5及2个.h5.truncated。

**核验依据：** 完整目录返回与分页末尾next_offset=157；归档大小来自目录stat。

**注意事项：** <CASE_ID>.h5.truncated、<CASE_ID>.h5.truncated需排除/核验；普通.h5尚未逐个打开；未解析HDF5内部键、维度或完整性。

**补充：** 归档大小819739559552字节（约819.74 GB，十进制）；多数H5为8724360192字节，部分更大；不据此推断当前可训练性。

### 08. SKM-TEA_ltr

**定位路径：** `<RAW_ROOT>/SKM-TEA_ltr`

**影像与格式：** image_files/*.h5（目录155条目）；dicoms/；image_files.tar.gz。标注样本注明双回波、512×512×160、0.3125×0.3125×0.8 mm。

**标注与监督：** segmentation_masks/dicom-track与raw-data-track中的.nii.gz；annotations/v1.0.0/train.json、val.json、test.json含3D bbox。

**元数据和辅助文件：** all_metadata.csv：患者/时间点、扫描器、序列、层厚、TR/TE、场强、线圈、翻转角等。

**数量口径：** image_files共155条目；JSON定义16类病变、4个上位病变类、6类组织名称；各split病灶数未统计。

**核验依据：** image_files末页；annotation JSON头部及bbox片段；两套mask文件抽查。

**注意事项：** 病灶bbox不是病灶体素mask；两套mask坐标轨道不同，不能只按同名直接叠加；本次未验证标签取值和坐标变换。

**补充：** 4上位类：半月板撕裂、韧带撕裂、软骨病变、积液。6组织名称：半月板、ACL、PCL、股骨软骨、髌骨软骨、胫骨软骨；这不等于已核实mask编码表。

### 09. SPIDER_cbb

**定位路径：** `<RAW_ROOT>/SPIDER_cbb/SPIDER`

**影像与格式：** images/*.mha；见T1、T2及T2_SPACE命名；保留images.zip。

**标注与监督：** masks/*.mha；radiological_gradings.csv含Modic、终板、滑脱、突出、狭窄、膨出、Pfirrman grade。

**元数据和辅助文件：** overview.csv含椎体/椎间盘数量、训练子集、部位及扫描参数；masks.zip。

**数量口径：** images目录447条目；masks目录447条目；不等于447名患者。

**核验依据：** 首尾分页，两个目录next_offset=447；两CSV表头。

**注意事项：** 同一患者多对比不可分到不同数据划分；447/447仅证明条目数一致，未做全量文件名配对及mask标签核验。

**补充：** 结构掩膜与椎间盘分级是不同粒度监督；不能把Pfirrmann分级直接当作逐体素异常分割。

### 10. SibBMS_ms

**定位路径：** `<RAW_ROOT>/SibBMS_ms/sibbms/Output`

**影像与格式：** MS/sub-*/ses-*/anat与Norm/sub-*/ses-*/anat；MS样本含FLAIR/T1w/T2w/ce-GAD_T1w；Norm样本含FLAIR/T1w/T2w。

**标注与监督：** Annotation共10个sub-*目录；抽查含Segmentation-label.nii.gz和FLAIR/T1/T1c/T2配套影像。

**元数据和辅助文件：** Cohort analysis_MS_data.xlsx（已确认存在，未解析表格）；上级有sibbms.zip及download.log。

**数量口径：** MS 93个受试者目录；Norm 100个受试者目录；Annotation 10个目录（非额外10名患者）。

**核验依据：** MS末页、Norm完整页、Annotation完整页及MS/Norm/Annotation样本。

**注意事项：** MS与Norm均从sub-<CASE_ID>编号，必须保留队列前缀；标注只覆盖部分病例；尚未数清全部session和标注与MS的对应关系。

**补充：** 不能把Annotation再计入独立患者总数；193为两组目录合计，不是跨库去重后患者数。

### 11. TotalSegmentator_MRI_cbb

**定位路径：** `<RAW_ROOT>/TotalSegmentator_MRI_cbb`

**影像与格式：** 未发现条目。

**标注与监督：** 未发现。

**元数据和辅助文件：** 未发现。

**数量口径：** 0个一级条目。

**核验依据：** list_files_page返回空且eof=true。

**注意事项：** 实际非空分支拼作TotalSegmentor_MRI_cbb，少了ator中的at；配置不要使用本空目录。

**补充：** 保留服务器原始拼写，不在本次盘点中重命名。

### 12. TotalSegmentor_MRI_cbb

**定位路径：** `<RAW_ROOT>/TotalSegmentor_MRI_cbb/TotalsegmentatorMRI_dataset_v200`

**影像与格式：** s<CASE_ID>–s0616/mri.nii.gz；上级保留TotalsegmentatorMRI_dataset_v200.zip。

**标注与监督：** 每例segmentations/；s0001实测50个独立结构.nii.gz。

**元数据和辅助文件：** meta.csv（已确认存在，未全表解析）。

**数量口径：** 根目录617条目：meta.csv+616个样本目录；样本s0001有50个mask文件。

**核验依据：** 首尾分页、s0001及其segmentations完整列表。

**注意事项：** 50个文件不代表50个结构在该扫描中均非空；需按FOV、序列和mask非空性核验，未逐例检查。

**补充：** 掩膜名称覆盖脏器、血管、骨骼、肌肉、脊髓/椎间盘等；不应直接当作疾病异常标签。

### 13. UCSF-BMSR_cbb

**定位路径：** `<RAW_ROOT>/UCSF-BMSR_cbb/UCSF-BMSR`

**影像与格式：** UCSF_BrainMetastases_TRAIN/<scan>/；样本含T1pre、T1post、FLAIR、subtraction、T2Synth。

**标注与监督：** 原始*_seg.nii.gz与部分病例*_BraTS-seg.nii.gz；CHANGELOG说明BraTS三类肿瘤相关标签。

**元数据和辅助文件：** CHANGELOG.md、TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx、v1.3.zip、.download.lock。

**数量口径：** 训练目录461个扫描条目；变更日志声明461例添加合成T2、324例添加BraTS分割（后者未逐例复核）。

**核验依据：** 训练目录末页、100101A完整列表、CHANGELOG全文。

**注意事项：** T2Synth是合成T2；两套分割编码不同；A/B/C等扫描后缀需患者级去重；.download.lock本身不能证明仍在下载。

**补充：** BraTS标签日志记载：1非增强核心、2周围非增强FLAIR高信号、3增强肿瘤；原始增强肿瘤对应原始标签1，不应混用编码。

### 14. UCSF-PDGM_lh

**定位路径：** `<RAW_ROOT>/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5`

**影像与格式：** 样本含T1/T1c/T2/FLAIR/DWI/ADC/ASL/SWI；DTI_eddy_noreg、FA、MD、L1/L2/L3及rotated_bvecs；另有*_bias版本。

**标注与监督：** tumor_segmentation、brain_segmentation、brain_parenchyma_segmentation。

**元数据和辅助文件：** metadata_v5.csv、metadata_glossary.csv、DTI-bval.zip、DTI-bvec.zip；临床字段含WHO分级、病理、MGMT、1p/19q、IDH、生存状态/OS、EOR、活检及BraTS关联。

**数量口径：** 501个病例/扫描目录条目；样本0004有24个文件；未跨目录/临床ID去重。

**核验依据：** v5目录首尾、0004完整列表、metadata_v5.csv表头。

**注意事项：** 不能默认501个独立患者或每例齐全；CSV ID如004与文件ID0004格式不同；glossary不是有效UTF-8；目录名含空格。

**补充：** 样本文件名包括bias版本，但未读取处理文档或逐图检查其具体处理含义；临床缺失值需单独核验。

### 15. derived

**定位路径：** `<RAW_ROOT>/derived`

**影像与格式：** 18个子目录、2个SKM-TEA门控日志；包括重采样、格式转换、推理、预处理、病灶清单等。

**标注与监督：** 含SynthSeg处理分支、nnU-Net/nnDetection任务数据、预测/派生标签入口；具体标注来源需逐任务确认。

**元数据和辅助文件：** aur/ssl_manifest_v1中有data_inventory.csv、split_leakage_report.json、samples_ssl.json、val_patients.json。

**数量口径：** 18个分支均已查看一级内容；nnunet/raw有8个Dataset901–908任务；既有SSL清单6130条序列记录。

**核验依据：** derived完整列表及18个分支列表；nnunet/raw列表；SSL报告全文。

**注意事项：** 不能与原始数据重复计数；模型输出不能自动升级为专家真值；已有ok=true只适用于当时那份SSL清单。

**补充：** aur_handoff_20261009为空；level_r_smoke存在疑似凭据文件，未读取内容；具体checkpoint及所有深层文件未穷举。

### 16. fastMRI_lh_brain_knee

**定位路径：** `<RAW_ROOT>/fastMRI_lh_brain_knee`

**影像与格式：** 转换后DICOM：brain/multicoil_train、knee/multicoil_train、knee/multicoil_val；原始H5：kspace/{brain,knee}/{multicoil_train,multicoil_val}。

**标注与监督：** Annotations/{brain,knee}.csv为slice/study_level/x/y/width/height/label；是二维框/检查级标签，不是病灶体素mask。

**元数据和辅助文件：** brain_file_list.csv、knee_file_list.csv、conversion.log、failed_conversions.txt；每例有.fastmri_dicom_complete.json。

**数量口径：** 历史日志处理5641个体积，记录1个失败；当前全量H5、DICOM和独立患者总数未重计。

**核验依据：** 各分支目录、DICOM和H5样本、两标注CSV头部、失败清单及conversion.log末尾。

**注意事项：** brain验证集存在于kspace，但转换后brain目录未见val入口；历史截断错误需复核当前副本；不要混淆原始采集DICOM与转换生成DICOM。

**补充：** 历史失败文件file_brain_<CASE_ID>.h5：日志记录实际eof=52639232、声明stored_eof=788021248；日志路径是原下载位置<REDACTED_PATH>，不等同当前副本状态。

### 17. fastMR_lh_bran_knee_rawdata

**定位路径：** `<RAW_ROOT>/fastMR_lh_bran_knee_rawdata`

**影像与格式：** 未发现条目。

**标注与监督：** 未发现。

**元数据和辅助文件：** 未发现。

**数量口径：** 0个一级条目。

**核验依据：** list_files_page返回空且eof=true。

**注意事项：** 原始fastMRI H5实际已见于fastMRI_lh_brain_knee/kspace，不要把本空目录作为训练入口。

**补充：** 本次保留fastMR与bran的原始拼写，未做重命名或移动。

## 三、derived 的18个分支

以下是实际读取的一级内容，不代表每个分支全部深层文件已经穷举。

|分支|类型|已确认内容|边界|
|---|---|---|---|
|aur|重采样/自监督|samples*.json、resampled_1mm_v1、ssl_manifest_v1、ssl_manifest_probe_nohcp、ssl_runs、eval_smoke_20261010|存在完整的既有SSL统计及划分报告；仅代表选入子集|
|aur_handoff_20261009|空目录|entries=[]，eof=true|没有可盘点的数据文件|
|brain_anatomy|脑解剖处理|cases.json、sim、preflight_best_*、eval_*、infer_smoke|未逐个解析病例清单和预测结果|
|brain_disease|脑疾病处理|bmsr_subjects.csv、glioma、infarct、metastasis、crossrun、infer_smoke|不是新增原始病例；标签来源与处理版本待逐任务核验|
|fastmri_knee|膝关节派生处理|leg2、leg2_gate0|只确认目录结构，未核实所有文件/标签定义|
|fastmri_nii|格式转换|brain|名字指示NIfTI转换分支；本次未逐文件检查|
|knee_infer|膝关节推理|<CASE_ID>、<CASE_ID>、dicom_<CASE_ID>、regress901_<CASE_ID>、world901_<CASE_ID>|多个处理/坐标版本不能重复计作病例|
|level_r|病灶/体积索引|lesions.json、volumes|未逐条解析病灶对象与标注来源|
|level_r_smoke|服务测试|level_r.sqlite、server.log、server.pid；另有疑似凭据文件|未读取凭据内容；不属于医学病例数据|
|nndet|检测任务数据|Task903_FastMRIBrainSmallLesion|任务组织目录；未穷举深层文件|
|nndet_models|检测模型目录|Task903_FastMRIBrainSmallLesion|未验证具体checkpoint、可加载性或训练完成状态|
|nndet_runs|检测运行结果|fold0_default.json、fold0_swept.json、gt.json、infer_smoke|gt命名不自动证明是人工体素真值|
|nndet_smoke|检测烟雾测试|data、models、runner|测试产物需与正式训练数据分开|
|nnunet|分割训练流水线|raw、preprocessed、results；raw中Dataset901–908共8个任务|任务名不证明全部标签是人工真值；具体内容来源待核验|
|relation|关系数据/索引入口|v1|本次仅确认v1目录存在，未解析其schema|
|skmtea|SKM-TEA派生处理|m1、m1_pre_audit_fixes、m1r、m1r_cache、m1r_pred|原始、缓存、备份和预测应分开，不应重复计数|
|synthseg|SynthSeg处理分支|bmsr、fastmri_brain、hcp、isles、pdgm、sibbms；2个run日志|模型生成分割/伪标签应与人工标注区分；未逐个检查输出|
|synthseg_smoke|SynthSeg测试|in、out_*及CPU/GPU/isles测试日志|只确认测试文件结构，未核验所有输出质量|

根目录另有 `skmtea_frames_gate_20260907_1126.log` 与 `skmtea_frames_gate_20260907_1221.log`，不计作数据集。`level_r_smoke` 中疑似凭据文件未读取内容。

### nnU-Net raw 已有的8个任务

|任务目录|名称指向的任务|
|---|---|
|`Dataset901_SKMTEAm1r`|SKM-TEA m1r|
|`Dataset902_SKMTEAlesion`|SKM-TEA lesion|
|`Dataset903_FastMRIBrainSmallLesion`|fastMRI脑小病灶|
|`Dataset904_PDGMGlioma`|PDGM胶质瘤|
|`Dataset905_BMSRMetastasis`|BMSR脑转移瘤|
|`Dataset906_ISLESInfarct`|ISLES脑梗死|
|`Dataset907_BrainAnatomyFLAIR`|FLAIR脑解剖|
|`Dataset908_FastMRIBrainOutline`|fastMRI脑轮廓|

这里只验证目录存在；具体标签来源、模型权重、训练完成状态或性能均未核实。

## 四、已有SSL清单：仅为选入子集

来源：`<RAW_ROOT>/derived/aur/ssl_manifest_v1/split_leakage_report.json`，以及同目录 `data_inventory.csv`。

|来源|训练序列|验证序列|序列合计|训练患者|验证患者|选入序列|
|---|---:|---:|---:|---:|---:|---|
|bmsr|954|123|1077|225|26|FLAIR/T1/T1c|
|hcp|2002|224|2226|1001|112|T1/T2|
|isles|360|40|400|180|20|ADC/DWI|
|pdgm|1436|164|1600|356|40|FLAIR/T1/T1c/T2|
|sibbms|749|78|827|133|15|FLAIR/T1/T2|

|仅序列合计|5501|629|6130|不跨来源合并|不跨来源合并|—|

这6,130条是序列样本，不是6,130名患者，也不是原始数据全库规模。ISLES清单只选入200名患者、ADC/DWI；PDGM清单中有400条/序列、396个报告内患者；SibBMS清单T2比FLAIR/T1少1条。不能因此推断原始数据缺失或已经全库去重。

既有报告中 `test_patient_rows_in_ssl=0`、`val_patient_rows_in_ssl_train=0`、`duplicate_images=0`、`ok=true`；这些是本次读取到的历史报告字段，**并非本次重新验证的全库结论**。报告还列出数值ID `<CASE_ID>`、`<CASE_ID>` 出现在两个来源中，这不能直接证明同一患者，应使用来源命名空间。

## 五、数量和证据台账

|数据集|对象|数量|单位|证据口径|来源|备注|
|---|---|---:|---|---|---|---|
|根目录|一级子目录|17|目录|本次目录实测|<RAW_ROOT>|完整列表，全部17项均已查看|
|AMOS|imagesTr目录条目|241|条目|本次分页末尾|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/imagesTr|含1个.DS_Store；dataset.json声明240个训练影像|
|AMOS|训练影像（清单声明）|240|扫描|本地dataset.json并与目录核对|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/dataset.json|CT与MRI混合，非240名MRI患者|
|AMOS|imagesVa|120|条目|本次分页末尾|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/imagesVa|未逐个解码|
|AMOS|imagesTs|240|条目|本次分页末尾|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/imagesTs|README test=120与此不一致|
|AMOS|labelsTr|240|条目|本次分页末尾|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/labelsTr|未逐文件配对|
|AMOS|labelsVa|120|条目|本次分页末尾|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/labelsVa|未逐文件配对|
|AMOS|labelsTs|0|条目|本次空目录实测|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/labelsTs|不存在可见测试标签|
|HCP|根目录条目|1115|条目|本次首尾分页|<RAW_ROOT>/HCP_lh_T1T2|末尾包含download.log及manifest.tsv|
|HCP|下载日志记录的受试者|1113|受试者|历史下载日志，目录及既有SSL报告交叉支持|<RAW_ROOT>/HCP_lh_T1T2/download.log|非本次逐病例全文件验证|
|HCP|下载日志记录的影像|2226|序列文件|历史下载日志|<RAW_ROOT>/HCP_lh_T1T2/download.log|T1w/T2w；当前逐文件完整性未核验|
|ISLES|本地README声明|250|病例|本地说明，非本次全量重计|<RAW_ROOT>/ISLES_ltr/ISLES-2022/README|每例三序列与一mask为文档声明|
|SKM-TEA|重建数据目录条目|157|条目|完整列表及分页末尾|<RAW_ROOT>/SKM-TEA/files_recon_calib-24|155个H5及2个truncated|
|SKM-TEA|普通H5|155|文件|完整目录列表|<RAW_ROOT>/SKM-TEA/files_recon_calib-24|不等于155个已验证可读文件|
|SKM-TEA|truncated文件|2|文件|完整目录列表|<RAW_ROOT>/SKM-TEA/files_recon_calib-24|<CASE_ID>、<CASE_ID>|
|SKM-TEA_ltr|image_files|155|条目|本次分页末尾|<RAW_ROOT>/SKM-TEA_ltr/image_files|抽查为H5，未逐文件解码|
|SKM-TEA_ltr|病变类别|16|类|读取annotation JSON分类定义|<RAW_ROOT>/SKM-TEA_ltr/annotations/v1.0.0/train.json|4个上位病变类；病灶是3D框|
|SPIDER|images|447|条目|本次分页末尾|<RAW_ROOT>/SPIDER_cbb/SPIDER/images|抽查.mha；不是患者数|
|SPIDER|masks|447|条目|本次分页末尾|<RAW_ROOT>/SPIDER_cbb/SPIDER/masks|未全量验证一一配对|
|SibBMS|MS|93|受试者目录|本次分页末尾|<RAW_ROOT>/SibBMS_ms/sibbms/Output/MS|会话数未统计|
|SibBMS|Norm|100|受试者目录|本次完整分页|<RAW_ROOT>/SibBMS_ms/sibbms/Output/Norm|与MS编号空间重叠，不能仅按sub-ID合并|
|SibBMS|Annotation|10|受试者目录|本次完整分页|<RAW_ROOT>/SibBMS_ms/sibbms/Output/Annotation|标注子集，不能追加到患者总数|
|TotalSegmentator MRI|根目录|617|条目|本次首尾分页|<RAW_ROOT>/TotalSegmentor_MRI_cbb/TotalsegmentatorMRI_dataset_v200|meta.csv及616个样本目录|
|TotalSegmentator MRI|样本目录|616|目录|首尾目录及meta.csv交叉确认|<RAW_ROOT>/TotalSegmentor_MRI_cbb/TotalsegmentatorMRI_dataset_v200|未逐病例核验内容|
|TotalSegmentator MRI|s<CASE_ID>/segmentations|50|mask文件|样本目录完整分页|<RAW_ROOT>/TotalSegmentor_MRI_cbb/TotalsegmentatorMRI_dataset_v200/s<CASE_ID>/segmentations|非全病例、非非空器官数|
|UCSF-BMSR|训练扫描目录|461|目录|本次首尾分页|<RAW_ROOT>/UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_TRAIN|不是独立患者数|
|UCSF-BMSR|BraTS分割覆盖（声明）|324|扫描|本地CHANGELOG v1.3|<RAW_ROOT>/UCSF-BMSR_cbb/UCSF-BMSR/CHANGELOG.md|本次未逐例复核覆盖|
|UCSF-PDGM|病例/扫描目录|501|目录条目|本次首尾分页|<RAW_ROOT>/UCSF-PDGM_lh/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5|未跨病例ID去重|
|fastMRI|历史转换任务总量|5641|体积任务|conversion.log末尾|<RAW_ROOT>/fastMRI_lh_brain_knee/conversion.log|不是当前整个kspace总量|
|fastMRI|历史转换失败|1|体积任务|failed_conversions.txt及conversion.log|<RAW_ROOT>/fastMRI_lh_brain_knee/failed_conversions.txt|当前副本是否修复待核验|
|derived|一级分支|18|目录|本次完整列表|<RAW_ROOT>/derived|另有2个日志；18分支均已查看一级内容|
|derived/nnunet/raw|任务目录|8|任务|本次完整列表|<RAW_ROOT>/derived/nnunet/raw|Dataset901–908|
|derived/aur|已有SSL清单|6130|序列记录|读取既有划分报告|<RAW_ROOT>/derived/aur/ssl_manifest_v1/split_leakage_report.json|训练5501，验证629，非全库总数|

## 六、风险与待核验事项

|优先级|事项|证据|处理重点|来源|
|---|---|---|---|---|
|高|空占位目录|CMRxRecon_ty、TotalSegmentator_MRI_cbb、fastMR_lh_bran_knee_rawdata均为空|按实际非空路径建索引；不要按目录名宣称已拥有数据|<RAW_ROOT>|
|高|AMOS测试标签缺失|labelsTs为空；imagesTs有240条目|有监督验证/测试不得误读空标签；区分有标签与无标签集合|<RAW_ROOT>/AMOS_cbb/AMOS22/amos22/labelsTs|
|高|SKM-TEA截断副本|<CASE_ID>.h5.truncated、<CASE_ID>.h5.truncated|排除.truncated；另验同名正常H5可读性、内部数组及校验|<RAW_ROOT>/SKM-TEA/files_recon_calib-24|
|高|fastMRI历史转换失败|failed_conversions.txt记录1个H5截断错误|核验当前H5及对应DICOM是否修复；不要把历史日志当当前全库健康证明|<RAW_ROOT>/fastMRI_lh_brain_knee/failed_conversions.txt|
|高|病灶框不是体素真值|SKM-TEA 3D bbox；fastMRI 2D bbox/study-level标签|框内外都不能直接视为精确异常/正常体素标签；训练策略区分强弱监督|<RAW_ROOT>/SKM-TEA_ltr/annotations/v1.0.0/train.json|
|高|MS/Norm重复编号|SibBMS两组均使用sub-<CASE_ID>等名称；MS93、Norm100|主键至少包含dataset/cohort/subject/session；核查既有合并清单是否保留组别|<RAW_ROOT>/SibBMS_ms/sibbms/Output|
|高|患者级泄漏风险|HCP双对比、OASIS纵向、BMSR重复扫描、多个数据集复用数值ID|按带来源命名空间的患者ID划分；同一患者所有序列/访视绑定|HCP manifest.tsv；OASIS-2目录；BMSR扫描目录；既有SSL划分报告|
|高|伪标签/预测不等于专家真值|derived下含SynthSeg、推理、缓存、nnU-Net/nnDetection等|记录标签来源、生成模型、版本和置信度；独立保留人工评测真值|<RAW_ROOT>/derived|
|中|模态与合成图像混入|AMOS含CT；BMSR T2Synth是合成T2|MRI专用任务显式筛选AMOS MRI；实采与合成通道分开标记|AMOS readme.md；UCSF-BMSR CHANGELOG.md|
|高|标签版本/编码冲突|BMSR原始seg与BraTS-seg标签含义不同|使用明确的label-map，不能按相同整数混并|<RAW_ROOT>/UCSF-BMSR_cbb/UCSF-BMSR/CHANGELOG.md|
|高|坐标/配准未核实|ISLES原生空间；SKM-TEA dicom-track/raw-data-track并存|核查shape、spacing、affine、方向、配准及变换链后再对齐训练|ISLES README；SKM-TEA_ltr/segmentation_masks|
|中|计数单位不一致|目录条目、序列体积、切片、访视、患者、归档不能相加|逐指标保留单位；不汇总成未经去重的总病例数|本报告数量与证据表|
|中|元数据解析/文件名对应|KMAR CSV含引号/空格及_0 vs _0.0；PDGM 004 vs 0004；glossary非UTF-8|建立规范化映射并记录编码；以实际路径/主键核验配对|KMAR TrainingCohort.csv；PDGM metadata目录|
|中|空子目录及标注覆盖|OASIS freesurfer为空；SibBMS Annotation仅10目录；derived handoff为空|不要按文件夹名称声称已有完整标注或交付成果|对应空目录与Annotation列表|
|中|归档与解压副本重复|多个数据集同时保留zip/tar.gz和解压数据|统计空间时分开计量；去重前不删除归档|压缩包清单|
|中|已有SSL报告的范围|6130条为已有5来源选入序列；ok=true为历史报告字段|不要把它当整个FM_data当前无泄漏或完整可读的证明|<RAW_ROOT>/derived/aur/ssl_manifest_v1/split_leakage_report.json|
|信息|没有全量完整性认证|本次未远程执行脚本、未全量解码影像、未计算总容量/哈希|进一步审计需文件级manifest、HDF5/NIfTI读取、配对及患者去重；本报告不虚报这些结果|本次工具能力及执行范围|

## 七、已观察到的27个压缩包

下表为目录stat返回的逻辑大小。未解压校验、未验证归档校验和；总和不等于全库占用，更不等于有效训练数据体积。

|数据集|相对根目录路径|字节数|GB（十进制）|
|---|---|---:|---:|
|AMOS|`AMOS_cbb/AMOS22/amos22.zip`|24,234,336,519|24.234|
|ISLES|`ISLES_ltr/ISLES-2022.zip`|1,692,717,470|1.693|
|KMAR|`KMAR-50K_ty/95w9f5tzz8-6.zip`|10,149,397,000|10.149|
|KMAR|`KMAR-50K_ty/xw7mrg7ntg-6.zip`|10,445,133,855|10.445|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc1.tar.gz`|1,380,527,970|1.381|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc2.tar.gz`|1,403,357,161|1.403|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc3.tar.gz`|1,343,992,875|1.344|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc4.tar.gz`|1,333,582,060|1.334|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc5.tar.gz`|1,362,916,370|1.363|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc6.tar.gz`|1,364,560,560|1.365|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc7.tar.gz`|1,390,914,852|1.391|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc8.tar.gz`|1,332,341,152|1.332|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc9.tar.gz`|1,292,937,504|1.293|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc10.tar.gz`|1,294,865,012|1.295|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc11.tar.gz`|1,247,923,584|1.248|
|OASIS-1|`OASIS_cbb/OASIS-1/raw/oasis_cross-sectional_disc12.tar.gz`|1,473,055,117|1.473|
|OASIS-2|`OASIS_cbb/OASIS-2/raw/OAS2_RAW_PART1.tar.gz`|10,874,461,059|10.874|
|OASIS-2|`OASIS_cbb/OASIS-2/raw/OAS2_RAW_PART2.tar.gz`|8,449,289,603|8.449|
|SKM-TEA|`SKM-TEA/files_recon_calib-24.tar.gz`|819,739,559,552|819.740|
|SKM-TEA_ltr|`SKM-TEA_ltr/image_files.tar.gz`|17,234,000,891|17.234|
|SPIDER|`SPIDER_cbb/SPIDER/images.zip`|3,700,562,886|3.701|
|SPIDER|`SPIDER_cbb/SPIDER/masks.zip`|58,222,897|0.058|
|SibBMS|`SibBMS_ms/sibbms.zip`|10,971,922,363|10.972|
|TotalSegmentator MRI|`TotalSegmentor_MRI_cbb/TotalsegmentatorMRI_dataset_v200.zip`|5,100,514,630|5.101|
|UCSF-BMSR|`UCSF-BMSR_cbb/UCSF-BMSR/UCSF_BrainMetastases_v1.3.zip`|8,133,241,929|8.133|
|UCSF-PDGM|`UCSF-PDGM_lh/metadata/UCSF-PDGM_DTI-bval.zip`|196|0.000|
|UCSF-PDGM|`UCSF-PDGM_lh/metadata/UCSF-PDGM_DTI-bvec.zip`|582|0.000|

## 八、对数据使用最重要的区分

已有数据同时包含器官/组织分割、病灶分割、病灶框、影像分级、临床分子字段、重建数据和派生预测，不能统一当作同一种监督信号。

尤其是：SKM-TEA病灶框与组织mask不是同一件事；fastMRI的框标注不能提供病灶精确边界；未标注区域不能自动当作严格正常组织；FSL/SynthSeg及其他模型结果需要单独记录来源。要评估“正常与异常组织体素级精确分离”，仍需明确每个病例的人工标签覆盖范围、体素空间及可靠正常区域。

尚待全量审计的内容包括：各病例序列齐全性、图像/标签文件名配对、HDF5键与shape、NIfTI/MHA/DICOM空间信息、mask类别与非空性、跨源/跨访视患者去重、训练/验证/测试泄漏，以及当前截断文件是否修复。本报告不将这些未执行项目标记为通过。
