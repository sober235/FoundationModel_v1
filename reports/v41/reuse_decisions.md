# T00 — 接手审计与复用裁决（匿名版）

日期：2026-11-02。性质：只读审计；未运行训练、推理或旧诊断；未改动旧仓库；未申请 GPU。
私有详细审计：`$PRIVATE_ROOT/audits/t00_reuse_audit/`（本仓库只保留匿名汇总）。

## 1. 研究目标（以 v4.1 为准）

同一 MRI 观察上独立验证体素级解剖身份 A、目标异常范围 U、病灶—解剖关系 R，以及有明确目标范围的“未见异常”；同一 checkpoint 支持独立二维与三维输入，二维/三维分别验收。
参照资格优先于模型：Dataset902 框填充、SynthSeg 伪 A、几何 R 只提供 `engineering_status`，不提供 `scientific_conclusion`。

依据（sha256 前 8 位）：Proposal `6bf1ed3f`、Contract `12205e46`、plan.md（本仓库首次提交版本）`b88598a6`、原盘点 `42cb8d18`（私有）。

## 2. 当前工程状态

| 项 | 结果 |
|---|---|
| 新仓库 | `sober235/FoundationModel_v1`，首次提交仅含四份文档与 `.gitignore`，无代码 |
| 旧仓库只读对象 | `sober235/Foundation_Model` 提交 `4deeead`；本地副本 HEAD 相对 4deeead 有 10 个文件 / +878 行的文档型增量，工作区干净；本审计只按 4deeead blob 读取 |
| 旧 CPU 测试套件 | `tests/` 207 个 `test_*.py`；交接文件记录约 1512–2080 passed / 5 skipped（UNVERIFIED，本次未运行） |
| 解释器 | `~/anaconda3/envs/nvgen/bin/python` 3.11.15；torch 2.5.1+cu121、numpy 2.4.4、scipy 1.17.1、nibabel 5.4.2、h5py 3.16.0、pydicom 3.0.2、monai 1.5.2、pytest 9.1.1、PyYAML 6.0.3；`jsonschema` 缺失（不升级环境，T01 自带最小校验器） |
| 旧仓库环境约定 | `PYTHONNOUSERSITE=1 PYTHONPATH=.`，测试加 `CUDA_VISIBLE_DEVICES=""`；沿用 |
| 私有输出根 | `/data2/congcong/code/FM_data`（mode 700，所属为当前用户，不在任何 Git 仓库内；可用空间 18T） |
| GPU（只读） | 8 卡，其中 5 卡有他人占用；`CUDA_VISIBLE_DEVICES` 未设置；T00–T08 不申请计算 |
| 旧完成证书/登记 | 旧仓库记录 run_v5 正式训练完成、最终评价未执行：**UNVERIFIED**（未读取私有 runtime，仅引用仓库文本） |

## 3. 复用裁决

裁决含义：REUSE＝按原文复制进 `anatobind/v41/` 并在源码注明 blob 哈希；WRAP＝复制核心逻辑并改接口/去耦合；REIMPLEMENT＝只借思路重写；LEGACY_ONLY＝不进入 v4.1 代码。新仓库不跨仓库 import。

| 组件 | 旧路径 @4deeead（blob） | 裁决 | 输入本体 | 初始化暴露 | 数值解码 | 评价角色 | 可验证证据 |
|---|---|---|---|---|---|---|---|
| `discrete_labels` + `LABEL_ROUND_ATOL=1e-5` | `anatobind/aur/labels.py` (303c510e) | REUSE（仅该函数） | 无本体耦合 | 无 | 近整数修复，拒绝概率图/非有限值/溢出 | 标签读入前置 | 旧 `test_aur_labels.py`；T05 重写等价测试 |
| 同文件 entity/host LUT（32 实体/13+1 host） | 同上 | LEGACY_ONLY | SynthSeg 脑本体 | — | — | — | 与 SKM A6 不可统一为同一 ID 表 |
| 裁剪/体素→世界坐标 | `anatobind/aur/crops.py` (ffe83ba6) | WRAP（去 `bias_field` 依赖） | 无 | 无 | 全 4×4 affine，无 0.5 偏移 | 几何工具 | 旧 `test_aur_crops.py` |
| 1 mm 重采样 | `anatobind/aur/resample.py` (3607a153) | WRAP（行 schema 需改） | 无 | 无 | 标签最近邻 + `discrete_labels` | 派生数据，不覆盖源 | 旧 `test_aur_resample.py`；v4.1 禁止薄软骨默认降到 1 mm |
| 脑 host 几何/距离 | `anatobind/eval/geometry.py` (3a1624df) | REIMPLEMENT | SynthSeg + fastMRI 2D 框 | — | — | R_geo | 仅 EDT 思路可移植 |
| 2× 面内降采样 | `anatobind/data_engine/resample.py` (65b49604) | LEGACY_ONLY（需要时再评） | 无 | — | 标签平局归标签 | — | 旧 `test_resample.py` |
| DICOM 读入 | `anatobind/aur/dicom.py` (1d2cab63) | WRAP（T10 之后） | 无 | — | 私有日志过滤 | 输入 | 旧 `test_aur_dicom.py` |
| checkpoint 保存/严格加载 | `anatobind/aur/ssl/checkpoint.py` (afc49f68) | WRAP（改 repo-root 假设，`weights_only=True`） | 无 | 记录 code_sha | — | 训练控制 | 旧 `test_aur_ssl_train.py` |
| 全量 RNG 捕获/恢复、原子 torch 保存 | `unified_train.py` (6230a2d2) / `unified_run.py` (457c83d0) | REUSE 三个小函数；恢复校验 REIMPLEMENT | 无 | — | — | 训练控制 | 旧 `test_aur_unified_run.py` |
| 训练控制器/登记/完成证书 | `unified_launcher.py` (6805fc50)、`unified_artifacts.py` (16b45901)、`scripts/aur_formal_supervised.py` (3364d30a) | REIMPLEMENT（借：不可变 JSON、`steps.jsonl`+`status.json`、execution_id、证书哈希集、`PrivateParser`） | 绑定旧 stage I/II/III | 旧 checkpoint 暴露史 UNVERIFIED → 不得进入严格 inductive 主结果 | — | T11 | 旧 `test_aur_unified_launcher.py` |
| 双模式输入适配 + Swin 骨干 | `anatobind/aur/unified.py` (44cf282f)、`swin.py` (1926c84f)、`rope.py` | WRAP（T22） | 骨干无本体 | 旧权重暴露史 UNVERIFIED | — | 模型 | 旧 `test_aur_unified_model.py` |
| 实体/事件/关系头 | `aur/model.py` (4daaa489)、`heads.py` | LEGACY_ONLY | 32 实体/14 host/64–128 事件 | — | — | — | 与 A6/多标签 U 不兼容 |
| 私有路径校验/原子 JSON/哈希/脱敏解析器 | `aur/data_contract.py` 44–92 行 (c29f2570)、`PrivateParser` | REUSE（抽出函数，`private_root` 显式传参，去模块级常量） | 无 | — | — | T01 | 旧 `test_aur_data_contract.py` |
| `data_contract.py` 其余 | 同上 | LEGACY_ONLY | 旧行 schema、硬编码路径 | — | — | — | — |
| 泄漏报告/患者分组/分层折 | `aur/ssl/samples.py` (bf0f3d79)、`data_contract.py`、`data_engine/splits.py` (c966bb55) | WRAP（T03；键名改为 v4.1） | 旧来源名 | — | — | 划分 | 旧 `test_aur_ssl_samples.py`、`test_splits.py` |
| SKM-TEA 引擎 | `data_engine/skmtea.py` (ed8f3e2d)、`nnunet/lesion_labels.py` (f3f6c4d0)、`docs/data_engine_skmtea.md` (c13e8080) | WRAP（T10；`normalise_box`、`seg_nifti_to_h5_frame`、重建、`boxes_to_label_map`） | A6 六类顺序需按源实查，不按整数猜 | — | bbox `[x,y,z,w,h,d]` 可为负需排序 | 框只生成定位目标 | 旧 `test_skmtea_*.py` |
| ISLES/PDGM/BMSR/SibBMS 表与同网格检查 | `nnunet/brain_disease.py` (c2e760ef)、`aur/samples.py` (4165d99d)、`anatomy/sources.py` | WRAP 序列/病灶值表与 `check_same_grid`；路径发现 REIMPLEMENT | SynthSeg 派生布局 | — | — | T10 | 旧 `test_aur_samples.py` |
| Dice / IoU 匹配 / 患者 bootstrap | `aur/eval.py` (bbf86a26)、`eval/matching.py` (f30951e9)、`aur/compare.py` | REUSE 小函数（`_dice`、`_dice_from_ids`、`iou3d`、`match`、`patient_bootstrap`） | 无 | — | — | T08 | 旧 `test_aur_eval.py` 等 |
| NSD / HD95 / 表面距离 | 旧仓库不存在 | REIMPLEMENT（T08；可用已装 monai 1.5.2 或自写） | — | — | — | T08 | 无旧证据 |
| 双模式 A/U/R 评价器 | `aur/unified_evaluation.py` (b53733cc) | LEGACY_ONLY | 32 实体/14 host | — | — | — | — |

## 4. 新科研验收与旧工程门的差异

- 旧门：伪参照（SynthSeg A、框填充 U、几何 R）通过、同网格一致、训练完成证书。新门：同例独立专家参照；Dataset902 框填充不得注册为 expert_voxel；`inspected=true`/空 mask 不自动给阴性资格；2D/3D 分别验收。
- 旧 A32（SynthSeg 实体）与 SKM A6 是两个本体，T04 分别版本化，不合并 ID 表。
- 旧 checkpoint 对现有测试影像的暴露史未核验：凡以旧权重初始化的模型只作兼容性对照，不进入严格 inductive 主结果。

## 5. 计划新增资产

按 plan.md §2：`anatobind/v41/`、`configs/v41/`、`schemas/v41/`、`tests/v41/`、`reports/v41/`；T01–T08 仅合成测试，不接触锁定测试。

## 6. 未核验事项（UNVERIFIED）

1. 旧仓库私有 runtime、run_v5 完成证书与登记哈希（未读取私有目录）。
2. 旧 CPU 测试套件当前通过数（未运行）。
3. 旧 checkpoint 的训练/验证/测试影像暴露史。
4. 旧仓库远端 `main` 是否仍指向 4deeead（`git ls-remote` 因网络失败）。
5. SKM-TEA 分割通道顺序、ISLES DWI/ADC 与 mask 配准：待 T04/T05 实查，不沿用旧文档整数。
