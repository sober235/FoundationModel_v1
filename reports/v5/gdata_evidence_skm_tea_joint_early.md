# G-DATA 证据：skm_tea_joint_early（skm_tea_annotations_v1.0.0）→ skm_joint_early

只含计数与哈希；不含病例 ID、路径或文件名。数据使用批准由研究者签署，不在本文件中。

- scope_hash：`f6e4b5f6e3c95a49f5f8cb45591b2a2c437b8db2d432551e1bd1e9e81935ec9f`；任务：joint
- 来源核验摘要 sha256：`a9b787bf285ba68e05f0d7f9a3beafcde02faf97369f51e8632a0c0dcfd13703`
- 冻结划分 split_sha256：`32820345ccef57fe3fc7a21ba87280f134db910d8d51c752ca0abc040c9c0fb9`

| 证据项 | 判定 | 依据（计数） |
| --- | --- | --- |
| readability | PASS | 准备记录 155，失败 0 |
| semantics | PASS | 标签取值 QC：pass/-/unexpected=False 155；通道核验 155/155 |
| identity | PASS | 划分覆盖样本 155/155，患者组 155 |
| pairing | PASS | incomplete 0；missing_label 0；paired 155 |
| task_geometry | PASS | 几何 pass 155；索引对齐 verified 155；物理几何 unknown 155 |
| provenance | PASS | 迁移 155/155，失败 0；release 绑定 `SKM-TEA v1.0.0` |

## 框导入（U 为框级，R 为组织级；并入 semantics、pairing 与 provenance 的判定）

- 框导入摘要 sha256：`488e50e3a5d526c5b553f59bd8226ba8a439b076f848f2f578a927ebba93e7ae`
- 框策略 `skm_box_reference_v1`，policy_sha256：`09b1003e63fb5e8517be7626ba7e522dba9a7397a53dcd4ef910226bc734f2a9`
- 官方划分文件：test 存在、train 存在、val 存在；登记扫描按来源划分：test 36、train 86、val 33
- 登记扫描 155，导入 155，失败 0
- 来源划分不是正式划分。本证据放行数据使用；训练、模型选择和评价仍按冻结划分的角色读取。
