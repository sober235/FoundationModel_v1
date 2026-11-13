# G-DATA 证据：isles_2022（isles2022_bids1.6.0）→ isles_u

只含计数与哈希；不含病例 ID、路径或文件名。数据使用批准由研究者签署，不在本文件中。

- scope_hash：`0ddcfb0347a2fd0a126393a4dc0f5227f4d3f2e58aab4e36b6d88f5074c5e5fb`；任务：U
- 来源核验摘要 sha256：`dc7658e36bee1e18cab42b3010504a4c42408f48e18c7757bff2c94fe1e62734`
- 冻结划分 split_sha256：`02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b`

| 证据项 | 判定 | 依据（计数） |
| --- | --- | --- |
| readability | PASS | 准备记录 250，失败 0 |
| semantics | PASS | 标签取值 QC：needs_review/ALL_ZERO/unexpected=False 3；pass/-/unexpected=False 247；通道核验 0/0 |
| identity | PASS | 划分覆盖样本 250/250，患者组 250 |
| pairing | PASS | incomplete 0；missing_label 0；paired 250 |
| task_geometry | PASS | 几何 pass 250；索引对齐 verified 250；物理几何 verified 250 |
| provenance | PASS | 迁移 250/250，失败 0；release 绑定 `ISLES-2022` |
