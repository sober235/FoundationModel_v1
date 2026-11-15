# 冻结划分：v51_isles_u_b2

只含计数、规则与哈希；不含样本或患者 ID。

- 规则：stratified；组（患者）数 250，样本数 250
- 被曝光阻断而移出 locked_test 的组：37（移到 `excluded`）
- `split_sha256`：`02bc40f0fbb63c9b77061c8db21b584483729b954aa66417812d5ed32a790a1b`

- fractions：`{'development_val': 0.15, 'locked_test': 0.15, 'train': 0.7}`
- min_stratum_groups：`5`
- seed：`20261108`
- small_stratum_rule：`merge`

## 各划分组数

| 划分 | 组数 |
| --- | --- |
| development_val | 38 |
| excluded | 37 |
| locked_test | 0 |
| train | 175 |
| unassigned | 0 |

## 分层

病灶体积三分位阈值（mL）：3.368、14.904

| 层 | 组数 |
| --- | --- |
| volume_t1 | 84 |
| volume_t2 | 83 |
| volume_t3 | 83 |
