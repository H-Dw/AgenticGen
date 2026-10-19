# CK2 官方材料示例：实际配置与验证

验证日期：2026-10-17。本报告从已保存的 manifest、result 和 summary 生成，不重新调用模型。

## 1. 材料与运行边界

已配置 SynCraft 论文 Figshare 完整 3,332 条参考语料、SimpRetro 51,720 个模板及过滤权重、用户选定的 SimpRetro 官方 eMolecules 库存、CK2 本地受体及 Vina/PLIP。库存为 23,081,629 条不同字符串；它不同于论文 Enamine 原料库，不将本次结果称为论文同协议复现。

| 材料 | 实际大小 MB（十进制） |
| --- | ---: |
| `official_stock` | 896.95 |
| `stock_index` | 1145.81 |
| `template_cache` | 646.62 |
| `templates` | 9.18 |
| `filter_weights` | 67.16 |
| `reference_corpus` | 6.18 |

来源 URL、精确哈希及镜像 ID 见 `configs/materials.official.lock.json`。库存索引采用上游默认的精确字符串 membership，不去盐、不合并立体异构体；模板指纹全部逐值确认可无损表示为 int16。两个库存加载点在隔离 worker 内替换为磁盘查询，未修改上游源码中的化学打分、过滤器或 MCTS。

本轮使用 `deepseek-flash`、双模式、每模式 K=1；每个独立分子的搜索预算 120 秒、单 CPU。全部模式提案在产物评价前冻结；生成历史增强关闭。这里是短预算工程/计算示例，不能估计总体性能。

## 2. CK2 配对结果

第一例为质量 TOP1，第二例为此前冻结的高 SA/高应力挑战组首例；第二例不是低应力优质集合。原始源分子的 affinity-head 分数与应力不复制到编辑产物；下表结合评价为重新运行的 Vina。

| 源分子 | 原 SA | 胜出者 SA | 原 Vina | 胜出者 Vina | 原路线状态 | 胜出者路线状态 | 模式 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| single_b004_final_027 | 2.8661 | 2.7550 | -11.8050 | -11.2480 | no_route_within_budget | no_route_within_budget | standard |
| single_b011_final_017 | 4.0512 | — | -10.8160 | — | no_route_within_budget | 无合格胜出者 | — |

代理胜出者须满足图审计、相似度>0.2、Vina 恶化≤1 kcal/mol，之后按 SA 升序选择。`route_validated_selection` 单独要求路线检查通过。SA 降低不计作路线救回，预算内未找到路线不等于不可合成。

| 运行计数 | 数值 |
| --- | ---: |
| `case_count` | 2 |
| `proposal_count` | 4 |
| `accepted_edit_count` | 1 |
| `unique_accepted_product_count` | 1 |
| `rejected_edit_count` | 3 |
| `interface_error_count` | 0 |
| `selected_case_count` | 1 |
| `route_validated_selected_case_count` | 0 |
| `verified_route_rescue_evaluable_case_count` | 2 |
| `verified_route_rescue_case_count` | 0 |
| `physical_retro_queries` | 3 |
| `physical_geometry_queries` | 3 |

无合格候选时保留失败，不删掉错误编辑指令后悄悄接受，也不重新抽取成功样本。

### single_b004_final_027 提案审计

| 提案 | 模式 | 状态 | 结果/拒绝原因 |
| ---: | --- | --- | --- |
| 0 | standard | accepted | 通过图审计；性质与路线独立评价 |
| 1 | interaction_aware | rejected | ADD_BOND needs an absent bond and a valid bond type |

### single_b011_final_017 提案审计

| 提案 | 模式 | 状态 | 结果/拒绝原因 |
| ---: | --- | --- | --- |
| 0 | standard | rejected | Edit references an absent or already deleted atom |
| 1 | interaction_aware | rejected | Edit references an absent or already deleted atom |

## 3. 独立逆合成工程对照

`outputs/official_full_template_control/`：完整模板及官方库存启动通过，库存内 CCO 为零步对照。

`outputs/official_nonzero_control/`：目标不在库存，规划器实际调用反应模型，找到单步路线；目标身份、路线图、库存叶节点和 RDChiral 模板回放通过。它是独立工程对照，不属于 CK2 优化结果。计算路线检查不覆盖反应条件、产率或真实实验成功。

## 4. 证据位置与重跑

- 本轮结果：`outputs/ck2_official_release_example/`；完整协议和计数见 `manifest.json`、`summary.json`。
- 每个案例：`cases/<case_id>/result.json`，并保留原始请求、响应、收据、编辑及搜索日志。
- 首次 3,272 条内置语料联调保存在 `outputs/ck2_official_dual_example/`；其每模式 K=2，与本轮不同，未混合候选或计数。
- 材料准备与运行命令：`examples/ck2_official/README.md`。

自动化验证：52 项测试通过、0 skipped；Ruff 检查与格式检查通过。包括双模式冻结/合并去重、接触指标口径、库存索引语义、材料完整性及既有图编辑/接口回归。

Enamine 论文库存仍需单独获授权并固定版本；无需因此重建大型商业近邻搜索。Flow 历史、steer 探索与 inpainting 迭代没有在本示例中运行。
