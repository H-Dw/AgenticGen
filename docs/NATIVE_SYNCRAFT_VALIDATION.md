# 原版 SynCraft 图重构接口与 CK2 复测

日期：2026-10-18。按用户要求移除 AgenticGen 的独立严格图审计，统一改用原版 SynCraft 执行语义。上游固定版本仍为 `557703e110b26a354afed2955ded5217837f6692`，未修改其源码。

## 接口变化

`factory.py` 和 `base_factory.py` 均使用 `SynCraftEditor`。`StrictSynCraftEditor` 与独立 `_reference_edit` 已从生产代码删除，不再保留严格模式开关。主配置 `editor` 默认为 `syncraft`；显式旧值 `strict` 需更新，历史结果的汇总仍能读取原协议标签。

适配器只调用上游 `reconstruct_from_sequence`，遵循其先统一删除原子、再应用其他编辑的行为：部分不存在的端点/键操作被忽略，重复加键被忽略，断开图可能只返回最大分子片段。适配器不再按独立参考图、操作/原子数量、保护邻域、立体化学或同位素保留规则拒绝原版产物。`constraints` 保留为提示指导，最终相似度等条件由选择协议检查。

原版有时在 sanitize 失败后仍返回 SMILES。若分子工具无法读取该输出，则记录 `error_stage=product_loading` 并保留原始字符串，不重构替代图、不调用后续性质工具。上游抛异常时记录 `upstream_reconstruction`。这两个状态是执行/数据错误，并非恢复了独立图编辑审计。

结果新增 `reconstruction_policy`、`upstream_smiles`、`upstream_mapped_smiles`、`was_fragmented`、`discarded_atom_maps`、`error_stage`。`deleted_atom_maps` 描述实际损失的源原子，`discarded_atom_maps` 另列裁片造成的非显式删除。JSON schema、运行 manifest、汇总和提示语义同步更新。已有实验结果未改写，旧严格执行源码仅在历史审计输出目录存档。

## 自动化与历史提案复测

51 项自动化测试通过、0 skipped。覆盖原版跳过不存在键/重复加键/已删原子操作、最大分子片段、实际原子损失、提示范围不作二次否决、原始输出保留、无法读取产物的完整运行路径，以及历史汇总协议标签。Ruff 检查及格式检查通过。

原样重放五批真实实验已保存的 26 条提案，并逐条与直接调用原版执行器核对。没有新增 LLM、docking 或逆合成任务；只重新计算可读取产物的 SA。

| 原实验 | 提案数 | 旧接受数 | 原版接受数 | 原版拒绝数 | 本批独立产物 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 TOP3 局部编辑 | 3 | 2 | 2 | 1 | 2 |
| 六分子 SA 分层筛查 | 6 | 5 | 6 | 0 | 6 |
| TOP1 standard/pass-5 | 5 | 3 | 4 | 1 | 1 |
| 3,272 条语料双模式 | 8 | 2 | 2 | 6 | 1 |
| 3,332 条语料双模式 | 4 | 1 | 3 | 1 | 3 |
| 合计 | 26 | 13 | 17 | 9 | 不跨实验累加独立产物 |

新增四条接受提案：

| 原实验/分子/提案 | 接受变化原因 | SA：源→新 |
| --- | --- | ---: |
| 六分子 / b000_final_034 / P0 | 原版允许删除六原子，不执行旧宿主两原子上限 | 4.0221→2.8248 |
| standard/pass-5 / b004_final_027 / P3 | 原版裁掉漏删的 O22 片段；与同批其他去酰胺产物重复 | 2.8661→2.7550 |
| 完整语料 / b004_final_027 / P1 | 原版忽略重复添加的 9–10 键 | 2.8661→3.4419 |
| 完整语料 / b011_final_017 / P0 | 原版忽略涉及已删除原子的键操作 | 4.0512→5.2074 |

接受数增加是接口行为变化，其中两项 SA 反而变差。剩余九项为原版输出无法通过分子工具读取，不能将这些返回字符串计为有效分子。

逐提案原始输出、比较和来源哈希：

- [复测摘要](../outputs/ck2_native_retest_20261018/summary.json)
- [26 条结果](../outputs/ck2_native_retest_20261018/results.json)
- [比较表](../outputs/ck2_native_retest_20261018/comparison.csv)

## 新的真实 CK2 双模式实验

当前运行目录为 `outputs/ck2_native_live_20261018/`。本轮采用完整 3,332 条参考语料、默认 deepseek-flash、每模式 K=1、两个源分子；Vina/PLIP 与 SimpRetro 全模板、官方 eMolecules 库存保持原配置，搜索预算仍为每分子 120 秒、单核。生成历史、steer 和 inpainting 关闭。

新实验除执行语义外，也更新了说明原版行为的宿主提示；因此是接口集成验证，不是仅改变执行器的严格配对性能比较。仅执行器变化的配对结果以上述 26 条固定提案回放为准。

运行已完成，manifest 为 `completed`。4 次真实 API、4 次 HTTP 请求、无接口错误；3 个可读取产物、1 个产物读取失败；实际执行 5 次 Vina/PLIP 和 5 次逆合成任务。没有化学失败重采样。Provider 报告输入 14,394 / 输出 2,203 tokens。

| 源分子 / 模式 | 原版产物 | SA：源→新 | Vina：源→新，kcal/mol | 选择结果 |
| --- | --- | ---: | ---: | --- |
| TOP1 / standard | 去酰胺 | 2.8661→2.7550 | −11.805→−11.248 | 合格，SA 次于去甲基产物 |
| TOP1 / interaction-aware | 去甲基 | 2.8661→2.7269 | −11.805→−11.561 | 本例胜出 |
| b011_final_017 / standard | 大幅简化为 `O=C(O)c1ccncc1` | 4.0512→1.6139 | −10.816→−5.513 | 相似度 0.1607，不合格；Vina 也超出损失预算 |
| b011_final_017 / interaction-aware | 原版返回不可读取产物 | — | — | `product_loading`，未进入性质工具 |

TOP1 去甲基产物精确保留 6/10 接触：保留 N21–VAL116 氢键及 HIS160 堆积，丢失 O22–VAL116 氢键；不是完整互作保留。去酰胺产物保留 5/10，丢失两项 VAL116 氢键。PLIP 的类型/残基 recall 与精确原子接触保留分开记录。

挑战分子的标准产物直接命中配置库存，因此其 `route_verified=true` 是零步库存可达检查，`reactions=[]`，不涉及非零反应验证。该产物结构保留和结合约束不通过，不能计为有效路线救回。两个原分子、TOP1 两个产物均在 120 秒预算内未找到路线，本轮 `route_validated_selected_case_count=0`、`verified_route_rescue_case_count=0`。预算内未找到不等于真实不可合成。

新接口允许原版得到可评价产物，最终选择仍然依赖独立的结构、结合和路线协议。本轮没有裁片产物；裁片行为由历史回放和完整运行测试覆盖。

- [本轮 manifest](../outputs/ck2_native_live_20261018/manifest.json)
- [本轮 summary](../outputs/ck2_native_live_20261018/summary.json)
- [TOP1 原始结果](../outputs/ck2_native_live_20261018/cases/single_b004_final_027/result.json)
- [挑战分子原始结果](../outputs/ck2_native_live_20261018/cases/single_b011_final_017/result.json)

## 重跑

在 AgenticGen 根目录执行，输出必须选择尚不存在的目录：

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/retest_syncraft_native.py --output outputs/native_regression_new
./scripts/run_ck2_official.ps1 -Output outputs/native_ck2_new
```

历史回放不消耗 API；最后一条命令进行新的真实调用及分子评价。库存仍为 eMolecules，不是 Enamine。图接受、SA 降低、Vina 阈值通过和路线救回分别报告。
