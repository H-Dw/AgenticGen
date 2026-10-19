# CK2 TOP3 工程示例

本例使用已有 CK2 原生 steered sampling 分析中冻结的 `top10_unique_graphs` 排名 1–3。`selected_candidates.csv` 同时包含三个不同 cohort，准备器明确过滤 `top10_unique_graphs`，避免将重复粒子或不同排名定义混入示例。

| 排名 | candidate ID | 预测 pIC50 | 原分析 SA | MMFF 局部松弛释放能/重原子 | 原分析 strain 字段 |
| --- | --- | --- | --- | --- | --- |
| 1 | `single_b004_final_027` | 8.625889 | 2.866127 | 0.631318 kcal/mol/重原子 | 18.044877 kcal/mol |
| 2 | `single_b017_final_021` | 8.584398 | 2.978317 | 0.389233 kcal/mol/重原子 | 13.538420 kcal/mol |
| 3 | `single_b007_final_026` | 8.354063 | 2.328472 | 0.567469 kcal/mol/重原子 | 23.135149 kcal/mol |

质量组来自先前分析：有效分子预测亲和力前十分位、完整 PoseBusters 通过、MMFF 松弛收敛，且每重原子局部松弛释放能不超过原有效群体的 Q75。表中保留的 `strain_MMFF94s_kcal_mol` 与局部松弛释放能不是同一个字段；本示例没有重新计算或混合这些三维指标。所谓 TOP3 指冻结质量组中的亲和力排名，不代表所有分子中应力最低的三例。

- `cases.json`：原子图、来源指标、实际 sampler seed、原始 slot、源文件哈希。
- `candidates.sdf`：三例原始解码坐标；没有优化、平移或重新构象化。
- `config.json`：默认真实 `deepseek-flash`，每例 1 个提案；`retro=disabled`。
- `replay.json`：人工指定的 N→C 编辑测试 fixture，不是 LLM 输出，不证明可合成性改善。

复建到新目录：

```powershell
agenticgen prepare-ck2 --workspace D:/llm_controlled_generation --output outputs/rebuilt_ck2 --count 3
```

准备器验证冻结 rank、质量标签、SMILES/SDF 图一致性，并保存原始 SDF 记录。`cases.json` 中源文件位置相对于工作区，运行配置中路径相对于配置文件；二者含义不同。

请按[运行指南](../../docs/RUN_GUIDE.md)运行。2026-10-15 三次真实 API 调用的结果见[验证报告](../../docs/VALIDATION.md)。三例是已选出的高质量工程样本，不是随机测试集，不能估计总体优化成功率或历史信息效用。
