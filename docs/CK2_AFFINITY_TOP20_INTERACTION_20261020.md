# CK2 全局 pIC50 去重前 20：interaction-aware 优化

2026-10-20 后续审计补充：本报告的互作保留率使用“原分子 Vina redocking pose”作为基准，
不等于相对原始 Steer pose 的保留率。`b011_final_043` 的前一口径为 10%，后一口径实测为
77.8%。同分子原始 pose→redocking 的差异、原版重构复核及提示消融见
[后续审计报告](CK2_TOP20_NATIVE_POSE_AUDIT_20261020.md)。原始实验结果未改写。

使用全部 1,000 个原 single-Steer 终态中的有效连通分子，按 canonical isomeric SMILES 去重。每图保留最高 pIC50 代表，再取前 20；不附加 PB、应力或 SA 门槛。961 个有效粒子对应 401 个独立图。

固定完整 3,332 条 SynCraft 参考语料、全队列检索排除、deepseek-flash、temperature=0、每例 K=1。interaction-aware 使用源分子重新 docking/PLIP 后的互作提示。执行原版图重构，关键互作未设硬冻结。

Vina 为原分子和有效产物各自重新构象化后独立 redocking 的最佳预测分数；相同 CK2 受体/盒子、seed=42、exhaustiveness=8、单 CPU。原始 pIC50 不复制给产物。2D 结构图仅为分子图展示；导出的 SDF 是实际 Vina 对接坐标。

本轮主评价是 SA 辅助指标与 Vina，不以 SA 改善宣称逆合成成功。起初完整流程联调的 SimpRetro 并行任务发生内存不足（exit 137），单独保留；主实验明确关闭逆合成，所有 not_run 均不解释为不可合成。

此前接口联调另有 2 次真实 API 调用，保留在 `outputs/ck2_affinity_top20_interaction_20261020/`。它们不计入主实验的 20 次调用，不与主实验候选合并，也不作为独立科学重复。大环 G0–G3 是 Meeko 闭环虚拟点，修复后仅从 PLIP 复合物中排除这些虚拟点，真实 CG0–CG3 碳原子仍要求映射完整。

20 个主实验案例均已结束。一个产物虽可解析，但原版 RDKit 构象生成失败（Bad Conformer Id），因此没有产物 Vina；该错误使整体 manifest 保持 partial，并非仍有案例在运行。

图接受不等于选优通过。代理筛选要求相似度>0.2、Vina 恶化≤1 kcal/mol，再按 SA 排序；另统计 SA 确实改善且通过结合预算的分子数。无法解析产物直接跳过，失败后没有化学重采样。

## 统计

- source_count: 20
- proposal_count: 20
- source_vina_count: 20
- paired_vina_count: 6
- valid_changed_graph_count: 7
- sa_improved_count: 5
- proxy_selected_count: 2
- sa_improved_with_binding_budget_count: 2
- sa_and_vina_both_improved_count: 0
- unparseable_product_count: 12
- unchanged_count: 1
- conformer_generation_failure_count: 1

## 全部结果

SA 越低越好；ΔVina=产物−原分子，正数表示预测结合变差。Vina 单位 kcal/mol。

| 排名 | 源分子 | 原 pIC50 | SA 原→新 | Vina 原→新 | ΔVina | 结果 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `single_b011_final_030` | 8.7048 | 2.3823→— | -11.867→— | — | SynCraft product_loading: Invalid or empty molecule |
| 2 | `single_b004_final_027` | 8.6259 | 2.8661→— | -11.805→— | — | SynCraft product_loading: Invalid or empty molecule |
| 3 | `single_b017_final_021` | 8.5844 | 2.9783→5.3857 | -10.986→— | — | conformer_generation_failed |
| 4 | `single_b011_final_027` | 8.5753 | 2.2758→2.1550 | -10.772→-10.258 | 0.514 | 代理筛选通过 |
| 5 | `single_b004_final_002` | 8.3799 | 2.8582→— | -11.809→— | — | SynCraft product_loading: Invalid or empty molecule |
| 6 | `single_b011_final_005` | 8.3557 | 3.3676→1.7147 | -11.167→-8.109 | 3.058 | vina_loss_budget |
| 7 | `single_b007_final_026` | 8.3541 | 2.3285→1.8781 | -12.319→-10.065 | 2.254 | vina_loss_budget |
| 8 | `single_b013_final_014` | 8.3493 | 2.2590→— | -11.611→— | — | SynCraft product_loading: Invalid or empty molecule |
| 9 | `single_b011_final_040` | 8.3394 | 2.8350→— | -11.982→— | — | SynCraft product_loading: Invalid or empty molecule |
| 10 | `single_b017_final_043` | 8.3351 | 2.5119→— | -12.254→— | — | SynCraft product_loading: Invalid or empty molecule |
| 11 | `single_b011_final_017` | 8.3245 | 4.0512→2.7640 | -10.816→-7.459 | 3.357 | vina_loss_budget |
| 12 | `single_b007_final_028` | 8.3060 | 2.3366→— | -12.016→— | — | SynCraft product_loading: Invalid or empty molecule |
| 13 | `single_b015_final_048` | 8.3001 | 2.6822→— | -12.257→— | — | SynCraft product_loading: Invalid or empty molecule |
| 14 | `single_b003_final_025` | 8.2951 | 2.3578→— | -10.589→— | — | SynCraft product_loading: Invalid or empty molecule |
| 15 | `single_b011_final_000` | 8.2926 | 2.4754→— | -11.533→— | — | SynCraft product_loading: Invalid or empty molecule |
| 16 | `single_b009_final_001` | 8.2846 | 2.3787→— | -11.446→— | — | SynCraft product_loading: Invalid or empty molecule |
| 17 | `single_b004_final_010` | 8.2811 | 2.9945→— | -11.615→— | — | SynCraft product_loading: Invalid or empty molecule |
| 18 | `single_b011_final_043` | 8.2798 | 2.4163→2.2521 | -11.185→-10.973 | 0.212 | 代理筛选通过 |
| 19 | `single_b011_final_016` | 8.2788 | 3.9817→— | -11.197→— | — | graph_not_accepted |
| 20 | `single_b000_final_034` | 8.2771 | 4.0221→4.3532 | -10.465→-5.039 | 5.426 | similarity_threshold |

## 通过代理筛选的方案及互作保留

| 分子 | SA 原→新 | Vina 原→新 | 精确接触保留率 | 类型/残基 recall |
|---|---:|---:|---:|---:|
| `single_b011_final_027` | 2.2758→2.1550 | -10.772→-10.258 | 8.3% | 71.4% |
| `single_b011_final_043` | 2.4163→2.2521 | -11.185→-10.973 | 10.0% | 44.4% |

选优只应用声明的相似度和 Vina 预算，不额外按互作保留率拒绝。两项获选方案的精确接触保留率仍低，不能将 SA 和 Vina 代理筛选通过解释为已保留原结合模式或已验证可合成性。

完整编辑指令、LLM 解释、失败原因、互作保留与原始结果哈希见运行目录的 `comparison.json` / `comparison.csv`；结构对照见 `molecules.html`。

运行目录：`D:\llm_controlled_generation\MolSteer\AgenticGen\outputs\ck2_affinity_top20_sa_vina_20261020`。输入：`D:\llm_controlled_generation\MolSteer\AgenticGen\examples\ck2_affinity_top20\cases.json`。
