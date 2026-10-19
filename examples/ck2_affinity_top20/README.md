# CK2 全局 pIC50 去重前 20：interaction-aware

本队列直接来自原 single-Steer 的全部 1,000 个终态。961 个有效连通粒子按
canonical isomeric SMILES 去重，得到 401 个化学图。每图保留 pIC50 最高的
粒子，随后取前 20；精确同分时按粒子 ID 排序。不增加 PB、应力或 SA 门槛，
不合并互变异构体或质子化状态。原始 SDF 块逐字节保留，输入在 API 调用前冻结。

- `cases.json`、`candidates.sdf`：20 个实际输入及来源、坐标、文件哈希。
- `selected_candidates.csv`、`selection.json`：完整入选指标、排序和去重定义。
- `interaction.sa_vina.json`：本轮主实验；DeepSeek、SynCraft interaction-aware、
  原版图重构、SA、独立 Vina redocking 和 PLIP。每例 K=1，失败后不化学重采样。
- `interaction.json`：带 120 秒/分子完整库存逆合成的配置；本地并行联调出现
  Docker 内存不足，未用其错误结果评价可合成性。需要足够内存或单独安排路线任务。

运行主实验（先设置进程级 `DEEPSEEK_API_KEY`，不将凭据写入配置）：

```powershell
.venv/Scripts/python.exe scripts/run_foundation_batch.py --config examples/ck2_affinity_top20/interaction.sa_vina.json --output outputs/ck2_affinity_top20_new --workers 2
.venv/Scripts/python.exe scripts/report_affinity_batch.py --run outputs/ck2_affinity_top20_new --report docs/CK2_AFFINITY_TOP20_NEW.md
```

每个独立子进程仍用全部 20 个输入排除参考库的相同图。并行只改变任务调度，
不改变提示、提案预算、图执行、评分或筛选。各案例独立保存 manifest 和真实 API 收据。
父 manifest 保存完整队列哈希、执行案例及子运行哈希；`--collect-only` 仅重建汇总。

Vina 使用同一 CK2 受体、盒子、seed=42、exhaustiveness=8、单 CPU。
原分子和产物分别重新构象化并 redock；不是对原始 FLOWR pose 进行 score-only。
原始 pIC50 是入选依据，不是修改产物的亲和力结论。

代理筛选采用相似度>0.2、Vina 恶化≤1 kcal/mol，并按 SA 排序。报告额外列出
SA 实际改善的数量，避免将“可选中”解释成“优于原分子”。关键互作是提示要求，
并非原子或接触的硬冻结；记录 PLIP 保留率和删除的互作原子。

报告和 CSV 包含所有 20 个输入；无法解析、无法构象化或对接失败的产物不赋造假分数。
结构对照 `molecules.html` 是二维重绘，`source_docked.sdf`、`product_docked.sdf`
则来自实际评价的对接坐标。SA 辅助分数改善不代表已验证逆合成路线。
