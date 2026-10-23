# CK2 原始 Steer 互作约束实验完成记录

核验时间：2026-10-23 09:10 北京时间。主控制器于 2026-10-23 06:39:10 北京时间正常退出（exit code 0）。最终维护队列已完成，账本与 manifest 均为 completed，无 pending/running/failed 任务。

## 覆盖与分母

- 固定20个原始Steer候选；20次设计均接受，共25次化学提案（5次重试）。复用已有DeepSeek high结果。
- 16个实验组，每组每分子100次尝试：共320个“分子×实验组”单元、8000个生成微批次、32000次尝试。
- 8000个评估微批次全部完成。28117个结果可解码，3883次解码失败；可解码不等于连通或构象合理。
- 27516个结果获得pIC50及Vina score-only；4484次尝试没有这些分数，其中包括3883次解码失败及601个已解码但无法评分的结果。未补样、去重或删除失败行。
- 全部最终评分使用实际生成构象；没有redock。原始Steer构象定义互作条件。窗口仍为0.1≤t<0.9，CE/Brier、两类条件及0.25/1/4权重网格不变。

## 描述性结果

每行分母均为2000次尝试。pIC50、Vina中位数仅针对该指标可评分的结果；PB列是PoseBusters通过数/2000，缺失检查不计作通过。不同组缺失率不同，不能把这些条件中位数直接解释为配对效果或统计显著优势。

| 实验组 | 可解码 | pIC50/Vina可评分 | PB通过 | pIC50中位数 | Vina中位数 kcal/mol |
|---|---:|---:|---:|---:|---:|
| fragment_graph | 556 | 207/207 | 0 | 5.916 | 18.887 |
| fragment_graph_pose | 538 | 288/288 | 0 | 5.944 | -7.718 |
| gradient_brier_graph_pose_w0p25 | 1937 | 1937/1937 | 1871 | 6.536 | -8.312 |
| gradient_brier_graph_pose_w1p0 | 1927 | 1927/1927 | 1850 | 6.480 | -8.148 |
| gradient_brier_graph_pose_w4p0 | 1927 | 1927/1927 | 1844 | 6.422 | -8.144 |
| gradient_brier_graph_w0p25 | 1919 | 1919/1919 | 1873 | 6.518 | -8.488 |
| gradient_brier_graph_w1p0 | 1927 | 1927/1927 | 1873 | 6.500 | -8.502 |
| gradient_brier_graph_w4p0 | 1914 | 1914/1914 | 1852 | 6.451 | -8.455 |
| gradient_ce_graph_pose_w0p25 | 1920 | 1920/1920 | 1861 | 6.556 | -8.383 |
| gradient_ce_graph_pose_w1p0 | 1925 | 1925/1925 | 1842 | 6.483 | -8.346 |
| gradient_ce_graph_pose_w4p0 | 1925 | 1924/1924 | 1839 | 6.481 | -8.360 |
| gradient_ce_graph_w0p25 | 1933 | 1932/1932 | 1867 | 6.570 | -8.364 |
| gradient_ce_graph_w1p0 | 1923 | 1923/1923 | 1832 | 6.478 | -8.361 |
| gradient_ce_graph_w4p0 | 1926 | 1926/1926 | 1831 | 6.488 | -8.328 |
| graph | 2000 | 2000/2000 | 0 | 6.090 | 30.120 |
| unguided | 1920 | 1920/1920 | 1864 | 6.590 | -8.443 |

原始20个Steer构象在统一协议下的pIC50中位数为8.136，Vina score-only中位数为−10.363 kcal/mol。LLM图编辑前后SA均值为2.818→2.654（越低越好）；这是设计图层面的辅助指标，不是完整逆合成验证。

三个硬约束组在本次设置下均没有PoseBusters通过构象，graph与fragment_graph组的Vina中位数为正。应保留并分析这些质量问题，不能用“可解码”代替合理构象。软引导组的预测亲和力中位数低于原始优选集合，也未表现出随权重增加而普遍改善的趋势；这份完成记录不声称任何引导组获胜。无引导对照与软引导有较好的PB通过率，但还需要按分子配对分析互作保留、编辑区域保持、SA变化和指标间权衡。

本次主要完成互作约束生成、区域奖励网格及实际构象评分；没有进行商业原料库路线搜索，SA不代表已证实可合成。

## 运行修复与保真审计

共102个评估批次经版本化overlay重评：v2恢复80批，v3恢复5批，v4恢复17批。错误涉及输入立体化学SMILES序列化、非四面体中心氢表示、以及不对称的返回图芳香性规范化。旧错误日志、execution和overlay全部保留。

最终审计确认102批的实际SDF哈希和affinity父记录均未改变；原先已成功的Vina分数和PLIP接触集合没有变化。修复没有重新生成分子、再次调用LLM或更改网格。v2/v3的failed状态是已被后续维护版本接替的历史记录，最终v4状态为completed，不代表最终账本仍失败。

代码回归测试228项通过；远端22个真实失败结构身份检查及实际Vina/PLIP验证通过，真正相反的指定手性仍被拒绝。

## 文件位置

远端完整报告：`/workspace/AgenticGen_experiments/ck2_original_pose_20261020_high/results/report/`。

本地副本：`outputs/ck2_original_pose_20261020_high/final_report/`，包含：

- `summary.json`、`completion_audit.json`：覆盖、状态和逐组汇总。
- `sources.csv`：20个原始分子与设计图SA。
- `proposals.csv`：逐次提案、理由和重试记录。
- `particles.csv`：全部32000次尝试，保留失败、不可评分与重复结果。
- `arms.csv`：320个分子×实验组单元的指标与有效分母。
- `campaign_manifest.json`、`campaign_status.json`：完成状态副本。

最终修复审计：`outputs/ck2_original_pose_20261020_high/environment/evaluation-recovery-audit-20261023T011040Z.json`。巡检完成快照：`heartbeat-20261023T010852Z-ledger.json`（以本地environment目录实际文件名为准）。

这些结果均为计算预测。该记录提供完成覆盖与描述性结果，尚未给出显著性检验或实验组因果结论。

