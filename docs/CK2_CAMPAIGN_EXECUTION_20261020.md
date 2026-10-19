# CK2 原始 Steer TOP20：执行记录

记录时间：2026-10-20（Asia/Shanghai）。正式实验正在运行，本文记录实现与运行验证，不是最终效果报告。

## 已完成的工作

- 源码已推送至 `H-Dw/AgenticGen`。实现提交为 `145b183`，预检数值修复为 `3196e38`。
- 启用 Torch 的完整本地测试：**220 passed**；Ruff 和 diff 检查通过。
- 远端 20 个原始构象的 Vina `score_only`、PLIP 和统一 native affinity 评分全部完成。
- DeepSeek high 设计完成：20 个可解析改造图，共 25 个化学提案，其中 5 个失败提案已记录。
- 真实 checkpoint 完整预检通过，并按完全相同的源文件部署正式运行版本。
- 8 个实际生成构象完成独立 Vina/PLIP、SA、PoseBusters、应力及原子谱系评价测试；输入 SDF 哈希保持不变。
- 正式实验已启动，8,000 个微批次排队，计划共 **32,000 次生成尝试**。
- 已建立本对话的 30 分钟巡检，健康且无变化时保持安静，异常时追溯、修复并续跑。

首个正式微批次 `single_b011_final_030 / graph / batch_000` 已完成：4 次尝试、4 个可解析结构、4 个实际终态 affinity 评分，均有落盘记录。首次微批次耗时约 45.995 秒，包含首次载模，不能直接据此推算整个实验时间。预检粒子与正式预算分开保存。

## 版本与位置

| 项目 | 已核验值 |
| --- | --- |
| 运行根目录 | `/workspace/AgenticGen_experiments/ck2_original_pose_20261020_high` |
| 控制器启动记录 | `environment/jobs/all-1792437566012742113.launch.json` |
| 控制器日志 | `environment/jobs/all-1792437566012742113.log` |
| 账本与状态 | `results/ledger.sqlite`、`results/status.json` |
| 正式源码快照 | `71a893cdbab3ae7021baa7ad5b68839150eb219a617d7062ec252a9e5585058c` |
| Python 包文件指纹 | `eccc9dac873cc5708bc33548d87a62f707963b0bdc1bad0583563cab50223808` |
| 协议指纹 | `2a4299c180a9aba3cd86eb107dd842019bb3a339fcef1207208264c2d0ea58fc` |
| 完整预检证据 | `results/preflight/execution_808346841ae7/result.json` |

旧 checkout、旧实验、失败预检和旧源码快照均保留。部署复用了预检同一 archive，逐文件核对源码、20 个原始构象和配置哈希。基线和已完成 API 内容没有重跑。

完整证据已回传本机 `outputs/ck2_original_pose_20261020_high/`，这些运行输出不纳入 Git。仓库提供[一个固定索引的小型诊断示例](../examples/ck2_original_pose_high/diagnostic_example/README.md)。

## 质量结果与解释边界

20 个编辑图的 SA 辅助分数中，13 个改善、7 个变差，平均值从 2.81796 变为 2.65395。该指标尚不能说明逆合成路线可达或三维结合保留；本配置不运行逆合成搜索。

原版 SynCraft 的 interaction-aware 指引属于软约束。10 个设计删除了原互作原子，15 个设计涉及互作原子的类型或局部键环境变化，均有逐分子记录。生成条件只锚定实际存活的 I，以及实际编辑与边界区域 E；不会凭空恢复已删除原子。

完整预检验证了两种真实梯度、三种硬条件、原始互作世界坐标、80 个活动步骤、步幅上限、配对 prior/RNG、零强度逐位一致性及 head 的种子/顺序不变性。两种局部条件在该诊断批次均为 0/4 可解码，完整保留为质量结果，不补样、不改网格。

独立评价诊断中，全图条件的 4 个结构均未通过 PoseBusters，Vina 为 +20.244 至 +38.997 kcal/mol；软 CE 条件的 4 个结构通过 PoseBusters，Vina 为 −9.36 至 −7.535 kcal/mol。这只是同一分子的少量预检样本，不能用于宣称某策略整体更优。正式比较需使用全部预定尝试及完整的缺失/失败分母。

运行协议、奖励算子、固定引导区间和恢复规则见[冻结协议](CK2_ORIGINAL_POSE_CAMPAIGN.md)；部署与报告命令见[远端指南](REMOTE_CK2_CAMPAIGN.md)。
