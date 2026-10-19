# 首版运行架构与 CK2 示例验证

2026-10-16 更新：当前完整 SynCraft 底座通过 46 项测试；真实 CK2 pass-5 API 与独立 CK2 docking/PLIP replay 联调的结果、路线 worker 的实际限制见 [SYNCRAFT_FOUNDATION.md](SYNCRAFT_FOUNDATION.md)。以下保留 2026-10-15 首版历史记录，其 31 项测试和 TOP3 结果不代表当前新增底座的全部范围。

验证日期：2026-10-15。环境：Windows、Python 3.13.3；依赖以 `uv.lock` 为准。此次验证覆盖单轮终态 T 基线，未开展 H/E 对照实验。

## 1. 自动化验证

31 项 pytest 测试全部通过，0 skipped。`ruff check` 和 `ruff format --check` 通过。

覆盖实际 SynCraft 重构、原子身份、元素和价态、断开片段拒绝、保护邻域、手性/同位素丢失拒绝、无效 stereo 指令拒绝；EvoMolSteer 实际 API 函数使用 HTTP MockTransport 验证精确模型、缓存哈希、截断失败、中立格式修复和无凭据失败。另有 CK2 数据完整性、完整 replay 流程、接口失败不替换产物、SimpRetro 搜索状态及计数语义测试。

MockTransport 和 replay 是工程测试，不计为真实 LLM 分子优化。独立 replay 命令的输出位于 `outputs/ck2_replay_20261015/`，真实调用位于 `outputs/ck2_live_20261015/`。外部代码和原候选数据未被修改。

## 2. 真实 deepseek-flash 示例

三个输入各进行一次逻辑提案。模型精确返回 `deepseek-flash`，每次 `finish_reason=stop`；总计 3 次 HTTP 请求，无网络重试、无格式修复、无模型或规则 fallback。Provider 收据共记录输入 11,428 tokens、输出 955 tokens。请求、输出、模型名称、用量和 SHA256 留存在各例 `calls/`。

| 原始候选 | 真实提案 | 约束结果 | 原 SA → 新 SA | Morgan 相似度 |
| --- | --- | --- | --- | --- |
| `single_b004_final_027` | `DEL_ATOM 1`，随后 STOP | accepted | 2.8661266425 → 2.7268863157 | 0.722222 |
| `single_b017_final_021` | `DEL_ATOM 23`，随后 STOP | accepted | 2.9783166208 → 2.7733963205 | 0.732143 |
| `single_b007_final_026` | `DEL_ATOM 23`，`MUTATE_ATOM 24 C`，STOP | rejected | 未评价产物 | 未评价 |

第三例把已经是碳的原子 24 再声明为碳，触发 `Redundant atom mutation`。没有删掉该指令后悄悄接受，也没有重新要求 LLM 生成另一个成功样本。整段指令的其他化学合理性未因这个首个失败而被认可。

两个 accepted 表示图重构与预先声明的约束通过，SA 辅助指标降低。它们不等于逆合成成功或保留结合能力。当前 SA 本来就较低，下降幅度有限，不能据三例高质量选样推断总体收益。

## 3. 衍生汇总修正

初次 summary 将“约束失败已有零分”也计入 `primary_route_evaluated_count`，与实际逆合成调用数混淆。已抽出独立 `reporting.py` 并添加回归测试，将约束失败、结果可判定和实际搜索次数分开统计。

通过 `agenticgen summarize` 仅重建了衍生 summary；`summary_rebuild.json` 保存修正前后摘要哈希、输入结果哈希和 `new_api_calls=0`。原始 API 请求、提案、收据、决策与分子结果未改写。修正后的真实运行统计：有效编辑 2、拒绝 1、接口错误 0、逆合成评价 0。最初真实运行发生在该汇总代码修正之前；最终工程代码另经自动化测试和完整 replay 验证，没有为重新抽取成功结果再次付费调用。

## 4. 尚未验证的能力

- SimpRetro 只完成命令/结果适配和单元测试，本机没有执行真实搜索；模板、目标库存和路线证书验证待接入。`route_verified` 不会凭统计结果被设为 true。
- 编辑产物没有重新构象化、inpainting、docking 或结合头预测，不能把原始 pIC50/应力沿用给新分子。
- H/E、共享动作池、多轮反馈、案例/skills 累计和跨分子迁移没有在此示例中运行。
- 本例排除了 RAG 中与输入完全相同的图，但未做 scaffold 层面的语料排除；它不是正式无泄漏 benchmark。
- 上游重构的手性/同位素限制采用拒绝策略处理；尚未提供通用的带手性编辑能力保证。

下一步正式实验仍应遵循[项目计划](PROJECT_PLAN.md)完成路线评价、数据划分和信息/预算对齐。
