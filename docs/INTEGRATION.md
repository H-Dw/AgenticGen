# SynCraft 与 EvoMolSteer 接入分析

2026-10-18：`StrictSynCraftEditor` 已从运行代码删除，改用原版 SynCraft 重构。当前适配差异与复测见 [NATIVE_SYNCRAFT_VALIDATION.md](NATIVE_SYNCRAFT_VALIDATION.md)；下文保留早期严格包装的来源记录。

2026-10-16：当前主运行链已升级为完整 SynCraft 底座，详见 [SYNCRAFT_FOUNDATION.md](SYNCRAFT_FOUNDATION.md)。本文以下保留 2026-10-15 旧版局部编辑适配的来源分析；其“首个提案为主选择”等行为不适用于新 `agenticgen.syncraft.v1` 的固定 pass-K 评价后选择。

核验日期：2026-10-15。本项目实现名称为 **AgenticGen strict SynCraft baseline**。

## 1. SynCraft 如何调用 Agent

本地核验版本为 `557703e110b26a354afed2955ded5217837f6692`。来源为 [SynCraft-Core](https://github.com/catalystforyou/SynCraft-Core/tree/557703e110b26a354afed2955ded5217837f6692)。

`src/inference_enhanced.py` 实现单次、无状态的 LLM 编辑策略：

1. `setup_globals` 加载 `assets/reasoning.json`，为示例计算 Morgan 指纹。
2. `get_similar_examples` 按相似度检索 cliff-pair 示例；`format_example` 构造 few-shot 文本。
3. `create_user_prompt` 给输入原子设置从 1 开始的 map ID。
4. `worker_task` 调用 `litellm.completion(model=..., messages=...)`。
5. `parse_llm_output` 从 `[REASONING]` 与 `[EDITS]` 文本块中读取解释和 JSON 指令。
6. `src/utils.py::reconstruct_from_sequence` 通过 RDKit 应用指令。

`pass_k` 是多次独立提案。原代码没有训练后的 skills 累计或生成轨迹理解模块。

`skill/syncraft/pipeline.py::edit_for_synthesizability` 进一步串联原分子 docking、PLIP、含相互作用信息的 LLM 编辑、编辑产物 redocking 和 SA 筛选。该入口内置模型列表包含 `deepseek-v4-pro`，没有 `deepseek-flash`；也不输出逆合成路线。当前 L1 不需要其强制 docking 流程，因此选择复用核心检索与编辑模块，并在 AgenticGen 控制面组合组件。

## 2. 当前真实复用和变更

| 部分 | AgenticGen 实现 | 差异与原因 |
| --- | --- | --- |
| 案例检索 | 调用上游 `get_similar_examples` | 固定 similarity 模式，排除本批输入的完全相同图；不是 scaffold 无泄漏评测 |
| 示例格式 | 调用上游 `format_example` | 示例作为参考数据，不能覆盖宿主指令 |
| Agent 提示 | `prompts/direction.txt` | 使用同一命令语言，输出 schema JSON、简洁依据、预期和风险 |
| LLM 传输 | `EvoMolSteerDeepSeek` | 不调用 SynCraft 的 LiteLLM completion；复用既有 EvoMolSteer API |
| 图执行 | 调用上游 `reconstruct_from_sequence` | 独立审计图核对实际返回值；不使用审计图替代上游失败产物 |
| 产物检查 | `StrictSynCraftEditor` | sanitize、连通性、显式 map、立体化学、同位素、保护原子及预算 |
| 结果选择 | 先保存所有提案及首选，再评价 | 首个提案是主选择；best-SA 索引是另列的事后辅助结果 |
| 原始三维坐标 | 保留在输入 SDF | 不能附给修改后的图，产物只输出图/SMILES |

SynCraft 在 sanitize 失败时可能继续，并可能只保留最大片段；初始骨架重建也未完整复制原来的手性/同位素。当前严格包装拒绝这些差异。已有手性或同位素输入若上游丢失信息，即使操作很小也会拒绝；这项能力限制不能通过放宽验收掩盖。隐式氢补全遵循上游 RDKit 重建语义；原有显式氢、形式电荷、同位素和手性仍纳入核对。

没有修改、复制或 vendoring SynCraft 源码。依赖代码与示例语料哈希写入每次运行 manifest。上游导入依赖 LiteLLM，但适配器强制该导入使用本地 cost map，并恢复上游改变的 RDKit 日志开关。

## 3. EvoMolSteer API 适配

入口为外部 `src/evomolsteer/deepseek_api.py::call_json`。检查到的 Git revision 为 `78f83fe741a7d17bde7ec93c432ed7881af68069`；实际使用文件的 SHA256 另行记录，以覆盖工作区代码差异。

默认调用 `https://api.deepseek.com/chat/completions`，模型精确为 `deepseek-flash`，`thinking.type=disabled`、`temperature=0`、`response_format=json_object`。AgenticGen 默认输出上限 4096 tokens、请求超时 240 秒、网络尝试最多 3 次、中立格式修复最多 1 次。接口继承 EvoMolSteer 的模型名称和 `finish_reason` 校验、请求/响应哈希、格式失败记录、用量及缓存核对。

格式修复用于 JSON/schema，不用于重新提出化学方案；图编辑失败记录为被拒绝，不要求 LLM 不断重抽到成功。API 异常不会切换模型、规则代理或 replay。

Agent 输出只包含 `case_id/action/edits/rationale/expected_effect/risks`。它不能选择 evaluator、开启隐藏历史、执行 shell 或覆盖宿主约束。`LLMPort` 可以被其他已授权 provider 适配器替换，而 `DirectionAgent` 与化学执行代码保持独立。

## 4. SimpRetro 接口边界

[官方 CLI](https://github.com/catalystforyou/SimpRetro/blob/5abbdd12d23bec3b0c524a3f5e59d160beac00f3/syntheseus/cli/search.py) 接受目标文件、库存、模板、时间预算和 MCTS 配置；该版本在内部固定 `set_random_seed(0)`。适配器不传递并不存在的 seed CLI 参数。

`SimpRetroCLI` 在独立 Python 环境中启动上述入口，将非零退出、超时、目标不匹配和缺失统计标记为 `infrastructure_error`。搜索结果中的有限解时间只能支持“搜索报告找到路线”，`route_verified` 仍为 false。路线反应/叶节点库存核验待实现，尚未具备主实验的合格路线成功判据。当前机器未安装该搜索环境、模板和已确认原料库，默认保持 `disabled`。

上游 CLI 使用含冒号的时间戳目录名，实际搜索优先部署在 Linux/WSL 环境中；本次只验证了命令及结果适配契约，没有验证本机 Windows 的完整 SimpRetro 搜索。以 eMolecules 替换 SynCraft 所用 Enamine 库会改变实验协议，不能称为相同设置复现。
