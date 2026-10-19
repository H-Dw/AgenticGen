# CK2 SynCraft 底座示例

本目录的当前运行配置使用先前冻结的 CK2 候选 `single_b004_final_027`。完整模块、选择语义及运行条件见 [SynCraft 底座说明](../../docs/SYNCRAFT_FOUNDATION.md)。

| 配置 | 用途 | 外部条件 |
| --- | --- | --- |
| `standard.engineering.json` | 原始格式 replay 与上游函数集成测试 | SynCraft 源码和参考语料；无 API |
| `standard.inference.json` | 真实 pass-5 推理诊断 | 默认 deepseek-flash；不评路线、不选择胜出者 |
| `standard.paper.json` | 路线可达性＋相似度选优 | SimpRetro Docker、固定模板与 `${SIMPRETRO_INVENTORY}` |
| `standard.history.json` | 同一标准底座＋生成历史 | 另需 `${FLOW_HISTORY_PACKETS}`，必须是真实冻结证据 |
| `interaction.live.json` | 原分子 docking/PLIP、pass-5、redocking、SA/结合约束选优 | docking Docker、`outputs/ck2_receptor_prepared/`、API |
| `interaction.engineering.json` | 对同一 CK2 的真实 docking＋显式人工提案 | 相同 CK2 受体，结果只能解释为工程测试 |

`ck2_one.json/.sdf` 绑定原始候选与坐标哈希。`native_replay.json` 中人工编写的提案只用于接口测试；不作为科学结果。历史证据没有提供伪造示例。

在仓库根目录执行：

```powershell
.venv/Scripts/python.exe -m agenticgen preflight --config examples/syncraft_foundation/standard.engineering.json
.venv/Scripts/python.exe -m agenticgen run --config examples/syncraft_foundation/standard.engineering.json --output outputs/foundation_replay
```

所有配置路径相对其 JSON 文件解析。新运行应使用新的输出目录；`summarize --run` 可以从保存结果重新计算汇总，不重复调用模型或化学工具。

受体准备入口为 `prepare-ck2-receptor --protein PDB --reference-ligand SDF --output outputs/ck2_receptor_prepared`。输入须已处于同一坐标系；默认使用参考配体范围加边距定义盒子，不自动推断整个蛋白的对接中心。SimpRetro 镜像已构建，但全模板检查尚未通过；小型零步工程检查与实际 CK2 路线评估严格区分。
