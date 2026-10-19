# CK2 TOP20：原版 SynCraft、提示和结合姿态审计

日期：2026-10-20。主实验原始记录保持不变。本次新增原始 pose/局部优化/固定 pose 复评共 60 次工具任务，以及 20 次去掉宿主编辑指引的真实 DeepSeek 提示消融；后者有 1 次达到 8192 token 上限，无格式修复或化学重采样。

## 结论

12 个解析失败都能直接用原版 SynCraft 重现，未发现独立严格图审计或保护原子的硬 gate 在这批失败中生效。但宿主确实附加了非原版指引，尤其将跳过操作和最大分子片段作为模型可见的执行语义；不能将这轮称为完全原版提示。

未修改的同一分子从原始 Steer pose 到 Vina redocking 已有明显姿态和原子级互作变化。此前低保留率使用的是原分子 Vina pose 基准，不是原始 Steer pose 基准，必须区分。结果证明两种预测姿态不一致，但不能据此认定哪种是实验真实结合模式。

## 1. 解析失败是否来自额外 gate

- 原版 `reconstruct_from_sequence` 与历史产物字符串逐条一致：20/20。
- 当前原生 system/user 两段与实际请求完全一致：20/20。
- 本机 RDKit 2026.09.1 与隔离 worker RDKit 2025.03.6 的逐案例可解析判定一致：20/20；均为 8 个可解析（包含 1 个 STOP）。
- 11 个失败案例报告 AtomKekulizeException（非环原子仍被标作芳香）；2 个伴随 AtomValenceException；另 1 个伴随 KekulizeException，分类有重叠。
- 上游复制源原子的芳香性和显式氢；删环/断键后不保证同步修复。`Chem.SanitizeMol` 异常被吞掉，随后仍返回 SMILES；返回字符串不等于有效分子。
- AgenticGen 对返回字符串的 `MolFromSmiles` 读取失败是下游工具的输入要求。原版 skill 的相似度、SA 和 docking 同样必须读取分子；不是重新引入独立图审计。相似度/Vina 的最终筛选没有参与这 12 次解析失败。

| 排名 | 分子 | 原版输出问题 | 非环芳香原子 map |
|---:|---|---|---|
| 1 | `single_b011_final_030` | AtomKekulizeException | [2, 3, 4, 5, 6] |
| 2 | `single_b004_final_027` | AtomKekulizeException | [8] |
| 5 | `single_b004_final_002` | AtomKekulizeException | [11, 12] |
| 8 | `single_b013_final_014` | AtomKekulizeException | [7] |
| 9 | `single_b011_final_040` | AtomKekulizeException | [24, 17] |
| 10 | `single_b017_final_043` | AtomKekulizeException | [2, 25, 10] |
| 12 | `single_b007_final_028` | AtomKekulizeException | [14, 13] |
| 13 | `single_b015_final_048` | AtomKekulizeException | [6] |
| 14 | `single_b003_final_025` | AtomValenceException, AtomKekulizeException | [10, 11] |
| 15 | `single_b011_final_000` | AtomKekulizeException | [14] |
| 16 | `single_b009_final_001` | AtomKekulizeException | [8] |
| 17 | `single_b004_final_010` | AtomValenceException, KekulizeException | [16] |

例：`b004_final_027` 的操作不只是去甲基，还删除了 7–8 键，打断芳香环，留下被标记为芳香的 map 8。`b004_final_010` 则出现价态超限和 Kekulé 分配失败。删除不存在的键或忽略重复加键不能修复这类化学不自洽。

实际检索的 100 次 few-shot 示例暴露（含重复）中，仅 84 次在当前重构器下得到可解析产物。示例本身均有原子映射，未发现本轮 0/1 索引混用。该版本下的示例—执行器兼容性仍是额外风险，不应简单把失败全部归因于 FLOWR 源图。

## 2. 额外提示及探索性消融

实际请求除上游 system/user 外，另有两段：

1. 宿主 Editing guidance：明确告知跳过不适用操作和最大分子片段；注入 max_operations=256、max_added_atoms=128、max_deleted_atoms=128、min_similarity=0、空 protected_atom_maps；要求简短化学解释。
2. JSON 传输封装：要求 `native_output` 字符串，内部仍按原版 REASONING/EDITS 输出。原主实验 20/20 均完成外层格式解析，没有因 JSON gate 拒绝这 12 个化学产物。

这些预算不是硬拒绝条件，反而过宽；暴露跳过/裁片行为可能鼓励破坏性大编辑，并与原版 minimal modifications 目标产生张力。这是需要清理的宿主提示差异，不能认为绝对中立。

| 设置 | 可解析且修改 | 可解析但未修改 | 化学无效 | 接口失败 |
|---|---:|---:|---:|---:|
| 原主实验：含宿主指引 | 7 | 1 | 12 | 0 |
| 去掉整段宿主指引，保留原版两段和同一 JSON 传输 | 5 | 0 | 14 | 1（长度上限） |

同模型、temperature=0、同参考示例、同原分子互作提示，每例各 1 次。该重测不是同期随机多重复因果实验；不能从 8→5 推断宿主指引有益，但它不支持“移除宿主 gate 就会解决大量解析失败”。JSON 传输本身没有在本次被消融，不能声称已验证完全原始 API 调用与当前调用等价。

## 3. 原始 Steer pose 的同分子对照

原始 SDF 坐标保持不变，仅通过整图同构分配当前提示使用的 map。19 例对应唯一，1 例有两个化学对称映射；对称置换检查不改变总体结论。所有控制使用同一受体、盒子、质子化/工具链及映射策略。固定 pose 的 score-only 不重新嵌入或移动重原子；local-only 允许局部优化；原分子 redocking 来自先前真实运行。

| 对照（20 例） | 精确接触 recall 中位数 | 类型/残基 recall 中位数 |
|---|---:|---:|
| 同一 Vina pose 固定重原子复评 | 100.0% | 100.0% |
| 原始 Steer → Vina local-only | 85.4% | 100.0% |
| 原始 Steer → 原分子 Vina redocking | 9.5% | 57.1% |

原始 Steer → redocking 的重原子 RMSD 中位数 **5.66 Å**（不作刚体拟合，已考虑化学对称）；14/20 大于 4 Å，12/20 的精确接触保留率不足 20%。质心位移中位数仅 0.54 Å，提示大量差异是口袋内朝向/构象变化。local-only 的 RMSD 中位数为 0.33 Å。

固定同一 Vina 重原子 pose 复评的精确接触 recall 均值为 98.7%；剩余少量差异含重新加氢与 PLIP 阈值敏感性。主要低保留率不能解释为普遍 atom-map 错位。原始 Steer score-only Vina 中位数 -10.3625、local-only -10.9845、完整 redocking -11.5720 kcal/mol；不同搜索模式的分数不等于实验结合自由能。

## 4. 两个先前选中方案，必须更正参照含义

| 分子 | 原 Steer→原分子 Vina 精确 recall | 原分子 Vina→产物 Vina 精确 recall | 原 Steer→产物 Vina 精确 recall | 原 Steer→产物 类型/残基 recall |
|---|---:|---:|---:|---:|
| `single_b011_final_027` | 0.0% | 8.3% | 0.0% | 77.8% |
| `single_b011_final_043` | 11.1% | 10.0% | 77.8% | 87.5% |

`b011_final_043` 的 10% 是相对原分子重新 docking 的姿态；相对真正原始 Steer pose 则保留 7/9=77.8%，方向性接触保留 2/4=50%。因此此前不能把 10% 解释成原始生成互作只保留了 10%。另一方面，`b011_final_027` 的原始原子级互作仍未保留，不能把所有编辑失败都归咎于对照姿态。

## 5. interaction-aware 为何仍无法保证保留

- 本轮给 LLM 的是**原分子 Vina redocking 后的互作**，不是原始 Steer pose 的互作。原始→redock 的不一致可以混淆生成姿态保留的评价，但不能独自解释 redock→产物的全部损失。
- 上游只在文本中要求关键接触保留或生物电子等排替换；没有冻结互作原子的身份、坐标或几何。本轮只有 2 个提案显式删除关键互作原子，其他方案也可因骨架变化和重新对接而丢失接触。
- `b011_final_043` 还有明确的解释—操作矛盾：LLM 已指出 map 1 参与 ASN118 氢键，随后却提交 DEL_ATOM 1，并声称 N1/N3 接触仍保留；没有添加替代基团。这属于模型结构识别/指令遵循失败，不能由原始 pose 与 Vina 不一致来解释。
- 原版产物独立嵌入、MMFF 和 Vina redock，没有沿用原结合姿态的锚点；较低 Vina 分数可以对应另一种结合朝向。能量近似保留不等于氢键、堆积或原子级接触保留。
- 选择协议只看相似度、Vina 损失预算和 SA，没有最低互作 recall；原版生物电子等排替换本来也可能改变原子集合，精确原子接触指标还应与残基/类型指标共同解释。本轮未运行 FLOWR inpainting。

## 6. 后续应如何区分问题

保持原版重构作为 baseline；不要用新硬 gate 掩盖失败。清理模型可见的实现细节提示，将化学有效性失败、构象失败、姿态切换和互作功能丢失分别记账。若目标是保留生成器的结合模式，应从原始 Steer pose 或其局部松弛态提取互作条件，对修改产物使用保留核心坐标的初始化/局部评价或已设计的 inpainting，并把自由 redocking 作为独立的替代姿态检验，同时保留零编辑同分子对照。未修改生产提示或上游重构代码；本轮只新增诊断、消融和控制评价。

原始审计：`D:\llm_controlled_generation\MolSteer\AgenticGen\outputs\ck2_top20_native_audit_20261020`；姿态控制：`D:\llm_controlled_generation\MolSteer\AgenticGen\outputs\ck2_top20_pose_audit_20261020`；汇总：`D:\llm_controlled_generation\MolSteer\AgenticGen\outputs\ck2_top20_audit_report_20261020`。

源码定位：`external/SynCraft-Core/src/utils.py::reconstruct_from_sequence`、`src/agenticgen/native.py::NativeSynCraftAgent.messages`、`src/agenticgen/editing.py::SynCraftEditor.apply`、`src/agenticgen/selection.py::select_candidates`、`external/SynCraft-Core/src/extract_interaction.py::format_interactions_for_llm`。
