# CK2 远端全流程验收（2026-10-19）

已在 `/workspace/AgenticGen` 完成真实 `deepseek-flash → SynCraft 原版图重建 → FLOWR Graph / Fragment inpainting → Vina / PLIP / SA / 几何 / SimpRetro` 示例。最终模型与评价运行清单为 `outputs/ck2_live_20261019_b/inpainting_final/manifest.json`，状态 `completed`；两种生成结果均通过固定条件检查，两个 Vina/PLIP 任务成功，逆合成没有接口错误。这表示工程执行完成，不表示得到可接受的优化候选。

[小型结果包](../examples/ck2_inpainting/results_20261019/summary.json)包含真实提案、数值、API 收据摘要与文件哈希；[结构图示](../examples/ck2_inpainting/results_20261019/structures.png)只展示二维分子图。三个 SDF 保留实际三维坐标：

- [原分子](../examples/ck2_inpainting/results_20261019/source.sdf)
- [Graph 生成结构](../examples/ck2_inpainting/results_20261019/graph_0000.sdf)
- [Fragment 生成结构](../examples/ck2_inpainting/results_20261019/fragment_0000.sdf)

## 实际预算与设计

使用此前冻结的 CK2 优质 TOP1 `single_b004_final_027`，未根据本次结果重新挑选源分子。两次真实 API 请求，各一次网络尝试，返回模型均为 `deepseek-flash`，合计 7,582 provider tokens；没有格式修复或模型回退。

检索池为固定上游仓库的 3,272 条可用参考语料，每次检索 5 条；哈希 `ca42144e2444c9762ae18089eff08d9610a278363f52d9dc65536a1867b81239`。完整发布的 3,332 对语料已另行备齐，可通过适配器配置选用；本例不冒充论文完全相同的语料/库存协议。

| 模式 | 原生编辑 | 图重建 |
| --- | --- | --- |
| standard，K=1 | 删除原子 1–7 | 产生非环芳香标记，无法解析；记录并跳过 |
| interaction_aware，K=1 | `DEL_ATOM 1` 后停止，即去甲基 | 可解析，进入两种原生 inpainting |

原互作规则按 SynCraft 提示执行：允许为合成修复替换互作位置，疏水为软偏好。Fragment 固定编辑后存活的互作原子和实际编辑区域；本例固定 12/24 个原子，固定坐标最大偏移为 0 Å。未被编辑的互作原子与可生成区之间的连接变化单独记录；编辑边界自身仍须保留。新生成原子使用独立 map 命名空间。

FLOWR 使用官方 `flowr_root_v2.ckpt`，种子 42，100 步，每模式一个样本、一次采样迭代，Graph 为原生 harmonic prior，两支均启用原生 `final_inpaint`。第三方子模块没有代码修改。固定片段坐标来自 SynCraft 原分子的 Vina 对接位姿，区别于归档的原始 FLOWR 终态位姿。

## 结果及边界

| 结构 | SA↓ | Vina score-only↓，kcal/mol | MMFF 松弛能差/重原子↓ | 固定库存逆合成 |
| --- | ---: | ---: | ---: | --- |
| 原分子 | 2.866 | −11.845 | 0.459 | 120 s 内未找到路线 |
| Graph | 2.727 | +28.783 | 4501.684 | 120 s 内未找到路线 |
| Fragment | 3.245 | −9.176 | 35.879 | 找到并验证 10 个反应节点的路线 |

结合评价直接使用实际生成 SDF，不重嵌入替代构象。原分子也采用相同 score-only 协议；本表不是旧 redocking 分数或新的 pIC50 预测。MMFF94s 在分子副本上松弛，输入 SDF 不变；能差表示同一分子的几何警示指标，不是跨分子的绝对能量排序。Graph 有 9 条极端键长，Fragment 虽无该类键长异常，仍有很高的松弛能差。

Graph 的 SA 改善约 0.139，但三维结构严重恶化。Fragment 的可达性改善与 SA 变化方向不同，且 Vina 损失 2.669 kcal/mol，超过配置中 SynCraft 的 1 kcal/mol 预算；Graph 也超过此预算。故本例没有整体合格的优化胜出分子。`accepted_for_evaluation` 只表示图/固定片段条件通过，不能解释为物理合理。

Fragment 的路线在搜索约 68.65 s 时首次出现。[验证记录](../examples/ck2_inpainting/results_20261019/fragment_0000.route.json)包含目标图一致性、9 个库存叶子及 RDChiral 模板回放。库存为官方 SimpRetro eMolecules 的 23,081,629 条记录、完整 51,720 模板，每图搜索预算 120 s；不同于论文 Enamine 协议。模板与库存校验不能保证实验条件、选择性或产率；未找到路线也不能判为不可合成。

已实际执行 `summarize` 重建 SynCraft 派生选择，[选择记录](../examples/ck2_inpainting/results_20261019/selection.json)中两支均因 `vina_loss_budget` 排除，整体及路线子集均未选中。110 项本地测试与 Ruff 检查通过；保存的选择实现哈希与原始采样清单分别记录，不重复模型或路线搜索。

本例不能证明 LLM 学会了流场规律。Fragment 除保留 LLM 去甲基结果外，还重建了其他区域的原子与连接，不能将路线改善单独归因于 LLM 的方向选择。正式实验仍需等预算的无轨迹、仅生成器、无图编辑及不同初始化对照；多轮经验累计尚未在此例中评价。固定上游种子也不保证按墙钟截断的 MCTS 每次得到相同路线与首解耗时。

## 工程修复与复核

早期失败记录保留，没有覆盖或当成分子失败隐去：

1. 官方 random graph prior 在未使用 hybridization 特征的 v2 模型上引用未赋值变量；示例改用已有 harmonic 入口，未修改模型实现。
2. V2000 会截断四位原子 map；输出改为 V3000，并增加真实 SDF 往返测试。
3. 不同 map 的化学等价磺酰氧导致多余的立体标记；移除 map 后重新清理 stereo，仍验证真实手性与完整映射。
4. 原版 SimpRetro 配置不兼容 Python 3.11；使用独立 Python 3.10 CPU worker，避免改上游或替换 DTK 栈。

完整模板的独立库存零步、库存外一步路线控制均找到路线并完成验证。安装与系统变更、包锁及材料哈希见 [远端环境记录](REMOTE_ENVIRONMENT_20261019.md)。其中 APT 曾连带升级系统库（glibc 2.35→2.39）；后续未再执行 APT，实际 DTK GPU 推理与本例已通过。

原始完整结果位于远端 `outputs/ck2_live_20261019_b/`；本地镜像与安装审计位于 ignored 的 `outputs/remote_20261019/`、`outputs/remote_environment_20261019/`。权重与 2.69 GB 库存/缓存不提交 Git。任务期间的外网通过临时 SSH 代理，后续 API 或下载需自行具备可用外网或重新建立代理。
