# AgenticGen

AgenticGen 研究分子生成过程中的信息，能否帮助冻结参数的 LLM Agent 选择更有效的局部图编辑，并在受约束探索中积累可迁移经验。

本项目是独立 Git 仓库。通过适配器复用 MolSteer、SynCraft、FLOWR.ROOT 和 SimpRetro；三个官方依赖以固定版本 Git 子模块提供，权重、库存和运行输出独立配置。

2026-10-19：新增完整的“SynCraft 设计 → Graph / Fragment inpainting → 实际生成结构评价”入口，以及不依赖兄弟项目路径的 `deepseek-flash` API。不可解析的图默认跳过并记录；互作替换规则与 SynCraft 一致。[可移植 CK2 示例](examples/ck2_inpainting/README.md) · [适配架构](docs/FLOWR_INPAINTING.md) · [远端环境](docs/REMOTE_ENVIRONMENT_20261019.md)。

远端真实示例已完成：两条 API 设计、一条有效图、两种原生结构生成及完整评价。Fragment 找到经验证的库存路线，但未得到同时保留结合与低应力的改进候选。[实际结果与失败记录](docs/REMOTE_EXAMPLE_VALIDATION_20261019.md) · [少量结构文件](examples/ck2_inpainting/results_20261019/README.md)。

```bash
git clone --recurse-submodules https://github.com/H-Dw/AgenticGen.git
cd AgenticGen
python -m pip install -e '.[syncraft,dev]'
# 模型环境、权重及受体配置见上方示例；普通笔记本可先执行无模型测试
python -m pytest -q
```

远端已有 FLOWR/DTK 环境时应复用现有 Torch，按环境文档安装控制层，不能直接套用通用 GPU 安装命令。`design` 保存提案和图但延后产物评价，`inpaint` 默认只规划，显式 `--execute` 才调用 FLOWR。

2026-10-18：图编辑已切换为原版 SynCraft 重构，移除独立严格审计。不存在的键操作、重复加键和最大分子片段选择由原版执行器处理；输出保存原始 SMILES、裁片标记及实际原子变化。编辑范围字段仅用于提示，最终相似度、SA、结合和路线筛选仍由评价流程负责。[接口变更与复测](docs/NATIVE_SYNCRAFT_VALIDATION.md)。

2026-10-17：已配置完整发布语料、SimpRetro 官方 eMolecules 库存、全模板低内存运行与双模式合并。[当前 CK2 示例](examples/ck2_official/README.md) · [实际验证与库存协议](docs/OFFICIAL_EXAMPLE_VALIDATION.md)。该库存不同于论文 Enamine，示例使用短搜索预算，不能直接比较论文救回率。

## 当前底座：完整 SynCraft 运行机制

已接入标准与结合约束两种模式：真实案例检索和原始提示/解析、默认 pass-5、原版 SynCraft 图执行、SimpRetro、Vina/PLIP 与原子映射、redocking、论文路线选择和发布 skill 的 SA/结合选择。Flow-matching 历史是默认关闭的独立增强输入。[底座架构与运行说明](docs/SYNCRAFT_FOUNDATION.md) · [CK2 配置](examples/syncraft_foundation/README.md)。

```powershell
# 无 API 的原始机制工程集成验证
.venv/Scripts/python.exe -m agenticgen run --config examples/syncraft_foundation/standard.engineering.json --output outputs/foundation_replay

# 真实 CK2 pass-5 提案诊断，需 DEEPSEEK_API_KEY
.venv/Scripts/python.exe -m agenticgen run --config examples/syncraft_foundation/standard.inference.json --output outputs/foundation_live
```

完整评价采用 `standard.paper.json` 或 `interaction.live.json`，需要对应固定库存、worker 或 CK2 受体。未配置评价器的 inference profile 不产生优化胜出分子。下方旧版 TOP3 示例继续作为兼容性回归。

## 项目文档

- [完整项目计划](docs/PROJECT_PLAN.md)：已确认需求、三级实验、评价指标、数据划分、对照、预算、里程碑和验收条件。
- [项目架构](docs/ARCHITECTURE.md)：模块职责、适配器、数据契约、实验隔离、图编辑、原生 steering、inpainting 和经验记忆。

## 已确认的实验顺序

| 级别 | 主要研究问题 | 执行方式 |
| --- | --- | --- |
| L1 | 加入生成中间态，是否改善优化方向选择？ | 单轮图编辑；路线可达性、编辑规模和结构保留评价 |
| L2 | 原始 steered sampling 的额外候选信息，是否进一步改善方向选择？ | 保持单轮图编辑；分离历史和探索信息的作用 |
| L3 | inpainting 迭代与经验累计，是否提高决策和跨分子迁移能力？ | 多轮图编辑及三维评价；冻结 LLM，累计经验证的案例与 skills |

默认优化目标是可合成性。SimpRetro 在固定原料库和搜索预算下的路线可达性是主要判据，SA score 是快速辅助指标。

## 历史 CK2 TOP3 示例（2026-10-15）

2026-10-15 已实现终态 T 基线：SynCraft 相似案例检索和图编辑、EvoMolSteer 的 `deepseek-flash` API 适配、严格图校验、SA 辅助评价、独立逆合成接口和可审计输出。示例采用此前冻结的 CK2 TOP10 优质不同分子图中的前 3 名，保留原始 SDF 坐标和来源哈希。

在本仓库目录运行：

```powershell
uv sync --frozen --extra syncraft --extra evomolsteer --extra dev

# 无 API 的工程集成检查，仍实际调用外部 SynCraft 的检索和重构函数
uv run --frozen --extra syncraft --extra evomolsteer agenticgen run --config examples/ck2_top3/config.json --backend replay --output outputs/ck2_replay

# 真实 API：脚本优先读取进程环境变量，再读取 Windows 用户级 DEEPSEEK_API_KEY
./scripts/run_ck2_example.ps1

# 单元测试和外部接口集成测试；不调用付费 API
uv run --frozen --extra syncraft --extra evomolsteer --extra dev pytest -q
```

输出目录必须是新目录，避免覆盖已有决策。默认每个分子只提交 1 个提案，共 3 次逻辑调用；失败不会自动切换模型或 replay。其他主机需在配置中修改 SynCraft/EvoMolSteer 的外部路径。`uv.lock` 固定本次验证的依赖版本。

- [运行指南](docs/RUN_GUIDE.md)：安装、预检、输入输出、可插拔接口、SimpRetro 配置和错误处理。
- [SynCraft / EvoMolSteer 接入分析](docs/INTEGRATION.md)：原始 Agent 调用方式、复用范围和严格包装的差异。
- [CK2 示例说明](examples/ck2_top3/README.md)：选样依据、来源指标、replay 与真实调用。
- [验证报告](docs/VALIDATION.md)：测试结果及三次真实 API 调用的结果、局限。

该历史示例默认不运行逆合成搜索，结果记录 `not_run`。当时真实 API 验证中 2 个编辑改善 SA，1 个被旧约束拒绝；这些是历史结果，不能用于证明轨迹有用。当前图编辑规则和新 inpainting 入口以上方 2026-10-19 文档为准；多轮记忆及正式对照仍待实验。

## 高亲和力候选的 SA 分层验证

新增 [CK2 SA 筛选报告](docs/CK2_SA_SCREENING.md)和[六分子示例](examples/ck2_sa_validation/README.md)：原高亲和力集合有 45 个独立图，严格质量集合有 24 个；选取质量组 SA 相对较高的 3 个与高应力、高 SA 挑战分子 3 个。六次真实调用中，5 个图编辑被接受，其中 3 个 SA 改善、2 个变差；1 个因超出删除预算被拒绝。

`prepare-ck2-sa` 支持可复核分层选样，`screen-retro` 可独立筛选原分子而不调用 Agent。历史报告按当时配置保留；当前完整库存协议见 `configs/materials.official.lock.json`，新的服务器验证按各自运行清单记录。
