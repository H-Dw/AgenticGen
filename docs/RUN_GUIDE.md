# AgenticGen 运行与开发指南

当前主配置为 `agenticgen.syncraft.v1`，覆盖完整 SynCraft 标准/结合模式和可选 Flow 证据，见[底座运行说明](SYNCRAFT_FOUNDATION.md)及 `examples/syncraft_foundation/`。下文 `ck2_top3/config.json` 是保留的旧 `agenticgen.run.v1` 工程基线；CLI 按 schema 分派，结果汇总也按对应 schema 重建。

2026-10-18 起两种运行入口都使用 `SynCraftEditor`，主配置 `editor` 默认为 `syncraft`；旧配置若显式写了 `editor: strict`，需改为 `syncraft`。`constraints` 仍可提供提示指导，但不再是独立图执行的否决条件。原始输出无法被分子工具读取时记录 `error_stage=product_loading`，不修复或替换产物。`was_fragmented` 和 `discarded_atom_maps` 记录原版裁片，详见[复测报告](NATIVE_SYNCRAFT_VALIDATION.md)。

## 1. 安装与外部依赖

在 AgenticGen 目录运行 `uv sync --frozen --extra syncraft --extra evomolsteer --extra dev`。当前锁文件在 Windows、Python 3.13.3 上验证；包要求 Python >=3.11，其他平台仍需实际安装检查。

SynCraft、EvoMolSteer 源码保持外部引用。默认示例配置沿用本工作区的相对布局；`syncraft.root` 和 `llm.evomolsteer_root` 可以指向别处。所有配置路径相对于**配置文件所在目录**，也支持 `${VARIABLE}` 环境变量引用。只在进程环境提供 `DEEPSEEK_API_KEY`；配置和运行文件不保存密钥。

首次初始化到另一台机器需要取得对应 SynCraft checkout、`assets/reasoning.json` 和 EvoMolSteer API 源码。缺失依赖会明确失败，不自动下载模型权重或改用其他 backend。

## 2. 预检与运行

```powershell
# 使用 replay 可在没有 API key 时检查依赖和输入
uv run --frozen --extra syncraft --extra evomolsteer agenticgen preflight --config examples/ck2_top3/config.json --backend replay

# 工程 fixture；不调用 LLM，仍执行实际 SynCraft 函数
uv run --frozen --extra syncraft --extra evomolsteer agenticgen run --config examples/ck2_top3/config.json --backend replay --output outputs/replay_example

# Windows 真实 API，自动读取进程或用户级环境变量
./scripts/run_ck2_example.ps1 -Output outputs/ck2_live_new

# 已设置进程级密钥的任意平台
uv run --frozen --extra syncraft --extra evomolsteer agenticgen run --config examples/ck2_top3/config.json --output outputs/ck2_live_other

# 仅重建衍生汇总，保留原始请求、提案和分子结果；不会发 API 请求
uv run --frozen agenticgen summarize --run outputs/ck2_live_other
```

`run` 要求新输出目录。默认 3 个分子 × 1 个提案，最多 3 次逻辑调用；每次最多 `(1+格式修复次数)×网络尝试次数` 次 HTTP 尝试。改变 `proposals_per_case` 后真实后端提交独立提案，可能产生重复方案；不得把重复提案视为独立统计样本。replay 文件仅含每例 1 个提案，增加数量必须同时提供新的明确测试记录。

预检只验证配置、依赖、输入、模型设置和凭据是否存在，不发 LLM 请求或启动搜索。返回码：0 表示工程流程完成，2 表示配置错误或部分调用/评价失败。完成状态可以包含化学编辑被拒绝；它不表示分子优化成功。

## 3. 输出和统计语义

```text
outputs/<run>/
├── manifest.json                   # 有效配置、版本、来源哈希、预算和运行状态
├── summary.json                    # 可重新计算的工程汇总
├── products.smi                    # 通过约束的产物图，不包含未验证的三维坐标
└── cases/<case_id>/
    ├── observation.json            # Agent 实际可见终态信息
    ├── decision.json               # 评价前保存的提案和首选索引
    ├── result.json                 # 编辑、辅助指标、逆合成状态
    ├── calls/                      # 请求、响应、收据、格式失败和按哈希存储的文本
    └── retro_*                     # 仅开启搜索时出现
```

`primary_proposal_index=0` 固定在评价前；`best_sa_proposal_index_auxiliary` 是事后 SA 筛选，不能冒充主选择。SA 降低由 `delta_sa=source−product > 0` 表示。结构无效/不满足约束与路线没有执行分别计数。`primary_route_evaluated_count` 只统计真正返回搜索结果的主提案；约束失败的零分另列。

输入 `source_metrics` 保存历史测量；当前 SA 由锁定的 RDKit 重新计算并单独存储。Agent 仅获得白名单终态指标；选样 rank、cohort 标签及未来结果不进入 Observation。终态示例当前只支持 `arm=T`，配置 `TH/THE` 会失败，避免将未接入的历史条件误记成成功实验。

收据用量是 provider 实际返回的 token 数；无收据的失败请求不推断用量。真实 API 原始模型响应由 EvoMolSteer 机制留存；默认 schema 接口保存解析后的 JSON、请求及校验哈希，不能宣称额外保存了未返回的内部推理。

## 4. 可插拔模块

| 文件 | 职责 | 可替换边界 |
| --- | --- | --- |
| `contracts.py` | Pydantic 输入输出值、观察、指令、结果 | schema 版本升级 |
| `ports.py` | LLM / Prompt / GraphEditor / Property / Retrosynthesis Protocol | 后端实现 |
| `config.py` | 有界、拒绝未知字段的配置 | composition root 的配置扩展 |
| `factory.py` | 创建组件和登记代码来源 | 内置适配器装配 |
| `agent.py` | 单一无状态方向 Agent、结构化输出协议 | PromptPort、LLMPort |
| `adapters/syncraft.py` | 上游检索/示例格式/重构 | 外部 SynCraft checkout |
| `editing.py` | 原版 SynCraft 重构、输出读取与裁片/实际图差异记录 | GraphEditorPort |
| `adapters/llm.py` | EvoMolSteer API、显式 ReplayLLM | LLMPort |
| `adapters/properties.py` | RDKit SA 辅助指标 | PropertyEvaluatorPort；本版语义固定 SA |
| `adapters/retro.py` | 禁用状态和 SimpRetro CLI | RetrosynthesisPort |
| `runner.py` | 单轮、有界、先决策后评价 | 注入 Components |
| `reporting.py` | 汇总、用量及无 API 重分析 | 分析阶段 |

Python 调用可直接注入符合 Protocol 的组件：

```python
from dataclasses import replace
from agenticgen.config import load_configuration
from agenticgen.dataset import load_dataset
from agenticgen.factory import build_components
from agenticgen.runner import run

config = load_configuration(config_path)
dataset = load_dataset(config.resolve(config.value.dataset))
components = build_components(config, dataset)
# my_retro 实现 evaluate(smiles, output) -> RetroResult
components = replace(components, retro=my_retro)
run(config, dataset, components, output_path)
```

自定义后端需将代码、版本、配置和资源哈希登记到 `components.provenance`，配置里反映实际协议。CLI 只开放明确支持的内置后端；新增 CLI 后端应扩展配置和 factory，不能从模型输出导入 Python 模块。组件结果按现有 schema 交换，不让 Agent 自行执行工具或修改预算。

## 5. SimpRetro 配置

将 `configs/simpretro.fragment.example.json` 的对象填入运行配置的 `retro` 字段。模板、原料库、独立 Python 环境须已可用。固定时间预算 1800 秒、MCTS、CPU 线程、源版本与资源 SHA256；上游版本固定 seed=0。`process_grace_seconds` 只用于模型加载及退出，搜索预算仍由上游 `time_limit_s` 控制。

本适配器已实现进程边界、搜索结果状态和目标核对，但没有路线化学与库存叶节点证书验证，因此 `route_verified=false`。它不能直接满足项目正式主终点验收；需要后续 route validator。没有可用环境/库存时保持 `backend=disabled`，不会改用 SA 伪造逆合成标签。

## 6. 验证与后续范围

```powershell
uv run --frozen --extra syncraft --extra evomolsteer --extra dev pytest -q
uv run --frozen --extra dev ruff check src tests scripts
uv run --frozen --extra dev ruff format --check src tests scripts
uv run --frozen python scripts/export_schemas.py
```

测试覆盖实际 SynCraft 图重构、手性/同位素丢失拒绝、结构约束、EvoMolSteer MockTransport、失败与缓存、不伪造搜索标签、CK2 输入完整性和完整 replay 流程。缺少外部 checkout 时相关测试会 skip；报告验证覆盖时必须公开 skipped 数。

当前没有实现正式 L1 共享动作池、历史特征、L2 native steering、L3 inpainting、共享 skills 记忆或模型微调。接下来优先完成 SimpRetro 实际运行与路线验证，再在相同 T 基线上接入可见性受控的 H。
