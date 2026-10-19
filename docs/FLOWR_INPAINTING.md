# 原生 FLOWR 适配边界

控制层 `design_runner.py` 复用 SynCraft 检索、原始提示/解析和图重建；`inpainting_runner.py` 把实际重建的图交给独立适配器，并在 FLOWR 之后评价。此前不含历史信息的 `foundation_run` 保留为对照。轨迹信息由已有增强接口额外注入，不替代 SynCraft 底座。

```mermaid
flowchart LR
    S[原分子与受体] --> I[原始 Vina / PLIP]
    I --> A[SynCraft Agent / DeepSeek]
    A --> G[原版图重建]
    G -->|无法解析| X[跳过并记录]
    G -->|可以解析| P[保留 map 的初始坐标]
    P --> R[Graph inpainting]
    P --> F[Fragment inpainting]
    R --> V[固定条件检查]
    F --> V
    V --> E[实际 pose 的 Vina / PLIP / 几何 / SA / 可选逆合成]
```

`inpainting_contracts.py` 定义输入、掩码、样本和运行结果的版本契约；`inpainting.py` 实现规划和子进程调用；`workers/flowr_inpainting.py` 在 FLOWR 的解释器中运行原生 `generate_from_pdb.py`。worker 只核对预处理的原子顺序和实际 keep-mask，不替代上游模型或修改第三方代码。三个 Git 子模块均固定 revision，FLOWR 无输出文件随仓库发布。

原子 map 与 RDKit 索引、FLOWR 输入 slot、输出 slot 明确区分。片段模式会把固定原子重排至输出前部；可生成区域的原子不得继承旧 map 身份，新原子分配独立命名空间。Graph 模式保留整个编辑图。两种模式均启用原生最终 inpaint；之后检查实际输出，不使用后处理伪造保留结果。

固定区内部连接、原子身份和片段坐标必须保留。与可生成区的外部连接，只有实际 LLM 修改原子及删除边界要求保持，以免撤销编辑；仅固定原互作原子的其他外部连接变化单独报告，允许由可生成区重建产生。互作是否恢复仍由 PLIP 评价。

Graph 默认使用官方 harmonic prior。固定上游版本的 random graph prior 在没有 hybridization 特征时会引用未赋值变量；此处只选择已提供的原生初始化路径，子模块保持原始版本。

两种方式默认从同一初始化结构平行采样。Fragment 固定 LLM 编辑区域的初始坐标，不能用它修复该区域自身的几何；Graph 分支才会重新生成该区域坐标。若以后采用 Graph 输出再接 Fragment，应单列为另一实验设置。

SynCraft `src/extract_interaction.py::format_interactions_for_llm` 允许为合成修复替换方向性互作原子，要求预期恢复互作几何；疏水等为软偏好。故默认 `interaction_policy=syncraft_compatible`，在 LLM 编辑完成后固定仍存活的互作原子和实际编辑区域。原互作删除/改变只记录，不触发默认硬拒绝。`strict` 是显式可选研究设置，不是默认值。片段采样期间的固定条件仍需保持。

`pose_initialization.py` 只负责初始条件：复制存活原子坐标，必要时为新增原子生成受约束初始坐标，并保存来源哈希。此步骤和 FLOWR 采样、最终合理性检查分别记录。生成后 `SynCraftGeometry.evaluate_pose` 直接处理 SDF；默认 score-only 不重嵌入或松弛重原子。其 protocol ID 包含 pose 哈希和评价模式，避免与普通 redocking 混用。MMFF 测量在副本上进行，原始坐标保持。

逐样本 SDF 使用 V3000，避免 V2000 的三位 atom-map 字段截断新原子身份。移除 map 后重新清理立体标记，避免两个化学等价的磺酰氧因不同 map 被误认为手性差异；真实手性仍需保留。对应图哈希采用 `canonical-isomeric-clean-stereo-after-unmapping-v2`；旧版增强包需要按新规范重新导出，不能直接混用旧图哈希。

单个工程例子只验证接口可运行。正式实验需要等预算无 inpainting 对照、固定掩码策略、多个采样种子，以及把结构有效率、可接受保留率、Vina/互作、SA 和固定库存路线结果分别报告。未配置库存时不能宣称完成逆合成验证。

`inpainting_selection.py` 将已评价结构交给现有 SynCraft 选择策略，分开保存代理指标选择和经验证路线子集。源与产物都使用 score-only 基线比较；MMFF 指标单独报告。`summarize` 可从冻结结果重建派生选择，不重复 API、采样或路线搜索。
