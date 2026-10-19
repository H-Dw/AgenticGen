# CK2：SynCraft → FLOWR inpainting → 评价

这是可独立克隆运行的小型工程示例。`live_cases.json` 使用此前冻结的 CK2 优质 TOP1；`cases.json` 额外包含一个高 SA、高应力挑战分子。后者用于验证不可解析编辑的跳过路径，不能混称为低应力优质分子。

`design.live.json` 对 TOP1 执行标准、互作感知两种原版 SynCraft 提案，各一次，使用真实 `deepseek-flash`。`design.replay.json` 重放两条 2026-10-18 的真实模型建议：TOP1 去甲基，以及挑战分子的不可解析编辑。重放不产生新 API 调用，也不作为新模型效果证据。原始结构、受体、来源及哈希见 `inputs/` 和 `replay.json`。

## 环境与执行

首次克隆必须拉取三个固定版本的子模块；FLOWR 输出、权重、库存均不在 Git 中。

```bash
git clone --recurse-submodules https://github.com/H-Dw/AgenticGen.git
cd AgenticGen
# 激活已配置的 FLOWR 环境，再安装控制层；服务器可用 --no-deps 保留已有 DTK 栈
python -m pip install --no-deps -e .
export AGENTICGEN_PYTHON="$(command -v python)"
export FLOWR_PYTHON="$AGENTICGEN_PYTHON"
export FLOWR_CHECKPOINT="$PWD/checkpoints/flowr_root_v2.ckpt"
export VINA_BINARY="$(command -v vina)"
# DEEPSEEK_API_KEY 由进程环境提供，不写入配置
bash scripts/run_inpainting_example.sh replay outputs/ck2_replay_example
bash scripts/run_inpainting_example.sh live outputs/ck2_live_example
```

可先不启动模型，仅检查掩码规划：

```bash
python -m agenticgen inpaint --run outputs/ck2_replay_example/design \
  --config examples/ck2_inpainting/flowr.json \
  --pocket-pdb examples/ck2_inpainting/inputs/receptor.pdb \
  --output outputs/ck2_masks_only
```

两阶段接口也可独立使用：`agenticgen design` 只计算原分子互作、调用 Agent、按 SynCraft 重建图；`agenticgen inpaint --execute` 才初始化编辑产物、生成结构并评价。已有 `agenticgen run` 继续支持不使用 FLOWR 的原版底座对照。

## 两种 inpainting 的边界

- Graph：固定编辑后的原子、键与电荷，重新生成坐标。
- Fragment：固定**编辑后仍存在**的原互作原子和实际 LLM 编辑区域的并集；删除操作固定存活边界。其余区域允许原生 FLOWR 生成。
- 与 SynCraft 一致，LLM 可以提出互作位置的替换。删除、改变的互作原子单独记录，不宣称其互作仍然存在。后续 PLIP 分别报告原子映射级保留与类型/残基级召回。
- 不可解析的编辑立即记录 `skipped`，不会调用初始化或 FLOWR。原版 SynCraft 的最大片段选择仍保留。
- 使用上游 `--final_inpaint`，再检测输出是否实际保留给定图/固定片段、编辑边界及固定坐标。未保留的输出仍保存在原始 SDF 中，并标记不进入评价。仅由原互作产生的固定原子，其与可生成区的连接变化单独报告；这类边界没有被 LLM 指定编辑时允许原生模型重新连接。
- 若编辑只删除原子，初始化逐 map 复制存活原子坐标；新增原子时使用受约束 RDKit 初始坐标。初始化收据明确其不是 FLOWR 输出，也不是构象合理性证明。

默认每种方式请求一个样本、100 步、一次采样迭代。生成失败不会自动放宽固定区域。两种模式改变的自由度不同，不能仅凭单个成功样本判断哪种更优。

Graph 显式采用官方 `harmonic` 初始化。固定版本中 `random` 路径在 v2 模型不使用 hybridization 特征时存在未初始化变量错误；本项目选择另一个官方入口，未修改 FLOWR 子模块。

## 评价与逆合成

默认 `evaluation.json` 运行 RDKit SA、Vina **score-only**、PLIP、MMFF94s 同图松弛能量差及基础键长检查。它评价实际生成坐标，不重新嵌入一个替代构象。原分子使用相同 score-only 协议作比较；旧 redocking 分数另存。`--redock` 可额外输出独立对接结果。

默认示例的逆合成为 `not_run`。完整库存校验需单独提供官方 SimpRetro eMolecules 库存及绑定元数据：

```bash
export SYNCRAFT_STOCK=/workspace/materials/simpretro/emolecules.txt
export SYNCRAFT_STOCK_INDEX=/workspace/materials/simpretro/stock.sqlite
export SYNCRAFT_FINGERPRINT_CACHE=/workspace/materials/simpretro/fingerprint_base.pkl
export SIMP_RETRO_PYTHON=/workspace/envs/agenticgen-simpretro/bin/python
bash scripts/run_inpainting_example.sh live outputs/ck2_live_with_retro retro
```

该设置使用所有 51,720 个模板、每图 120 秒搜索预算，并检查路线图、库存叶子和模板重放；不是论文 Enamine 库存协议。`no_route_within_budget` 不能解释为不可合成。SA、Vina 和单个例子均不足以证明方法优于基线。

控制层与 FLOWR 使用 Python 3.11 环境；固定 SimpRetro 的配置 dataclass 不兼容 Python 3.11，因此逆合成使用独立 Python 3.10 CPU worker。三个 Python 路径分别配置，避免替换 DTK 依赖。上方 worker 路径以实际环境报告为准。

输出包含原始提案、掩码与 map/slot 映射、模型版本/权重哈希、原始及逐样本 SDF、固定区检测、评价结果和运行日志。只有符合所配选择策略的结果才会进入选择清单；策略胜出也不等于实验性质得到验证。

生成后复用原有 SynCraft 选择策略，以相同 score-only 模式比较源与产物；路线验证子集另行选择。MMFF 指标单独报告，不暗加到原版选择门槛中。保存结果可用 `agenticgen summarize --run RUN/inpainting` 重建派生选择，无须重复 API、模型或逆合成调用。[本次真实验收结果](../../docs/REMOTE_EXAMPLE_VALIDATION_20261019.md)同时保留未整体成功的结构与原因。
