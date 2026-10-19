# 远端环境准备

安装脚本只补充 AgenticGen/SynCraft 的额外依赖。模型权重与完整原料库须单独获取和校验，`verify_materials.py` 可核查已获授权复制的官方 SimpRetro 材料。先在目标机器读取环境，再决定是否安装；不能把脚本已经生成等同于远端环境已经就绪。

## 先盘点

使用目标 `molsteer-flowr-dtk` 环境自己的解释器运行：

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate molsteer-flowr-dtk
ENV_PREFIX=/opt/miniforge3/envs/molsteer-flowr-dtk
"$ENV_PREFIX/bin/python" scripts/remote_setup/inspect_environment.py \
  --output /workspace/agenticgen-environment/before.json
```

上面的前缀已在 2026-10-19 本次实例核对。激活脚本会加载 DTK 动态库路径，只调用绝对 Python 路径不能替代激活。报告记录 Python、包版本、Torch/设备、工具位置和磁盘空间，不读取环境变量中的凭据。

## 增量安装

```bash
bash scripts/remote_setup/bootstrap_syncraft.sh \
  --prefix "$ENV_PREFIX" --with-chemistry
```

`guarded_pip.py` 先检查 pip 解析结果，再执行安装。它锁定已有 Torch、PyG、DTK、NumPy、SciPy、RDKit 等包；拒绝替换 GPU 栈，也拒绝修改通过 `.pth` 暴露的环境前缀外包。新增依赖与当前版本冲突时会停止，改用单独 CPU worker。默认不执行未经审核的 `uv sync`、FLOWR GPU extras 或 Torch 重装。

本机既有 OpenAI SDK 为 3.24.0；当前 LiteLLM 最新依赖会将其降级。配方采用满足 SynCraft 下限、且已完成导入验证的 LiteLLM 1.53.1，保留既有 OpenAI SDK。若更换版本，应重新核查 pip 计划和导入检查。

当前实例已有 `/usr/bin/vina`；默认安装与上面的示例均不调用 APT。只有在另一个环境中显式指定 `--with-system-tools` 才进入可选系统工具步骤，且使用 `--no-remove`。若实例预置的操作系统和软件源不一致导致依赖求解要求删除系统库，应停止此步骤，改为独立工具环境或显式审核操作系统变更；本次实例不再执行系统包修改。

如果当前环境缺少 RDKit，应先确认与现有 NumPy 的兼容性，再用 guard 添加；若已经有合适版本，直接复用。AgenticGen 本身与 vendored SynCraft/SimpRetro 代码在 clone 后使用 `pip install --no-deps` 接入，不执行它们未受约束的全量依赖解析。

SimpRetro 除 Python 依赖外还需要固定版本源码、反应模板、fast-filter 权重、匹配库存和可选指纹缓存。`rdchiral` Python 实现适合小型工具测试；实际全模板运行优先使用与历史结果一致的 `rdchiral_cpp` CPU worker。不能将少量示例库存的结果标记为商业库存覆盖的逆合成验证，也不能将库缺失视为路线失败。

本次确认官方 SimpRetro 的搜索配置在 Python 3.11 会因 dataclass 的可变默认值报错。控制层和 FLOWR 保留既有 Python 3.11，逆合成单独使用 Python 3.10 CPU worker，配方复用 `environments/simpretro-worker/environment.yml`，不修改官方源码：

```bash
bash scripts/remote_setup/bootstrap_simpretro_cpu.sh \
  --prefix /workspace/envs/agenticgen-simpretro --repo /workspace/AgenticGen
export SIMP_RETRO_PYTHON=/workspace/envs/agenticgen-simpretro/bin/python
```

该步骤只使用 Conda 与当前固定仓库源码，不调用 APT，不替换 DTK 环境。外网不可达时由调用者提供临时网络代理；脚本不存储代理凭据。

同目录 `simpretro-linux-64.explicit.txt` 从此前通过完整路线检查的本地 Docker 镜像 `sha256:b74285f58129bc7970e6a378b5908021aa7364cdcc0196e11773d82cc0810afb` 导出，包含 Linux x86-64 包 URL 与 MD5。重建优先使用此锁文件，避免当前频道优先级或滚动元数据改变求解结果；不存在锁文件时才按 YAML 配方以明确的 flexible priority 求解。本次服务器最初按 YAML 求解，实际部署包列表另行导出为 `cpu-worker-explicit.txt`，不能直接宣称所有底层包与旧镜像逐项相同。

## 自检

```bash
export PATH="$ENV_PREFIX/bin:$PATH"
"$ENV_PREFIX/bin/python" scripts/remote_setup/smoke_tools.py \
  --output /workspace/agenticgen-environment/tools-smoke.json
```

源码接入 SimpRetro 后可增加 `--require-retro`。自检只运行分子读取、固定种子构象生成、Meeko/OpenBabel 转换、RDChiral 小反应与工具版本查询；它不调用 LLM、Vina 对接、FLOWR 采样或库存搜索，不能替代完整 example。

```bash
"$ENV_PREFIX/bin/python" scripts/remote_setup/verify_materials.py \
  /workspace/materials/simpretro \
  --output /workspace/agenticgen-environment/materials-smoke.json
```

材料校验会读取三个文件的完整 SHA-256、检查库存与索引的哈希绑定，并使用只读 SQLite 查询；不会反序列化未校验的 pickle，也不会自动启动路线搜索。

工具与材料完成后，可运行全模板的库存零步和非零路线工程控制：

```bash
"$ENV_PREFIX/bin/python" scripts/remote_setup/retro_smoke.py \
  --repo /workspace/AgenticGen --materials /workspace/materials/simpretro \
  --worker-python "$SIMP_RETRO_PYTHON" \
  --output /workspace/agenticgen-environment/retro_controls_cpu
```

工程控制使用已有固定示例，与 CK2 的优化结果分开报告；必须同时满足找到路线与路线图、库存末端、模板回放验证，才能算通过。

远端安装与 example 的实际完成状态记录在 `docs/REMOTE_ENVIRONMENT_20261019.md`。所有密码、API key 和 SSH 私钥均应由终端或外部 secret 提供，不写进此目录或 Git 仓库。
