# 远端环境准备

这些脚本只安装 AgenticGen/SynCraft 的额外依赖，不安装模型权重、训练语料或完整原料库。先在目标机器读取环境，再决定是否安装；不能把脚本已经生成等同于远端环境已经就绪。

## 先盘点

使用目标 `molsteer-flowr-dtk` 环境自己的解释器运行：

```bash
ENV_PREFIX=/root/private_data/.miniforge3/envs/molsteer-flowr-dtk
"$ENV_PREFIX/bin/python" scripts/remote_setup/inspect_environment.py \
  --output /workspace/agenticgen-environment/before.json
```

上面的前缀来自既往部署记录，必须先核对本次实例。报告记录 Python、包版本、Torch/设备、工具位置和磁盘空间，不读取环境变量中的凭据。

## 增量安装

```bash
bash scripts/remote_setup/bootstrap_syncraft.sh \
  --prefix "$ENV_PREFIX" --with-chemistry --with-system-tools
```

`guarded_pip.py` 先检查 pip 解析结果，再执行安装。它锁定已有 Torch、PyG、DTK、NumPy、SciPy、RDKit 等包；拒绝替换 GPU 栈，也拒绝修改通过 `.pth` 暴露的环境前缀外包。新增依赖与当前版本冲突时会停止，改用单独 CPU worker。默认不执行未经审核的 `uv sync`、FLOWR GPU extras 或 Torch 重装。

如果当前环境缺少 RDKit，应先确认与现有 NumPy 的兼容性，再用 guard 添加；若已经有合适版本，直接复用。AgenticGen 本身与 vendored SynCraft/SimpRetro 代码在 clone 后使用 `pip install --no-deps` 接入，不执行它们未受约束的全量依赖解析。

SimpRetro 除 Python 依赖外还需要固定版本源码、反应模板、fast-filter 权重、匹配库存和可选指纹缓存。`rdchiral` Python 实现适合小型工具测试；实际全模板运行优先使用与历史结果一致的 `rdchiral_cpp` CPU worker。不能将少量示例库存的结果标记为商业库存覆盖的逆合成验证，也不能将库缺失视为路线失败。

## 自检

```bash
export PATH="$ENV_PREFIX/bin:$PATH"
"$ENV_PREFIX/bin/python" scripts/remote_setup/smoke_tools.py \
  --output /workspace/agenticgen-environment/tools-smoke.json
```

源码接入 SimpRetro 后可增加 `--require-retro`。自检只运行分子读取、固定种子构象生成、Meeko/OpenBabel 转换、RDChiral 小反应与工具版本查询；它不调用 LLM、Vina 对接、FLOWR 采样或库存搜索，不能替代完整 example。

远端安装与 example 的实际完成状态记录在 `docs/REMOTE_ENVIRONMENT_20261019.md`。所有密码、API key 和 SSH 私钥均应由终端或外部 secret 提供，不写进此目录或 Git 仓库。
