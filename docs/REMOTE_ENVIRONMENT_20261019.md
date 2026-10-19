# 远端 SynCraft 环境记录（2026-10-19）

## 环境与工具

授权入口 `root@ksai.scnet.cn:10399` 最初尚未启动；恢复后已成功连接。在 `/workspace` 配置额外依赖，未重装 Torch、PyG 或 DTK。安装前后报告保存在远端 `/workspace/agenticgen-environment/`。

| 项目 | 本次实测 |
| --- | --- |
| Python | `/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python`，3.11.16 |
| Torch | 运行时 2.9.0；包元数据 2.9.0+das.opt1.dtk2604；系统镜像模块 |
| 加速设备 | HIP 6.3.26093；1 张 K500SM_AI；激活后 GPU 可用 |
| PyG / NumPy / SciPy / RDKit | 2.8.0.post1 / 1.26.4 / 1.15.3 / 2026.3.6，均保留原版本 |
| 既有 API 基础包 | Pydantic 2.12.5、HTTPX 0.28.1、OpenAI 3.24.0 |
| 新增 API / 逆合成依赖 | LiteLLM 1.53.1、more-itertools 11.2.1、RDChiral 1.1.0（Python 实现） |
| 独立逆合成 worker | `/workspace/envs/agenticgen-simpretro/bin/python`，Python 3.10.22、Torch 2.5.1 CPU、RDKit 2024.09.6、RDChiral_CPP 1.1.3 |
| 新增几何依赖 | Meeko 0.7.1、PLIP 3.0.0、lxml 6.1.3 |
| 复用几何依赖 | Gemmi 0.7.5、Biopython 1.88、OpenBabel 3.1.0 |
| Vina | `/usr/bin/vina`，1.2.5 |
| PLIP / OpenBabel CLI | 环境的 `bin/plip`、`bin/obabel` |
| 空间 | 初始 `/workspace` 所在系统卷可用约 383 GB；持久个人卷仅余约 962 MB |

必须先加载 Conda 并激活环境；其 activate.d 脚本会加载 DTK 动态库路径。未激活直接调用绝对 Python 会缺失 `libgalaxyhip.so.5`：

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate molsteer-flowr-dtk
```

普通 SSH 命令未继承 `DEEPSEEK_API_KEY`，交互 Bash 环境内确认存在。只检查了存在性，没有输出值。需要 API 的批处理通过 `bash -ic` 载入既有配置。登录密码未写入脚本、文档或仓库。

## 安装与变更边界

pip 先用受保护包约束解析计划，再执行安装；保留 Torch、PyG、NumPy、SciPy、RDKit、Lightning 的版本。LiteLLM 最新版本要求降级原有 OpenAI SDK，因此改为 SynCraft 声明允许且已验证可导入的 1.53.1。

Vina 等系统工具从实例预先配置的 Ubuntu Noble 软件源安装，APT 事务包含 37 个包升级、28 个新增及 46 个系统依赖替换/删除；下载约 31.7 MB（另有软件源索引约 62.5 MB），glibc 从 2.35 变为 2.39。该事务没有重装 DTK/PyTorch，但不等于仅增加一个二进制。随后 GPU 导入/可用性和化学工具检查通过；这只覆盖已检查入口，不能证明所有预装程序不受影响。完整记录在 `apt-tools.log`、`apt-transaction.json` 和 `apt-transaction.txt`。复用安装配方默认不触碰系统依赖，显式启用时也增加 `--no-remove`。后续不再进行 APT 修改。

## 已通过的真实工具检查

- RDKit、Meeko、Gemmi、OpenBabel、PLIP、LiteLLM、RDChiral、Torch、PyG 实际导入。
- 固定种子乙醇构象、Meeko PDBQT 写出及 OpenBabel PDB/PDBQT 插件加载。
- 与 SynCraft 一致的 `prepare(mol)` 后调用旧 `write_pdbqt_file(path)` 实际写出成功，仅有弃用提示。
- Vina 版本、PLIP 帮助入口和 OpenBabel 版本查询。
- FLOWR `generate_from_pdb --help` 实际加载成功；这一步没有进行模型采样。
- DTK GPU 实际执行 `torch.ones(4, device='cuda').sum()` 得到 4.0，并完成同步。

报告为 `tools-smoke.json`、`before.json`、`after.json` 与 `flowr-help.txt`。这些检查不能替代完整 LLM→inpainting→评价实验。

## FLOWR 与材料

已复用 `/opt/MolSteer/flowr_root`，Git revision 为 `739a33c2aab74d074dd24605b9a2e34dc809ae9c`，工作区无改动。

- 既有 v2.2 checkpoint：`/opt/MolSteer/flowr_root/checkpoints/flowr_root_v2.2.ckpt`，1,007,216,503 bytes，MD5 `d227b736471c6f6fd90488951571a6fa`，与官方 Zenodo 发布元数据一致。
- v2.0：已从官方 Drive 文件 id `16dXF7C6iAFNpv1nT0W_N-6VtiqdDNWch` 完整重新下载并放入 `/workspace/materials/flowr_root_v2.ckpt`，677,580,035 bytes，SHA-256 `f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0`，MD5 `65a8d2d93c02ddb0ce495947fa160f4a`。与旧持久目录文件完整哈希一致。旁边 `.json` 记录官方来源与本次实测校验。
- 官方 SimpRetro eMolecules、SQLite 索引、完整模板指纹缓存及各自 JSON 收据已复制到 `/workspace/materials/simpretro`，三个完整 SHA-256、文件大小、索引与原库存哈希绑定及只读库存查询均通过，见 `materials-smoke.json`。数据文件总大小 2,689,377,966 bytes。库存不是 SynCraft 论文的 Enamine 库。

材料均放 Git 仓库之外；最终 checkout 的 ignored checkpoints 可使用链接复用权重。未下载训练语料，没有检索其他生物相关知识库。

## 网络与后续验收

实例原生外网不可达。任务期间使用仅监听远端 localhost 的临时反向代理，已实测 PyPI 与 GitHub HTTP 200；没有改写系统全局代理配置。该隧道关闭后需按实际网络重新配置，不能作为永久运行条件。

SimpRetro 固定版本源码已在 `/workspace/AgenticGen/external/SimpRetro` 使用 `--no-deps --no-build-isolation` 接入，Git SHA 为 `5abbdd12d23bec3b0c524a3f5e59d160beac00f3`。带 `--require-retro` 的工具检查通过，但这只覆盖依赖导入：首次完整搜索先发现缺少 Python graphviz binding，补齐后发现官方搜索配置不兼容 Python 3.11 的 dataclass 可变默认值规则；均记录为 infrastructure_error，未归类为分子不可合成。

现已按原有 `environments/simpretro-worker/environment.yml` 在独立 CPU Python 3.10 环境重建，不改上游代码。宿主 mamba 默认 strict channel priority 会使原配方求解冲突；最终使用命令行 `--channel-priority flexible` 完成 190 包、约 508 MB 下载的事务，未改全局 Conda 配置，也未调用 APT。worker 内源码 editable 安装及 RDChiral 模板执行通过，实际包列表为 `cpu-worker-explicit.txt`。另从此前验收的 Docker 镜像导出带 MD5 的 Linux 包锁文件，供后续免求解重建；不能将其与本次滚动求解的全部底层包等同。

完整 51,720 模板的两项工程控制已在独立 worker 通过，报告为远端 `/workspace/agenticgen-environment/retro_controls_cpu/summary.json`：

| 控制 | 结果 | 验证 | 总耗时 |
| --- | --- | --- | --- |
| 库存零步 CCO | route_found | route_verified=true | 48.1 s（首次求解 0 s） |
| 固定的库存外非零目标 | route_found，1 步 | route_verified=true；库存末端和模板 21828 回放通过 | 151.0 s（首次求解 5.32 s） |

搜索预算为 120 s；总耗时包括完整模型/模板初始化及单独的路线验证，不能将总耗时等同于搜索求解时间。非零控制是此前已冻结的工程目标，不是 CK2 优化结果，也不证明实验反应条件或产率。

summary、results、protocol 和两个 route 验证 JSON 已同步到本地 `outputs/remote_environment_20261019/retro_controls_cpu/`。CPU 实际包清单与盘点亦在该本地目录上一级；与此前 Docker 的 190 个包 URL 比对，仅 ca-certificates、libsqlite 和 xorg-libxtst 三个底层包滚动更新，核心化学/数值模型包相同。项目 CK2 example 的实际 FLOWR/LLM 运行记录另行保存，不将工具控制与真实优化收益混合报告。
