# 远端 SynCraft 环境记录（2026-10-19）

## 当前状态

远端依赖安装与真实 example **尚未执行**。授权入口为 `root@ksai.scnet.cn:10399`。首次检查时，该域名解析为 `223.113.240.64`；Windows OpenSSH、Python TCP 与 WSL OpenSSH 直连均在 SSH banner 前被拒绝。通过已配置的本机 HTTP CONNECT 代理时收到连接确认，但随后对端关闭，仍未进入 SSH 认证。没有使用或保存登录密码，没有修改远端文件或软件环境。

因此本记录暂时不能提供本次实例的 Python、DTK、GPU、磁盘或包版本，也不能声称环境已兼容。待入口恢复后，应先收集实际报告，再按最小改动安装。

## 已准备的复用脚本

- `scripts/remote_setup/inspect_environment.py`：无秘密环境盘点。
- `scripts/remote_setup/guarded_pip.py`：先解析依赖，拒绝替换已有 GPU/数值栈与环境外包。
- `scripts/remote_setup/bootstrap_syncraft.sh`：增量依赖与可选 Vina/Graphviz 系统工具安装。
- `scripts/remote_setup/smoke_tools.py`：RDKit、Meeko、OpenBabel、PLIP、Vina 和可选 RDChiral 的工具级检查。

## 既往部署参考，尚未核实本次实例

先前 MolSteer 文档记录的环境为 `/root/private_data/.miniforge3/envs/molsteer-flowr-dtk`，Python 3.11.16；PyTorch `2.9.0+das.opt1.dtk2604` 从系统镜像导入，FLOWR 权重位于旧 MolSteer checkout。该记录不能替代当前检查。

需保留现有 DTK/Torch/PyG 组合，不运行 FLOWR 上游完整 GPU 环境同步。若化学工具与其冲突，在独立 CPU worker 安装，之后通过 AgenticGen 的外部工具接口调用。现有 checkpoint、库存和模板优先复用；任何新增大文件需要在审计中注明来源、大小及校验值。

## 后续验收条件

1. 保存本次实例安装前后报告与受保护包对比。
2. 工具级 smoke 通过；明确 Vina、PLIP、RDChiral 与源码路径。
3. 在 `/workspace/AgenticGen` 拉取已发布的固定 Git commit 和官方 submodules。
4. 运行小型 example，记录图解析失败的跳过、LLM 设计结果、两种 inpainting 输出及最终评价。
5. 区分真实模型采样、离线回放、工具级检查和缺失材料；缺库存或权重不能标记为完整流程成功。
