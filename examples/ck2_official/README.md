# CK2：公开参考语料、双模式编辑与官方库存逆合成示例

本示例使用论文 Figshare 发布的完整 3,332 条参考语料、真实 `deepseek-flash`、SynCraft 原始提示/解析/图执行、Vina/PLIP，以及 SimpRetro 全量 51,720 模板。原料清单是用户选择的 **SimpRetro 官方 eMolecules 库存**，不是 SynCraft 论文的 Enamine 库存，结果不得直接比较论文救回率。

2026-10-18 起默认使用原版图重构（`editor=syncraft`），忽略和裁片行为遵循外部 SynCraft。历史严格审计实验仍保留原记录。当前复测目录为 `outputs/ck2_native_live_20261018/`，已保存提案的无 API 对照为 `outputs/ck2_native_retest_20261018/`；运行时请另选尚不存在的输出目录。[变更与结果](../../docs/NATIVE_SYNCRAFT_VALIDATION.md)。

`cases.json` 在新调用前冻结两个源分子：质量 TOP1 `single_b004_final_027`，以及此前选定的高 SA/高应力挑战组首例 `single_b011_final_017`。第二例不属于低应力优质集合；原 SDF 分子块与坐标保持不变，文件哈希写入案例清单。

## 环境和材料

在项目根目录使用已有 `.venv`。第三方源码版本和 Docker 配方见 `docs/SYNCRAFT_FOUNDATION.md`。首先构建或确认这两个镜像：

```powershell
docker build -t agenticgen-simpretro-worker:20261016 environments/simpretro-worker
docker build -t agenticgen-syncraft-worker:20261015 environments/syncraft-worker
.venv/Scripts/python.exe scripts/prepare_simpretro_assets.py
```

材料准备脚本下载官方库存与 Figshare 发布数据、检查数据完整性、建立精确字符串库存索引，并生成经过全值无损检查的模板指纹缓存。已有材料先核对哈希，不重复下载。参考语料保存在 `data/raw/syncraft_release/cliff_pairs.json`，不会自动建设新的商业近邻搜索库。

库存索引与上游 `canonicalize_inventory=false` 规则一致，不去盐、不忽略立体化学、不扩大或缩小原料集合。索引避免将 2,308 万条 SMILES 转成常驻内存的分子对象；反应打分、模板枚举、过滤权重及 MCTS 保持上游实现。缓存只改变整数指纹的存储类型，仍使用上游 float32 模型输入。

## CK2 运行

受体与盒子尚未准备时，从本项目既有 CK2 输入生成：

```powershell
.venv/Scripts/python.exe -m agenticgen prepare-ck2-receptor --protein ../../EvoMolSteer/data/reference_terminal/original/inputs/3PE1_protein_aligned.pdb --reference-ligand ../../EvoMolSteer/data/reference_terminal/original/inputs/3PE1_ligand_aligned.sdf --output outputs/ck2_receptor_prepared
```

设置进程环境 `DEEPSEEK_API_KEY`，或让以下启动脚本读取已有用户级凭据；凭据不写入配置和报告。

```powershell
.venv/Scripts/python.exe -m agenticgen preflight --config examples/ck2_official/dual.json
powershell -File scripts/run_ck2_official.ps1 -Output outputs/ck2_official_release_example
```

输出目录必须尚不存在。预检检查路径、版本、材料哈希和配置，不执行搜索或 API 请求。

`mode=dual` 表示同一源分子分别执行 standard 和 interaction-aware 两组；`pass_k=1` 是**每组**一次，即两例共 4 个逻辑提案。源分子 docking/PLIP 共享，两组提案全部冻结后才评价产物。每个不同分子的路线搜索预算为 120 秒，单 CPU，另外最多 300 秒启动/收尾宽限；这是一项短预算示例，不是论文每分子 1,800 秒、每模式 K=10 的复现。

每例输出包含独立模式选优、合并选优和额外的 `route_validated_selection`：

- 代理选优：相似度>0.2、Vina 恶化≤1 kcal/mol，在合格候选中取最低 SA，平分取更高相似度。
- 路线选优：从上述规则中进一步限于搜索找到路线且库存/模板检查通过的候选。
- 原分子没有路线、代理胜出者没有路线时，不计为路线救回。
- 产物按图去重，保留每个提案来源；相同图的重复提案不冒充不同产物。
- 同时输出严格原子映射接触保留率，以及论文口径的残基/类型 recall、Jaccard。

`dual.fixture.json` 保留首次工程联调设置：3,272 条代码内置示例、每模式 K=2。它与完整发布语料示例分别保存；不能混合两轮提案或用后者替换前者的原始结果。

## 非零路线工程对照

```powershell
.venv/Scripts/python.exe -m agenticgen screen-retro --config examples/ck2_official/dual.json --targets examples/ck2_official/nonzero_control.smi --output outputs/official_nonzero_control
```

该对照独立于 CK2：目标不在官方库存，而预先指定的两个前体在库存中。它用于检查规划器是否实际调用反应模型，以及非零路线能否通过库存和模板回放；不计入 CK2 优化效果。规划器可以找到不同于预设前体的替代路线。验证范围不包含反应条件、产率或真实实验成功。

实际结果和材料大小见 `docs/OFFICIAL_EXAMPLE_VALIDATION.md`。
