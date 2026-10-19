# CK2 高亲和力、SA 分层示例

来源：先前 CK2 原生 single steering 的 1,000 个终态。固定 24 个独立质量图和 45 个独立高亲和力图；选择质量组 SA 最大的 3 个，以及 PB 通过但高应力、SA≥3 的挑战分子 3 个。挑战组不作为质量合格分子解释。

[完整统计与真实调用结果](../../docs/CK2_SA_SCREENING.md) · [入选指标](validation_selection.csv) · [来源与阈值](selection.json)

`candidates.sdf` 保留此前导出的原始 pose 字节；`cases.json` 固定输入及来源哈希。`config.json` 每个分子仅一个真实提案，共 6 个；无 replay 数据。输出中接受图编辑不代表保留亲和力或改善路线。

在 AgenticGen 根目录运行，输出目录需尚不存在：

```powershell
# 重新制备时使用新的目标目录；不会覆盖冻结输入
.venv/Scripts/python.exe -m agenticgen prepare-ck2-sa --workspace ../.. --output examples/ck2_sa_new --count-per-stratum 3

# 设置 DEEPSEEK_API_KEY 后，实际调用 6 个提案
.venv/Scripts/python.exe -m agenticgen run --config examples/ck2_sa_validation/config.json --output outputs/ck2_sa_new

# 只从既有结果生成 CSV，不调用 API
.venv/Scripts/python.exe scripts/summarize_ck2_sa.py --run outputs/ck2_sa_validation_live
```

## 独立逆合成筛选

已有清单：`retro_quality_unique.smi` 共 24 个；`retro_high_affinity_unique.smi` 共 45 个。每行一个 canonical SMILES，配套 `_index.csv` 将零基序号映射回分子 ID。两份有重叠，不应不加区分地重复搜索。真实运行的 5 个合格产物已导出到 `outputs/ck2_sa_validation_live/retro_products.smi`。

`config.simpretro.json` 是待配置的真实后端模板，并不表示工具已可用。需要设置 `SIMPRETRO_ROOT`、`SIMPRETRO_PYTHON`、`SIMPRETRO_TEMPLATES`、`SIMPRETRO_INVENTORY`。commit 固定为 `5abbdd12d23bec3b0c524a3f5e59d160beac00f3`，1800 秒/分子、单核、上游固定 seed=0。建议使用 Linux 部署上游环境；还需其 C++ RDChiral 依赖。自行替换原料库会改变实验协议，结果必须分开记录。

```powershell
# 不加载 LLM，无需 DEEPSEEK_API_KEY；开始前校验依赖路径与 commit
.venv/Scripts/python.exe -m agenticgen screen-retro --config examples/ck2_sa_validation/config.simpretro.json --targets examples/ck2_sa_validation/retro_quality_unique.smi --output outputs/ck2_quality_retro
```

路径可以改为 Linux 运行环境中的绝对路径，或在该环境内设置同名变量后运行 Python 命令。不要从 Windows 配置中填写无法在当前主机执行的远端 Python 路径。

搜索统计目前仅支持 `route_found` / `no_route_within_budget` / `infrastructure_error` 等状态；`route_verified` 始终保持 false，直到接入独立路线核查。原分子和产物必须使用同一库存、模板与预算；当前本地未执行这一步。
