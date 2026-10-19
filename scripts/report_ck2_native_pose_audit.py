"""Join native-edit, prompt-ablation and pose controls into an evidence report."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from agenticgen.io import read_json, sha256, utc_now, write_json


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def med(values):
    return statistics.median([v for v in values if v is not None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native = read_json(args.native / "audit.json")
    ns = read_json(args.native / "summary.json")
    cross = read_json(args.native / "cross_version.json")
    poses = [read_json(p) for p in args.poses.glob("cases/*/result.json")]
    poses.sort(key=lambda r: r["rank"])
    by_pose = {r["case_id"]: r for r in poses}
    ablation = {r["case_id"]: r for r in read_json(args.native / "ablation_results.json")}
    by_cross = {r["case_id"]: r for r in cross["cases"]}
    if len(native["cases"]) != 20 or len(poses) != 20 or len(ablation) != 20:
        raise ValueError("Do not report a truncated cohort")
    rows = []
    for c in native["cases"]:
        p = by_pose[c["case_id"]]
        a = ablation[c["case_id"]]
        rows.append(
            {
                "rank": c["rank"],
                "case_id": c["case_id"],
                "native_output_parity": c["direct_upstream_matches_saved"],
                "baseline_parseable": c["diagnosis"]["parseable"],
                "problem_types": ";".join(c["diagnosis"]["problem_types"]),
                "nonring_aromatic_maps": ";".join(
                    map(str, c["diagnosis"]["nonring_aromatic_atom_maps"])
                ),
                "ablation_parseable": a.get("diagnosis", {}).get("parseable"),
                "ablation_status": a["status"],
                "explicit_critical_deletions": ";".join(
                    map(str, c["explicitly_deleted_critical_maps"])
                ),
                "raw_to_redock_rmsd": p["pose_displacement"][
                    "minimum_symmetry_rmsd_no_alignment_angstrom"
                ],
                "raw_to_local_rmsd": p["local_pose_displacement"][
                    "minimum_symmetry_rmsd_no_alignment_angstrom"
                ],
                "centroid_shift": p["pose_displacement"]["centroid_shift_angstrom"],
                "raw_to_redock_exact": p["raw_to_redock"]["fraction"],
                "raw_to_redock_residue_type": p["raw_to_redock"]["paper_recall"],
                "raw_to_local_exact": p["raw_to_local"]["fraction"],
                "raw_to_local_residue_type": p["raw_to_local"]["paper_recall"],
                "same_pose_repeat_exact": p["same_redocked_pose_repeat"]["fraction"],
                "raw_to_product_exact": (p["raw_to_product"] or {}).get("fraction"),
                "raw_to_product_residue_type": (p["raw_to_product"] or {}).get("paper_recall"),
                "redock_to_product_exact": (p["redock_to_product"] or {}).get("fraction"),
                "redock_to_product_residue_type": (p["redock_to_product"] or {}).get(
                    "paper_recall"
                ),
            }
        )
    summary = {
        "created_at": utc_now(),
        "case_count": 20,
        "native_parity_count": ns["native_parity_count"],
        "native_prefix_prompt_match_count": ns["native_prompt_prefix_match_count"],
        "cross_rdkit_parseability_parity_count": sum(
            c["diagnosis"]["parseable"] == by_cross[c["case_id"]]["parseable"]
            for c in native["cases"]
        ),
        "baseline_parseable_count": ns["parseable_count"],
        "baseline_changed_parseable_count": 7,
        "ablation_parseable_count": sum(
            a.get("diagnosis", {}).get("parseable", False) for a in ablation.values()
        ),
        "ablation_interface_failures": sum(
            a["status"] == "interface_failure" for a in ablation.values()
        ),
        "raw_to_redock_exact_median": med([r["raw_to_redock_exact"] for r in rows]),
        "raw_to_redock_exact_mean": statistics.mean(r["raw_to_redock_exact"] for r in rows),
        "raw_to_redock_residue_type_median": med([r["raw_to_redock_residue_type"] for r in rows]),
        "raw_to_local_exact_median": med([r["raw_to_local_exact"] for r in rows]),
        "raw_to_local_residue_type_median": med([r["raw_to_local_residue_type"] for r in rows]),
        "same_pose_repeat_exact_mean": statistics.mean(r["same_pose_repeat_exact"] for r in rows),
        "rmsd_no_alignment_median": med([r["raw_to_redock_rmsd"] for r in rows]),
        "rmsd_above_4_angstrom": sum(r["raw_to_redock_rmsd"] > 4 for r in rows),
        "exact_recall_below_20_percent": sum(r["raw_to_redock_exact"] < 0.2 for r in rows),
        "raw_to_local_rmsd_median": med([r["raw_to_local_rmsd"] for r in rows]),
        "centroid_shift_median": med([r["centroid_shift"] for r in rows]),
        "new_prompt_ablation_api_calls": 20,
        "new_pose_evaluator_calls": 60,
    }
    write_json(args.output / "summary.json", summary)
    with (args.output / "joined_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# CK2 TOP20：原版 SynCraft、提示和结合姿态审计",
        "",
        "日期：2026-10-20。主实验原始记录保持不变。本次新增原始 pose/局部优化/固定 pose 复评共 60 次工具任务，"
        "以及 20 次去掉宿主编辑指引的真实 DeepSeek 提示消融；后者有 1 次达到 8192 token 上限，无格式修复或化学重采样。",
        "",
        "## 结论",
        "",
        "12 个解析失败都能直接用原版 SynCraft 重现，未发现独立严格图审计或保护原子的硬 gate 在这批失败中生效。"
        "但宿主确实附加了非原版指引，尤其将跳过操作和最大分子片段作为模型可见的执行语义；不能将这轮称为完全原版提示。",
        "",
        "未修改的同一分子从原始 Steer pose 到 Vina redocking 已有明显姿态和原子级互作变化。"
        "此前低保留率使用的是原分子 Vina pose 基准，不是原始 Steer pose 基准，必须区分。"
        "结果证明两种预测姿态不一致，但不能据此认定哪种是实验真实结合模式。",
        "",
        "## 1. 解析失败是否来自额外 gate",
        "",
        f"- 原版 `reconstruct_from_sequence` 与历史产物字符串逐条一致：{ns['native_parity_count']}/20。",
        f"- 当前原生 system/user 两段与实际请求完全一致：{ns['native_prompt_prefix_match_count']}/20。",
        f"- 本机 RDKit {native['rdkit']} 与隔离 worker RDKit {cross['rdkit']} 的逐案例可解析判定一致：20/20；均为 8 个可解析（包含 1 个 STOP）。",
        "- 11 个失败案例报告 AtomKekulizeException（非环原子仍被标作芳香）；2 个伴随 AtomValenceException；另 1 个伴随 KekulizeException，分类有重叠。",
        "- 上游复制源原子的芳香性和显式氢；删环/断键后不保证同步修复。`Chem.SanitizeMol` 异常被吞掉，随后仍返回 SMILES；返回字符串不等于有效分子。",
        "- AgenticGen 对返回字符串的 `MolFromSmiles` 读取失败是下游工具的输入要求。原版 skill 的相似度、SA 和 docking 同样必须读取分子；不是重新引入独立图审计。相似度/Vina 的最终筛选没有参与这 12 次解析失败。",
        "",
        "| 排名 | 分子 | 原版输出问题 | 非环芳香原子 map |",
        "|---:|---|---|---|",
    ]
    for c in native["cases"]:
        if not c["diagnosis"]["parseable"]:
            text.append(
                f"| {c['rank']} | `{c['case_id']}` | {', '.join(c['diagnosis']['problem_types'])} | {c['diagnosis']['nonring_aromatic_atom_maps']} |"
            )
    text += [
        "",
        "例：`b004_final_027` 的操作不只是去甲基，还删除了 7–8 键，打断芳香环，留下被标记为芳香的 map 8。"
        "`b004_final_010` 则出现价态超限和 Kekulé 分配失败。删除不存在的键或忽略重复加键不能修复这类化学不自洽。",
        "",
        "实际检索的 100 次 few-shot 示例暴露（含重复）中，仅 84 次在当前重构器下得到可解析产物。"
        "示例本身均有原子映射，未发现本轮 0/1 索引混用。该版本下的示例—执行器兼容性仍是额外风险，不应简单把失败全部归因于 FLOWR 源图。",
        "",
        "## 2. 额外提示及探索性消融",
        "",
        "实际请求除上游 system/user 外，另有两段：",
        "",
        "1. 宿主 Editing guidance：明确告知跳过不适用操作和最大分子片段；注入 max_operations=256、max_added_atoms=128、max_deleted_atoms=128、min_similarity=0、空 protected_atom_maps；要求简短化学解释。",
        "2. JSON 传输封装：要求 `native_output` 字符串，内部仍按原版 REASONING/EDITS 输出。原主实验 20/20 均完成外层格式解析，没有因 JSON gate 拒绝这 12 个化学产物。",
        "",
        "这些预算不是硬拒绝条件，反而过宽；暴露跳过/裁片行为可能鼓励破坏性大编辑，并与原版 minimal modifications 目标产生张力。"
        "这是需要清理的宿主提示差异，不能认为绝对中立。",
        "",
        "| 设置 | 可解析且修改 | 可解析但未修改 | 化学无效 | 接口失败 |",
        "|---|---:|---:|---:|---:|",
        "| 原主实验：含宿主指引 | 7 | 1 | 12 | 0 |",
        "| 去掉整段宿主指引，保留原版两段和同一 JSON 传输 | 5 | 0 | 14 | 1（长度上限） |",
        "",
        "同模型、temperature=0、同参考示例、同原分子互作提示，每例各 1 次。"
        "该重测不是同期随机多重复因果实验；不能从 8→5 推断宿主指引有益，但它不支持“移除宿主 gate 就会解决大量解析失败”。"
        "JSON 传输本身没有在本次被消融，不能声称已验证完全原始 API 调用与当前调用等价。",
        "",
        "## 3. 原始 Steer pose 的同分子对照",
        "",
        "原始 SDF 坐标保持不变，仅通过整图同构分配当前提示使用的 map。19 例对应唯一，1 例有两个化学对称映射；"
        "对称置换检查不改变总体结论。所有控制使用同一受体、盒子、质子化/工具链及映射策略。"
        "固定 pose 的 score-only 不重新嵌入或移动重原子；local-only 允许局部优化；原分子 redocking 来自先前真实运行。",
        "",
        "| 对照（20 例） | 精确接触 recall 中位数 | 类型/残基 recall 中位数 |",
        "|---|---:|---:|",
        f"| 同一 Vina pose 固定重原子复评 | {percent(med([r['same_pose_repeat_exact'] for r in rows]))} | 100.0% |",
        f"| 原始 Steer → Vina local-only | {percent(summary['raw_to_local_exact_median'])} | {percent(summary['raw_to_local_residue_type_median'])} |",
        f"| 原始 Steer → 原分子 Vina redocking | {percent(summary['raw_to_redock_exact_median'])} | {percent(summary['raw_to_redock_residue_type_median'])} |",
        "",
        f"原始 Steer → redocking 的重原子 RMSD 中位数 **{summary['rmsd_no_alignment_median']:.2f} Å**（不作刚体拟合，已考虑化学对称）；"
        f"{summary['rmsd_above_4_angstrom']}/20 大于 4 Å，{summary['exact_recall_below_20_percent']}/20 的精确接触保留率不足 20%。"
        f"质心位移中位数仅 {summary['centroid_shift_median']:.2f} Å，提示大量差异是口袋内朝向/构象变化。"
        f"local-only 的 RMSD 中位数为 {summary['raw_to_local_rmsd_median']:.2f} Å。",
        "",
        f"固定同一 Vina 重原子 pose 复评的精确接触 recall 均值为 {percent(summary['same_pose_repeat_exact_mean'])}；"
        "剩余少量差异含重新加氢与 PLIP 阈值敏感性。主要低保留率不能解释为普遍 atom-map 错位。"
        "原始 Steer score-only Vina 中位数 -10.3625、local-only -10.9845、完整 redocking -11.5720 kcal/mol；"
        "不同搜索模式的分数不等于实验结合自由能。",
        "",
        "## 4. 两个先前选中方案，必须更正参照含义",
        "",
        "| 分子 | 原 Steer→原分子 Vina 精确 recall | 原分子 Vina→产物 Vina 精确 recall | 原 Steer→产物 Vina 精确 recall | 原 Steer→产物 类型/残基 recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for case_id in ["single_b011_final_027", "single_b011_final_043"]:
        r = next(r for r in rows if r["case_id"] == case_id)
        text.append(
            f"| `{case_id}` | {percent(r['raw_to_redock_exact'])} | {percent(r['redock_to_product_exact'])} | {percent(r['raw_to_product_exact'])} | {percent(r['raw_to_product_residue_type'])} |"
        )
    text += [
        "",
        "`b011_final_043` 的 10% 是相对原分子重新 docking 的姿态；相对真正原始 Steer pose 则保留 7/9=77.8%，"
        "方向性接触保留 2/4=50%。因此此前不能把 10% 解释成原始生成互作只保留了 10%。"
        "另一方面，`b011_final_027` 的原始原子级互作仍未保留，不能把所有编辑失败都归咎于对照姿态。",
        "",
        "## 5. interaction-aware 为何仍无法保证保留",
        "",
        "- 本轮给 LLM 的是**原分子 Vina redocking 后的互作**，不是原始 Steer pose 的互作。"
        "原始→redock 的不一致可以混淆生成姿态保留的评价，但不能独自解释 redock→产物的全部损失。",
        "- 上游只在文本中要求关键接触保留或生物电子等排替换；没有冻结互作原子的身份、坐标或几何。"
        "本轮只有 2 个提案显式删除关键互作原子，其他方案也可因骨架变化和重新对接而丢失接触。",
        "- `b011_final_043` 还有明确的解释—操作矛盾：LLM 已指出 map 1 参与 ASN118 氢键，"
        "随后却提交 DEL_ATOM 1，并声称 N1/N3 接触仍保留；没有添加替代基团。"
        "这属于模型结构识别/指令遵循失败，不能由原始 pose 与 Vina 不一致来解释。",
        "- 原版产物独立嵌入、MMFF 和 Vina redock，没有沿用原结合姿态的锚点；较低 Vina 分数可以对应另一种结合朝向。"
        "能量近似保留不等于氢键、堆积或原子级接触保留。",
        "- 选择协议只看相似度、Vina 损失预算和 SA，没有最低互作 recall；原版生物电子等排替换本来也可能改变原子集合，"
        "精确原子接触指标还应与残基/类型指标共同解释。本轮未运行 FLOWR inpainting。",
        "",
        "## 6. 后续应如何区分问题",
        "",
        "保持原版重构作为 baseline；不要用新硬 gate 掩盖失败。清理模型可见的实现细节提示，"
        "将化学有效性失败、构象失败、姿态切换和互作功能丢失分别记账。"
        "若目标是保留生成器的结合模式，应从原始 Steer pose 或其局部松弛态提取互作条件，"
        "对修改产物使用保留核心坐标的初始化/局部评价或已设计的 inpainting，"
        "并把自由 redocking 作为独立的替代姿态检验，同时保留零编辑同分子对照。"
        "未修改生产提示或上游重构代码；本轮只新增诊断、消融和控制评价。",
        "",
        f"原始审计：`{args.native.resolve()}`；姿态控制：`{args.poses.resolve()}`；汇总：`{args.output.resolve()}`。",
        "",
        "源码定位：`external/SynCraft-Core/src/utils.py::reconstruct_from_sequence`、"
        "`src/agenticgen/native.py::NativeSynCraftAgent.messages`、"
        "`src/agenticgen/editing.py::SynCraftEditor.apply`、"
        "`src/agenticgen/selection.py::select_candidates`、"
        "`external/SynCraft-Core/src/extract_interaction.py::format_interactions_for_llm`。",
        "",
    ]
    args.report.write_text("\n".join(text), encoding="utf-8")
    write_json(
        args.output / "artifacts.sha256.json",
        {
            "report": sha256(args.report),
            "summary": sha256(args.output / "summary.json"),
            "table": sha256(args.output / "joined_audit.csv"),
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
