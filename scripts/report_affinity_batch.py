"""Export all affinity-cohort outcomes, including failures, without new inference."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from agenticgen.base_config import load_foundation
from agenticgen.dataset import load_dataset
from agenticgen.io import read_json, sha256, utc_now, write_json


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def structure(smiles):
    if not smiles:
        return '<div class="empty">无有效产物</div>'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '<div class="empty">产物无法解析</div>'
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    AllChem.Compute2DCoords(mol)
    draw = rdMolDraw2D.MolDraw2DSVG(460, 280)
    draw.DrawMolecule(mol)
    draw.FinishDrawing()
    return draw.GetDrawingText().split("?>", 1)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    manifest = read_json(output / "manifest.json")
    config = load_foundation(Path(manifest["config"]))
    dataset = load_dataset(config.resolve(config.value.dataset))
    rows, details, cards = [], [], []
    source_writer = Chem.SDWriter(str(output / "source_docked.sdf"))
    product_writer = Chem.SDWriter(str(output / "product_docked.sdf"))
    source_writer.SetForceV3000(True)
    product_writer.SetForceV3000(True)
    for case in dataset.cases:
        folder = output / "runs" / case.case_id / "cases" / case.case_id
        path = folder / "result.json"
        if not path.is_file():
            raise ValueError(f"Missing case result: {case.case_id}; no silent cohort truncation")
        result = read_json(path)
        for candidate in result["candidates"]:
            index = candidate["proposal_index"]
            graph = candidate.get("graph", {})
            prop = candidate.get("properties", {})
            before = result["source_binding"].get("score_kcal_mol")
            after = candidate.get("binding", {}).get("score_kcal_mol")
            delta_vina = after - before if before is not None and after is not None else None
            delta_sa = candidate.get("delta_sa")
            interactions = result["source_binding"].get("interactions", [])
            interaction_maps = {m for i in interactions for m in i["atom_maps"]}
            directional_maps = {
                m
                for i in interactions
                if i["kind"]
                in {"HydrogenBond", "SaltBridge", "PiStacking", "PiCation", "HalogenBond"}
                for m in i["atom_maps"]
            }
            deleted = set(graph.get("deleted_atom_maps", []))
            exclusions = result["selection"]["excluded"]
            selected = result["selection"]["selected_proposal_index"] == index
            reason = next((r["reason"] for r in exclusions if r["proposal_index"] == index), "")
            geometry_log = folder / f"geometry_product_{index:02d}/worker.log"
            if geometry_log.exists() and "Bad Conformer Id" in geometry_log.read_text(
                encoding="utf-8"
            ):
                reason = "conformer_generation_failed"
            row = {
                "rank": case.selection_rank,
                "case_id": case.case_id,
                "source_pic50": case.source_metrics["pic50_on_rescore"],
                "source_pb_pass": case.source_metrics["pb_full_pass"],
                "source_quality_group": case.source_metrics["quality_tail"],
                "source_relief_per_heavy": case.source_metrics["mmff_relief_per_heavy"],
                "proposal_index": index,
                "mode": candidate["proposal_mode"],
                "status": candidate["status"],
                "was_fragmented": graph.get("was_fragmented", False),
                "discarded_atom_maps": json.dumps(graph.get("discarded_atom_maps", [])),
                "source_sa": result["source_sa"].get("value"),
                "product_sa": prop.get("value"),
                "delta_sa_source_minus_product": delta_sa,
                "source_vina": before,
                "product_vina": after,
                "source_geometry_status": result["source_binding"]["status"],
                "product_geometry_status": candidate.get("binding", {}).get(
                    "status", "not_evaluated"
                ),
                "delta_vina_product_minus_source": delta_vina,
                "similarity": candidate.get("selection_similarity"),
                "selected_proxy": selected,
                "sa_improved_with_binding_budget": bool(
                    selected and delta_sa is not None and delta_sa > 0
                ),
                "exact_contact_retention": candidate.get("contact_retention", {}).get("fraction"),
                "paper_contact_recall": candidate.get("contact_retention", {}).get("paper_recall"),
                "deleted_interaction_maps": json.dumps(sorted(deleted & interaction_maps)),
                "deleted_directional_maps": json.dumps(sorted(deleted & directional_maps)),
                "source_smiles": case.smiles,
                "product_smiles": graph.get("product_smiles"),
                "edits": json.dumps(candidate.get("proposal", {}).get("edits")),
                "reason": graph.get("reason") or candidate.get("reason") or reason,
                "source_retrosynthesis": result["source_retrosynthesis"]["status"],
                "product_retrosynthesis": candidate.get("retrosynthesis", {}).get(
                    "status", "not_evaluated"
                ),
                "result_sha256": sha256(path),
            }
            rows.append(row)
            details.append(
                {"metrics": row, "proposal": candidate.get("proposal"), "result": str(path)}
            )
            for pose_path, writer, kind in (
                (folder / "geometry_original/docked_pose.sdf", source_writer, "source"),
                (
                    folder / f"geometry_product_{index:02d}/docked_pose.sdf",
                    product_writer,
                    "product",
                ),
            ):
                if pose_path.exists():
                    mol = Chem.SDMolSupplier(str(pose_path), removeHs=False)[0]
                    if mol is None:
                        raise ValueError("Saved evaluated pose is unreadable")
                    mol.SetProp("case_id", case.case_id)
                    mol.SetProp("rank", str(case.selection_rank))
                    mol.SetProp("kind", kind)
                    mol.SetProp("vina_kcal_mol", str(before if kind == "source" else after))
                    writer.write(mol)
            rationale = (candidate.get("proposal") or {}).get("rationale", row["reason"] or "")
            cards.append(
                "<article><h2>"
                + html.escape(f"{case.selection_rank}. {case.case_id}")
                + "</h2>"
                + "<p>"
                + html.escape(
                    f"pIC50 {case.source_metrics['pic50_on_rescore']:.4f}; "
                    f"SA {number(row['source_sa'])} → {number(row['product_sa'])}; "
                    f"Vina {number(before)} → {number(after)} kcal/mol; {candidate['status']}; "
                    f"selected={selected}; {row['reason']}"
                )
                + '</p><div class="pair">'
                + structure(case.smiles)
                + structure(graph.get("product_smiles"))
                + "</div>"
                + "<details><summary>LLM 提案与解释</summary><pre>"
                + html.escape(
                    json.dumps(candidate.get("proposal", {}).get("edits"), ensure_ascii=False)
                )
                + "</pre><p>"
                + html.escape(rationale)
                + "</p></details></article>"
            )
    source_writer.close()
    product_writer.close()
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparison.json", details)
    counts = {
        "source_count": len(dataset.cases),
        "proposal_count": len(rows),
        "source_vina_count": sum(r["source_vina"] is not None for r in rows),
        "paired_vina_count": sum(r["product_vina"] is not None for r in rows),
        "valid_changed_graph_count": sum(r["status"] == "accepted" for r in rows),
        "sa_improved_count": sum(
            r["delta_sa_source_minus_product"] is not None
            and r["delta_sa_source_minus_product"] > 0
            for r in rows
        ),
        "proxy_selected_count": sum(r["selected_proxy"] for r in rows),
        "sa_improved_with_binding_budget_count": sum(
            r["sa_improved_with_binding_budget"] for r in rows
        ),
        "sa_and_vina_both_improved_count": sum(
            r["delta_sa_source_minus_product"] is not None
            and r["delta_sa_source_minus_product"] > 0
            and r["delta_vina_product_minus_source"] is not None
            and r["delta_vina_product_minus_source"] < 0
            for r in rows
        ),
        "unparseable_product_count": sum(r["status"] == "rejected" for r in rows),
        "unchanged_count": sum(r["status"] == "unchanged" for r in rows),
        "conformer_generation_failure_count": sum(
            r["reason"] == "conformer_generation_failed" for r in rows
        ),
    }
    write_json(output / "paired_metrics.json", {"generated_at": utc_now(), **counts})
    header = [
        "# CK2 全局 pIC50 去重前 20：interaction-aware 优化",
        "",
        "使用全部 1,000 个原 single-Steer 终态中的有效连通分子，按 canonical isomeric SMILES 去重。"
        "每图保留最高 pIC50 代表，再取前 20；不附加 PB、应力或 SA 门槛。961 个有效粒子对应 401 个独立图。",
        "",
        "固定完整 3,332 条 SynCraft 参考语料、全队列检索排除、deepseek-flash、temperature=0、每例 K=1。"
        "interaction-aware 使用源分子重新 docking/PLIP 后的互作提示。执行原版图重构，关键互作未设硬冻结。",
        "",
        "Vina 为原分子和有效产物各自重新构象化后独立 redocking 的最佳预测分数；"
        "相同 CK2 受体/盒子、seed=42、exhaustiveness=8、单 CPU。原始 pIC50 不复制给产物。"
        "2D 结构图仅为分子图展示；导出的 SDF 是实际 Vina 对接坐标。",
        "",
        "本轮主评价是 SA 辅助指标与 Vina，不以 SA 改善宣称逆合成成功。"
        "起初完整流程联调的 SimpRetro 并行任务发生内存不足（exit 137），单独保留；"
        "主实验明确关闭逆合成，所有 not_run 均不解释为不可合成。",
        "",
        "此前接口联调另有 2 次真实 API 调用，保留在 `outputs/ck2_affinity_top20_interaction_20261020/`。"
        "它们不计入主实验的 20 次调用，不与主实验候选合并，也不作为独立科学重复。"
        "大环 G0–G3 是 Meeko 闭环虚拟点，修复后仅从 PLIP 复合物中排除这些虚拟点，真实 CG0–CG3 碳原子仍要求映射完整。",
        "",
        "20 个主实验案例均已结束。一个产物虽可解析，但原版 RDKit 构象生成失败（Bad Conformer Id），"
        "因此没有产物 Vina；该错误使整体 manifest 保持 partial，并非仍有案例在运行。",
        "",
        "图接受不等于选优通过。代理筛选要求相似度>0.2、Vina 恶化≤1 kcal/mol，再按 SA 排序；"
        "另统计 SA 确实改善且通过结合预算的分子数。无法解析产物直接跳过，失败后没有化学重采样。",
        "",
        "## 统计",
        "",
        *[f"- {k}: {v}" for k, v in counts.items()],
        "",
        "## 全部结果",
        "",
        "SA 越低越好；ΔVina=产物−原分子，正数表示预测结合变差。Vina 单位 kcal/mol。",
        "",
        "| 排名 | 源分子 | 原 pIC50 | SA 原→新 | Vina 原→新 | ΔVina | 结果 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        status = "代理筛选通过" if r["selected_proxy"] else r["reason"] or r["status"]
        header.append(
            f"| {r['rank']} | `{r['case_id']}` | {number(r['source_pic50'], 4)} | "
            f"{number(r['source_sa'], 4)}→{number(r['product_sa'], 4)} | "
            f"{number(r['source_vina'])}→{number(r['product_vina'])} | "
            f"{number(r['delta_vina_product_minus_source'])} | {status} |"
        )
    header += [
        "",
        "## 通过代理筛选的方案及互作保留",
        "",
        "| 分子 | SA 原→新 | Vina 原→新 | 精确接触保留率 | 类型/残基 recall |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| `{r['case_id']}` | {number(r['source_sa'], 4)}→{number(r['product_sa'], 4)} | "
            f"{number(r['source_vina'])}→{number(r['product_vina'])} | "
            f"{number(r['exact_contact_retention'] * 100, 1)}% | {number(r['paper_contact_recall'] * 100, 1)}% |"
            for r in rows
            if r["selected_proxy"]
        ],
        "",
        "选优只应用声明的相似度和 Vina 预算，不额外按互作保留率拒绝。"
        "两项获选方案的精确接触保留率仍低，不能将 SA 和 Vina 代理筛选通过解释为已保留原结合模式或已验证可合成性。",
        "",
        "完整编辑指令、LLM 解释、失败原因、互作保留与原始结果哈希见运行目录的 "
        "`comparison.json` / `comparison.csv`；结构对照见 `molecules.html`。",
        "",
        f"运行目录：`{output}`。输入：`{config.resolve(config.value.dataset)}`。",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(header), encoding="utf-8")
    (output / "molecules.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>CK2 interaction-aware TOP20</title>'
        "<style>body{font:16px system-ui;margin:32px;background:#f5f6f7;color:#17212b}"
        "article{background:white;padding:24px;margin:24px 0;border-radius:12px}"
        ".pair{display:flex;flex-wrap:wrap;align-items:center;gap:24px}.empty{width:460px;text-align:center}"
        "pre,p{overflow-wrap:anywhere;white-space:pre-wrap}svg{max-width:100%;height:auto}</style>"
        "<h1>CK2 pIC50 去重前 20：原分子 / 修改产物</h1>"
        "<p>左：原分子；右：修改产物。图为二维重绘，Vina 来自独立三维 redocking；SA 不能替代路线验证。</p>"
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
