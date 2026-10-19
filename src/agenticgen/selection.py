"""Explicit SynCraft paper and released-skill selection rules."""

from __future__ import annotations

from agenticgen.base_config import SelectionConfig
from agenticgen.chemistry import graph_hash
from agenticgen.contracts import BindingResult


def select_candidates(
    candidates: list[dict], settings: SelectionConfig, original_binding: BindingResult
) -> dict:
    eligible, excluded, seen_graphs = [], [], set()
    for record in candidates:
        index = record["proposal_index"]
        reason = None
        if record["status"] != "accepted":
            reason = "graph_not_accepted"
        elif settings.policy == "proposal_only":
            # Inference-only execution intentionally produces no optimization winner.
            reason = "evaluation_disabled"
        elif record["selection_similarity"] <= settings.similarity_threshold:
            reason = "similarity_threshold"
        elif settings.policy == "paper_route_similarity":
            retro = record["retrosynthesis"]
            if retro["status"] != "route_found":
                reason = "route_not_found_or_unavailable"
            elif settings.require_verified_route and not retro["route_verified"]:
                reason = "route_not_verified"
        elif settings.policy == "skill_sa_binding":
            binding = record["binding"]
            if original_binding.status != "ok" or binding["status"] != "ok":
                reason = "docking_unavailable"
            elif (
                binding["score_kcal_mol"] - original_binding.score_kcal_mol
                > settings.vina_loss_budget
            ):
                reason = "vina_loss_budget"
            elif (
                record["properties"]["status"] != "ok" or record["properties"].get("value") is None
            ):
                reason = "sa_unavailable"  # Never silently switch to affinity-only ranking.
        if reason:
            excluded.append({"proposal_index": index, "reason": reason})
        else:
            graph = record.get("graph", {}).get("product_smiles")
            if graph:
                key = graph_hash(graph)
                if key in seen_graphs:
                    excluded.append({"proposal_index": index, "reason": "duplicate_product_graph"})
                    continue
                seen_graphs.add(key)
            eligible.append(record)
    if settings.policy == "paper_route_similarity":
        ranked = sorted(eligible, key=lambda r: (-r["selection_similarity"], r["proposal_index"]))
    else:
        ranked = sorted(
            eligible,
            key=lambda r: (
                r["properties"]["value"],
                -r["selection_similarity"],
                r["proposal_index"],
            ),
        )
    return {
        "policy": settings.policy,
        "selected_proposal_index": ranked[0]["proposal_index"] if ranked else None,
        "eligible_proposal_indices": [r["proposal_index"] for r in ranked],
        "excluded": excluded,
        "settings": settings.model_dump(mode="json"),
        "selection_timing": "after_fixed_pass_k_proposals_and_external_evaluation",
    }


def contact_retention(original: BindingResult, product: BindingResult) -> dict:
    if original.status != "ok" or product.status != "ok":
        return {"status": "unavailable", "fraction": None}

    def keys(binding):
        return {(i.kind, i.residue, tuple(sorted(i.atom_maps))) for i in binding.interactions}

    source, edited = keys(original), keys(product)
    paper_source = {(i.kind, i.residue) for i in original.interactions if i.kind != "MetalComplex"}
    paper_edited = {(i.kind, i.residue) for i in product.interactions if i.kind != "MetalComplex"}
    return {
        "status": "ok",
        "source_contact_count": len(source),
        "retained_count": len(source & edited),
        "fraction": len(source & edited) / len(source) if source else None,
        "paper_recall": len(paper_source & paper_edited) / len(paper_source)
        if paper_source
        else None,
        "paper_jaccard": len(paper_source & paper_edited) / len(paper_source | paper_edited)
        if paper_source | paper_edited
        else None,
        "paper_definition": "interaction type/residue sets, without ligand-atom IDs; excludes metal contacts",
        "definition": "exact contact type/residue/source atom-map set; consistent input maps are required",
    }
