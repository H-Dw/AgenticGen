"""SynCraft pass-K inference, evaluation and selection; optional flow evidence is orthogonal."""

from __future__ import annotations

from pathlib import Path

from agenticgen.base_config import FoundationConfiguration
from agenticgen.base_factory import FoundationComponents
from agenticgen.base_reporting import foundation_summary
from agenticgen.chemistry import graph_hash, mapped_smiles, similarity
from agenticgen.contracts import CaseDataset, Observation
from agenticgen.io import object_hash, sha256, utc_now, write_json
from agenticgen.selection import contact_retention, select_candidates


def foundation_run(
    config: FoundationConfiguration,
    dataset: CaseDataset,
    components: FoundationComponents,
    output: Path,
) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = config.value
    manifest = {
        "schema_version": "agenticgen.foundation-run.v1",
        "started_at": utc_now(),
        "status": "running",
        "effective_config": settings.model_dump(mode="json"),
        "dataset_sha256": sha256(config.resolve(settings.dataset)),
        "dependencies": components.provenance,
        "budget": {
            "logical_proposals": len(dataset.cases) * settings.total_pass_k,
            "proposal_modes": list(settings.proposal_modes),
            "pass_k_per_mode": settings.pass_k,
            "maximum_http_attempts": len(dataset.cases)
            * settings.total_pass_k
            * settings.llm.network_attempts
            if settings.llm.backend != "replay"
            else 0,
            "max_geometry_jobs": len(dataset.cases) * (1 + settings.total_pass_k)
            if settings.geometry.backend != "disabled"
            else 0,
            "max_retro_jobs": len(dataset.cases) * (1 + settings.total_pass_k)
            if settings.retro.backend != "disabled"
            else 0,
        },
        "llm_backend": settings.llm.backend,
        "method": "SynCraft foundation with native upstream graph reconstruction",
        "graph_reconstruction": "syncraft_native",
        "edit_constraints_scope": "prompt_guidance_only",
    }
    manifest["protocol_sha256"] = object_hash(
        {
            "config": manifest["effective_config"],
            "dataset": manifest["dataset_sha256"],
            "dependencies": manifest["dependencies"],
        }
    )
    write_json(output / "manifest.json", manifest)
    results = []
    # Deduplicate evaluations by exact graph/mapped graph AND evaluator protocol.
    # A binding cache retains map identities for contact comparisons.
    retro_cache, binding_cache = {}, {}

    def retro(smiles, folder):
        key = graph_hash(smiles)
        hit = key in retro_cache
        if not hit:
            retro_cache[key] = components.retro.evaluate(smiles, folder)
        return retro_cache[key], hit

    def binding(smiles, folder):
        key = smiles
        hit = key in binding_cache
        if not hit:
            binding_cache[key] = components.geometry.evaluate(smiles, folder)
        return binding_cache[key], hit

    try:
        for case in dataset.cases:
            folder = output / "cases" / case.case_id
            folder.mkdir(parents=True)
            maps = mapped_smiles(case.smiles)
            original_binding, source_binding_hit = binding(maps, folder / "geometry_original")
            write_json(folder / "binding_original.json", original_binding.model_dump(mode="json"))
            observation = Observation(
                case_id=case.case_id,
                target=case.target,
                source_smiles=case.smiles,
                mapped_smiles=maps,
                graph_sha256=graph_hash(case.smiles),
                terminal_metrics={},
                constraints=settings.constraints,
            )
            write_json(folder / "observation.json", observation.model_dump(mode="json"))
            records = []
            for index in range(settings.total_pass_k):
                mode = settings.proposal_modes[index // settings.pass_k]
                agent = (
                    components.agent.for_mode(mode) if settings.mode == "dual" else components.agent
                )
                try:
                    if mode == "interaction_aware" and original_binding.status != "ok":
                        raise ValueError(
                            "Source interaction preparation failed; standard fallback prohibited"
                        )
                    proposal = agent.propose(
                        observation,
                        original_binding,
                        index,
                        folder / "calls" / f"proposal_{index:02d}.json",
                    )
                    records.append(
                        {
                            "proposal_index": index,
                            "proposal_mode": mode,
                            "mode_proposal_index": index % settings.pass_k,
                            "status": "proposed",
                            "proposal": proposal.model_dump(mode="json"),
                        }
                    )
                except Exception as error:
                    records.append(
                        {
                            "proposal_index": index,
                            "proposal_mode": mode,
                            "mode_proposal_index": index % settings.pass_k,
                            "status": "interface_error",
                            "error_type": type(error).__name__,
                            "reason": str(error)
                            if isinstance(error, ValueError)
                            else "Agent failure; inspect call artifacts",
                        }
                    )
            # The complete fixed pass-K candidate set is frozen before any product score or route is seen.
            write_json(
                folder / "decision.json",
                {
                    "created_at": utc_now(),
                    "primary_proposal_index": 0,
                    "policy": "fixed_pass_k_before_evaluation",
                    "proposals": records,
                },
            )
            original_sa = components.properties.evaluate(case.smiles)
            original_retro, source_retro_hit = retro(case.smiles, folder / "retro_original")
            for record in records:
                if record["status"] != "proposed":
                    continue
                edited = components.editor.apply(
                    maps, record["proposal"]["edits"], settings.constraints
                )
                record.update(status=edited.status, graph=edited.model_dump(mode="json"))
                if edited.status != "accepted":
                    continue
                properties = components.properties.evaluate(edited.product_smiles)
                product_retro, retro_hit = retro(
                    edited.product_smiles, folder / f"retro_product_{record['proposal_index']:02d}"
                )
                product_binding, binding_hit = binding(
                    edited.product_mapped_smiles,
                    folder / f"geometry_product_{record['proposal_index']:02d}",
                )
                delta = (
                    original_sa.value - properties.value
                    if original_sa.status == properties.status == "ok"
                    and original_sa.value is not None
                    and properties.value is not None
                    else None
                )
                record.update(
                    properties=properties.model_dump(mode="json"),
                    delta_sa=delta,
                    sa_improved=delta > 0 if delta is not None else None,
                    retrosynthesis=product_retro.model_dump(mode="json"),
                    binding=product_binding.model_dump(mode="json"),
                    contact_retention=contact_retention(original_binding, product_binding),
                    cache_hits={"retro": retro_hit, "binding": binding_hit},
                    selection_similarity=similarity(
                        case.smiles, edited.product_smiles, settings.selection.similarity_bits
                    ),
                )
            selection = select_candidates(records, settings.selection, original_binding)
            per_mode = {
                mode: select_candidates(
                    [r for r in records if r["proposal_mode"] == mode],
                    settings.selection,
                    original_binding,
                )
                for mode in settings.proposal_modes
            }
            # SA+binding ranking remains a proxy winner. Independently rank the
            # subset with verified routes; never label an unverified proxy winner a rescue.
            route_records = [
                r
                for r in records
                if r.get("retrosynthesis", {}).get("status") == "route_found"
                and r["retrosynthesis"]["route_verified"]
            ]
            route_selection = select_candidates(route_records, settings.selection, original_binding)
            per_mode_routes = {
                mode: select_candidates(
                    [r for r in route_records if r["proposal_mode"] == mode],
                    settings.selection,
                    original_binding,
                )
                for mode in settings.proposal_modes
            }
            write_json(folder / "selection.json", selection)
            selected = next(
                (r for r in records if r["proposal_index"] == selection["selected_proposal_index"]),
                None,
            )
            missing = any(
                r["status"] == "interface_error"
                or r.get("retrosynthesis", {}).get("status") == "infrastructure_error"
                or r.get("binding", {}).get("status") == "infrastructure_error"
                or r.get("properties", {}).get("status") == "error"
                for r in records
            )
            result = {
                "case_id": case.case_id,
                "source": case.model_dump(mode="json"),
                "status": "partial"
                if missing
                or original_retro.status == "infrastructure_error"
                or original_binding.status == "infrastructure_error"
                or original_sa.status != "ok"
                else "completed",
                "source_sa": original_sa.model_dump(mode="json"),
                "source_retrosynthesis": original_retro.model_dump(mode="json"),
                "source_binding": original_binding.model_dump(mode="json"),
                "source_cache_hits": {"retro": source_retro_hit, "binding": source_binding_hit},
                "candidates": records,
                "selection": selection,
                "per_mode_selection": per_mode,
                "route_validated_selection": route_selection,
                "per_mode_route_validated_selection": per_mode_routes,
                "verified_route_rescue": (
                    route_selection["selected_proposal_index"] is not None
                    if original_retro.status == "no_route_within_budget"
                    else None
                ),
                "selected_product_smiles": selected["graph"]["product_smiles"]
                if selected
                else None,
                "route_rescue": (
                    bool(selected)
                    if settings.selection.policy == "paper_route_similarity"
                    and original_retro.status == "no_route_within_budget"
                    else None
                ),
            }
            write_json(folder / "result.json", result)
            results.append(result)
        summary = foundation_summary(results, settings, output)
        write_json(output / "summary.json", summary)
        (output / "accepted_candidates.smi").write_text(
            "".join(
                f"{candidate['graph']['product_smiles']}\t{case['case_id']}_p{candidate['proposal_index']}\n"
                for case in results
                for candidate in case["candidates"]
                if candidate["status"] == "accepted"
            ),
            encoding="utf-8",
        )
        (output / "products.smi").write_text(
            "".join(
                f"{r['selected_product_smiles']}\t{r['case_id']}\n"
                for r in results
                if r["selected_product_smiles"]
            ),
            encoding="utf-8",
        )
        manifest["status"] = (
            "partial" if any(r["status"] != "completed" for r in results) else "completed"
        )
        return summary
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        manifest["completed_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
