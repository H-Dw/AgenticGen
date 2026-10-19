"""Freeze native SynCraft proposals and reconstruct graphs before product evaluation."""

from pathlib import Path

from agenticgen.base_config import FoundationConfiguration
from agenticgen.base_factory import FoundationComponents
from agenticgen.chemistry import graph_hash, mapped_smiles
from agenticgen.contracts import CaseDataset, Observation
from agenticgen.io import sha256, utc_now, write_json


def design_run(
    config: FoundationConfiguration,
    dataset: CaseDataset,
    components: FoundationComponents,
    output: Path,
) -> dict:
    """Run source contacts -> LLM -> native edit. All product evaluation is deferred.

    This deliberately does not rank designs or label an SA/route improvement. It
    supplies the same native agent, prompt retrieval, pass-K and dual-mode policy
    as the complete SynCraft foundation, for the inpainting workflow to consume.
    """
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = config.value
    manifest = {
        "schema_version": "agenticgen.design-run.v1",
        "status": "running",
        "started_at": utc_now(),
        "effective_config": settings.model_dump(mode="json"),
        "dataset_sha256": sha256(config.resolve(settings.dataset)),
        "dependencies": components.provenance,
        "execution_kind": "engineering_replay" if settings.llm.backend == "replay" else "real_api",
        "product_evaluation": "deferred_until_after_inpainting",
        "graph_reconstruction": "syncraft_native",
        "pass_k_per_mode": settings.pass_k,
        "proposal_modes": list(settings.proposal_modes),
    }
    write_json(output / "manifest.json", manifest)
    results = []
    try:
        for case in dataset.cases:
            folder = output / "cases" / case.case_id
            folder.mkdir(parents=True)
            maps = mapped_smiles(case.smiles)
            binding = components.geometry.evaluate(maps, folder / "geometry_original")
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
                try:
                    if mode == "interaction_aware" and binding.status != "ok":
                        raise ValueError("Source contacts unavailable; no silent prompt fallback")
                    agent = (
                        components.agent.for_mode(mode)
                        if settings.mode == "dual"
                        else components.agent
                    )
                    proposal = agent.propose(
                        observation, binding, index, folder / "calls" / f"proposal_{index:02d}.json"
                    )
                    records.append(
                        {
                            "proposal_index": index,
                            "proposal_mode": mode,
                            "status": "proposed",
                            "proposal": proposal.model_dump(mode="json"),
                        }
                    )
                except Exception as error:
                    records.append(
                        {
                            "proposal_index": index,
                            "proposal_mode": mode,
                            "status": "interface_error",
                            "reason": str(error)
                            if isinstance(error, ValueError)
                            else type(error).__name__,
                        }
                    )
            write_json(
                folder / "decision.json",
                {
                    "policy": "fixed_pass_k_before_product_evaluation",
                    "proposals": records,
                },
            )
            for record in records:
                if record["status"] != "proposed":
                    continue
                graph = components.editor.apply(
                    maps, record["proposal"]["edits"], settings.constraints
                )
                record.update(status=graph.status, graph=graph.model_dump(mode="json"))
                if graph.status == "rejected":
                    record["next_stage"] = "ignored_unparseable_graph"
                else:
                    record["next_stage"] = "inpainting_before_product_evaluation"
            result = {
                "case_id": case.case_id,
                "source": case.model_dump(mode="json"),
                "source_binding": binding.model_dump(mode="json"),
                "candidates": records,
            }
            write_json(folder / "result.json", result)
            results.append(result)
        records = [r for case in results for r in case["candidates"]]
        summary = {
            "schema_version": "agenticgen.design-summary.v1",
            "case_count": len(results),
            "proposal_count": len(records),
            "accepted_graph_count": sum(r["status"] == "accepted" for r in records),
            "ignored_unparseable_graph_count": sum(r["status"] == "rejected" for r in records),
            "interface_error_count": sum(r["status"] == "interface_error" for r in records),
            "product_evaluations": 0,
            "optimization_winner": None,
            "execution_kind": manifest["execution_kind"],
        }
        write_json(output / "summary.json", summary)
        manifest["status"] = "partial" if summary["interface_error_count"] else "completed"
        return summary
    except BaseException as error:
        manifest.update(status="failed", failure_type=type(error).__name__)
        raise
    finally:
        manifest["completed_at"] = utc_now()
        write_json(output / "manifest.json", manifest)
