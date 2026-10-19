"""Audit saved prompts/reconstruction and optionally remove only host edit guidance."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, rdBase

from agenticgen.base_config import load_foundation
from agenticgen.base_factory import build_foundation
from agenticgen.chemistry import mapped_smiles
from agenticgen.contracts import BindingResult, Observation
from agenticgen.dataset import load_dataset
from agenticgen.deepseek_native import DeepSeekNative
from agenticgen.io import read_json, sha256, utc_now, write_json


def chemistry_diagnosis(smiles):
    parsed = Chem.MolFromSmiles(smiles) if smiles else None
    raw = Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
    diagnostics = []
    nonring = []
    if raw is not None:
        raw.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(raw)
        nonring = [
            a.GetAtomMapNum() for a in raw.GetAtoms() if a.GetIsAromatic() and not a.IsInRing()
        ]
        for problem in Chem.DetectChemistryProblems(raw):
            item = {"type": problem.GetType(), "message": problem.Message()}
            indices = (
                [problem.GetAtomIdx()]
                if hasattr(problem, "GetAtomIdx")
                else list(problem.GetAtomIndices())
                if hasattr(problem, "GetAtomIndices")
                else []
            )
            item["atom_maps"] = [raw.GetAtomWithIdx(i).GetAtomMapNum() for i in indices]
            diagnostics.append(item)
    return {
        "parseable": parsed is not None,
        "problem_types": [p["type"] for p in diagnostics],
        "problems": diagnostics,
        "nonring_aromatic_atom_maps": nonring,
        "fragment_count": len(Chem.GetMolFrags(raw)) if raw is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-prompt-ablation", action="store_true")
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    config = load_foundation(Path(read_json(run / "manifest.json")["config"]))
    dataset = load_dataset(config.resolve(config.value.dataset))
    components = build_foundation(config, dataset)
    output.mkdir(parents=True, exist_ok=False)
    backend = components.editor.backend
    cases = []
    examples = []
    for case in dataset.cases:
        folder = run / "runs" / case.case_id / "cases" / case.case_id
        result = read_json(folder / "result.json")
        candidate = result["candidates"][0]
        observation = Observation.model_validate(read_json(folder / "observation.json"))
        binding = BindingResult.model_validate(result["source_binding"])
        request = read_json(folder / "calls/proposal_00.request.json")["body"]
        expected = components.agent.messages(observation, binding)
        messages = request["messages"]
        upstream = backend.editor.reconstruct_from_sequence(
            mapped_smiles(case.smiles), copy.deepcopy(candidate["proposal"]["edits"])
        )
        diagnosis = chemistry_diagnosis(upstream["smiles_with_maps"])
        interactions = binding.interactions
        critical = {
            m
            for i in interactions
            if i.kind in {"HydrogenBond", "SaltBridge", "PiStacking", "PiCation", "HalogenBond"}
            for m in i.atom_maps
        }
        explicit_deletions = {op[1] for op in candidate["proposal"]["edits"] if op[0] == "DEL_ATOM"}
        audit = {
            "case_id": case.case_id,
            "rank": case.selection_rank,
            "saved_status": candidate["status"],
            "upstream_mapped_smiles": upstream["smiles_with_maps"],
            "upstream_plain_smiles": upstream["smiles"],
            "direct_upstream_matches_saved": upstream["smiles_with_maps"]
            == candidate["graph"]["upstream_mapped_smiles"]
            and upstream["smiles"] == candidate["graph"]["upstream_smiles"],
            "diagnosis": diagnosis,
            "upstream_was_fragmented": upstream["was_fragmented"],
            "native_system_and_user_exact_match": messages[:2] == expected[:2],
            "extra_host_message": messages[2],
            "transport_message": messages[3],
            "request_sha256": sha256(folder / "calls/proposal_00.request.json"),
            "explicitly_deleted_critical_maps": sorted(explicit_deletions & critical),
            "edits": candidate["proposal"]["edits"],
        }
        cases.append(audit)
        for example in backend.retrieve(case.smiles, config.value.syncraft.few_shot_count):
            reconstructed = backend.editor.reconstruct_from_sequence(
                example["source"], copy.deepcopy(example["edits"])
            )
            source = Chem.MolFromSmiles(example["source"])
            examples.append(
                {
                    "case_id": case.case_id,
                    "source": example["source"],
                    "source_has_maps": all(a.GetAtomMapNum() > 0 for a in source.GetAtoms()),
                    "native_parseable": Chem.MolFromSmiles(reconstructed["smiles"]) is not None,
                    "has_zero_index_edit": any(
                        isinstance(x, int) and x == 0 for op in example["edits"] for x in op[1:]
                    ),
                }
            )
    write_json(
        output / "audit.json",
        {
            "rdkit": rdBase.rdkitVersion,
            "cases": cases,
            "retrieved_examples": examples,
            "dependencies": components.provenance,
            "created_at": utc_now(),
        },
    )
    write_json(
        output / "requests.json",
        [
            {"case_id": c["case_id"], "smiles": mapped_smiles(d.smiles), "edits": c["edits"]}
            for c, d in zip(cases, dataset.cases, strict=True)
        ],
    )
    summary = {
        "case_count": len(cases),
        "native_parity_count": sum(c["direct_upstream_matches_saved"] for c in cases),
        "native_prompt_prefix_match_count": sum(
            c["native_system_and_user_exact_match"] for c in cases
        ),
        "parseable_count": sum(c["diagnosis"]["parseable"] for c in cases),
        "problem_counts": dict(
            Counter(p for c in cases for p in set(c["diagnosis"]["problem_types"]))
        ),
        "retrieved_example_exposures": len(examples),
        "parseable_example_exposures": sum(e["native_parseable"] for e in examples),
        "unmapped_example_exposures": sum(not e["source_has_maps"] for e in examples),
        "zero_index_example_exposures": sum(e["has_zero_index_edit"] for e in examples),
        "explicit_critical_deletion_case_count": sum(
            bool(c["explicitly_deleted_critical_maps"]) for c in cases
        ),
    }
    write_json(output / "summary.json", summary)
    print(summary, flush=True)
    if not args.native_prompt_ablation:
        return
    write_json(
        output / "ablation_manifest.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "change": "Remove only host Editing guidance message; retain exact native system/user and JSON transport envelope",
            "provider_settings": config.value.llm.model_dump(mode="json"),
            "planned_new_api_calls": len(cases),
            "comparison": "Exploratory rerun against saved baseline, not a contemporaneously randomized replicated experiment",
        },
    )

    def rerun(case):
        folder = run / "runs" / case.case_id / "cases" / case.case_id
        request = read_json(folder / "calls/proposal_00.request.json")["body"]
        api = DeepSeekNative(config.value.llm)
        path = output / "ablation" / case.case_id / "proposal_00.json"
        try:
            text = api.generate(request["messages"][:2], case.case_id, 0, path)
        except (ValueError, RuntimeError) as error:
            value = {"case_id": case.case_id, "status": "interface_failure", "error": str(error)}
            write_json(path.parent / "result.json", value)
            return value
        parsed = backend.inference.parse_llm_output(text)
        if parsed.get("error"):
            value = {
                "case_id": case.case_id,
                "status": "output_format_failure",
                "error": parsed["error"],
            }
        else:
            upstream = backend.editor.reconstruct_from_sequence(
                mapped_smiles(case.smiles), copy.deepcopy(parsed["edits"])
            )
            value = {
                "case_id": case.case_id,
                "status": "reconstructed",
                "edits": parsed["edits"],
                "upstream_smiles": upstream["smiles"],
                "upstream_mapped_smiles": upstream["smiles_with_maps"],
                "diagnosis": chemistry_diagnosis(upstream["smiles_with_maps"]),
                "unchanged": parsed["edits"] == [["STOP"]],
            }
        write_json(path.parent / "result.json", value)
        return value

    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(rerun, case) for case in dataset.cases]):
            value = future.result()
            outcomes.append(value)
            print(
                {
                    "ablation_finished": len(outcomes),
                    "case_id": value["case_id"],
                    "parseable": value.get("diagnosis", {}).get("parseable"),
                },
                flush=True,
            )
    write_json(output / "ablation_results.json", outcomes)
    manifest = read_json(output / "ablation_manifest.json")
    manifest.update(status="completed", ended_at=utc_now(), completed_cases=len(outcomes))
    write_json(output / "ablation_manifest.json", manifest)


if __name__ == "__main__":
    main()
