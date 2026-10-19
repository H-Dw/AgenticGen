"""Command-line entry points; preflight performs no API or retrosynthesis job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agenticgen import __version__
from agenticgen.base_config import load_foundation
from agenticgen.base_factory import build_foundation
from agenticgen.base_reporting import rebuild_foundation_summary
from agenticgen.base_runner import foundation_run
from agenticgen.cohorts import prepare_ck2_sa
from agenticgen.config import load_configuration
from agenticgen.corpus import build_corpus
from agenticgen.dataset import load_dataset, prepare_ck2
from agenticgen.design_runner import design_run
from agenticgen.factory import build_components, build_retrosynthesis
from agenticgen.inpainting_runner import inpainting_run, load_inpainting
from agenticgen.inpainting_selection import rebuild_inpainting_selection
from agenticgen.io import read_json
from agenticgen.receptor import prepare_receptor
from agenticgen.reporting import rebuild_summary
from agenticgen.retro_screen import screen_targets
from agenticgen.runner import run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="agenticgen")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "design"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument(
            "--backend", choices=("deepseek_api", "evomolsteer_deepseek", "litellm", "replay")
        )
        if name in {"run", "design"}:
            child.add_argument("--output", type=Path, required=True)
    inpaint = commands.add_parser("inpaint")
    inpaint.add_argument("--run", type=Path, required=True)
    inpaint.add_argument("--config", type=Path, required=True)
    inpaint.add_argument("--pocket-pdb", type=Path, required=True)
    inpaint.add_argument("--output", type=Path, required=True)
    inpaint.add_argument("--execute", action="store_true")
    inpaint.add_argument("--evaluation-config", type=Path)
    inpaint.add_argument("--redock", action="store_true")
    prepare = commands.add_parser("prepare-ck2")
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=3)
    cohort = commands.add_parser("prepare-ck2-sa")
    cohort.add_argument("--workspace", type=Path, required=True)
    cohort.add_argument("--output", type=Path, required=True)
    cohort.add_argument("--count-per-stratum", type=int, default=3)
    retro = commands.add_parser("screen-retro")
    retro.add_argument("--config", type=Path, required=True)
    retro.add_argument("--targets", type=Path, required=True)
    retro.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run", type=Path, required=True)
    corpus = commands.add_parser("build-corpus")
    corpus.add_argument("--syncraft-root", type=Path, required=True)
    corpus.add_argument("--pairs", type=Path, required=True)
    corpus.add_argument("--output", type=Path, required=True)
    receptor = commands.add_parser("prepare-ck2-receptor")
    receptor.add_argument("--protein", type=Path, required=True)
    receptor.add_argument("--reference-ligand", type=Path, required=True)
    receptor.add_argument("--output", type=Path, required=True)
    receptor.add_argument("--image", default="agenticgen-syncraft-worker:20261015")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "inpaint":
            summary = inpainting_run(
                args.run,
                load_inpainting(args.config),
                args.pocket_pdb,
                args.output,
                execute=args.execute,
                evaluation=load_foundation(args.evaluation_config)
                if args.evaluation_config
                else None,
                redock=args.redock,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if read_json(args.output / "manifest.json")["status"] == "completed" else 2
        if args.command == "prepare-ck2-receptor":
            print(
                json.dumps(
                    prepare_receptor(
                        args.protein.resolve(),
                        args.reference_ligand.resolve(),
                        args.output.resolve(),
                        args.image,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.command == "build-corpus":
            print(
                json.dumps(
                    build_corpus(
                        args.syncraft_root.resolve(), args.pairs.resolve(), args.output.resolve()
                    ),
                    indent=2,
                )
            )
            return 0
        if args.command == "summarize":
            if (
                read_json(args.run / "manifest.json").get("schema_version")
                == "agenticgen.inpainting-run.v1"
            ):
                print(
                    json.dumps(
                        rebuild_inpainting_selection(args.run.resolve()),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            summarize = (
                rebuild_foundation_summary
                if read_json(args.run / "manifest.json").get("schema_version")
                == "agenticgen.foundation-run.v1"
                else rebuild_summary
            )
            print(json.dumps(summarize(args.run.resolve()), ensure_ascii=False, indent=2))
            return 0
        if args.command == "prepare-ck2":
            dataset = prepare_ck2(args.workspace.resolve(), args.output.resolve(), args.count)
            print(json.dumps({"case_ids": [c.case_id for c in dataset.cases]}, indent=2))
            return 0
        if args.command == "prepare-ck2-sa":
            dataset = prepare_ck2_sa(
                args.workspace.resolve(), args.output.resolve(), args.count_per_stratum
            )
            print(json.dumps({"case_ids": [c.case_id for c in dataset.cases]}, indent=2))
            return 0
        if args.command == "screen-retro":
            config = (
                load_foundation(args.config)
                if read_json(args.config).get("schema_version") == "agenticgen.syncraft.v1"
                else load_configuration(args.config)
            )
            if config.value.retro.backend == "disabled":
                raise ValueError("Configure a real retrosynthesis backend before screening")
            evaluator = build_retrosynthesis(config)
            summary = screen_targets(
                args.targets.resolve(), args.output.resolve(), evaluator, evaluator.protocol
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if read_json(args.output / "manifest.json")["status"] == "completed" else 2
        if read_json(args.config).get("schema_version") == "agenticgen.syncraft.v1":
            config = load_foundation(args.config, args.backend)
            dataset = load_dataset(config.resolve(config.value.dataset))
            components = build_foundation(config, dataset)
            ready = config.value.llm.backend not in {
                "deepseek_api",
                "evomolsteer_deepseek",
            } or bool(os.environ.get("DEEPSEEK_API_KEY"))
            if args.command == "preflight":
                print(
                    json.dumps(
                        {
                            "ready": ready,
                            "case_count": len(dataset.cases),
                            "mode": config.value.mode,
                            "selection_policy": config.value.selection.policy,
                            "max_logical_proposals": len(dataset.cases) * config.value.total_pass_k,
                            "proposal_modes": list(config.value.proposal_modes),
                            "enhancements_enabled": config.value.enhancements.enabled,
                            "dependencies": components.provenance,
                            "jobs_executed": 0,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0 if ready else 2
            if not ready:
                raise ValueError("DEEPSEEK_API_KEY is absent")
            runner = design_run if args.command == "design" else foundation_run
            summary = runner(config, dataset, components, args.output)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if read_json(args.output / "manifest.json")["status"] == "completed" else 2
        if args.command == "design":
            raise ValueError("design requires a SynCraft foundation profile")
        config = load_configuration(args.config, args.backend)
        dataset = load_dataset(config.resolve(config.value.dataset))
        components = build_components(config, dataset)
        key_ready = bool(os.environ.get("DEEPSEEK_API_KEY"))
        ready = config.value.llm.backend == "replay" or key_ready
        if args.command == "preflight":
            print(
                json.dumps(
                    {
                        "ready": ready,
                        "case_count": len(dataset.cases),
                        "llm_backend": config.value.llm.backend,
                        "model": config.value.llm.model,
                        "api_key_present": key_ready,
                        "retrosynthesis": config.value.retro.backend,
                        "max_logical_proposals": len(dataset.cases)
                        * config.value.proposals_per_case,
                        "dependencies": components.provenance,
                        "jobs_executed": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if ready else 2
        if not ready:
            raise ValueError(
                "DEEPSEEK_API_KEY is absent. Configure it, or explicitly select --backend replay"
            )
        summary = run(config, dataset, components, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if read_json(args.output / "manifest.json")["status"] == "completed" else 2
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
