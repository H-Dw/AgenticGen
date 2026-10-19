"""Composition root. New backends implement ports; orchestration needs no backend branches."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from agenticgen.adapters.llm import EvoMolSteerDeepSeek, ReplayLLM
from agenticgen.adapters.properties import RDKitSAEvaluator
from agenticgen.adapters.retro import DisabledRetrosynthesis, SimpRetroCLI
from agenticgen.adapters.syncraft import SynCraftBackend
from agenticgen.agent import DirectionAgent, SynCraftPromptBuilder
from agenticgen.config import Configuration
from agenticgen.contracts import CaseDataset
from agenticgen.editing import SynCraftEditor
from agenticgen.errors import DependencyError
from agenticgen.io import sha256, source_fingerprint
from agenticgen.ports import GraphEditorPort, PropertyEvaluatorPort, RetrosynthesisPort


@dataclass(frozen=True)
class Components:
    agent: DirectionAgent
    editor: GraphEditorPort
    properties: PropertyEvaluatorPort
    retro: RetrosynthesisPort
    provenance: dict[str, Any]


def build_retrosynthesis(config: Configuration) -> RetrosynthesisPort:
    """Construct the evaluator without loading an LLM or SynCraft."""
    settings = config.value.retro
    if settings.backend == "disabled":
        return DisabledRetrosynthesis()
    return SimpRetroCLI(
        root=config.resolve(settings.root),
        python=config.resolve(settings.python) if settings.python else None,
        execution=settings.execution,
        image=settings.image,
        inventory=config.resolve(settings.inventory),
        templates=config.resolve(settings.templates),
        fingerprint_cache=config.resolve(settings.fingerprint_cache)
        if settings.fingerprint_cache
        else None,
        inventory_index=config.resolve(settings.inventory_index)
        if settings.inventory_index
        else None,
        time_limit_seconds=settings.time_limit_seconds,
        process_grace_seconds=settings.process_grace_seconds,
        cpu_threads=settings.cpu_threads,
        expected_revision=settings.expected_revision,
    )


def build_components(config: Configuration, dataset: CaseDataset) -> Components:
    settings = config.value
    syncraft = SynCraftBackend(
        config.resolve(settings.syncraft.root),
        examples_file=config.resolve(settings.syncraft.examples_file)
        if settings.syncraft.examples_file
        else None,
        excluded_smiles={case.smiles for case in dataset.cases},
    )
    if (
        settings.syncraft.expected_revision
        and syncraft.provenance["revision"] != settings.syncraft.expected_revision
    ):
        raise DependencyError("SynCraft revision differs from configured revision")
    if settings.syncraft.few_shot_count > len(syncraft.examples):
        raise DependencyError(
            "Insufficient eligible SynCraft examples for the configured retrieval count"
        )
    provenance = {
        "syncraft": syncraft.provenance,
        "versions": {name: version(name) for name in ("agenticgen", "rdkit", "pydantic", "httpx")},
    }
    llm_config = settings.llm
    source_dir = Path(__file__).resolve().parent
    provenance["agenticgen_source"] = {
        path.relative_to(source_dir).as_posix(): sha256(path)
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".txt"}
    }
    if llm_config.backend == "evomolsteer_deepseek":
        root = config.resolve(llm_config.evomolsteer_root)
        llm = EvoMolSteerDeepSeek(
            root,
            model=llm_config.model,
            timeout=llm_config.timeout_seconds,
            attempts=llm_config.network_attempts,
            max_output_tokens=llm_config.max_output_tokens,
            max_format_repairs=llm_config.max_format_repairs,
        )
        provenance["evomolsteer"] = source_fingerprint(
            root,
            [
                "src/evomolsteer/deepseek_api.py",
                "src/evomolsteer/io.py",
                "src/evomolsteer/json_syntax.py",
            ],
        )
    else:
        if not llm_config.replay_file:
            raise ValueError("Explicit replay backend requires replay_file")
        llm = ReplayLLM(config.resolve(llm_config.replay_file))
        for case in dataset.cases:
            if (
                len(llm.data.get("responses", {}).get(case.case_id, []))
                < settings.proposals_per_case
            ):
                raise DependencyError(
                    "Replay fixture does not cover every configured case/proposal"
                )
    retro = build_retrosynthesis(config)
    if isinstance(retro, SimpRetroCLI):
        provenance["retrosynthesis"] = retro.protocol
    prompts = SynCraftPromptBuilder(syncraft, settings.syncraft.few_shot_count)
    return Components(
        agent=DirectionAgent(llm, prompts),
        editor=SynCraftEditor(syncraft),
        properties=RDKitSAEvaluator(),
        retro=retro,
        provenance=provenance,
    )
