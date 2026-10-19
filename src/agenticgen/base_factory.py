"""Compose all SynCraft foundation modules independently of flow evidence."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from agenticgen.adapters.geometry import DisabledGeometry, SynCraftGeometry
from agenticgen.adapters.properties import RDKitSAEvaluator
from agenticgen.adapters.syncraft import SynCraftBackend
from agenticgen.base_config import FoundationConfiguration
from agenticgen.contracts import CaseDataset
from agenticgen.deepseek_native import DeepSeekNative
from agenticgen.editing import SynCraftEditor
from agenticgen.enhancements import EvidenceInjector
from agenticgen.factory import build_retrosynthesis
from agenticgen.io import sha256, source_fingerprint
from agenticgen.native import EvoNativeLLM, LiteLLMNative, NativeReplay, NativeSynCraftAgent
from agenticgen.ports import (
    GeometryPort,
    GraphEditorPort,
    PropertyEvaluatorPort,
    RetrosynthesisPort,
)


@dataclass(frozen=True)
class FoundationComponents:
    agent: NativeSynCraftAgent
    editor: GraphEditorPort
    properties: PropertyEvaluatorPort
    retro: RetrosynthesisPort
    geometry: GeometryPort
    provenance: dict


def build_foundation(config: FoundationConfiguration, dataset: CaseDataset) -> FoundationComponents:
    settings = config.value
    root = config.resolve(settings.syncraft.root)
    backend = SynCraftBackend(
        root,
        examples_file=config.resolve(settings.syncraft.examples_file)
        if settings.syncraft.examples_file
        else None,
        excluded_smiles={c.smiles for c in dataset.cases},
    )
    if (
        settings.syncraft.expected_revision
        and backend.provenance["revision"] != settings.syncraft.expected_revision
    ):
        raise ValueError("SynCraft revision differs from the configured foundation")
    if settings.syncraft.few_shot_count > len(backend.examples):
        raise ValueError("Insufficient eligible SynCraft corpus examples")
    evidence = EvidenceInjector(
        settings.enhancements,
        config.resolve(settings.enhancements.packets_file)
        if settings.enhancements.enabled
        else None,
        dataset,
    )
    provenance = {
        "versions": {name: version(name) for name in ("agenticgen", "rdkit", "pydantic", "httpx")},
        "syncraft": backend.provenance,
        "enhancements": evidence.provenance,
        "native_upstream": source_fingerprint(
            root,
            [
                "src/inference_enhanced.py",
                "src/inference_bioactivity_constrain.py",
                "src/utils.py",
                "src/extract_interaction.py",
                "skill/syncraft/pipeline.py",
            ],
        ),
    }
    llm = settings.llm
    if llm.backend == "deepseek_api":
        provider = DeepSeekNative(llm)
        provenance["transport"] = {
            "backend": "deepseek_api",
            "wire_format": "json_envelope_for_native_output",
            "protocol": "deepseek-flash, thinking disabled, temperature 0; EvoMolSteer-compatible",
        }
    elif llm.backend == "evomolsteer_deepseek":
        provider = EvoNativeLLM(config.resolve(llm.evomolsteer_root), llm)
        provenance["transport"] = {
            "backend": llm.backend,
            "wire_format": "json_envelope_for_native_output",
            "evo": source_fingerprint(
                config.resolve(llm.evomolsteer_root), ["src/evomolsteer/deepseek_api.py"]
            ),
        }
    elif llm.backend == "litellm":
        provider = LiteLLMNative(llm)
        provenance["transport"] = {"backend": "litellm", "wire_format": "native_text"}
    else:
        provider = NativeReplay(config.resolve(llm.replay_file))
        for case in dataset.cases:
            if (
                len(provider.data.get("responses", {}).get(case.case_id, []))
                < settings.total_pass_k
            ):
                raise ValueError("Native replay fixture does not cover every pass")
        provenance["transport"] = {"backend": "engineering_replay", "scientific_result": False}
    retro = build_retrosynthesis(config)
    geometry = (
        DisabledGeometry() if settings.geometry.backend == "disabled" else SynCraftGeometry(config)
    )
    if settings.retro.backend != "disabled":
        provenance["retrosynthesis"] = retro.protocol
    if settings.geometry.backend != "disabled":
        provenance["geometry"] = geometry.protocol
    source_dir = Path(__file__).parent
    provenance["source_files"] = {
        p.relative_to(source_dir).as_posix(): sha256(p)
        for p in sorted(source_dir.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".txt"}
    }
    return FoundationComponents(
        NativeSynCraftAgent(
            backend,
            provider,
            settings.syncraft.few_shot_count,
            settings.proposal_modes[0],
            evidence,
        ),
        SynCraftEditor(backend),
        RDKitSAEvaluator(),
        retro,
        geometry,
        provenance,
    )
