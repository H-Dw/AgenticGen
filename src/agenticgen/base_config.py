"""SynCraft foundation profiles; independent of optional generation evidence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenticgen.config import RetroConfig, SynCraftConfig
from agenticgen.contracts import EditConstraints, Record
from agenticgen.io import read_json


class FoundationLLM(Record):
    backend: Literal["deepseek_api", "evomolsteer_deepseek", "litellm", "replay"] = "deepseek_api"
    model: str = "deepseek-flash"
    evomolsteer_root: str | None = None
    timeout_seconds: int = Field(default=240, ge=1, le=1200)
    network_attempts: int = Field(default=3, ge=1, le=5)
    max_output_tokens: int = Field(default=8192, ge=512, le=65536)
    temperature: float = Field(default=0, ge=0, le=2)
    reasoning_effort: str | None = None
    replay_file: str | None = None

    @model_validator(mode="after")
    def transport_contract(self):
        if self.backend == "deepseek_api" and (
            self.model != "deepseek-flash"
            or self.temperature != 0
            or self.reasoning_effort not in {None, "high"}
        ):
            raise ValueError(
                "DeepSeek API requires deepseek-flash, configured temperature=0, "
                "and explicit high or legacy disabled thinking"
            )
        if self.backend == "evomolsteer_deepseek" and (
            not self.evomolsteer_root
            or self.model != "deepseek-flash"
            or self.temperature != 0
            or self.reasoning_effort is not None
        ):
            raise ValueError(
                "EvoMolSteer requires deepseek-flash, temperature=0, disabled thinking"
            )
        if self.backend == "replay" and not self.replay_file:
            raise ValueError("Replay requires an explicit engineering fixture")
        if self.backend == "litellm" and not self.model.strip():
            raise ValueError("LiteLLM requires an explicit model ID")
        return self


class GeometryConfig(Record):
    backend: Literal["disabled", "syncraft_worker"] = "disabled"
    execution: Literal["python", "docker"] = "python"
    python: str | None = None
    image: str = "agenticgen-syncraft-worker:20261015"
    receptor_pdbqt: str | None = None
    receptor_pdb: str | None = None
    vina_config: str | None = None
    vina_binary: str = "vina"
    timeout_seconds: int = Field(default=1200, ge=30)
    cpu_threads: int = Field(default=1, ge=1, le=64)
    seed: int = Field(default=42, ge=1, le=2147483647)
    exhaustiveness: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def require_receptor(self):
        if self.backend == "syncraft_worker" and (
            not all((self.receptor_pdbqt, self.receptor_pdb, self.vina_config))
            or (self.execution == "python" and not self.python)
        ):
            raise ValueError("Geometry requires receptor PDBQT/PDB, box, and a worker runtime")
        return self


class SelectionConfig(Record):
    policy: Literal["paper_route_similarity", "skill_sa_binding", "proposal_only"] = (
        "paper_route_similarity"
    )
    similarity_threshold: float = Field(default=0.5, ge=0, le=1)
    similarity_bits: Literal[2048, 4096] = 4096
    vina_loss_budget: float = Field(default=1, ge=0)
    require_verified_route: bool = False


class EnhancementConfig(Record):
    enabled: bool = False
    packets_file: str | None = None
    allowed_kinds: tuple[Literal["flow_history", "native_exploration", "case_memory"], ...] = (
        "flow_history",
    )
    max_characters: int = Field(default=16000, ge=1, le=100000)

    @model_validator(mode="after")
    def require_packets(self):
        if self.enabled and not self.packets_file:
            raise ValueError("Enabled generation evidence requires a frozen packets file")
        return self


class FoundationConfig(Record):
    schema_version: Literal["agenticgen.syncraft.v1"] = "agenticgen.syncraft.v1"
    dataset: str
    mode: Literal["standard", "interaction_aware", "dual"] = "standard"
    pass_k: int = Field(default=5, ge=1, le=20)
    syncraft: SynCraftConfig
    llm: FoundationLLM
    retro: RetroConfig = RetroConfig()
    geometry: GeometryConfig = GeometryConfig()
    selection: SelectionConfig = SelectionConfig()
    # Prompt guidance; the native reconstruction adapter does not enforce these limits.
    constraints: EditConstraints = EditConstraints(
        max_operations=256, max_added_atoms=128, max_deleted_atoms=128, min_similarity=0
    )
    editor: Literal["syncraft"] = "syncraft"
    enhancements: EnhancementConfig = EnhancementConfig()

    @model_validator(mode="after")
    def full_method_dependencies(self):
        if self.mode in {"interaction_aware", "dual"} and self.geometry.backend == "disabled":
            raise ValueError(
                "Interaction-aware mode requires geometry; no silent standard fallback"
            )
        if self.selection.policy == "paper_route_similarity" and self.retro.backend == "disabled":
            raise ValueError("Paper selection requires a real retrosynthesis backend")
        if self.selection.policy == "skill_sa_binding" and self.geometry.backend == "disabled":
            raise ValueError("Skill selection requires original and product docking")
        return self

    @property
    def proposal_modes(self) -> tuple[str, ...]:
        return ("standard", "interaction_aware") if self.mode == "dual" else (self.mode,)

    @property
    def total_pass_k(self) -> int:
        """pass_k is per mode; dual never hides its doubled proposal/API budget."""
        return self.pass_k * len(self.proposal_modes)


@dataclass(frozen=True)
class FoundationConfiguration:
    value: FoundationConfig
    path: Path

    def resolve(self, value: str) -> Path:
        expanded = os.path.expandvars(value)
        if "${" in expanded or ("%" in expanded and expanded.count("%") >= 2):
            raise ValueError("Unresolved environment variable in configured path")
        p = Path(expanded).expanduser()
        return p.resolve() if p.is_absolute() else (self.path.parent / p).resolve()


def load_foundation(path: Path, backend: str | None = None) -> FoundationConfiguration:
    raw = read_json(path)
    if backend:
        raw["llm"]["backend"] = backend
    return FoundationConfiguration(FoundationConfig.model_validate(raw), path.resolve())
