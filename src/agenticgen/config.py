"""Strict host configuration. Relative paths are relative to the configuration file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenticgen.contracts import EditConstraints, Record
from agenticgen.io import read_json


class SynCraftConfig(Record):
    root: str
    examples_file: str | None = None
    few_shot_count: int = Field(default=5, ge=0, le=20)
    expected_revision: str | None = None


class LLMConfig(Record):
    backend: Literal["evomolsteer_deepseek", "replay"] = "evomolsteer_deepseek"
    evomolsteer_root: str
    model: Literal["deepseek-flash"] = "deepseek-flash"
    timeout_seconds: int = Field(default=240, ge=1, le=1200)
    network_attempts: int = Field(default=3, ge=1, le=5)
    max_format_repairs: int = Field(default=1, ge=0, le=2)
    max_output_tokens: int = Field(default=4096, ge=512, le=65536)
    replay_file: str | None = None


class RetroConfig(Record):
    backend: Literal["disabled", "simpretro_cli"] = "disabled"
    root: str | None = None
    python: str | None = None
    execution: Literal["python", "docker"] = "python"
    image: str = "agenticgen-simpretro-worker:20261016"
    inventory: str | None = None
    templates: str | None = None
    fingerprint_cache: str | None = None
    inventory_index: str | None = None
    time_limit_seconds: int = Field(default=1800, ge=1)
    process_grace_seconds: int = Field(default=300, ge=1)
    cpu_threads: int = Field(default=1, ge=1, le=64)
    seed: Literal[0] = 0  # Audited upstream search.py calls set_random_seed(0).
    expected_revision: str | None = None

    @model_validator(mode="after")
    def require_paths(self) -> RetroConfig:
        if self.backend == "simpretro_cli" and (
            not all((self.root, self.inventory, self.templates))
            or (self.execution == "python" and not self.python)
        ):
            raise ValueError(
                "SimpRetro requires source, inventory, templates and an execution runtime"
            )
        return self


class RunConfig(Record):
    schema_version: Literal["agenticgen.run.v1"] = "agenticgen.run.v1"
    dataset: str
    arm: Literal["T"] = "T"
    proposals_per_case: int = Field(default=1, ge=1, le=5)
    syncraft: SynCraftConfig
    llm: LLMConfig
    retro: RetroConfig = RetroConfig()
    constraints: EditConstraints = EditConstraints()


@dataclass(frozen=True)
class Configuration:
    value: RunConfig
    path: Path

    def resolve(self, value: str) -> Path:
        expanded = os.path.expandvars(value)
        if "${" in expanded or ("%" in expanded and expanded.count("%") >= 2):
            raise ValueError("An environment variable in a configured path is unresolved")
        path = Path(expanded).expanduser()
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()


def load_configuration(path: Path, backend: str | None = None) -> Configuration:
    raw = read_json(path)
    if backend is not None:
        raw["llm"]["backend"] = backend
    return Configuration(RunConfig.model_validate(raw), path.resolve())
