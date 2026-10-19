"""Dependency-injection boundaries. Agents never execute chemistry or choose evaluators."""

from pathlib import Path
from typing import Any, Protocol

from agenticgen.contracts import (
    BindingResult,
    CallResult,
    EditConstraints,
    GraphEditResult,
    Observation,
    PropertyResult,
    RetroResult,
)


class LLMPort(Protocol):
    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], output: Path
    ) -> CallResult: ...


class PromptPort(Protocol):
    def build(self, observation: Observation, proposal_index: int) -> list[dict[str, str]]: ...


class GraphEditorPort(Protocol):
    def apply(
        self, mapped_smiles: str, commands: list[list[Any]], constraints: EditConstraints
    ) -> GraphEditResult: ...


class PropertyEvaluatorPort(Protocol):
    def evaluate(self, smiles: str) -> PropertyResult: ...


class RetrosynthesisPort(Protocol):
    def evaluate(self, smiles: str, output: Path) -> RetroResult: ...


class GeometryPort(Protocol):
    def evaluate(self, mapped_smiles: str, output: Path) -> BindingResult: ...


class PoseGeometryPort(GeometryPort, Protocol):
    def evaluate_pose(
        self, mapped_smiles: str, pose_sdf: Path, output: Path, *, mode: str = "score_only"
    ) -> BindingResult: ...
