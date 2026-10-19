"""Versioned values exchanged between independently replaceable components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceCase(Record):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    target: str
    smiles: str
    selection_rank: int = Field(ge=1)
    source_metrics: dict[str, float | str | bool]
    provenance: dict[str, Any]
    sdf_record_index: int = Field(ge=0)


class CaseDataset(Record):
    schema_version: Literal["agenticgen.cases.v1"] = "agenticgen.cases.v1"
    purpose: Literal["engineering_example"] = "engineering_example"
    selection_rule: str
    source_files: dict[str, str]
    sdf_file: str
    sdf_sha256: str
    cases: list[SourceCase]


class EditConstraints(Record):
    """Prompt guidance; native SynCraft reconstruction does not enforce these limits."""

    max_operations: int = Field(default=6, ge=1, le=256)
    max_added_atoms: int = Field(default=2, ge=0)
    max_deleted_atoms: int = Field(default=2, ge=0)
    min_similarity: float = Field(default=0.5, ge=0, le=1)
    allowed_elements: tuple[str, ...] = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
    protected_atom_maps: tuple[int, ...] = ()


class Observation(Record):
    """Terminal-only v1. Historical/route labels cannot enter via arbitrary fields."""

    schema_version: Literal["agenticgen.observation.v1"] = "agenticgen.observation.v1"
    case_id: str
    target: str
    source_smiles: str
    mapped_smiles: str
    graph_sha256: str
    terminal_metrics: dict[str, float | str | bool]
    constraints: EditConstraints


class Proposal(Record):
    case_id: str
    action: Literal["edit", "abstain"]
    edits: list[list[Any]] = Field(min_length=1, max_length=257)
    rationale: str = Field(min_length=1, max_length=262144)
    expected_effect: str = Field(min_length=1, max_length=1000)
    risks: list[str] = Field(max_length=10)

    @field_validator("edits")
    @classmethod
    def require_commands(cls, value: list[list[Any]]) -> list[list[Any]]:
        if any(not command or not isinstance(command[0], str) for command in value):
            raise ValueError("Every edit must be a nonempty command list")
        return value


class GraphEditResult(Record):
    status: Literal["accepted", "rejected", "unchanged"]
    source_smiles: str
    product_smiles: str | None = None
    product_mapped_smiles: str | None = None
    product_graph_sha256: str | None = None
    similarity: float | None = None
    operations: int = 0
    added_atom_maps: list[int] = []
    deleted_atom_maps: list[int] = []
    changed_atom_maps: list[int] = []
    reconstruction_policy: Literal["syncraft_native"] = "syncraft_native"
    upstream_smiles: str | None = None
    upstream_mapped_smiles: str | None = None
    was_fragmented: bool = False
    discarded_atom_maps: list[int] = []
    error_stage: Literal["upstream_reconstruction", "product_loading"] | None = None
    reason: str | None = None


class PropertyResult(Record):
    status: Literal["ok", "unavailable", "error"]
    metric: str = "sa_score"
    value: float | None = None
    direction: Literal["lower_is_better"] = "lower_is_better"
    method: str = "rdkit.contrib.SA_Score"
    reason: str | None = None


class RetroResult(Record):
    status: Literal[
        "not_run", "route_found", "no_route_within_budget", "invalid_input", "infrastructure_error"
    ]
    protocol_id: str
    route_verified: bool = False
    elapsed_seconds: float | None = None
    search_statistics: dict[str, Any] = {}
    artifact_paths: list[str] = []
    reason: str | None = None


class CallResult(Record):
    payload: dict[str, Any]
    backend: str
    receipt_path: str


class Interaction(Record):
    kind: str
    residue: str
    atom_maps: list[int] = Field(min_length=1)


class BindingResult(Record):
    status: Literal["ok", "not_run", "infrastructure_error"]
    protocol_id: str
    graph_smiles: str
    score_kcal_mol: float | None = None
    interactions: list[Interaction] = []
    prompt: str = ""
    artifact_paths: list[str] = []
    reason: str | None = None
    evaluation_mode: Literal["redock", "score_only", "local_only"] = "redock"
    input_pose_sha256: str | None = None


class EnhancementPacket(Record):
    case_id: str
    source_graph_sha256: str
    kind: Literal["flow_history", "native_exploration", "case_memory"]
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
