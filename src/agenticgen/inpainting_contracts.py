"""Data contracts for native FLOWR runs, kept independent of the proposal workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from agenticgen.contracts import GraphEditResult, Record

InpaintingMode = Literal["graph", "fragment"]


class InpaintingConfig(Record):
    root: str
    python: str
    checkpoint: str
    expected_revision: str | None = "739a33c2aab74d074dd24605b9a2e34dc809ae9c"
    modes: tuple[InpaintingMode, ...] = Field(default=("graph", "fragment"), min_length=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    samples_per_mode: int = Field(default=1, ge=1, le=256)
    integration_steps: int = Field(default=100, ge=1, le=10000)
    max_sample_iter: int = Field(default=0, ge=0, le=100)
    num_workers: int = Field(default=0, ge=0, le=64)
    batch_cost: int = Field(default=200, ge=1)
    gpus: int = Field(default=1, ge=0, le=1)
    timeout_seconds: int = Field(default=1800, ge=1, le=86400)
    graph_initialization: Literal["random", "conformer", "harmonic"] = "harmonic"
    cut_pocket: bool = True
    pocket_cutoff: float = Field(default=7.0, gt=0, le=20)
    coordinate_tolerance_angstrom: float = Field(default=0.05, gt=0, le=1)
    interaction_policy: Literal["syncraft_compatible", "strict"] = "syncraft_compatible"

    @field_validator("modes")
    @classmethod
    def distinct_modes(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Inpainting modes must be distinct")
        return value


class InpaintingRequest(Record):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    proposal_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    source_mapped_smiles: str
    graph_result: GraphEditResult
    interaction_atom_maps: tuple[int, ...] = ()
    llm_optimized_atom_maps: tuple[int, ...] = ()
    product_pose_sdf: str | None = None
    product_sdf_record_index: int = Field(default=0, ge=0)
    pocket_pdb: str | None = None


class AtomSlot(Record):
    atom_map: int
    source_rdkit_index: int | None
    product_rdkit_index: int
    input_slot: int
    flowr_output_slot: int | None
    fixed: bool


class MaskPlan(Record):
    mode: InpaintingMode
    status: Literal["ready", "skipped"]
    reason: str | None = None
    warnings: list[str] = []
    product_mapped_smiles: str | None = None
    interaction_atom_maps: list[int] = []
    interaction_policy: Literal["syncraft_compatible", "strict"] = "syncraft_compatible"
    interaction_surviving_atom_maps: list[int] = []
    interaction_deleted_atom_maps: list[int] = []
    interaction_changed_atom_maps: list[int] = []
    optimized_atom_maps: list[int] = []
    deleted_atom_maps: list[int] = []
    deletion_boundary_atom_maps: list[int] = []
    fixed_atom_maps: list[int] = []
    fixed_input_slots: list[int] = []
    regenerate_input_slots: list[int] = []
    atom_slots: list[AtomSlot] = []
    # The native CLI selects the area to regenerate; fragment_mask=True means keep.
    cli_substructure_indices: list[int] = []
    fixed_pair_bonds: list[list[int | str]] = []
    fixed_attachment_bonds: dict[int, list[str]] = {}


class PoseArtifact(Record):
    pose_id: str
    mode: InpaintingMode
    sdf_path: str
    record_index: int = 0
    smiles: str
    mapped_smiles: str
    accepted_for_evaluation: bool
    preservation_report_path: str
    coordinate_frame: Literal["pocket_input"] = "pocket_input"


class InpaintingModeResult(Record):
    mode: InpaintingMode
    status: Literal["planned", "completed", "skipped", "failed"]
    seed: int
    requested_samples: int
    delivered_samples: int = 0
    eligible_samples: int = 0
    poses: list[PoseArtifact] = []
    raw_sdf_path: str | None = None
    artifact_paths: list[str] = []
    reason: str | None = None


class InpaintingResult(Record):
    schema_version: Literal["agenticgen.inpainting.v1"] = "agenticgen.inpainting.v1"
    case_id: str
    proposal_id: str
    status: Literal["planned", "completed", "partial", "skipped", "failed"]
    plans: list[MaskPlan]
    mode_results: list[InpaintingModeResult] = []
    artifact_paths: list[str] = []
    reason: str | None = None
