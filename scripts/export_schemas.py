"""Export the authoritative Python contracts for other worker implementations."""

from pathlib import Path

from agenticgen.agent import proposal_schema
from agenticgen.base_config import FoundationConfig
from agenticgen.config import RunConfig
from agenticgen.contracts import (
    BindingResult,
    CaseDataset,
    EnhancementPacket,
    GraphEditResult,
    Observation,
    RetroResult,
)
from agenticgen.inpainting_contracts import InpaintingConfig, InpaintingRequest, InpaintingResult
from agenticgen.io import write_json


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "schemas"
    for name, model in (
        ("run", RunConfig),
        ("cases", CaseDataset),
        ("observation", Observation),
        ("graph-edit-result", GraphEditResult),
        ("retro-result", RetroResult),
        ("syncraft-foundation", FoundationConfig),
        ("binding-result", BindingResult),
        ("enhancement-packet", EnhancementPacket),
        ("inpainting-config", InpaintingConfig),
        ("inpainting-request", InpaintingRequest),
        ("inpainting-result", InpaintingResult),
    ):
        write_json(output / f"{name}.schema.json", model.model_json_schema())
    write_json(output / "proposal.schema.json", proposal_schema())


if __name__ == "__main__":
    main()
