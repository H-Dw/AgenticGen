"""Frozen protocol for the CK2 original-pose campaign, separate from legacy runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenticgen.contracts import Record
from agenticgen.io import read_json


class CampaignConfig(Record):
    schema_version: Literal["agenticgen.ck2-campaign.v1"] = "agenticgen.ck2-campaign.v1"
    foundation_config: str
    flowr_root: str
    flowr_python: str
    checkpoint: str
    pocket_pdb: str
    checkpoint_sha256: str = "f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0"
    flowr_revision: str = "739a33c2aab74d074dd24605b9a2e34dc809ae9c"
    seed: int = Field(default=20261020, ge=0, le=2147483647)
    chemical_retries: Literal[5] = 5
    samples_per_arm: Literal[100] = 100
    microbatch_size: Literal[4] = 4
    integration_steps: Literal[100] = 100
    guidance_window: tuple[float, float] = (0.1, 0.9)
    guidance_weights: tuple[float, ...] = (0.25, 1.0, 4.0)
    guidance_families: tuple[Literal["ce", "brier"], ...] = ("ce", "brier")
    geometry_workers: int = Field(default=4, ge=1, le=8)
    design_workers: int = Field(default=2, ge=1, le=4)
    generation_timeout_seconds: int = Field(default=7200, ge=60, le=86400)
    affinity_timeout_seconds: int = Field(default=7200, ge=60, le=86400)

    @model_validator(mode="after")
    def frozen_experiment(self):
        if self.guidance_window != (0.1, 0.9):
            raise ValueError("This campaign fixes the guidance window to [0.1, 0.9)")
        if self.guidance_weights != (0.25, 1.0, 4.0):
            raise ValueError("The predeclared strength grid must not change on resume")
        if self.guidance_families != ("ce", "brier"):
            raise ValueError("The predeclared reward grid must not change on resume")
        return self


class CampaignArm(Record):
    arm_id: str
    mode: Literal["graph", "fragment_graph", "fragment_graph_pose", "gradient", "unguided"]
    family: Literal["ce", "brier"] | None = None
    coordinates: bool = False
    weight: float = 0.0


def campaign_arms(config: CampaignConfig) -> list[CampaignArm]:
    arms = [
        CampaignArm(arm_id=m, mode=m)
        for m in ("graph", "fragment_graph", "fragment_graph_pose", "unguided")
    ]
    for family in config.guidance_families:
        for coordinates in (False, True):
            for weight in config.guidance_weights:
                label = str(weight).replace(".", "p")
                arms.append(
                    CampaignArm(
                        arm_id=f"gradient_{family}_{'graph_pose' if coordinates else 'graph'}_w{label}",
                        mode="gradient",
                        family=family,
                        coordinates=coordinates,
                        weight=weight,
                    )
                )
    return arms


def paired_seed(seed: int, case_id: str, batch_index: int, *, preflight: bool = False) -> int:
    """Arm is deliberately absent: matched arms receive the same batch seed."""
    payload = f"{seed}:{case_id}:{batch_index}:{'preflight' if preflight else 'production'}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") % 2147483647


def load_campaign(path: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(read_json(path))


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
