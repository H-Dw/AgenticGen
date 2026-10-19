"""Validate explicit DeepSeek thinking profiles while preserving historical defaults."""

import pytest
from pydantic import ValidationError

from agenticgen.base_config import (
    FoundationConfig,
    FoundationConfiguration,
    FoundationLLM,
    SelectionConfig,
)
from agenticgen.base_factory import build_foundation
from agenticgen.config import SynCraftConfig
from agenticgen.dataset import load_dataset


def test_existing_deepseek_config_keeps_disabled_thinking_defaults():
    settings = FoundationLLM()
    assert settings.backend == "deepseek_api" and settings.model == "deepseek-flash"
    assert settings.reasoning_effort is None and settings.temperature == 0
    assert settings.max_output_tokens == 8192 and settings.timeout_seconds == 240
    assert settings.network_attempts == 3


def test_explicit_high_campaign_profile_is_valid():
    settings = FoundationLLM(
        reasoning_effort="high",
        max_output_tokens=65536,
        timeout_seconds=1200,
        network_attempts=3,
    )
    assert settings.reasoning_effort == "high"
    assert settings.max_output_tokens == 65536 and settings.timeout_seconds == 1200


@pytest.mark.parametrize("effort", ["low", "medium", "max", "", "none", "disabled"])
def test_deepseek_does_not_silently_map_another_reasoning_profile(effort):
    with pytest.raises(ValidationError, match="explicit high"):
        FoundationLLM(reasoning_effort=effort)


def test_legacy_evo_transport_remains_disabled_only():
    with pytest.raises(ValidationError, match="EvoMolSteer"):
        FoundationLLM(
            backend="evomolsteer_deepseek",
            evomolsteer_root="legacy-source",
            reasoning_effort="high",
        )


@pytest.mark.parametrize("effort", [None, "high"])
@pytest.mark.integration
def test_foundation_factory_records_effective_deepseek_protocol(
    repo, tmp_path, syncraft_root, effort
):
    dataset_path = repo / "examples/syncraft_foundation/ck2_one.json"
    llm = FoundationLLM(
        reasoning_effort=effort,
        max_output_tokens=65536,
        timeout_seconds=1200,
        network_attempts=3,
    )
    config = FoundationConfiguration(
        FoundationConfig(
            dataset=str(dataset_path),
            syncraft=SynCraftConfig(root=str(syncraft_root), few_shot_count=0),
            llm=llm,
            selection=SelectionConfig(policy="proposal_only"),
        ),
        tmp_path / "config.json",
    )
    components = build_foundation(config, load_dataset(dataset_path))
    transport = components.provenance["transport"]
    assert transport["thinking"] == ("enabled" if effort else "disabled")
    assert transport["reasoning_effort"] == effort
    assert transport["temperature"] == (None if effort else 0)
    assert transport["max_output_tokens"] == 65536
    assert transport["timeout_seconds"] == 1200 and transport["network_attempts"] == 3
