"""Public Python entry point for the configured SynCraft foundation."""

from pathlib import Path

from agenticgen.base_config import load_foundation
from agenticgen.base_factory import FoundationComponents, build_foundation
from agenticgen.base_runner import foundation_run
from agenticgen.dataset import load_dataset


def optimize(
    config: str | Path,
    output: str | Path,
    *,
    backend: str | None = None,
    components: FoundationComponents | None = None,
) -> dict:
    """Run a fixed, auditable profile. Optional injected components support host integrations.

    No credentials, model changes, downloads or fallback backends are inferred.
    Output must be a new directory. The return value is the saved run summary.
    """
    settings = load_foundation(Path(config), backend)
    dataset = load_dataset(settings.resolve(settings.value.dataset))
    return foundation_run(
        settings, dataset, components or build_foundation(settings, dataset), Path(output)
    )
