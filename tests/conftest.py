import os
from pathlib import Path

import pytest

from agenticgen.adapters.loading import load_file
from agenticgen.editing import SynCraftEditor

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parents[1]


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def syncraft_root():
    path = Path(os.environ.get("SYNCRAFT_ROOT", REPO / "external/SynCraft-Core"))
    if not (path / "src/utils.py").is_file():
        pytest.skip("External SynCraft checkout unavailable; set SYNCRAFT_ROOT")
    return path


@pytest.fixture(scope="session")
def evo_root():
    path = Path(os.environ.get("EVOMOLSTEER_ROOT", WORKSPACE / "EvoMolSteer"))
    if not (path / "src/evomolsteer/deepseek_api.py").is_file():
        pytest.skip("External EvoMolSteer checkout unavailable; set EVOMOLSTEER_ROOT")
    return path


@pytest.fixture(scope="session")
def editor(syncraft_root):
    module = load_file(syncraft_root / "src/utils.py", "test_syncraft_editor")

    class Backend:
        @staticmethod
        def reconstruct(smiles, commands):
            return module.reconstruct_from_sequence(smiles, commands)

    return SynCraftEditor(Backend())
