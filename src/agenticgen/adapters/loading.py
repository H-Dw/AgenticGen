"""Explicit, isolated loading of versioned external source checkouts."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from agenticgen.errors import DependencyError
from agenticgen.io import sha256


def load_file(path: Path, prefix: str) -> ModuleType:
    if not path.is_file():
        raise DependencyError(f"Missing external module: {path}")
    name = f"_agenticgen_{prefix}_{sha256(path)[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DependencyError(f"Cannot load external module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_evomolsteer(root: Path) -> ModuleType:
    expected = (root / "src/evomolsteer/deepseek_api.py").resolve()
    if not expected.is_file():
        raise DependencyError(f"Missing EvoMolSteer API module: {expected}")
    source = str((root / "src").resolve())
    sys.path.insert(0, source)
    try:
        module = importlib.import_module("evomolsteer.deepseek_api")
    finally:
        sys.path.remove(source)
    if Path(module.__file__).resolve() != expected:
        raise DependencyError("Another EvoMolSteer checkout is already imported in this process")
    return module
