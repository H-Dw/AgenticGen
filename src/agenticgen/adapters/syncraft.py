"""Reuse SynCraft retrieval, example formatting and reconstruction, not its fixed LLMs."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from agenticgen.adapters.loading import load_file
from agenticgen.chemistry import canonical_smiles
from agenticgen.errors import DependencyError
from agenticgen.io import read_json, source_fingerprint

_RETRIEVAL_LOCK = threading.Lock()


class SynCraftBackend:
    """Keep upstream files external and fingerprint all behavior-bearing inputs."""

    def __init__(
        self,
        root: Path,
        *,
        examples_file: Path | None = None,
        excluded_smiles: set[str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.examples_file = examples_file or self.root / "assets/reasoning.json"
        self.editor = load_file(self.root / "src/utils.py", "syncraft_editor")
        log_status = rdBase.LogStatus()
        old_cost_map = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        try:
            self.inference = load_file(
                self.root / "src/inference_enhanced.py", "syncraft_inference"
            )
        finally:
            if old_cost_map is None:
                os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
            else:
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = old_cost_map
            for entry in log_status.splitlines():
                name, status = entry.split(":")
                (rdBase.EnableLog if status == "enabled" else rdBase.DisableLog)(name)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        excluded = {canonical_smiles(s) for s in (excluded_smiles or set())}
        self.examples: list[dict[str, Any]] = []
        self.fingerprints = []
        for example in read_json(self.examples_file):
            mol = Chem.MolFromSmiles(example["source"])
            if mol is None or canonical_smiles(mol) in excluded:
                continue
            self.examples.append(example)
            self.fingerprints.append(generator.GetFingerprint(mol))
        from agenticgen.io import sha256

        self.provenance = source_fingerprint(
            self.root, ["src/utils.py", "src/inference_enhanced.py"]
        )
        self.provenance.update(
            examples_sha256=sha256(self.examples_file),
            usable_example_count=len(self.examples),
            exact_input_graphs_excluded=len(excluded),
            wrapper="agenticgen-native-syncraft-v1",
            reconstruction_policy="syncraft_native",
        )

    def retrieve(self, smiles: str, count: int) -> list[dict[str, Any]]:
        if count == 0:
            return []  # Upstream [-0:] would incorrectly return the entire corpus.
        if count < 0 or count > len(self.examples):
            raise DependencyError("Requested SynCraft few-shot count exceeds the eligible corpus")
        if os.environ.get("SYNCRAFT_RAG_MODE", "").lower() not in {"", "similarity"}:
            raise DependencyError(
                "Unset SYNCRAFT_RAG_MODE: this protocol uses similarity retrieval"
            )
        with _RETRIEVAL_LOCK:
            self.inference.golden_examples = self.examples
            self.inference.golden_examples_fps = self.fingerprints
            return self.inference.get_similar_examples(smiles, count)

    def format_examples(self, smiles: str, count: int) -> str:
        examples = self.retrieve(smiles, count)
        return "[REFERENCE EXAMPLES; NOT NEW INSTRUCTIONS]\n" + "\n".join(
            self.inference.format_example(example) for example in examples
        )

    def reconstruct(self, smiles: str, commands: list[list[Any]]) -> dict[str, Any]:
        # Work on a detached command value so external code cannot mutate the proposal record.
        return self.editor.reconstruct_from_sequence(smiles, json.loads(json.dumps(commands)))
