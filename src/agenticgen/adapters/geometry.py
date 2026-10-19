"""Run upstream Vina/PLIP in an isolated worker and verify its molecular identity."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from agenticgen.base_config import FoundationConfiguration
from agenticgen.chemistry import canonical_smiles, molecule
from agenticgen.contracts import BindingResult
from agenticgen.errors import DependencyError
from agenticgen.io import object_hash, read_json, sha256, write_json


class DisabledGeometry:
    def evaluate(self, mapped_smiles: str, output: Path) -> BindingResult:
        return BindingResult(
            status="not_run", protocol_id="disabled", graph_smiles=canonical_smiles(mapped_smiles)
        )


class SynCraftGeometry:
    def __init__(self, config: FoundationConfiguration):
        self.config, self.settings = config, config.value.geometry
        self.root = config.resolve(config.value.syncraft.root)
        self.worker = Path(__file__).resolve().parents[1] / "workers/syncraft_geometry.py"
        self.files = {
            "receptor_pdbqt": config.resolve(self.settings.receptor_pdbqt),
            "receptor_pdb": config.resolve(self.settings.receptor_pdb),
            "vina_config": config.resolve(self.settings.vina_config),
        }
        for path in (*self.files.values(), self.worker, self.root / "src/extract_interaction.py"):
            if not path.is_file():
                raise DependencyError(f"Geometry input unavailable: {path}")
        self.image_id = None
        if self.settings.execution == "docker":
            check = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", self.settings.image],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            if check.returncode:
                raise DependencyError("Build the configured SynCraft worker Docker image first")
            self.image_id = check.stdout.strip()
        elif not config.resolve(self.settings.python).is_file():
            raise DependencyError("Geometry worker Python does not exist")
        self.protocol = {
            "backend": "syncraft_worker",
            "settings": self.settings.model_dump(mode="json"),
            "inputs": {key: sha256(p) for key, p in self.files.items()},
            "worker_sha256": sha256(self.worker),
            "alignment_sha256": sha256(self.worker.with_name("plip_alignment.py")),
            "docker_image_id": self.image_id,
            "upstream_sha256": sha256(self.root / "src/extract_interaction.py"),
        }
        self.protocol_id = object_hash(self.protocol)

    def evaluate(self, mapped_smiles: str, output: Path) -> BindingResult:
        return self._evaluate(mapped_smiles, output)

    def evaluate_pose(
        self, mapped_smiles: str, pose_sdf: Path, output: Path, *, mode: str = "score_only"
    ) -> BindingResult:
        """Evaluate supplied coordinates without embedding a replacement conformation."""
        if mode not in {"score_only", "local_only", "redock"}:
            raise ValueError("Unknown pose evaluation mode")
        pose_sdf = pose_sdf.resolve()
        if not pose_sdf.is_file():
            raise DependencyError(f"Input pose unavailable: {pose_sdf}")
        return self._evaluate(mapped_smiles, output, pose_sdf=pose_sdf, mode=mode)

    def _evaluate(
        self,
        mapped_smiles: str,
        output: Path,
        *,
        pose_sdf: Path | None = None,
        mode: str = "redock",
    ) -> BindingResult:
        original = molecule(mapped_smiles)
        output.mkdir(parents=True, exist_ok=False)
        protocol = dict(self.protocol)
        if pose_sdf:
            protocol.update(pose_evaluation_mode=mode, input_pose_sha256=sha256(pose_sdf))
        protocol_id = object_hash(protocol)
        job = {
            "smiles": mapped_smiles,
            "protocol_id": protocol_id,
            "seed": self.settings.seed,
            "cpu_threads": self.settings.cpu_threads,
            "exhaustiveness": self.settings.exhaustiveness,
            "vina_binary": os.path.expandvars(self.settings.vina_binary),
            "pose_mode": mode,
        }
        if pose_sdf:
            job["input_pose_sha256"] = protocol["input_pose_sha256"]
        if self.settings.execution == "docker":
            # Mount only the source checkout, this worker, and copied receptor/job inputs.
            for key, path in self.files.items():
                filename = key + path.suffix
                shutil.copyfile(path, output / filename)
                job[key] = "/job/" + filename
            job.update(syncraft_root="/syncraft", output="/job")
            if pose_sdf:
                shutil.copyfile(pose_sdf, output / "input_pose.sdf")
                job["pose_sdf"] = "/job/input_pose.sdf"
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                str(self.settings.cpu_threads),
                "--mount",
                f"type=bind,source={self.root},target=/syncraft,readonly",
                "--mount",
                f"type=bind,source={self.worker.parent},target=/workers,readonly",
                "--mount",
                f"type=bind,source={output.resolve()},target=/job",
                self.image_id,
                "python",
                "/workers/syncraft_geometry.py",
                "/job/request.json",
            ]
        else:
            job.update({key: str(path) for key, path in self.files.items()})
            job.update(syncraft_root=str(self.root), output=str(output.resolve()))
            if pose_sdf:
                job["pose_sdf"] = str(pose_sdf)
            command = [
                str(self.config.resolve(self.settings.python)),
                str(self.worker),
                str(output.resolve() / "request.json"),
            ]
        write_json(output / "request.json", job)
        write_json(output / "protocol.json", protocol)
        try:
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.settings.timeout_seconds,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError("SynCraft geometry worker failed; inspect worker.log")
            response = read_json(output / "response.json")
            binding = BindingResult.model_validate(response)
            if binding.protocol_id != protocol_id or canonical_smiles(
                binding.graph_smiles
            ) != canonical_smiles(original):
                raise ValueError("Geometry response molecular identity/protocol mismatch")
            maps = {a.GetAtomMapNum() for a in original.GetAtoms()}
            if any(not set(i.atom_maps) <= maps for i in binding.interactions):
                raise ValueError("PLIP contact references a different atom map")
            if binding.status == "ok" and binding.score_kcal_mol is None:
                raise ValueError("Successful geometry result is missing its docking score")
            return binding
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            return BindingResult(
                status="infrastructure_error",
                protocol_id=protocol_id,
                graph_smiles=canonical_smiles(original),
                reason=str(error),
                evaluation_mode=mode,
                input_pose_sha256=protocol.get("input_pose_sha256"),
            )
