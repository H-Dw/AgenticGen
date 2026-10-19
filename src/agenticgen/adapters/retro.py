"""Optional isolated SimpRetro CLI execution, with honest search/evidence statuses.

An isolated worker checks freshly generated graphs, inventory leaves and template
replay. Verification does not certify reaction conditions, yield or experimental feasibility.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from agenticgen.chemistry import canonical_smiles, molecule
from agenticgen.contracts import RetroResult
from agenticgen.errors import DependencyError
from agenticgen.io import object_hash, read_json, sha256, source_fingerprint, write_json


class DisabledRetrosynthesis:
    def evaluate(self, smiles: str, output: Path) -> RetroResult:
        return RetroResult(
            status="not_run",
            protocol_id="disabled",
            reason="No route search was requested; SA score cannot establish route reachability",
        )


class SimpRetroCLI:
    def __init__(
        self,
        *,
        root: Path,
        python: Path | None,
        inventory: Path,
        templates: Path,
        time_limit_seconds: int = 1800,
        process_grace_seconds: int = 300,
        cpu_threads: int = 1,
        expected_revision: str | None = None,
        execution: str = "python",
        image: str = "agenticgen-simpretro-worker:20261016",
        fingerprint_cache: Path | None = None,
        inventory_index: Path | None = None,
    ) -> None:
        self.root, self.python, self.inventory, self.templates = root, python, inventory, templates
        self.execution, self.image = execution, image
        self.fingerprint_cache, self.inventory_index = fingerprint_cache, inventory_index
        self.time_limit, self.grace, self.cpu_threads = (
            time_limit_seconds,
            process_grace_seconds,
            cpu_threads,
        )
        paths = [inventory, templates, root / "syntheseus/cli/search.py"]
        if execution == "python":
            paths.append(python)
        elif execution != "docker":
            raise DependencyError("Unsupported SimpRetro execution runtime")
        for path in paths:
            if path is None:
                raise DependencyError("SimpRetro Python runtime is missing")
            if not path.is_file():
                raise DependencyError(f"SimpRetro dependency is unavailable: {path}")
        for asset, expected_key, expected_hash in (
            (fingerprint_cache, "templates_sha256", sha256(templates)),
            (inventory_index, "inventory_sha256", sha256(inventory)),
        ):
            if asset:
                metadata = read_json(asset.with_suffix(asset.suffix + ".json"))
                if metadata[expected_key] != expected_hash or metadata["sha256"] != sha256(asset):
                    raise DependencyError("Prepared SimpRetro asset differs from bound inputs")
                if asset == fingerprint_cache and not metadata.get("all_values_losslessly_checked"):
                    raise DependencyError("Fingerprint cache lacks a lossless conversion audit")
        self.image_id = None
        if execution == "docker":
            inspected = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if inspected.returncode:
                raise DependencyError("Build/configure the SimpRetro Docker worker image first")
            self.image_id = inspected.stdout.strip()
        fingerprint = source_fingerprint(root, ["syntheseus/cli/search.py"])
        if expected_revision and fingerprint["revision"] != expected_revision:
            raise DependencyError("SimpRetro revision differs from configured revision")
        self.protocol = {
            "backend": "simpretro_cli",
            "source": fingerprint,
            "inventory_sha256": sha256(inventory),
            "templates_sha256": sha256(templates),
            "time_limit_seconds": time_limit_seconds,
            "process_grace_seconds": process_grace_seconds,
            "route_verification_budget_seconds": process_grace_seconds,
            "cpu_threads": cpu_threads,
            "search_algorithm": "mcts",
            "seed": 0,
            "use_gpu": False,
            "route_validation": "graph_inventory_and_template_replay_v1_when_graph_available",
            "validator_sha256": sha256(
                Path(__file__).resolve().parents[1] / "workers/retro_route.py"
            ),
            "execution": execution,
            "docker_image_id": self.image_id,
            "internal_stock_alignment": "runtime emolecules.txt uses configured inventory",
            "fast_filter_sha256": sha256(root / "fast_filter/model_smoothbce.pth")
            if (root / "fast_filter/model_smoothbce.pth").is_file()
            else None,
            "fingerprint_cache_sha256": sha256(fingerprint_cache) if fingerprint_cache else None,
            "inventory_index_sha256": sha256(inventory_index) if inventory_index else None,
            "inventory_membership": "exact strings; upstream canonicalize_inventory=false",
            "memory_adapter": source_fingerprint(
                Path(__file__).resolve().parents[1],
                ["workers/retro_search.py", "workers/stock_index.py"],
            )
            if inventory_index
            else None,
        }
        self.protocol_id = object_hash(self.protocol)

    def command(self, targets: Path, results: Path) -> list[str]:
        docker = self.execution == "docker"
        prefix = self._docker_prefix(targets.parent) if docker else [str(self.python)]
        return [
            *prefix,
            *(
                [
                    "/workers/retro_search.py"
                    if docker
                    else str(Path(__file__).resolve().parents[1] / "workers/retro_search.py")
                ]
                if self.inventory_index
                else []
            ),
            "/simpretro/syntheseus/cli/search.py"
            if docker
            else str(self.root / "syntheseus/cli/search.py"),
            f"inventory_smiles_file={'/job/runtime/emolecules.txt' if docker else self.inventory.as_posix()}",
            f"search_targets_file={'/job/targets.smi' if docker else targets.as_posix()}",
            "model_class=SimpRetro",
            f"model_dir={'/templates/templates.json' if docker else self.templates.as_posix()}",
            f"time_limit_s={self.time_limit}",
            "search_algorithm=mcts",
            "use_gpu=false",
            "canonicalize_inventory=false",
            "num_routes_to_plot=0",
            "save_graph=true",
            f"results_dir={'/job/search' if docker else results.as_posix()}",
        ]

    def _docker_prefix(self, output: Path) -> list[str]:
        workers = Path(__file__).resolve().parents[1] / "workers"
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            str(self.cpu_threads),
            "--mount",
            f"type=bind,source={self.root.resolve()},target=/simpretro,readonly",
            "--mount",
            f"type=bind,source={self.templates.resolve()},target=/templates/templates.json,readonly",
            "--mount",
            f"type=bind,source={workers},target=/workers,readonly",
            "--mount",
            f"type=bind,source={output.resolve()},target=/job",
            "--workdir",
            "/job/runtime",
            "--env",
            "PYTHONPATH=/simpretro",
            "--env",
            f"OMP_NUM_THREADS={self.cpu_threads}",
            "--env",
            f"MKL_NUM_THREADS={self.cpu_threads}",
            "--env",
            f"OPENBLAS_NUM_THREADS={self.cpu_threads}",
            *(
                [
                    "--mount",
                    f"type=bind,source={self.inventory_index.resolve()},target=/stock/index.sqlite,readonly",
                    "--env",
                    "AGENTICGEN_STOCK_INDEX=/stock/index.sqlite",
                ]
                if self.inventory_index
                else []
            ),
            self.image_id,
            "python",
        ]

    def validation_command(self, graph: Path, output: Path, smiles: str) -> list[str]:
        if self.execution == "docker":
            return [
                *self._docker_prefix(output),
                "/workers/retro_route.py",
                "/job/" + graph.relative_to(output).as_posix(),
                "/job/runtime/emolecules.txt",
                "/templates/templates.json",
                smiles,
                "/job/route.json",
                *(["/stock/index.sqlite"] if self.inventory_index else []),
            ]
        return [
            str(self.python),
            str(Path(__file__).resolve().parents[1] / "workers/retro_route.py"),
            str(graph.resolve()),
            str(self.inventory),
            str(self.templates),
            smiles,
            str(output.resolve() / "route.json"),
            *([str(self.inventory_index.resolve())] if self.inventory_index else []),
        ]

    def evaluate(self, smiles: str, output: Path) -> RetroResult:
        try:
            molecule(smiles)
        except ValueError:
            return RetroResult(status="invalid_input", protocol_id=self.protocol_id)
        output.mkdir(parents=True, exist_ok=False)
        runtime = output / "runtime"
        runtime.mkdir()
        # Upstream ASScore opens emolecules.txt relative to CWD independently of
        # inventory_smiles_file. Bind both to the same stock without editing upstream.
        for source, destination in [
            (self.inventory, runtime / "emolecules.txt"),
            *[
                (p, runtime / "fast_filter" / p.name)
                for p in (self.root / "fast_filter").glob("*")
                if p.is_file() and (p.name != "fingerprint_base.pkl" or not self.fingerprint_cache)
            ],
            *(
                [(self.fingerprint_cache, runtime / "fast_filter/fingerprint_base.pkl")]
                if self.fingerprint_cache
                else []
            ),
        ]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        targets = output / "targets.smi"
        targets.write_text(smiles + "\n", encoding="utf-8")
        command = self.command(targets, output / "search")
        write_json(output / "request.json", {"command": command, "protocol": self.protocol})
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.root) + os.pathsep + env.get("PYTHONPATH", "")
        if self.inventory_index:
            env["AGENTICGEN_STOCK_INDEX"] = str(self.inventory_index.resolve())
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[variable] = str(self.cpu_threads)
        start = time.monotonic()
        try:
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=runtime,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.time_limit + self.grace,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"SimpRetro exited with status {result.returncode}")
            stats_files = list((output / "search").rglob("stats.json"))
            if len(stats_files) != 1:
                raise ValueError("Expected exactly one per-target SimpRetro stats file")
            stats = read_json(stats_files[0])
            if canonical_smiles(stats["smiles"]) != canonical_smiles(smiles):
                raise ValueError("SimpRetro statistics belong to another molecule")
            solution_time = float(stats["soln_time_wallclock"])
            if math.isnan(solution_time) or solution_time < 0:
                raise ValueError("Invalid SimpRetro solution time")
            found = math.isfinite(solution_time) and solution_time <= self.time_limit
            route_verified = False
            graphs = list((output / "search").rglob("graph.pkl"))
            validation_reason = "No graph artifact available for route verification"
            if found and len(graphs) == 1:
                validation = output / "route.json"
                with (output / "route_validation.log").open("w", encoding="utf-8") as log:
                    try:
                        checked = subprocess.run(
                            self.validation_command(graphs[0], output, smiles),
                            cwd=runtime,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=self.grace,
                            check=False,
                        )
                        if checked.returncode == 0 and validation.is_file():
                            route_verified = read_json(validation).get("route_verified") is True
                            validation_reason = (
                                "Route graph, inventory leaves and template replay checked"
                            )
                        else:
                            validation_reason = "Route export/verification failed; search result retained separately"
                    except subprocess.TimeoutExpired:
                        validation_reason = "Route verification exceeded its separate budget"
            clean_stats = {
                key: None if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in stats.items()
            }
            return RetroResult(
                status="route_found" if found else "no_route_within_budget",
                protocol_id=self.protocol_id,
                route_verified=route_verified,
                elapsed_seconds=time.monotonic() - start,
                search_statistics=clean_stats,
                artifact_paths=[
                    str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
                ],
                reason=validation_reason
                if found
                else "No route reported within the fixed search budget",
            )
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, RuntimeError) as error:
            return RetroResult(
                status="infrastructure_error",
                protocol_id=self.protocol_id,
                elapsed_seconds=time.monotonic() - start,
                reason=str(error),
            )
