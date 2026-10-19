"""Resumable CK2 source-pose campaign; no outcome-dependent experiment selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem

from agenticgen.base_config import load_foundation
from agenticgen.base_factory import build_foundation
from agenticgen.campaign_contracts import campaign_arms, load_campaign, paired_seed, resolve_path
from agenticgen.campaign_evaluation import evaluate_particles, posebusters
from agenticgen.campaign_inputs import write_graph_placeholder, write_original_pose
from agenticgen.campaign_ledger import JobLedger, campaign_lock
from agenticgen.chemistry import graph_hash, mapped_smiles
from agenticgen.contracts import BindingResult, GraphEditResult, Observation, Proposal
from agenticgen.dataset import load_dataset
from agenticgen.io import object_hash, read_json, sha256, utc_now, write_json
from agenticgen.pose_quality import measure_pose


def code_fingerprint() -> str:
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))}
    return object_hash(files)


def _job_id(stage: str, case: str, suffix: str = "") -> str:
    return ":".join(x for x in (stage, case, suffix) if x)


def particle_rows(path: Path) -> list[dict]:
    value = read_json(path)
    if value.get("schema_version") != "agenticgen.flowr-particles.v1":
        raise ValueError("Unexpected particle receipt schema")
    rows = value["particles"]
    if len(rows) != value["sample_count"]:
        raise ValueError("Raw attempted particle accounting is incomplete")
    return rows


def verify_input_hashes(payload: dict) -> None:
    for name, expected in payload.get("input_sha256", {}).items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen task input changed: {name}")


class Campaign:
    def __init__(self, config_path: Path, root: Path):
        self.config_path, self.root = config_path.resolve(), root.resolve()
        self.config = load_campaign(self.config_path)
        self.foundation = load_foundation(
            resolve_path(self.config_path, self.config.foundation_config)
        )
        self.dataset_path = self.foundation.resolve(self.foundation.value.dataset)
        self.dataset = load_dataset(self.dataset_path)
        self.cases = {case.case_id: case for case in self.dataset.cases}
        self.ledger = JobLedger(self.root / "ledger.sqlite")
        self._components = None
        self._gpu_session = None

    @property
    def components(self):
        if self._components is None:
            self._components = build_foundation(self.foundation, self.dataset)
        return self._components

    def source_path(self, case_id: str) -> Path:
        return self.root / "cases" / case_id / "source.sdf"

    def prepare(self) -> dict:
        llm = self.foundation.value.llm
        if (
            llm.backend != "deepseek_api"
            or llm.model != "deepseek-flash"
            or llm.reasoning_effort != "high"
            or llm.max_output_tokens != 65536
            or llm.timeout_seconds != 1200
        ):
            raise ValueError("Production campaign requires the approved DeepSeek high profile")
        if self.foundation.value.mode != "interaction_aware":
            raise ValueError("Campaign requires original-pose interaction-aware proposals")
        if len(self.dataset.cases) != 20:
            raise ValueError("The frozen campaign contains exactly 20 original candidates")
        self.root.mkdir(parents=True, exist_ok=True)
        frozen = {
            "campaign_config_sha256": sha256(self.config_path),
            "foundation_config_sha256": sha256(self.foundation.path),
            "dataset_sha256": sha256(self.dataset_path),
            "source_sdf_sha256": self.dataset.sdf_sha256,
            "pocket_sha256": sha256(resolve_path(self.config_path, self.config.pocket_pdb)),
        }
        for name in ("receptor_pdb", "receptor_pdbqt", "vina_config"):
            frozen[name + "_sha256"] = sha256(
                self.foundation.resolve(getattr(self.foundation.value.geometry, name))
            )
        if self.foundation.value.syncraft.examples_file:
            frozen["retrieval_examples_sha256"] = sha256(
                self.foundation.resolve(self.foundation.value.syncraft.examples_file)
            )
        for label, root, expected in (
            (
                "flowr",
                resolve_path(self.config_path, self.config.flowr_root),
                self.config.flowr_revision,
            ),
            (
                "syncraft",
                self.foundation.resolve(self.foundation.value.syncraft.root),
                self.foundation.value.syncraft.expected_revision,
            ),
        ):
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != expected:
                raise ValueError(f"{label} revision differs from the pinned protocol")
            frozen[label + "_revision"] = revision
        manifest_path = self.root / "manifest.json"
        checkpoint = resolve_path(self.config_path, self.config.checkpoint)
        if sha256(checkpoint) != self.config.checkpoint_sha256:
            raise ValueError("FLOWR v2 checkpoint hash differs from approved material")
        if manifest_path.exists():
            old = read_json(manifest_path)
            if old["frozen_inputs"] != frozen:
                raise ValueError("Frozen campaign inputs/configuration changed")
        else:
            write_json(
                manifest_path,
                {
                    "schema_version": "agenticgen.ck2-campaign-run.v1",
                    "status": "prepared",
                    "created_at": utc_now(),
                    "config": str(self.config_path),
                    "frozen_inputs": frozen,
                    "protocol": self.config.model_dump(mode="json"),
                    "code_sha256": code_fingerprint(),
                    "case_ids": list(self.cases),
                    "maximum_generation_attempts": 32000,
                    "interaction_source": "original Steer pose; no redocking",
                    "scoring": "actual final pose; SC-on zero-noise pIC50 and Vina score_only",
                    "reasoning": "deepseek-flash high; old host guidance retained",
                },
            )
        source_sdf = self.dataset_path.parent / self.dataset.sdf_file
        supplier = Chem.SDMolSupplier(str(source_sdf), removeHs=False)
        poses = []
        for case in self.dataset.cases:
            path = self.source_path(case.case_id)
            if not path.exists():
                write_original_pose(supplier[case.sdf_record_index], case.smiles, path)
            if sha256(path) != read_json(path.with_suffix(".mapping.json"))["output_sha256"]:
                raise ValueError("Mapped source pose changed after preparation")
            self.ledger.add(
                _job_id("source_geometry", case.case_id),
                "source_geometry",
                case.case_id,
                {"source_pose_sdf": str(path)},
                path.parent / "source_geometry",
            )
            self.ledger.add(
                _job_id("design", case.case_id),
                "design",
                case.case_id,
                {"maximum_chemical_proposals": 6},
                path.parent / "design_execution",
            )
            poses.append(
                {"pose_id": case.case_id, "sdf_path": str(path), "particle_index": len(poses)}
            )
        self.ledger.add(
            "source_affinity:all",
            "source_affinity",
            "all",
            {"poses": poses},
            self.root / "source_affinity",
        )
        return self.status("prepared")

    def status(self, phase: str | None = None) -> dict:
        counts = self.ledger.counts()
        value = {
            "schema_version": "agenticgen.campaign-status.v1",
            "updated_at": utc_now(),
            "controller_pid": os.getpid(),
            "phase": phase or "inspection",
            "counts": counts,
            "case_count": len(self.cases),
            "guidance_window": [0.1, 0.9],
            "active_guidance_steps": 80,
            "code_sha256": code_fingerprint(),
        }
        write_json(self.root / "status.json", value)
        return value

    def _source_binding(self, case_id: str) -> BindingResult:
        job = self.ledger.get(_job_id("source_geometry", case_id))
        if job["status"] != "completed":
            raise RuntimeError("Original Steer interaction measurement has not completed")
        return BindingResult.model_validate(read_json(Path(job["result_path"]))["binding"])

    def _worker(self, module: str, task: dict, work: Path, timeout: int) -> None:
        task_path = work / "task.json"
        write_json(task_path, task)
        env = os.environ.copy()
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = "1"
        if module == "agenticgen.workers.flowr_campaign":
            from agenticgen.campaign_worker_client import FlowrSession

            if self._gpu_session is None or self._gpu_session.process.poll() is not None:
                self._gpu_session = FlowrSession(
                    resolve_path(self.config_path, self.config.flowr_python),
                    env,
                    self.root / "runtime",
                )
            try:
                response = self._gpu_session.run(task_path, timeout)
                write_json(work / "worker_receipt.json", response)
            except Exception:
                self._gpu_session.close()
                self._gpu_session = None
                raise
            return
        with (work / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    str(resolve_path(self.config_path, self.config.flowr_python)),
                    "-m",
                    module,
                    str(task_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{module} exited {result.returncode}; see {work / 'worker.log'}")

    def _flowr_task(self, work: Path, seed: int) -> dict:
        return {
            "flowr_root": str(resolve_path(self.config_path, self.config.flowr_root)),
            "checkpoint": str(resolve_path(self.config_path, self.config.checkpoint)),
            "pocket_pdb": str(resolve_path(self.config_path, self.config.pocket_pdb)),
            "seed": seed,
            "output_dir": str(work),
            "cut_pocket": True,
            "pocket_cutoff": 7.0,
        }

    def _source_geometry(self, job: dict, work: Path) -> dict:
        case = self.cases[job["case_id"]]
        source = self.source_path(case.case_id)
        binding = self.components.geometry.evaluate_pose(
            mapped_smiles(case.smiles), source, work / "geometry", mode="score_only"
        )
        if binding.status != "ok":
            raise RuntimeError(binding.reason or "Source score-only/PLIP failed")
        return {
            "case_id": case.case_id,
            "source_pose_sha256": sha256(source),
            "binding": binding.model_dump(mode="json"),
            "properties": self.components.properties.evaluate(case.smiles).model_dump(mode="json"),
            "pose_quality": measure_pose(source),
            "posebusters": posebusters(source, self.components.geometry.files["receptor_pdb"]),
        }

    def _affinity(self, job: dict, work: Path) -> dict:
        task = self._flowr_task(work, self.config.seed)
        payload = job["payload"]
        if "generation_task_id" in payload:
            if (
                sha256(Path(payload["generation_result_path"]))
                != payload["generation_result_sha256"]
            ):
                raise ValueError("Affinity parent execution receipt changed")
            output = Path(read_json(Path(payload["generation_result_path"]))["output_dir"])
            task["poses"] = particle_rows(output / "poses.json")
        else:
            task["poses"] = payload["poses"]
        self._worker(
            "agenticgen.campaign_affinity", task, work, self.config.affinity_timeout_seconds
        )
        result = read_json(work / "affinity.json")
        if result["status"] != "completed":
            raise RuntimeError("Affinity batch did not complete")
        if any(row.get("status") == "error" for row in result.get("poses", [])):
            raise RuntimeError("Affinity scoring has recoverable error rows; see affinity.json")
        return result

    def _cached_proposal(self, native_output: str, case_id: str) -> Proposal:
        parsed = self.components.agent.backend.inference.parse_llm_output(native_output)
        if parsed.get("error"):
            raise ValueError("SynCraft native output parsing failed")
        return Proposal(
            case_id=case_id,
            action="abstain" if parsed["edits"] == [["STOP"]] else "edit",
            edits=parsed["edits"],
            rationale=parsed["reasoning"],
            expected_effect="Native SynCraft proposal; evaluate externally",
            risks=[],
        )

    def _design(self, job: dict, work: Path) -> dict:
        if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise RuntimeError("DEEPSEEK_API_KEY is missing; this is not a chemical rejection")
        case = self.cases[job["case_id"]]
        binding = self._source_binding(case.case_id)
        observation = Observation(
            case_id=case.case_id,
            target=case.target,
            source_smiles=case.smiles,
            mapped_smiles=mapped_smiles(case.smiles),
            graph_sha256=graph_hash(case.smiles),
            terminal_metrics=case.source_metrics,
            constraints=self.foundation.value.constraints,
        )
        case_root = self.source_path(case.case_id).parent
        write_json(case_root / "observation.json", observation.model_dump(mode="json"))
        attempts = []
        for attempt_index in range(self.config.chemical_retries + 1):
            attempt = case_root / "proposals" / f"attempt_{attempt_index:02d}"
            attempt.mkdir(parents=True, exist_ok=True)
            saved_result = attempt / "result.json"
            if saved_result.exists():
                result = read_json(saved_result)
            else:
                result = {
                    "attempt_index": attempt_index,
                    "maximum_proposals": 6,
                    "started_at": utc_now(),
                    "prompt_changed_for_retry": False,
                }
                # A transport crash can leave an immutable request but no completion.
                # Keep its receipt and use a separate explicitly recorded transport execution.
                responses = sorted(attempt.glob("transport_*/response.json"))
                if responses:
                    response_path = responses[0]
                else:
                    number = len(list(attempt.glob("transport_*")))
                    response_path = attempt / f"transport_{number:03d}" / "response.json"
                try:
                    if response_path.exists():
                        proposal = self._cached_proposal(
                            read_json(response_path)["native_output"], case.case_id
                        )
                    else:
                        proposal = self.components.agent.propose(
                            observation, binding, attempt_index, response_path
                        )
                    result.update(
                        proposal=proposal.model_dump(mode="json"),
                        response_path=str(response_path),
                        response_sha256=sha256(response_path),
                    )
                    if proposal.action == "abstain":
                        result["status"] = "abstained"
                    else:
                        graph = self.components.editor.apply(
                            observation.mapped_smiles,
                            proposal.edits,
                            self.foundation.value.constraints,
                        )
                        result.update(status=graph.status, graph=graph.model_dump(mode="json"))
                except ValueError as error:
                    result.update(
                        status="unreadable_output",
                        reason=str(error),
                        response_path=str(response_path),
                    )
                except RuntimeError:
                    # Transport exhaustion is infrastructure, not a chemical proposal.
                    raise
                result["completed_at"] = utc_now()
                write_json(saved_result, result)
            attempts.append(result)
            if result["status"] in {"accepted", "unchanged", "abstained"}:
                break
        last = attempts[-1]
        result = {
            "case_id": case.case_id,
            "attempts": attempts,
            "chemical_proposal_count": len(attempts),
            "retry_count": len(attempts) - 1,
            "status": last["status"]
            if last["status"] in {"accepted", "unchanged", "abstained"}
            else "chemical_retries_exhausted",
        }
        if result["status"] == "accepted":
            from agenticgen.region_conditions import build_region_conditions

            graph = GraphEditResult.model_validate(last["graph"])
            product = case_root / "edited_placeholder.sdf"
            if not product.exists():
                write_graph_placeholder(
                    graph,
                    self.source_path(case.case_id),
                    product,
                    paired_seed(self.config.seed, case.case_id, 0),
                )
            interactions = sorted({m for i in binding.interactions for m in i.atom_maps})
            conditions = build_region_conditions(
                observation.mapped_smiles,
                graph,
                interactions,
                self.source_path(case.case_id),
                product,
            )
            path = case_root / "conditions.json"
            write_json(path, conditions)
            result.update(
                graph=graph.model_dump(mode="json"),
                product_pose_sdf=str(product),
                conditions_file=str(path),
                interaction_atom_maps=interactions,
                properties=self.components.properties.evaluate(graph.product_smiles).model_dump(
                    mode="json"
                ),
            )
        return result

    def enqueue_generation(self) -> int:
        from agenticgen.region_conditions import mode_channel_masks

        count = 0
        for design in self.ledger.list(stage="design", status="completed"):
            result = read_json(Path(design["result_path"]))
            if result["status"] != "accepted":
                continue
            case_id = design["case_id"]
            conditions = read_json(Path(result["conditions_file"]))
            for arm in campaign_arms(self.config):
                mask = mode_channel_masks(conditions, arm.mode)
                if arm.mode == "gradient" and (
                    not conditions.get("fixed_atom_input_slots")
                    or (arm.coordinates and not conditions.get("interaction_input_slots"))
                ):
                    mask = {"status": "not_applicable", "reason": "Gradient target region is empty"}
                for batch_index in range(
                    self.config.samples_per_arm // self.config.microbatch_size
                ):
                    payload = {
                        **self._flowr_task(
                            Path("."), paired_seed(self.config.seed, case_id, batch_index)
                        ),
                        "schema_version": "agenticgen.flowr-campaign-task.v1",
                        "case_id": case_id,
                        "arm_id": arm.arm_id,
                        "source_pose_sdf": str(self.source_path(case_id)),
                        "product_pose_sdf": result["product_pose_sdf"],
                        "conditions_file": result["conditions_file"],
                        "mode": arm.mode,
                        "sample_count": self.config.microbatch_size,
                        "integration_steps": self.config.integration_steps,
                        "batch_index": batch_index,
                        "score_affinity": True,
                        "guidance": {
                            "family": arm.family or "ce",
                            "coordinates": arm.coordinates,
                            "weight": arm.weight,
                            "window": list(self.config.guidance_window),
                        },
                        "not_applicable": mask.get("status") in {"not_applicable", "skipped"},
                        "condition_reason": mask.get("reason"),
                    }
                    input_files = [
                        self.source_path(case_id),
                        Path(result["product_pose_sdf"]),
                        Path(result["conditions_file"]),
                    ]
                    payload["input_sha256"] = {
                        str(p): sha256(p) for p in input_files if p.is_file()
                    }
                    payload.pop("output_dir")
                    name = f"{arm.arm_id}:batch_{batch_index:03d}"
                    self.ledger.add(
                        _job_id("generation", case_id, name),
                        "generation",
                        case_id,
                        payload,
                        self.root
                        / "cases"
                        / case_id
                        / "arms"
                        / arm.arm_id
                        / f"batch_{batch_index:03d}",
                    )
                    count += 1
        return count

    def _generation(self, job: dict, work: Path) -> dict:
        self.require_preflight()
        if job["payload"].get("not_applicable"):
            return {
                "status": "not_applicable",
                "reason": job["payload"]["condition_reason"],
                "attempted": 0,
                "output_dir": str(work),
            }
        task = {**job["payload"], "output_dir": str(work)}
        self._worker(
            "agenticgen.workers.flowr_campaign", task, work, self.config.generation_timeout_seconds
        )
        runtime = read_json(work / "runtime.json")
        if runtime.get("status") == "not_applicable":
            return {
                "status": "not_applicable",
                "reason": runtime.get("reason"),
                "attempted": 0,
                "output_dir": str(work),
                "runtime": runtime,
            }
        rows = particle_rows(work / "poses.json")
        if len(rows) != self.config.microbatch_size:
            raise RuntimeError("Generation did not account for the exact attempted microbatch")
        return {
            "status": "completed",
            "output_dir": str(work),
            "attempted": len(rows),
            "decoded": sum(p["status"] == "decoded" for p in rows),
            "runtime": runtime,
        }

    def enqueue_evaluations(self) -> None:
        children = defaultdict(list)
        for child in self.ledger.list():
            if child["stage"] in {"affinity", "evaluation"}:
                children[child["payload"].get("generation_task_id")].append(child)
        for job in self.ledger.list(stage="generation", status="completed"):
            result = read_json(Path(job["result_path"]))
            output = Path(result["output_dir"])
            generation_hash = sha256(Path(job["result_path"]))
            parent = {
                "generation_task_id": job["task_id"],
                "generation_result_path": job["result_path"],
                "generation_result_sha256": generation_hash,
            }
            for child in children[job["task_id"]]:
                if (
                    child["stage"] in {"affinity", "evaluation"}
                    and child["payload"].get("generation_task_id") == job["task_id"]
                    and child["payload"].get("generation_result_sha256") != generation_hash
                    and child["status"] != "superseded"
                ):
                    self.ledger.supersede(child["task_id"], "Parent generation execution changed")
            affinity_path = output / "affinity.json"
            native_head = read_json(affinity_path) if affinity_path.is_file() else {}
            good = native_head.get("status") == "completed" and not any(
                r.get("status") == "error" for r in native_head.get("poses", [])
            )
            if not good:
                affinity_id = "affinity:" + job["task_id"] + ":" + generation_hash[:16]
                self.ledger.add(
                    affinity_id,
                    "affinity",
                    job["case_id"],
                    parent,
                    output / "affinity_recovery",
                )
                recovery = self.ledger.get(affinity_id)
                if recovery["status"] != "completed":
                    continue
                affinity_path = Path(recovery["result_path"])
            parent["affinity_result_path"] = str(affinity_path)
            parent["affinity_result_sha256"] = sha256(affinity_path)
            version = object_hash(parent)[:16]
            child_id = "evaluation:" + job["task_id"] + ":" + version
            for old in children[job["task_id"]]:
                if (
                    old["stage"] == "evaluation"
                    and old["payload"].get("generation_task_id") == job["task_id"]
                    and old["task_id"] != child_id
                    and old["status"] != "superseded"
                ):
                    self.ledger.supersede(
                        old["task_id"], "Parent generation/affinity version changed"
                    )
            self.ledger.add(
                child_id, "evaluation", job["case_id"], parent, output / ("evaluations_" + version)
            )

    def _evaluation(self, job: dict, work: Path) -> dict:
        payload = job["payload"]
        for name in ("generation_result", "affinity_result"):
            if sha256(Path(payload[name + "_path"])) != payload[name + "_sha256"]:
                raise ValueError("Evaluation parent receipt changed")
        output = Path(read_json(Path(payload["generation_result_path"]))["output_dir"])
        particles = particle_rows(output / "poses.json")
        affinity = read_json(Path(payload["affinity_result_path"])).get("poses", [])
        rows = evaluate_particles(
            self.components.geometry,
            self._source_binding(job["case_id"]),
            self.source_path(job["case_id"]),
            self.cases[job["case_id"]].smiles,
            particles,
            work,
            affinity=affinity,
        )
        return {
            "status": "completed",
            "attempted": len(rows),
            "particles": rows,
            "generation_task_id": payload["generation_task_id"],
        }

    def execute(self, task_id: str) -> bool:
        job = self.ledger.claim(task_id)
        if job is None:
            return False
        work = Path(job["workdir"]) / f"execution_{job['attempts']:03d}"
        work.mkdir(parents=True, exist_ok=False)
        write_json(
            work / "execution.json",
            {
                "task_id": task_id,
                "started_at": utc_now(),
                "code_sha256": code_fingerprint(),
                "payload_sha256": job["payload_sha256"],
                "controller_pid": os.getpid(),
            },
        )
        handlers = {
            "source_geometry": self._source_geometry,
            "source_affinity": self._affinity,
            "design": self._design,
            "generation": self._generation,
            "evaluation": self._evaluation,
            "affinity": self._affinity,
        }
        try:
            verify_input_hashes(job["payload"])
            result = handlers[job["stage"]](job, work)
            result.update(task_id=task_id, completed_at=utc_now(), code_sha256=code_fingerprint())
            path = work / "result.json"
            write_json(path, result)
            self.ledger.finish(task_id, path, skipped=result.get("status") == "not_applicable")
            return True
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            write_json(
                work / "failure.json",
                {
                    "reason": detail,
                    "traceback": traceback.format_exc(),
                    "failed_at": utc_now(),
                    "code_sha256": code_fingerprint(),
                },
            )
            self.ledger.fail(task_id, detail)
            print(
                json.dumps({"task_id": task_id, "status": "failed", "reason": detail}), flush=True
            )
            return False

    def phase(self, stage: str, workers: int = 1) -> None:
        jobs = self.ledger.list(stage=stage, status="pending")
        self.status(stage)
        if workers == 1:
            for job in jobs:
                self.execute(job["task_id"])
                self.status(stage)
        else:
            # Instantiate the immutable backend once, before sharing its retrieval corpus.
            _ = self.components
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self.execute, job["task_id"]) for job in jobs]
                for future in as_completed(futures):
                    future.result()
                    self.status(stage)

    def run(self, phase: str = "all") -> dict:
        try:
            return self._run(phase)
        finally:
            if self._gpu_session is not None:
                self._gpu_session.close()
                self._gpu_session = None

    def _run(self, phase: str = "all") -> dict:
        with campaign_lock(self.root / "controller.lock"):
            self.prepare()
            if phase in {"all", "baseline"}:
                self.phase("source_geometry", self.config.geometry_workers)
                self.phase("source_affinity")
            if phase in {"all", "design"}:
                self.phase("design", self.config.design_workers)
            if phase in {"all", "preflight", "generation"}:
                self.preflight()
            if phase in {"all", "generation"}:
                self.enqueue_generation()
                self.phase("generation")
            if phase in {"all", "evaluate"}:
                self.enqueue_evaluations()
                self.phase("affinity")
                self.enqueue_evaluations()
                self.phase("evaluation", self.config.geometry_workers)
            state = self.status("finished" if phase == "all" else f"{phase}_finished")
            manifest = read_json(self.root / "manifest.json")
            counts = state["counts"]
            unfinished = any(
                n
                for stages in counts.values()
                for status, n in stages.items()
                if status in {"pending", "running", "failed"}
            )
            manifest.update(
                status="partial" if unfinished or phase != "all" else "completed",
                last_controller_exit_at=utc_now(),
                last_phase=phase,
            )
            write_json(self.root / "manifest.json", manifest)
            return state

    def preflight(self) -> None:
        """Production is not released before the real-model verification receipt passes."""
        path = self.root / "preflight/result.json"
        if path.exists():
            try:
                self.require_preflight()
                return
            except RuntimeError:
                pass
        accepted = [
            j
            for j in self.ledger.list(stage="design", status="completed")
            if read_json(Path(j["result_path"]))["status"] == "accepted"
        ]
        if not accepted:
            if len(self.ledger.list(stage="design", status="completed")) != len(self.cases):
                raise RuntimeError("Design phase is incomplete; real-model preflight cannot start")
            write_json(
                path,
                {
                    "status": "not_applicable",
                    "reason": "No changed parsable graph",
                    "code_sha256": code_fingerprint(),
                    "protocol_sha256": object_hash(self.config.model_dump(mode="json")),
                },
            )
            return
        # Implementation is kept in a separate optional-Torch module.
        from agenticgen.campaign_preflight import run_preflight

        def diagnostic_coverage(job):
            result = read_json(Path(job["result_path"]))
            condition = read_json(Path(result["conditions_file"]))
            anchors = bool(condition.get("interaction_input_slots"))
            partial = (
                0 < len(condition.get("fixed_atom_input_slots", [])) < len(condition["atom_maps"])
            )
            return (
                not (anchors and partial),
                not anchors,
                not partial,
                self.cases[job["case_id"]].selection_rank,
            )

        # This chooses preflight coverage, never a production molecule or winner.
        selected = min(accepted, key=diagnostic_coverage)
        run_preflight(self, selected, self.root / "preflight")
        self.require_preflight()

    def require_preflight(self) -> None:
        path = self.root / "preflight/result.json"
        receipt = read_json(path) if path.exists() else {}
        if (
            receipt.get("status") != "passed"
            or receipt.get("code_sha256") != code_fingerprint()
            or receipt.get("protocol_sha256") != object_hash(self.config.model_dump(mode="json"))
        ):
            raise RuntimeError("A current passing real-model preflight receipt is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "requeue", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "baseline", "design", "preflight", "generation", "evaluate"),
        default="all",
    )
    parser.add_argument("--task-id")
    parser.add_argument("--reason")
    args = parser.parse_args()
    campaign = Campaign(args.config, args.output)
    if args.command == "prepare":
        value = campaign.prepare()
    elif args.command == "run":
        value = campaign.run(args.phase)
    elif args.command == "requeue":
        if not args.task_id or not args.reason:
            parser.error("requeue requires --task-id and --reason")
        with campaign_lock(campaign.root / "controller.lock"):
            campaign.ledger.requeue(args.task_id, args.reason)
        value = campaign.status("recovery_prepared")
    elif args.command == "report":
        from agenticgen.campaign_report import report_campaign

        value = report_campaign(campaign)
    else:
        value = {"counts": campaign.ledger.counts(), "root": str(campaign.root)}
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
