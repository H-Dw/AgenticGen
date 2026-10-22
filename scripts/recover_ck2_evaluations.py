"""Repair only failed pose-identity evaluations beside a healthy frozen controller.

The separate overlay and lock leave the production source and pending queue alone.
Run with the campaign interpreter, activation, and frozen code on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sqlite3
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from agenticgen.adapters.geometry import SynCraftGeometry
from agenticgen.campaign import Campaign, code_fingerprint
from agenticgen.campaign_ledger import campaign_lock
from agenticgen.io import object_hash, read_json, sha256, utc_now, write_json

ERROR = "ValueError: Input pose connectivity or specified stereochemistry differs"
METADATA_ERROR = "ValueError: Docking preparation changed source identity/stereochemistry"


def import_overlay(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_failure(job: dict) -> bool:
    work = Path(job["workdir"]) / f"execution_{job['attempts']:03d}"
    receipt = work / "evaluations.json"
    if not receipt.is_file():
        return False
    rows = read_json(receipt)["particles"]
    errors = [row for row in rows if row["status"] == "infrastructure_error"]
    return bool(errors) and all(
        (work / row["pose_id"] / "score_only/worker.log").is_file()
        and any(
            error in (work / row["pose_id"] / "score_only/worker.log").read_text()
            for error in (ERROR, METADATA_ERROR)
        )
        for row in errors
    )


class RecoveryCampaign(Campaign):
    def _evaluation(self, job: dict, work: Path) -> dict:
        result = super()._evaluation(job, work)
        result["evaluation_recovery"] = {
            **self.recovery_provenance,
            "prior_execution": str(
                Path(job["workdir"]) / f"execution_{job['attempts'] - 1:03d}"
            ),
            "generation_and_affinity_repeated": False,
            "coordinates_repaired": False,
        }
        return result


def run(args) -> None:
    directory = args.receipts.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "status.json"
    state = {"status": "starting", "pid": os.getpid(), "started_at": utc_now(), "recovered": []}

    def save(**changes):
        state.update(updated_at=utc_now(), **changes)
        write_json(state_path, state)

    with campaign_lock(directory / "recovery.lock"):
        try:
            release = read_json(args.overlay / "release.json")
            if code_fingerprint() != release["production_code_sha256"]:
                raise ValueError("The frozen production code changed")
            for name, expected in release["files"].items():
                if sha256(args.overlay / name) != expected:
                    raise ValueError(f"Overlay source changed: {name}")
            # Imports affect this maintenance process only, never the live controller.
            alignment = import_overlay("ck2_recovery_alignment", args.overlay / "plip_alignment.py")
            original_alignment = importlib.import_module("agenticgen.workers.plip_alignment")
            original_alignment.pose_graph_smiles = alignment.pose_graph_smiles
            original_alignment.pose_request_smiles = alignment.pose_request_smiles
            evaluation = import_overlay(
                "ck2_recovery_evaluation", args.overlay / "campaign_evaluation.py"
            )
            importlib.import_module("agenticgen.campaign").evaluate_particles = (
                evaluation.evaluate_particles
            )
            campaign = RecoveryCampaign(args.config, args.output)
            if campaign.ledger.counts().get("generation") != {"completed": 8000}:
                raise ValueError("Recovery requires all frozen generation jobs to be complete")
            config_hashes = {
                str(path): sha256(path) for path in (args.config, campaign.foundation.path)
            }
            if config_hashes != release["config_hashes"]:
                raise ValueError("Frozen configuration changed")
            geometry = SynCraftGeometry(campaign.foundation)
            geometry.worker = args.overlay / "syncraft_geometry.py"
            geometry.protocol = {
                **geometry.protocol,
                "worker_sha256": sha256(geometry.worker),
                "alignment_sha256": sha256(args.overlay / "plip_alignment.py"),
                "pose_identity_serialization": "explicit_h_graph_copy_v3_heavy_atom_request",
            }
            geometry.protocol_id = object_hash(geometry.protocol)
            campaign._components = SimpleNamespace(geometry=geometry)
            campaign.recovery_provenance = {
                "overlay": str(args.overlay),
                "overlay_files": release["files"],
                "recovery_driver_sha256": sha256(Path(__file__)),
                "geometry_protocol_id": geometry.protocol_id,
                "config_hashes": config_hashes,
                "reason": "SDF/request/Meeko stereochemistry serialization; non-tetrahedral H retained implicitly in requests",
            }
            with campaign.ledger.connect() as source:
                with sqlite3.connect(directory / "ledger.before.sqlite") as backup:
                    source.backup(backup)
            save(status="watching", provenance=campaign.recovery_provenance)
            while True:
                failed = campaign.ledger.list(stage="evaluation", status="failed")
                for job in failed:
                    if not eligible_failure(job):
                        save(status="needs_attention", task_id=job["task_id"], reason=job["reason"])
                        raise RuntimeError("An evaluation error is outside the verified repair scope")
                    if job["attempts"] != 1:
                        raise RuntimeError("A repaired evaluation failed again; no automatic retry")
                    save(status="repairing", task_id=job["task_id"])
                    write_json(directory / (object_hash(job["task_id"]) + ".before.json"), job)
                    campaign.ledger.requeue(
                        job["task_id"],
                        "Verified pose SMILES serialization defect; preserve old execution and "
                        f"rescore the same four actual poses with overlay {args.overlay}",
                    )
                    if not campaign.execute(job["task_id"]):
                        raise RuntimeError("Corrected evaluation failed; prior evidence retained")
                    state["recovered"].append(job["task_id"])
                    with campaign.ledger.connect() as db:
                        db.execute(
                            "INSERT INTO events(task_id,time,kind,detail) VALUES(?,?,?,?)",
                            (
                                job["task_id"],
                                utc_now(),
                                "pose_serialization_evaluation_recovered",
                                json.dumps(campaign.recovery_provenance),
                            ),
                        )
                    save(status="watching", counts=campaign.ledger.counts())
                counts = campaign.ledger.counts()
                evaluations = counts.get("evaluation", {})
                controller = read_json(args.controller_status)
                if evaluations == {"completed": 8000}:
                    if controller["status"] == "completed":
                        with campaign_lock(args.output / "controller.lock"):
                            if any(
                                n
                                for stages in campaign.ledger.counts().values()
                                for status, n in stages.items()
                                if status in {"failed", "pending", "running"}
                            ):
                                raise RuntimeError("Other campaign tasks remain unresolved")
                            manifest_path = args.output / "manifest.json"
                            manifest = read_json(manifest_path)
                            write_json(directory / "manifest.before-finalization.json", manifest)
                            manifest.update(
                                status="completed", evaluation_recovery_completed_at=utc_now()
                            )
                            write_json(manifest_path, manifest)
                            campaign.status("finished")
                        save(status="completed", finished_at=utc_now(), counts=counts)
                        return
                if controller["status"] in {"failed", "interrupted"}:
                    raise RuntimeError("The main controller exited unexpectedly")
                if controller["status"] == "completed":
                    raise RuntimeError("Main controller finished with unresolved evaluations")
                save(status="watching", counts=counts)
                time.sleep(args.poll_seconds)
        except BaseException as error:
            save(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
                finished_at=utc_now(),
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--controller-status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    run(parser.parse_args())
