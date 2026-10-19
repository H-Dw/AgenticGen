"""Small real-checkpoint checks, stored separately from production denominators."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

from agenticgen.campaign_contracts import paired_seed
from agenticgen.io import object_hash, read_json, utc_now, write_json
from agenticgen.region_conditions import mode_channel_masks


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield value


def _check_trace(output: Path) -> dict:
    rows = read_json(output / "guidance_trace.json")
    if len(rows) != 80 or [r["time"] for r in rows] != [round(i / 100, 2) for i in range(10, 90)]:
        raise RuntimeError("Actual native guidance clock differs from the frozen 80-step window")
    if not all(math.isfinite(n) for row in rows for n in _numbers(row)):
        raise RuntimeError("Guidance trace contains a nonfinite value")
    for name in ("logit_gradient_rms", "coordinate_gradient_rms", "injected_rms_angstrom"):
        if not any(value > 0 for row in rows for value in _numbers(row[name])):
            raise RuntimeError(f"Real guidance did not exercise {name}")
    maximum = max(value for row in rows for value in _numbers(row["injected_max_atom_angstrom"]))
    if maximum > 0.025001:
        raise RuntimeError("Coordinate injection exceeded the specified per-atom cap")
    probe = read_json(output / "guidance_probe.json")
    if probe.get("status") != "completed" or not probe.get("report", {}).get("passed"):
        raise RuntimeError(f"Real-model derivative verification failed: {probe}")
    return {
        "active_steps": len(rows),
        "maximum_injection_angstrom": maximum,
        "finite_difference": probe["report"],
    }


def run_preflight(campaign, design_job: dict, output: Path) -> dict:
    import torch

    from agenticgen.campaign import code_fingerprint, particle_rows

    design = read_json(Path(design_job["result_path"]))
    conditions = read_json(Path(design["conditions_file"]))
    execution = output / ("execution_" + uuid.uuid4().hex[:12])
    execution.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "agenticgen.campaign-preflight.v1",
        "status": "running",
        "started_at": utc_now(),
        "code_sha256": code_fingerprint(),
        "protocol_sha256": object_hash(campaign.config.model_dump(mode="json")),
        "case_id": design_job["case_id"],
        "execution_dir": str(execution),
        "production_particles": 0,
        "checks": {},
    }
    write_json(output / "result.json", receipt)
    seed = paired_seed(campaign.config.seed, design_job["case_id"], 0, preflight=True)
    jobs = [
        ("unguided", "unguided", "ce", False, 0),
        ("zero_guidance", "gradient", "ce", False, 0),
        ("ce_graph", "gradient", "ce", False, 1),
        ("brier", "gradient", "brier", bool(conditions["interaction_input_slots"]), 1),
    ]
    jobs += [
        (mode, mode, "ce", False, 0) for mode in ("graph", "fragment_graph", "fragment_graph_pose")
    ]
    tensors, runtimes, paths = {}, {}, {}
    try:
        for label, mode, family, coordinates, weight in jobs:
            channels = mode_channel_masks(conditions, mode)
            if channels["status"] != "ready":
                receipt["checks"][label] = {"status": "not_applicable", **channels}
                continue
            folder = execution / label
            folder.mkdir()
            task = {
                **campaign._flowr_task(folder, seed),
                "schema_version": "agenticgen.flowr-campaign-task.v1",
                "source_pose_sdf": str(campaign.source_path(design_job["case_id"])),
                "product_pose_sdf": design["product_pose_sdf"],
                "conditions_file": design["conditions_file"],
                "mode": mode,
                "sample_count": 4,
                "integration_steps": 100,
                "score_affinity": False,
                "guidance_probe": mode == "gradient" and weight > 0,
                "guidance": {
                    "family": family,
                    "coordinates": coordinates,
                    "weight": weight,
                    "window": [0.1, 0.9],
                },
            }
            campaign._worker(
                "agenticgen.workers.flowr_campaign",
                task,
                folder,
                campaign.config.generation_timeout_seconds,
            )
            runtime = read_json(folder / "runtime.json")
            if runtime.get("status") != "completed" or not runtime.get("generation_completed"):
                raise RuntimeError(f"Preflight {label} did not actually generate")
            if len(particle_rows(folder / "poses.json")) != 4:
                raise RuntimeError("Preflight lost an attempted particle")
            raw = torch.load(folder / "raw_final.pt", map_location="cpu", weights_only=True)
            if any(not torch.isfinite(t).all() for t in raw.values()):
                raise RuntimeError(f"Nonfinite final tensors in {label}")
            paths[label], tensors[label], runtimes[label] = folder, raw, runtime
            detail = {
                "status": "passed",
                "raw_particles": 4,
                "decoded": runtime.get("decoded_count"),
                "output_dir": str(folder),
            }
            if mode == "gradient" and weight > 0:
                detail.update(_check_trace(folder))
            if mode not in {"gradient", "unguided"}:
                reference = torch.load(
                    folder / "raw_reference.pt", map_location="cpu", weights_only=True
                )
                for key in ("atomics", "charges"):
                    slots = channels["fixed_atom_input_slots"]
                    if not torch.equal(raw[key][:, slots], reference[key][:, slots]):
                        raise RuntimeError(f"Hard {label} failed its terminal {key} condition")
                for first, second in channels["fixed_bond_pairs"]:
                    if not torch.equal(
                        raw["bonds"][:, first, second], reference["bonds"][:, first, second]
                    ):
                        raise RuntimeError(f"Hard {label} failed its terminal edge condition")
                slots = channels["fixed_coordinate_input_slots"]
                if slots:
                    # _generate returns world Angstrom; the collated posterior is
                    # normalized around the pocket COM and is not a pose target.
                    targets = raw["coords"].new_tensor(conditions["interaction_coordinates_world"])
                    targets = targets.unsqueeze(0).expand(raw["coords"].shape[0], -1, -1)
                    if not torch.allclose(raw["coords"][:, slots], targets, atol=1e-4, rtol=0):
                        raise RuntimeError(
                            "Hard original interaction coordinates were not retained"
                        )
                detail["terminal_conditions_checked"] = True
            receipt["checks"][label] = detail
            write_json(output / "result.json", receipt)
        if not tensors.keys() >= {"unguided", "zero_guidance", "ce_graph", "brier", "graph"}:
            raise RuntimeError(
                "The selected preflight graph lacks required test degrees of freedom"
            )
        for key, value in tensors["unguided"].items():
            if not torch.equal(value, tensors["zero_guidance"][key]):
                raise RuntimeError(f"Zero guidance changed the native output tensor {key}")
        for label, runtime in runtimes.items():
            for name in ("base_prior_sha256", "rng_before_generation_sha256"):
                if runtime[name] != runtimes["unguided"][name]:
                    raise RuntimeError(f"{label} changed matched prior/RNG initialization")
        if (
            runtimes["ce_graph"]["final_tensor_sha256"]
            == runtimes["unguided"]["final_tensor_sha256"]
        ):
            raise RuntimeError("Active real-model guidance had no effect on final tensors")
        receipt["checks"]["zero_strength_native_parity"] = {"status": "passed"}
        receipt["checks"]["matched_priors_and_rng"] = {"status": "passed"}
        # The identical SDF is rescored twice in reverse order under different
        # sampling seeds. Fixed singleton native SC must make all four equal.
        source = str(campaign.source_path(design_job["case_id"]))
        values = []
        for offset in (0, 19):
            folder = execution / f"affinity_invariance_{offset}"
            folder.mkdir()
            task = {
                **campaign._flowr_task(folder, seed + offset),
                "poses": [
                    {"pose_id": p, "sdf_path": source}
                    for p in (("a", "b") if offset == 0 else ("b", "a"))
                ],
            }
            campaign._worker(
                "agenticgen.campaign_affinity",
                task,
                folder,
                campaign.config.affinity_timeout_seconds,
            )
            scores = read_json(folder / "affinity.json")
            if scores["status"] != "completed" or any(r["status"] != "ok" for r in scores["poses"]):
                raise RuntimeError("Affinity invariance probe did not return actual scores")
            values.extend(r["pic50"] for r in scores["poses"])
        if max(values) - min(values) > 1e-6:
            raise RuntimeError(f"Affinity depends on task seed/order: {values}")
        receipt["checks"]["affinity_seed_order_invariance"] = {"status": "passed", "scores": values}
        receipt.update(status="passed", completed_at=utc_now())
    except BaseException as error:
        receipt.update(
            status="failed", failed_at=utc_now(), reason=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        write_json(execution / "result.json", receipt)
        write_json(output / "result.json", receipt)
    return receipt
