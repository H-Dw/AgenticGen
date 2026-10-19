"""Uniform, zero-noise affinity rescoring of actual final poses using native FLOWR."""

from __future__ import annotations

import contextlib
import importlib
import math
import random
import sys
from pathlib import Path

from agenticgen.io import sha256, write_json


@contextlib.contextmanager
def _scoring_rng(torch, seed: int = 42):
    """Singleton native priors are reproducible and do not advance sampling RNG."""
    import numpy as np

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextlib.contextmanager
def _prediction_state(model):
    """The same explicit SC-on head protocol applies to every generated/source pose."""
    overrides = {
        "inpainting_mode": False,
        "graph_inpainting": False,
        "_inpaint_self_condition": False,
        "self_condition": True,
    }
    previous = {key: getattr(model, key) for key in overrides if hasattr(model, key)}
    try:
        for key, value in overrides.items():
            if hasattr(model, key):
                setattr(model, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(model, key, value)


def _arguments(native, task: dict, first_pose: str):
    argv = [
        str(native.__file__),
        "--pdb_file",
        task["pocket_pdb"],
        "--ligand_file",
        first_pose,
        "--arch",
        "pocket",
        "--pocket_type",
        "holo",
        "--ckpt_path",
        task["checkpoint"],
        "--save_dir",
        str(Path(task["output_dir"]) / "affinity"),
        "--gpus",
        "1",
        "--num_workers",
        "0",
        "--batch_cost",
        "200",
        "--coord_noise_scale",
        "0.0",
        "--seed",
        "42",
        "--pocket_cutoff",
        str(task.get("pocket_cutoff", 7.0)),
    ]
    if task.get("cut_pocket", True):
        argv.append("--cut_pocket")
    old = sys.argv
    try:
        sys.argv = argv
        return native.get_args()
    finally:
        sys.argv = old


def score_pose_files(task: dict, poses: list[dict], loaded: tuple | None = None) -> list[dict]:
    """Reuse a loaded model when supplied; never decode a replacement graph as scored pose.

    Every input receives a row, including unreadable/unsupported structures. Native
    preprocessing constructs the exact molecular tensor from each input SDF. Scores
    are terminal predictions, not per-step forecasts from the generation trajectory.
    """
    root = Path(task["flowr_root"]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from rdkit import Chem

    native = importlib.import_module("flowr.predict.predict_from_pdb")
    if not Path(native.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Affinity module is outside the pinned FLOWR checkout")
    rows, eligible = [], []
    for index, pose in enumerate(poses):
        identity = pose.get("pose_id", f"particle_{pose.get('particle_index', index):04d}")
        row = {
            "pose_id": identity,
            "particle_index": pose.get("particle_index", index),
            "status": "unavailable",
            "pic50": None,
            "coordinate_noise_angstrom": 0.0,
            "ligand_time": 0.9999,
            "self_condition": True,
            "sc_initialization": "native_prior_seed42",
            "scoring_seed": 42,
            "scoring_batch_size": 1,
            "pose_sdf": pose.get("sdf_path"),
        }
        rows.append(row)
        if not pose.get("sdf_path") or not Path(pose["sdf_path"]).is_file():
            row["reason"] = "No decoded input pose"
            continue
        path = Path(pose["sdf_path"])
        mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
        if mol is None or mol.GetNumConformers() != 1:
            row["reason"] = "Input pose is unreadable"
            continue
        if len(Chem.GetMolFrags(mol)) != 1:
            row["reason"] = "Disconnected graph; no chemical affinity interpretation"
            continue
        row["input_pose_sha256"] = sha256(path)
        eligible.append((row, path))
    if not eligible:
        return rows
    with _scoring_rng(torch):
        _score_eligible(torch, native, task, eligible, loaded)
    for row, path in eligible:
        if sha256(path) != row["input_pose_sha256"]:
            raise RuntimeError("Affinity evaluation changed its input SDF")
        if row["status"] == "unavailable":
            row.update(status="error", reason="Input was not returned by native batching")
    return rows


def _score_eligible(torch, native, task, eligible, loaded):
    args = _arguments(native, task, str(eligible[0][1]))
    if loaded is None:
        loaded = native.load_model(args)
    model, hparams, vocab, charges, hybridization, aromatic, *_ = loaded
    from flowr.util.device import resolve_device

    device = resolve_device(args)
    model = model.to(device)
    model.eval()
    transform, interpolant = native.load_util(
        args, hparams, vocab, charges, hybridization, aromatic
    )
    for row, path in eligible:
        try:
            # Seed before preprocessing/collation: the first native self-conditioning
            # pass uses the randomly drawn prior, even when coordinate noise is zero.
            with _scoring_rng(torch), _prediction_state(model), torch.inference_mode():
                args.ligand_file = str(path)
                args.ligand_idx = 0
                system = native.load_data_from_pdb(
                    args,
                    remove_hs=hparams["remove_hs"],
                    remove_aromaticity=hparams["remove_aromaticity"],
                    ligand_idx=0,
                )
                if system is None:
                    raise ValueError("Native affinity preprocessing returned no system")
                system.metadata["system_id"] = row["pose_id"]
                dataset = native.get_dataset([system], transform, vocab, interpolant, args, hparams)
                dataloader = native.get_dataloader(args, dataset, interpolant)
                values = _predict_singleton(torch, model, dataloader, device)
                if len(values) != 1 or not math.isfinite(values[0]):
                    raise ValueError("Affinity singleton produced missing or nonfinite pIC50")
                row.update(status="ok", pic50=values[0])
        except (ValueError, RuntimeError, OSError, TypeError) as error:
            row.update(status="error", reason=f"Native affinity scoring: {error}")


def _predict_singleton(torch, model, dataloader, device):
    values = []
    for prior, posterior, _, _ in dataloader:
        from flowr.util.device import dict_to_device

        lig_prior = model.builder.extract_ligand_from_complex(prior)
        lig_data = model.builder.extract_ligand_from_complex(posterior)
        for data, whole in ((lig_prior, prior), (lig_data, posterior)):
            data["interactions"] = whole["interactions"]
            data["fragment_mask"] = whole["fragment_mask"]
            if "fragment_mode" in whole:
                data["fragment_mode"] = whole["fragment_mode"]
        pocket = model.builder.extract_pocket_from_complex(posterior)
        pocket["interactions"] = posterior["interactions"]
        pocket["complex"] = posterior["complex"]
        lig_prior = dict_to_device(lig_prior, device)
        lig_data = dict_to_device(lig_data, device)
        pocket = dict_to_device(pocket, device)
        batch_size = lig_data["coords"].shape[0]
        times = [
            torch.full((batch_size,), 0.9999, device=device),
            torch.full((batch_size,), 0.9999, device=device),
            torch.ones(batch_size, device=device),
        ]
        prediction = model._predict_affinity(lig_prior, lig_data, pocket, times)
        if "affinity" not in prediction or "pic50" not in prediction["affinity"]:
            raise RuntimeError("Checkpoint has no pIC50 affinity head")
        batch_values = prediction["affinity"]["pic50"].detach().reshape(-1).cpu().tolist()
        if len(batch_values) != 1:
            raise RuntimeError("Affinity batch/structure cardinality mismatch")
        values.extend(batch_values)
    return values


def run_task(task_path: Path) -> None:
    from agenticgen.io import read_json

    task = read_json(task_path)
    out = Path(task["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": "agenticgen.pose-affinity.v1", "status": "running"}
    try:
        result["poses"] = score_pose_files(task, task["poses"])
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "affinity.json", result)


if __name__ == "__main__":
    run_task(Path(sys.argv[1]))
