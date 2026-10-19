"""Isolated native FLOWR launcher. This file deliberately imports no agenticgen code.

It runs under the existing FLOWR/DTK interpreter. The two wrappers below observe and
validate native preprocessing/masks, without changing tensors or model behavior.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
import traceback
from pathlib import Path


def run(task_path: Path) -> None:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task["schema_version"] != "agenticgen.flowr_worker.v1":
        raise ValueError("Unsupported FLOWR worker task schema")
    root = Path(task["root"]).resolve()
    receipt_path = Path(task["runtime_receipt"])
    receipt = {
        "schema_version": "agenticgen.flowr_runtime.v1",
        "status": "starting",
        "python": sys.executable,
        "python_version": sys.version,
        "root": str(root),
        "mode": task["mode"],
        "seed": task["seed"],
        "native_masks": [],
    }

    def save():
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save()
    try:
        # FLOWR's native extension import order stays isolated from all agent modules.
        sys.path.insert(0, str(root))
        native = importlib.import_module("flowr.gen.generate_from_pdb")
        if not Path(native.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Imported FLOWR module is outside the configured checkout")
        import numpy as np
        import torch
        from rdkit import Chem

        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task["seed"])
        receipt.update(
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            torch_cuda_version=torch.version.cuda,
            torch_hip_version=getattr(torch.version, "hip", None),
            native_entrypoint=str(Path(native.__file__).resolve()),
        )
        source = Chem.SDMolSupplier(task["input_ligand"], removeHs=True)[0]
        if source is None:
            raise ValueError("Prepared input SDF is unreadable")
        ref_coords = np.asarray(source.GetConformer().GetPositions())
        ref_atomics = np.asarray([a.GetAtomicNum() for a in source.GetAtoms()])
        ref_charges = np.asarray([a.GetFormalCharge() for a in source.GetAtoms()])
        original_load_data = native.util.load_data_from_pdb

        def checked_load_data(*args, **kwargs):
            system = original_load_data(*args, **kwargs)
            if system is None:
                raise RuntimeError("Native pocket preprocessing returned no system")
            ligand = system.ligand
            if len(ligand) != len(ref_atomics):
                raise RuntimeError("Native ligand preprocessing changed the heavy-atom count")
            if not np.array_equal(ligand.atomics.cpu().numpy(), ref_atomics):
                raise RuntimeError("Native ligand atom order/types differ from the planned slots")
            if not np.array_equal(ligand.charges.cpu().numpy(), ref_charges):
                raise RuntimeError("Native ligand charges differ from the planned graph")
            if not np.allclose(ligand.coords.cpu().numpy(), ref_coords, atol=0.002, rtol=0):
                raise RuntimeError("Native ligand preprocessing reordered or moved mapped atoms")
            receipt["input_slot_order_verified"] = True
            save()
            return system

        native.util.load_data_from_pdb = checked_load_data
        from flowr.data.interpolate import GeometricInterpolant

        original_build = GeometricInterpolant._build_inference_prior
        plan = task["mask_plan"]
        expected_fixed = plan["fixed_input_slots"]

        def checked_build(self, to_mol, rdkit_mol, mode, mask, is_local):
            actual_fixed = torch.where(mask)[0].cpu().tolist()
            expected_mode = "graph" if task["mode"] == "graph" else "substructure_inpainting"
            if mode != expected_mode:
                raise RuntimeError(f"Native mode differs: expected {expected_mode}, got {mode}")
            if actual_fixed != expected_fixed:
                raise RuntimeError(
                    f"Native keep-mask differs: expected {expected_fixed}, got {actual_fixed}"
                )
            prior = original_build(self, to_mol, rdkit_mol, mode, mask, is_local)
            output_fixed = torch.where(prior.fragment_mask)[0].cpu().tolist()
            expected_output = (
                expected_fixed if task["mode"] == "graph" else list(range(len(expected_fixed)))
            )
            if output_fixed != expected_output:
                raise RuntimeError("Native fixed output slots differ from planned identity mapping")
            if task["mode"] == "fragment" and not torch.allclose(
                prior.coords[output_fixed], to_mol.coords[expected_fixed], atol=0.0001, rtol=0
            ):
                raise RuntimeError("Native fragment prior did not retain fixed atom coordinates")
            if len(receipt["native_masks"]) < 32:
                receipt["native_masks"].append(
                    {
                        "mode": mode,
                        "fixed_input_slots": actual_fixed,
                        "fixed_output_slots": output_fixed,
                    }
                )
                save()
            return prior

        GeometricInterpolant._build_inference_prior = checked_build
        sys.argv = [str(native.__file__), *task["native_arguments"]]
        args = native.get_args()
        # Native mode selection and sampling remain the upstream implementation.
        native.evaluate(args)
        if not receipt.get("input_slot_order_verified") or not receipt["native_masks"]:
            raise RuntimeError("Native mask/slot instrumentation was not exercised")
        receipt["status"] = "completed"
        save()
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: flowr_inpainting.py task.json")
    run(Path(sys.argv[1]))
