"""One physical FLOWR batch with independent atom, bond and coordinate channels.

The native checkpoint, forward implementation and integration loop are reused.
All final tensors are saved before any decoding. No resampling, deduplication,
largest-fragment pruning, geometry optimization or hidden source-pose initialization
is performed. Launch this file with the existing isolated FLOWR interpreter.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import random
import sys
import traceback
import types
from pathlib import Path

MODEL_CACHE = {}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_fingerprint(tensors: dict, torch) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode("utf-8"))
            digest.update(str((array.shape, array.dtype)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def native_condition_channels(model, conditioner):
    """Route every native clamp, SC call, clock and fixed embedding independently."""
    old_builder = model.builder.inpaint_graph
    old_times = model._get_times
    old_gen = model.gen.forward
    old_flags = {
        name: getattr(model, name)
        for name in (
            "graph_inpainting",
            "inpainting_mode",
            "inpainting_mode_inf",
        )
    }

    def clamp(builder, data, prediction, **kwargs):
        del builder, data
        return conditioner.apply(
            prediction,
            feature_keys=kwargs.get("feature_keys"),
            overwrite_with_zeros=kwargs.get("overwrite_with_zeros", False),
        )

    def clocks(self, times, lig_mask, inpaint_mask):
        del self, inpaint_mask
        return conditioner.times(times, lig_mask)

    def generator(self, *args, **kwargs):
        del self
        if getattr(model, "use_is_fixed_embed", False):
            kwargs["is_fixed_node"] = conditioner.atom_mask.long()
            kwargs["is_fixed_position"] = conditioner.coordinate_mask.long()
        return old_gen(*args, **kwargs)

    model.builder.inpaint_graph = types.MethodType(clamp, model.builder)
    model._get_times = types.MethodType(clocks, model)
    model.gen.forward = types.MethodType(generator, model.gen)
    # Use the native graph branch to reach the custom independent-channel clamp.
    # The fragment branch would copy all Q coordinates and couple the clocks.
    model.graph_inpainting = True
    model.inpainting_mode = False
    model.inpainting_mode_inf = "graph"
    try:
        yield
    finally:
        model.builder.inpaint_graph = old_builder
        model._get_times = old_times
        model.gen.forward = old_gen
        for name, value in old_flags.items():
            setattr(model, name, value)


def _cpu_tensors(data, torch):
    return {
        key: value.detach().cpu().clone() for key, value in data.items() if torch.is_tensor(value)
    }


def _load_model_cached(native, args, task, cache):
    checkpoint = Path(task["checkpoint"]).resolve()
    stat = checkpoint.stat()
    key = (
        str(Path(task["flowr_root"]).resolve()),
        str(checkpoint),
        stat.st_size,
        stat.st_mtime_ns,
        100,
    )
    hit = cache is not None and key in cache
    if hit:
        loaded = cache[key]
    else:
        loaded = native.load_model(args)
        if cache is not None:
            # A serving process owns one checkpoint at a time. Replacing its
            # registered checkpoint does not accumulate stale models in GPU RAM.
            cache.clear()
            cache[key] = loaded
    model = loaded[0]
    snapshot_key = "_agenticgen_campaign_initial_configuration"
    if not hasattr(model, snapshot_key):
        # Only configuration is copied: never duplicate checkpoint tensors or
        # optimizer/module objects in the persistent GPU worker.
        setattr(
            model,
            snapshot_key,
            {
                "hparams": copy.deepcopy(dict(loaded[1])),
                "model_hparams": copy.deepcopy(dict(model.hparams))
                if hasattr(model, "hparams")
                else None,
                "model_flags": {
                    name: copy.deepcopy(getattr(model, name))
                    for name in (
                        "graph_inpainting",
                        "inpainting_mode",
                        "inpainting_mode_inf",
                        "_inpaint_self_condition",
                        "self_condition",
                        "self_condition_mode",
                        "self_condition_prob",
                        "use_is_fixed_embed",
                    )
                    if hasattr(model, name)
                },
                "builder_flags": {
                    name: copy.deepcopy(getattr(model.builder, name))
                    for name in (
                        "ligand_valence_repair",
                        "ligand_valence_repair_allow_bond_deletion",
                        "ligand_valence_repair_max_edits",
                        "ligand_valence_repair_top_k",
                        "ligand_valence_repair_max_states",
                    )
                    if hasattr(model.builder, name)
                },
            },
        )
    initial = getattr(model, snapshot_key)
    loaded[1].clear()
    loaded[1].update(copy.deepcopy(initial["hparams"]))
    loaded[1]["save_dir"] = args.save_dir
    if initial["model_hparams"] is not None:
        model.hparams.clear()
        model.hparams.update(copy.deepcopy(initial["model_hparams"]))
        model.hparams["save_dir"] = args.save_dir
    for name, value in initial["model_flags"].items():
        setattr(model, name, copy.deepcopy(value))
    for name, value in initial["builder_flags"].items():
        setattr(model.builder, name, copy.deepcopy(value))
    if hasattr(model.builder, "save_dir"):
        model.builder.save_dir = args.save_dir
    return loaded, hit


@contextlib.contextmanager
def capture_native_probe_state(model, destination: Path, torch):
    """Save a real t=.4 native forward input without drawing any random numbers."""
    original = model.forward
    captured = {}

    def capture(self, batch, pocket_batch, times, *args, **kwargs):
        del self
        if not captured and abs(float(times[0][0]) - 0.4) < 0.00001:

            def detached(value):
                if torch.is_tensor(value):
                    return value.detach().clone()
                if isinstance(value, dict):
                    return {key: detached(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [detached(item) for item in value]
                return value

            captured.update(
                batch=detached(batch),
                pocket_batch=detached(pocket_batch),
                times=detached(times),
                cond_batch=detached(kwargs.get("cond_batch")),
                forward_kwargs={
                    key: detached(value)
                    for key, value in kwargs.items()
                    if key not in {"cond_batch", "training"}
                },
            )

            def cpu_tree(value):
                if torch.is_tensor(value):
                    return value.detach().cpu().clone()
                if isinstance(value, dict):
                    return {key: cpu_tree(item) for key, item in value.items() if key != "complex"}
                if isinstance(value, (list, tuple)):
                    return [cpu_tree(item) for item in value]
                if value is None or isinstance(value, (bool, int, float, str)):
                    return value
                raise TypeError(f"Unsupported native probe-state value: {type(value).__name__}")

            durable = {
                "batch": cpu_tree(captured["batch"]),
                "pocket_batch": cpu_tree(captured["pocket_batch"]),
                "times": cpu_tree(captured["times"]),
                "cond_batch": cpu_tree(captured["cond_batch"]),
                "forward_kwargs": cpu_tree(captured["forward_kwargs"]),
                "pocket_com_world": [
                    system.com.detach().cpu() for system in pocket_batch.get("complex", [])
                ],
                "time": 0.4,
            }
            torch.save(durable, destination)
        return original(batch, pocket_batch, times, *args, **kwargs)

    model.forward = types.MethodType(capture, model)
    try:
        yield captured
    finally:
        model.forward = original


def _make_product_ligand(conditions, hparams, torch):
    """Featurize a graph at dummy zero coordinates. No RDKit embedding can gate it."""
    from flowr.util import rdkit as native_rdkit
    from flowr.util.molrepr import GeometricMol
    from rdkit import Chem

    mol = Chem.MolFromSmiles(conditions["product_mapped_smiles"])
    if mol is None:
        raise ValueError("Edited product graph is unparseable; campaign input is skipped")
    if [a.GetAtomMapNum() for a in mol.GetAtoms()] != conditions["atom_maps"]:
        raise ValueError("Product RDKit order differs from the condition slot order")
    if hparams.get("remove_aromaticity", False):
        Chem.Kekulize(mol, clearAromaticFlags=True)
    bonds = native_rdkit.retrieve_bonds_from_mol(mol, infer_bonds=False)
    indices = torch.tensor([b[:2] for b in bonds], dtype=torch.long).reshape(-1, 2)
    hybridization = torch.tensor(
        [
            native_rdkit.ADD_FEAT_IDX_MAP["hybridization"][a.GetHybridization()]
            for a in mol.GetAtoms()
        ],
        dtype=torch.long,
    )
    return GeometricMol(
        coords=torch.zeros((mol.GetNumAtoms(), 3)),
        atomics=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
        charges=torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]),
        hybridization=hybridization,
        bond_indices=indices,
        bond_types=torch.tensor([b[2] for b in bonds]),
        str_id=conditions["product_mapped_smiles"],
    )


def _build_batch(task, conditions, loaded, native, args, torch):
    from flowr.data.datamodules import GeometricInterpolantDM

    model, hparams, vocab, charge_vocab, hybridization_vocab, aromatic_vocab, *_ = loaded
    transform, interpolant = native.util.load_util(
        args, hparams, vocab, charge_vocab, hybridization_vocab, aromatic_vocab
    )
    if interpolant.rotation_alignment or interpolant.permutation_alignment:
        raise RuntimeError("Campaign prior must not align to reference ligand coordinates")
    if interpolant.sample_mol_sizes or getattr(interpolant, "virtual_atom_p", 0):
        raise RuntimeError("Campaign batching cannot change the requested slot count")
    system = native.util.load_data_from_pdb(
        args,
        remove_hs=hparams["remove_hs"],
        remove_aromaticity=hparams["remove_aromaticity"],
        ligand_idx=0,
        chain_id=None,
        canonicalize_conformer=False,
    )
    if system is None:
        raise ValueError("Native source-pocket preprocessing failed")
    system.ligand = _make_product_ligand(conditions, hparams, torch)
    # Original source coordinates are used only for deterministic pocket extraction.
    # The posterior ligand geometry is a zero-valued placeholder and cannot enter
    # a free-coordinate prior or SC channel as the original molecular pose.
    system.interactions = None
    dataset = native.get_dataset(system, transform, vocab, interpolant, args, hparams)
    if len(dataset) != task["sample_count"]:
        raise RuntimeError("Native dataset changed the requested particle count")
    systems = [dataset[index] for index in range(task["sample_count"])]
    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        task["sample_count"],
        task["sample_count"],
        test_interpolant=interpolant,
        num_workers=0,
        pad_to_bucket=False,
    )
    prior_complex, data, _, _ = dm._collate(systems, dataset="test")
    prior = model.builder.extract_ligand_from_complex(prior_complex)
    prior.update(
        interactions=prior_complex["interactions"],
        fragment_mask=prior_complex["fragment_mask"],
        fragment_mode=prior_complex["fragment_mode"],
    )
    reference = model.builder.extract_ligand_from_complex(data)
    pocket = model.builder.extract_pocket_from_complex(data)
    pocket.update(interactions=data["interactions"], complex=data["complex"])
    device = next(model.parameters()).device
    prior, reference, pocket = [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in values.items()
        }
        for values in (prior, reference, pocket)
    ]
    n = len(conditions["atom_maps"])
    if prior["coords"].shape[:2] != (task["sample_count"], n):
        raise RuntimeError("Native batch shape differs from requested independent particles")
    if not torch.all(reference["mask"].sum(dim=1) == n):
        raise RuntimeError("Native reference featurization changed molecule size")
    # Every coordinate target is derived from ORIGINAL source I coordinates.
    # No E/product pose coordinates are read, even if a product_pose_sdf was supplied.
    if conditions.get("interaction_coordinates_world") is not None:
        for b in range(task["sample_count"]):
            com = pocket["complex"][b].com.reshape(-1).to(device)
            for slot, xyz in zip(
                conditions["interaction_input_slots"],
                conditions["interaction_coordinates_world"],
                strict=True,
            ):
                world = reference["coords"].new_tensor(xyz)
                reference["coords"][b, slot] = (world - com) / float(model.coord_scale)
    return prior, reference, pocket


def particle_ids(task):
    ids = task.get("particle_ids")
    if ids is None:
        ids = [f"{task['mode']}_seed{task['seed']}_p{i:04d}" for i in range(task["sample_count"])]
    if len(ids) != task["sample_count"] or len(set(ids)) != len(ids):
        raise ValueError("Task particle IDs must be unique and match sample_count")
    return ids


def export_particles(task, conditions, output, decoded, channels, vocab, torch):
    """Retain decoder failures and tensor-slot provenance, without selection gates."""
    from rdkit import Chem

    folder = Path(task["output_dir"]) / "particles"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    ids = particle_ids(task)
    if len(decoded) != task["sample_count"]:
        raise RuntimeError("Decoder changed the number of independent particle slots")
    fixed_slots = set(channels.get("fixed_atom_input_slots", []))
    namespace = max(conditions["atom_maps"]) + 1000
    noatom = getattr(vocab, "token_idx_map", {}).get("NOATOM")
    reference_graph = Chem.MolFromSmiles(conditions.get("product_mapped_smiles", ""))
    if noatom is None:
        noatom = getattr(vocab, "token_idx_map", {}).get("<NOATOM>")
    for i, mol in enumerate(decoded):
        active = output["mask"][i].bool().nonzero(as_tuple=False).flatten().tolist()
        if noatom is not None:
            selected = output["atomics"][i].argmax(dim=-1)
            active = [slot for slot in active if int(selected[slot]) != noatom]
        record = {
            "particle_index": i,
            "pose_id": ids[i],
            "status": "decode_failed",
            "mode": task["mode"],
            "raw_tensor_particle_index": i,
            "input_slot_count": len(conditions["atom_maps"]),
            "raw_active_slot_count": len(active),
            "raw_active_input_slots": active,
            "raw_final_tensors": "raw_final.pt",
            "decoded_atom_count": mol.GetNumAtoms() if mol is not None else None,
            "tensor_slot_to_reference_atom_map": {
                slot: conditions["atom_maps"][slot]
                for slot in active
                if slot < len(conditions["atom_maps"])
            },
        }
        if mol is None:
            record["reason"] = "Native argmax tensor decoding returned no molecule"
            records.append(record)
            continue
        if mol.GetNumAtoms() != len(active):
            record["reason"] = (
                "Decoded size differs from the known raw active slots; atom identity not guessed"
            )
            records.append(record)
            continue
        for atom, slot in zip(mol.GetAtoms(), active, strict=True):
            atom.SetAtomMapNum(
                conditions["atom_maps"][slot] if slot in fixed_slots else namespace + slot
            )
        try:
            Chem.SanitizeMol(mol)
            sdf = folder / f"particle_{i:04d}.sdf"
            with Chem.SDWriter(str(sdf)) as writer:
                writer.SetForceV3000(True)
                writer.write(mol)
            reloaded = Chem.SDMolSupplier(str(sdf), removeHs=True)[0]
            if reloaded is None or [a.GetAtomMapNum() for a in reloaded.GetAtoms()] != [
                a.GetAtomMapNum() for a in mol.GetAtoms()
            ]:
                raise ValueError("Generated SDF atom identities did not survive V3000 roundtrip")
            mapped = Chem.MolToSmiles(reloaded, isomericSmiles=True)
            plain = Chem.Mol(reloaded)
            for atom in plain.GetAtoms():
                atom.SetAtomMapNum(0)
            record.update(
                status="decoded",
                sdf_path=str(sdf.resolve()),
                mapped_smiles=mapped,
                smiles=Chem.MolToSmiles(plain, isomericSmiles=True),
                component_count=len(Chem.GetMolFrags(reloaded)),
                coordinates_modified_after_sampling=False,
                tensor_slot_to_atom_map={
                    slot: a.GetAtomMapNum()
                    for slot, a in zip(active, reloaded.GetAtoms(), strict=True)
                },
            )
            lineage = []
            for slot, atom in zip(active, reloaded.GetAtoms(), strict=True):
                if slot >= len(conditions["atom_maps"]):
                    continue
                target = (
                    reference_graph.GetAtomWithIdx(slot) if reference_graph is not None else None
                )
                local_match = target is not None
                if target is not None:
                    expected_neighbors = {
                        bond.GetOtherAtomIdx(slot): str(bond.GetBondType())
                        for bond in target.GetBonds()
                    }
                    actual_neighbors = {
                        active[bond.GetOtherAtomIdx(atom.GetIdx())]: str(bond.GetBondType())
                        for bond in atom.GetBonds()
                    }
                    local_match = expected_neighbors == actual_neighbors
                lineage.append(
                    {
                        "input_slot": slot,
                        "reference_atom_map": conditions["atom_maps"][slot],
                        "generated_atom_map": atom.GetAtomMapNum(),
                        "element_matches": atom.GetAtomicNum()
                        == (
                            target.GetAtomicNum()
                            if target is not None
                            else conditions["atomic_numbers"][slot]
                        ),
                        "charge_matches": atom.GetFormalCharge()
                        == (
                            target.GetFormalCharge()
                            if target is not None
                            else conditions["formal_charges"][slot]
                        ),
                        "local_graph_matches": bool(local_match),
                        "hard_atom_condition": slot in fixed_slots,
                    }
                )
            record["slot_lineage"] = lineage
        except (ValueError, RuntimeError) as error:
            record["reason"] = f"Native output could not be serialized/read: {error}"
        records.append(record)
    return {
        "schema_version": "agenticgen.flowr-particles.v1",
        "sample_count": task["sample_count"],
        "particles": records,
    }


def run(task_path: Path, *, model_cache=None) -> None:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    output_dir = Path(task["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "runtime.json").exists():
        raise FileExistsError("Use a fresh campaign execution directory")
    receipt = {
        "schema_version": "agenticgen.flowr-campaign-runtime.v1",
        "status": "starting",
        "mode": task.get("mode"),
        "seed": task.get("seed"),
        "sample_count": task.get("sample_count"),
        "python": sys.executable,
        "no_resampling": True,
        "no_deduplication": True,
        "no_geometry_optimization": True,
        "generation_completed": False,
        "sampling_calls": 0,
    }

    def save():
        write_json(output_dir / "runtime.json", receipt)

    save()
    try:
        if task.get("schema_version") != "agenticgen.flowr-campaign-task.v1":
            raise ValueError("Unsupported campaign task schema")
        if task.get("mode") not in {
            "graph",
            "fragment_graph",
            "fragment_graph_pose",
            "gradient",
            "unguided",
        }:
            raise ValueError("Unknown campaign mode")
        if not isinstance(task.get("sample_count"), int) or not 1 <= task["sample_count"] <= 4:
            raise ValueError("A physical campaign batch contains one to four independent particles")
        if task.get("integration_steps") != 100:
            raise ValueError("Campaign requires exactly 100 integration steps")
        particle_ids(task)
        sys.path.insert(0, str(Path(task["flowr_root"]).resolve()))
        native = importlib.import_module("flowr.gen.generate_from_pdb")
        if not Path(native.__file__).resolve().is_relative_to(Path(task["flowr_root"]).resolve()):
            raise RuntimeError("Imported FLOWR is outside the configured official checkout")
        import numpy as np
        import torch

        # Package import is deferred until after native extensions load (DTK safety).
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from agenticgen.region_conditions import RegionConditioner, mode_channel_masks

        conditions = json.loads(Path(task["conditions_file"]).read_text(encoding="utf-8"))
        if conditions.get("schema_version") != "agenticgen.region-conditions.v1":
            raise ValueError("Unsupported condition schema")
        channels = mode_channel_masks(conditions, task["mode"])
        write_json(output_dir / "channel_masks.json", channels)
        if channels["status"] != "ready":
            receipt.update(status="not_applicable", reason=channels["reason"])
            save()
            return
        guidance = task.get("guidance") or {}
        if task["mode"] == "gradient" and (
            not conditions["fixed_atom_input_slots"]
            or (guidance.get("coordinates") and not conditions["interaction_input_slots"])
        ):
            receipt.update(
                status="not_applicable", reason="Requested gradient condition has no target region"
            )
            save()
            return
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task["seed"])
        argv = [
            "--pdb_file",
            str(Path(task["pocket_pdb"]).resolve()),
            "--ligand_file",
            str(Path(task["source_pose_sdf"]).resolve()),
            "--ckpt_path",
            str(Path(task["checkpoint"]).resolve()),
            "--save_dir",
            str(output_dir),
            "--arch",
            "pocket",
            "--pocket_type",
            "holo",
            "--gpus",
            "1",
            "--num_workers",
            "0",
            "--integration_steps",
            "100",
            "--sample_n_molecules_per_target",
            str(task["sample_count"]),
            "--seed",
            str(task["seed"]),
            "--cut_pocket",
            "--pocket_cutoff",
            "7.0",
            "--no_ligand_valence_repair",
        ]
        previous_argv = sys.argv
        try:
            sys.argv = [str(native.__file__), *argv]
            args = native.get_args()
        finally:
            sys.argv = previous_argv
        loaded, cache_hit = _load_model_cached(native, args, task, model_cache)
        receipt["model_cache_hit"] = cache_hit
        model = loaded[0]
        model.to(native.resolve_device(args)).eval()
        # Reset after model construction so all arm priors use the same RNG stream.
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task["seed"])
        prior, reference, pocket = _build_batch(task, conditions, loaded, native, args, torch)
        receipt.update(
            torch_version=torch.__version__,
            torch_hip_version=getattr(torch.version, "hip", None),
            native_entrypoint=str(native.__file__),
            native_arguments=argv,
            input_slot_count=len(conditions["atom_maps"]),
            batch_particle_ids=particle_ids(task),
            native_categorical_representation="Kekule"
            if loaded[1].get("remove_aromaticity")
            else "aromatic",
            base_prior_sha256=tensor_fingerprint(prior, torch),
        )
        torch.save(_cpu_tensors(prior, torch), output_dir / "raw_prior.pt")
        torch.save(_cpu_tensors(reference, torch), output_dir / "raw_reference.pt")
        receipt["native_reference_sha256"] = tensor_fingerprint(reference, torch)
        receipt["raw_reference_path"] = "raw_reference.pt"
        rng = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            for i, state in enumerate(torch.cuda.get_rng_state_all()):
                rng[f"gpu_{i}"] = state
        torch.save(rng, output_dir / "rng_before_generation.pt")
        receipt["rng_before_generation_sha256"] = tensor_fingerprint(rng, torch)
        conditioner = RegionConditioner(channels, reference)
        hard = task["mode"] not in {"gradient", "unguided"}
        if hard:
            prior = conditioner.apply(prior)
        receipt["conditioned_prior_sha256"] = tensor_fingerprint(prior, torch)
        model._agenticgen_pocket_data = pocket
        model._agenticgen_conditions = conditions
        model._agenticgen_condition_reference = {
            key: value[:1] if torch.is_tensor(value) else value for key, value in reference.items()
        }
        controller = None
        if task["mode"] == "gradient":
            from agenticgen.gradient_guidance import build_controller

            controller = build_controller(task, conditions, model)
            write_json(output_dir / "reward_design_ir.json", controller.reward_design_ir())
        times = [torch.zeros(task["sample_count"], device=prior["coords"].device) for _ in range(3)]
        probe_state = None
        with contextlib.ExitStack() as stack:
            if task.get("guidance_probe", False):
                probe_state = stack.enter_context(
                    capture_native_probe_state(model, output_dir / "guidance_probe_state.pt", torch)
                )
            if hard:
                stack.enter_context(native_condition_channels(model, conditioner))
            if controller is not None:
                stack.enter_context(controller.install(model))
            receipt.update(status="sampling", sampling_calls=1)
            save()
            raw = model._generate(
                prior,
                pocket,
                steps=100,
                times=times,
                strategy="linear",
                solver="euler",
                corr_iters=0,
                final_inpaint=hard,
                apply_guidance=False,
                save_traj=False,
            )
        # This snapshot MUST precede every call to _generate_mols or RDKit.
        torch.save(_cpu_tensors(raw, torch), output_dir / "raw_final.pt")
        receipt.update(
            generation_completed=True,
            final_tensor_sha256=tensor_fingerprint(raw, torch),
            raw_final_path="raw_final.pt",
            raw_particle_count=int(raw["coords"].shape[0]),
        )
        save()
        if int(raw["coords"].shape[0]) != task["sample_count"]:
            raise RuntimeError("Native generator changed the physical particle count")
        write_json(
            output_dir / "raw_particles.json",
            {
                "schema_version": "agenticgen.raw-particles.v1",
                "raw_final_path": "raw_final.pt",
                "particles": [
                    {"particle_index": i, "pose_id": p, "tensor_index": i}
                    for i, p in enumerate(particle_ids(task))
                ],
            },
        )
        if task.get("guidance_probe", False):
            from agenticgen.gradient_guidance import build_controller, probe_model_reward

            try:
                if not probe_state:
                    raise RuntimeError(
                        "Native sampling never exposed the registered t=.4 probe state"
                    )
                probe_controller = controller or build_controller(
                    {
                        **task,
                        "mode": "gradient",
                        "guidance": task.get("guidance")
                        or {
                            "family": "ce",
                            "coordinates": False,
                            "weight": 1.0,
                            "window": [0.1, 0.9],
                        },
                    },
                    conditions,
                    model,
                )
                report = probe_model_reward(
                    probe_controller,
                    model,
                    probe_state["batch"],
                    probe_state["pocket_batch"],
                    probe_state["times"],
                    cond_batch=probe_state["cond_batch"],
                    epsilon=float(task.get("probe_epsilon", 0.001)),
                    **probe_state["forward_kwargs"],
                )
                receipt["probe_status"] = "completed"
                write_json(
                    output_dir / "guidance_probe.json",
                    {
                        "schema_version": "agenticgen.guidance-probe.v1",
                        "status": "completed",
                        "report": report,
                    },
                )
            except Exception as error:
                receipt.update(
                    probe_status="failed", probe_error=f"{type(error).__name__}: {error}"
                )
                write_json(
                    output_dir / "guidance_probe.json",
                    {
                        "schema_version": "agenticgen.guidance-probe.v1",
                        "status": "failed",
                        "error": receipt["probe_error"],
                    },
                )
        if controller is not None:
            write_json(output_dir / "guidance_trace.json", controller.trace)
            receipt["guidance"] = controller.provenance()
        decoded = []
        decode_errors = {}
        for i in range(task["sample_count"]):
            one = {
                key: value[i : i + 1] if torch.is_tensor(value) else value
                for key, value in raw.items()
            }
            try:
                values = model._generate_mols(one, valence_repair=False)
                decoded.append(values[0] if len(values) == 1 else None)
            except Exception as error:
                decoded.append(None)
                decode_errors[i] = f"{type(error).__name__}: {error}"
        particles = export_particles(task, conditions, raw, decoded, channels, loaded[2], torch)
        for record in particles["particles"]:
            if record["particle_index"] in decode_errors:
                record["reason"] = decode_errors[record["particle_index"]]
        write_json(output_dir / "poses.json", particles)
        receipt.update(
            status="completed",
            decoded_count=sum(p["status"] == "decoded" for p in particles["particles"]),
            decode_failed_count=sum(p["status"] == "decode_failed" for p in particles["particles"]),
        )
        if task.get("score_affinity", True):
            try:
                from agenticgen.campaign_affinity import score_pose_files

                affinity = score_pose_files(task, particles["particles"], loaded=loaded)
                write_json(
                    output_dir / "affinity.json",
                    {
                        "schema_version": "agenticgen.pose-affinity.v1",
                        "status": "completed",
                        "poses": affinity,
                    },
                )
                receipt["affinity_status"] = "completed"
            except Exception as error:
                receipt.update(
                    affinity_status="failed", affinity_error=f"{type(error).__name__}: {error}"
                )
                write_json(
                    output_dir / "affinity.json",
                    {
                        "schema_version": "agenticgen.pose-affinity.v1",
                        "status": "failed",
                        "error": receipt["affinity_error"],
                        "poses": [],
                    },
                )
        save()
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        traceback.print_exc()
        raise


def serve(input_stream=None, output_stream=None):
    """JSON-lines control channel. Native logs go to stderr, never into replies."""
    input_stream, output_stream = input_stream or sys.stdin, output_stream or sys.stdout
    for line in input_stream:
        request = {}
        try:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                output_stream.write(
                    json.dumps({"status": "stopped", "request_id": request.get("request_id")})
                    + "\n"
                )
                output_stream.flush()
                return
            if request.get("op") == "ping":
                response = {"status": "ready", "cached_models": len(MODEL_CACHE)}
            else:
                task_path = Path(request["task_path"])
                with contextlib.redirect_stdout(sys.stderr):
                    run(task_path, model_cache=MODEL_CACHE)
                task = json.loads(task_path.read_text(encoding="utf-8"))
                runtime_path = Path(task["output_dir"]) / "runtime.json"
                receipt = json.loads(runtime_path.read_text(encoding="utf-8"))
                response = {
                    "status": "completed"
                    if receipt["status"] in {"completed", "not_applicable"}
                    else "failed",
                    "runtime_status": receipt["status"],
                    "runtime_path": str(runtime_path),
                    "generation_completed": receipt.get("generation_completed", False),
                    "affinity_status": receipt.get("affinity_status"),
                    "probe_status": receipt.get("probe_status"),
                }
        except Exception as error:
            response = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        response["request_id"] = request.get("request_id")
        output_stream.write(json.dumps(response, allow_nan=False) + "\n")
        output_stream.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    elif len(sys.argv) == 2:
        run(Path(sys.argv[1]))
    else:
        raise SystemExit("Usage: flowr_campaign.py task.json | --serve")
