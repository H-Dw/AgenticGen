"""Portable worker protocol and durable-particle tests without a FLOWR install."""

from __future__ import annotations

import importlib
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from agenticgen.workers import flowr_campaign as worker  # noqa: E402


def test_serve_protocol_reuses_cache_and_keeps_logs_out_of_json(tmp_path, monkeypatch):
    task_path = tmp_path / "task.json"
    output = tmp_path / "result"
    task_path.write_text(json.dumps({"output_dir": str(output)}))
    cache_ids = []

    def run(path, *, model_cache):
        assert path == task_path
        cache_ids.append(id(model_cache))
        print("native progress is not protocol")
        worker.write_json(
            output / "runtime.json", {"status": "completed", "generation_completed": True}
        )

    monkeypatch.setattr(worker, "run", run)
    requests = [
        {"op": "ping", "request_id": "p"},
        {"task_path": str(task_path), "request_id": "a"},
        {"task_path": str(task_path), "request_id": "b"},
        {"task_path": str(tmp_path / "missing.json"), "request_id": "bad"},
        {"op": "shutdown", "request_id": "s"},
    ]
    stream = io.StringIO()
    worker.serve(io.StringIO("\n".join(json.dumps(row) for row in requests)), stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["status"] for row in rows] == [
        "ready",
        "completed",
        "completed",
        "failed",
        "stopped",
    ]
    assert rows[1]["request_id"] == "a" and rows[1]["generation_completed"]
    assert len(set(cache_ids)) == 1
    assert "native progress" not in stream.getvalue()


def test_four_particle_worker_saves_raw_before_decode_and_zero_weight_matches_unguided(
    tmp_path, monkeypatch
):
    root = tmp_path / "native"
    root.mkdir()
    checkpoint = root / "model.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    conditions = {
        "schema_version": "agenticgen.region-conditions.v1",
        "status": "ready",
        "product_mapped_smiles": "[CH3:7][OH:501]",
        "atom_maps": [7, 501],
        "atomic_numbers": [6, 8],
        "formal_charges": [0, 0],
        "bond_types": [[0, 1], [1, 0]],
        "fixed_atom_input_slots": [0],
        "fixed_bond_pairs": [[0, 0]],
        "interaction_input_slots": [0],
        "interaction_coordinates_world": [[1.0, 2.0, 3.0]],
        "optimized_input_slots": [],
    }
    conditions_path = tmp_path / "conditions.json"
    conditions_path.write_text(json.dumps(conditions))
    calls = []

    class Vocabulary:
        token_idx_map = {"C": 0, "O": 1}

        def indices_from_tokens(self, values):
            return [self.token_idx_map[value] for value in values]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.vocab = Vocabulary()
            self.vocab_charges = SimpleNamespace(
                indices_from_tokens=lambda values: [0 for _ in values]
            )
            self.hparams = {"remove_aromaticity": False}
            self.builder = SimpleNamespace(save_dir="", ligand_valence_repair=False)
            self.graph_inpainting = self.inpainting_mode = False
            self.inpainting_mode_inf = None

        def _generate(self, prior, pocket, **kwargs):
            assert kwargs["steps"] == 100 and not kwargs["final_inpaint"]
            assert prior["coords"].shape == (4, 2, 3)
            calls.append("sample")
            return {**prior, "coords": prior["coords"] + torch.rand_like(prior["coords"])}

        def _generate_mols(self, one, **kwargs):
            destination = Path(self.hparams["save_dir"])
            assert (destination / "raw_final.pt").is_file()
            assert (destination / "raw_particles.json").is_file()
            assert kwargs["valence_repair"] is False
            calls.append("decode")
            if calls.count("decode") % 4 == 2:
                raise RuntimeError("one physically sampled particle is unreadable")
            mol = Chem.MolFromSmiles("CO")
            conf = Chem.Conformer(2)
            conf.Set3D(True)
            for i, xyz in enumerate(one["coords"][0].tolist()):
                conf.SetAtomPosition(i, xyz)
            mol.AddConformer(conf)
            return [mol]

    loads = []

    def load_model(args):
        loads.append(args)
        random.random()
        np.random.random()
        torch.rand(123)
        model = Model()
        return model, {"remove_aromaticity": False}, model.vocab, model.vocab_charges, *([None] * 4)

    def arguments():
        return SimpleNamespace(save_dir=sys.argv[sys.argv.index("--save_dir") + 1])

    native = SimpleNamespace(
        __file__=str(root / "entry.py"),
        load_model=load_model,
        get_args=arguments,
        resolve_device=lambda args: "cpu",
    )
    original_import = importlib.import_module
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: native if name == "flowr.gen.generate_from_pdb" else original_import(name),
    )

    def batch(task, conditions, loaded, native, args, torch):
        # Every RNG stream matters, including native transforms which use NumPy.
        coords = torch.rand(4, 2, 3) + random.random() + float(np.random.random())
        atomics = torch.eye(2).unsqueeze(0).repeat(4, 1, 1)
        bonds = (
            torch.nn.functional.one_hot(torch.tensor([[0, 1], [1, 0]]), 5)
            .float()
            .unsqueeze(0)
            .repeat(4, 1, 1, 1)
        )
        reference = {
            "coords": torch.zeros_like(coords),
            "atomics": atomics,
            "charges": torch.ones(4, 2, 1),
            "bonds": bonds,
            "mask": torch.ones(4, 2, dtype=torch.long),
        }
        return {**reference, "coords": coords}, reference, {}

    monkeypatch.setattr(worker, "_build_batch", batch)
    cache = {}
    results = []
    for mode in ("unguided", "gradient"):
        output_dir = tmp_path / mode
        task = {
            "schema_version": "agenticgen.flowr-campaign-task.v1",
            "mode": mode,
            "flowr_root": str(root),
            "checkpoint": str(checkpoint),
            "pocket_pdb": "source.pdb",
            "source_pose_sdf": "source.sdf",
            "conditions_file": str(conditions_path),
            "output_dir": str(output_dir),
            "seed": 31,
            "sample_count": 4,
            "integration_steps": 100,
            "score_affinity": False,
            "guidance": {"weight": 0.0, "coordinates": False, "family": "ce", "window": [0.1, 0.9]},
        }
        task_path = tmp_path / f"{mode}.json"
        task_path.write_text(json.dumps(task))
        worker.run(task_path, model_cache=cache)
        runtime = json.loads((output_dir / "runtime.json").read_text())
        poses = json.loads((output_dir / "poses.json").read_text())
        assert runtime["status"] == "completed" and runtime["sampling_calls"] == 1
        assert runtime["decoded_count"] == 3 and runtime["decode_failed_count"] == 1
        assert len(poses["particles"]) == 4
        assert [row["particle_index"] for row in poses["particles"]] == [0, 1, 2, 3]
        assert (output_dir / "raw_reference.pt").is_file()
        assert runtime["model_cache_hit"] == (mode == "gradient")
        results.append(torch.load(output_dir / "raw_final.pt", weights_only=True))
    assert len(loads) == 1 and calls.count("sample") == 2 and calls.count("decode") == 8
    for key in results[0]:
        assert torch.equal(results[0][key], results[1][key]), key
