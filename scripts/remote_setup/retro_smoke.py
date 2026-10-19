"""Check full-template SimpRetro and route replay using two engineering controls.

The nonzero control is frozen from the prior official-stock example. These checks
are infrastructure acceptance, not CK2 design results or experimental validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--materials", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-limit", default=120, type=int)
    parser.add_argument("--worker-python", default=Path(sys.executable), type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    from agenticgen.adapters.retro import SimpRetroCLI
    from agenticgen.io import write_json

    args.output.mkdir(parents=True, exist_ok=False)
    source = args.repo / "external/SimpRetro"
    evaluator = SimpRetroCLI(
        root=source,
        python=args.worker_python,
        inventory=args.materials / "emolecules.txt",
        inventory_index=args.materials / "stock.sqlite",
        fingerprint_cache=args.materials / "fingerprint_base.pkl",
        templates=source / "SimpRetro_templates.json",
        time_limit_seconds=args.time_limit,
        process_grace_seconds=300,
        cpu_threads=1,
        expected_revision="5abbdd12d23bec3b0c524a3f5e59d160beac00f3",
    )
    targets = {
        "stock_zero_step": "CCO",
        "nonzero_route": "CC(C)(C)c1cc(C(=O)Nc2cc(Cl)cc(C(F)(F)F)c2)cc(C(C)(C)C)c1",
    }
    write_json(args.output / "protocol.json", evaluator.protocol)
    records = {}
    for name, smiles in targets.items():
        print(f"Starting full-template engineering control: {name}", flush=True)
        result = evaluator.evaluate(smiles, args.output / name)
        records[name] = {"smiles": smiles, "retrosynthesis": result.model_dump(mode="json")}
        write_json(args.output / "results.json", records)
        print(
            json.dumps(
                {
                    "control": name,
                    "status": result.status,
                    "route_verified": result.route_verified,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            ),
            flush=True,
        )
    passed = all(
        row["retrosynthesis"]["status"] == "route_found" and row["retrosynthesis"]["route_verified"]
        for row in records.values()
    )
    summary = {
        "ok": passed,
        "engineering_controls_only": True,
        "agent_calls": 0,
        "inventory_identity": "SimpRetro official eMolecules, not Enamine",
        "worker_python": str(args.worker_python),
        "runtime_note": "Worker package versions are recorded in the environment audit",
        "controls": records,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"ok": passed, "output": str(args.output)}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
