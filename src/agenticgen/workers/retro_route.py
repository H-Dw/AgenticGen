"""Export and verify only a graph freshly produced by this job's trusted search process."""

import json
import pickle
import sys
from pathlib import Path


def verify(graph_path, inventory_path, templates_path, target, inventory_index=None):
    from rdchiral.main import rdchiralRunText
    from rdkit import Chem
    from syntheseus.search.analysis.route_extraction import iter_routes_time_order

    def canonical(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid molecule in route/inventory")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def components(smiles):
        return tuple(sorted(canonical(s) for s in smiles.split(".")))

    with Path(graph_path).open("rb") as stream:
        graph = pickle.load(
            stream
        )  # Only generated job output; never accept user-supplied pickle inputs.
    if inventory_index:
        from stock_index import SQLiteStock

        inventory = SQLiteStock(inventory_index)
    else:
        inventory = {
            canonical(line.strip())
            for line in Path(inventory_path).read_text().splitlines()
            if line.strip()
        }
    root = canonical(graph.root_mol.smiles)
    if root != canonical(target):
        raise ValueError("Route root does not match requested target")
    nodes = next(iter_routes_time_order(graph, max_routes=1), None)
    if nodes is None:
        raise ValueError("Search reported a solution but none can be extracted")
    if len(nodes) == 1 and root in inventory:
        return {
            "route_verified": True,
            "target": root,
            "reactions": [],
            "leaves": [root],
            "verification_scope": "zero-step target is in configured inventory",
        }
    synthesis = graph.to_synthesis_graph(nodes=nodes)
    synthesis.assert_validity()
    leaves = sorted(canonical(m.smiles) for m in synthesis.get_starting_molecules())
    stock_ok = bool(leaves) and all(s in inventory for s in leaves)
    templates = json.loads(Path(templates_path).read_text())
    reactions = []
    for rxn in synthesis.nodes():
        product = canonical(rxn.product.smiles)
        reactants = tuple(sorted(canonical(m.smiles) for m in rxn.reactants))
        matched = None
        for index, template in enumerate(templates):
            try:
                outcomes = rdchiralRunText(template, product)
                if any(components(outcome) == reactants for outcome in outcomes):
                    matched = index
                    break
            except (ValueError, RuntimeError):
                continue
        reactions.append(
            {"product": product, "reactants": reactants, "matching_template_index": matched}
        )
    return {
        "route_verified": stock_ok
        and all(r["matching_template_index"] is not None for r in reactions),
        "target": root,
        "reactions": reactions,
        "step_count": len(reactions),
        "leaves": leaves,
        "inventory_leaves_verified": stock_ok,
        "verification_scope": "graph consistency, inventory membership, RDChiral template replay; no experimental conditions/yield guarantee",
    }


if __name__ == "__main__":
    graph, inventory, templates, target, output = sys.argv[1:6]
    Path(output).write_text(
        json.dumps(verify(graph, inventory, templates, target, *sys.argv[6:]), indent=2),
        encoding="utf-8",
    )
