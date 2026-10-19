"""Adapt native SynCraft reconstruction without a second graph-edit interpreter."""

from __future__ import annotations

import copy
from typing import Any, Protocol

from rdkit import Chem

from agenticgen.chemistry import canonical_smiles, graph_hash, molecule, similarity
from agenticgen.contracts import EditConstraints, GraphEditResult


class ReconstructionBackend(Protocol):
    def reconstruct(self, smiles: str, commands: list[list[Any]]) -> dict[str, Any]: ...


def _atoms(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    return {atom.GetAtomMapNum(): atom for atom in mol.GetAtoms()}


def _signature(atom: Chem.Atom) -> tuple:
    """Describe changes for reporting; this does not veto the upstream product."""
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        str(atom.GetChiralTag()),
        tuple(
            sorted(
                (bond.GetOtherAtom(atom).GetAtomMapNum(), str(bond.GetBondType()))
                for bond in atom.GetBonds()
            )
        ),
    )


class SynCraftEditor:
    """Use the upstream product, including skipped operations and largest-fragment pruning.

    EditConstraints remain prompt guidance. No independent edit program, capacity,
    protected-neighborhood, similarity or stereo-preservation audit runs here.
    Loading the returned SMILES is required by downstream evaluators; an unreadable
    upstream product is recorded without substituting or repairing a molecule.
    """

    policy = "syncraft_native"

    def __init__(self, backend: ReconstructionBackend) -> None:
        self.backend = backend

    def apply(
        self, mapped_smiles: str, commands: list[list[Any]], constraints: EditConstraints
    ) -> GraphEditResult:
        del constraints  # Historical interface retained; these are no longer execution gates.
        metadata: dict[str, Any] = {"reconstruction_policy": self.policy}
        stage = "upstream_reconstruction"
        try:
            upstream = self.backend.reconstruct(mapped_smiles, copy.deepcopy(commands))
            metadata.update(
                upstream_smiles=upstream.get("smiles"),
                upstream_mapped_smiles=upstream.get("smiles_with_maps"),
                was_fragmented=bool(upstream.get("was_fragmented", False)),
            )
            stage = "product_loading"
            raw = metadata["upstream_mapped_smiles"]
            if not isinstance(raw, str):
                raise ValueError("SynCraft returned no mapped product SMILES")
            actual = molecule(raw)
            source, product = canonical_smiles(mapped_smiles), canonical_smiles(actual)
            before, after = _atoms(molecule(mapped_smiles)), _atoms(actual)
            declared_deletions = {c[1] for c in commands if len(c) > 1 and c[0] == "DEL_ATOM"}
            return GraphEditResult(
                status="unchanged" if source == product else "accepted",
                source_smiles=source,
                product_smiles=product,
                product_mapped_smiles=canonical_smiles(actual, maps=True),
                product_graph_sha256=graph_hash(product),
                similarity=similarity(source, product),
                operations=len(commands) - sum(c == ["STOP"] for c in commands),
                added_atom_maps=sorted(after.keys() - before.keys()),
                deleted_atom_maps=sorted(before.keys() - after.keys()),
                changed_atom_maps=sorted(
                    i
                    for i in before.keys() & after.keys()
                    if _signature(before[i]) != _signature(after[i])
                ),
                discarded_atom_maps=sorted(before.keys() - after.keys() - declared_deletions)
                if metadata["was_fragmented"]
                else [],
                **metadata,
            )
        except (ValueError, RuntimeError, KeyError, TypeError, IndexError) as error:
            return GraphEditResult(
                status="rejected",
                source_smiles=mapped_smiles,
                operations=max(0, len(commands) - sum(c == ["STOP"] for c in commands)),
                error_stage=stage,
                reason=f"SynCraft {stage}: {error}",
                **metadata,
            )
