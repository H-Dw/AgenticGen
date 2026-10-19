"""Build executable reference examples using SynCraft's actual iterative MCS extractor."""

from pathlib import Path

from agenticgen.adapters.syncraft import SynCraftBackend
from agenticgen.chemistry import canonical_smiles, mapped_smiles
from agenticgen.io import read_json, sha256, write_json


def build_corpus(root: Path, pairs: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Reference corpus already exists")
    backend = SynCraftBackend(root)
    records, rejected = [], []
    for index, row in enumerate(read_json(pairs)):
        try:
            if not isinstance(row["reasoning"], str) or not row["reasoning"].strip():
                raise ValueError(
                    "Provide a chemical explanation; no fabricated rationale is generated"
                )
            if not row.get("evidence_ids"):
                raise ValueError("Reference pairs require source/target evidence identifiers")
            source = mapped_smiles(row["source"])
            extraction = backend.editor.IterativeMCSExtractor(
                source, row["target"]
            ).generate_editing_data()
            if extraction is None:
                raise ValueError("Upstream MCS extraction failed")
            commands = extraction["editing_sequence"]
            product = backend.reconstruct(source, commands)
            if product["was_fragmented"] or canonical_smiles(product["smiles"]) != canonical_smiles(
                row["target"]
            ):
                raise ValueError("Extracted edit path does not reconstruct the exact target")
            records.append(
                {
                    "source": source,
                    "target": canonical_smiles(row["target"]),
                    "reasoning": row["reasoning"],
                    "edits": commands,
                    "evidence_ids": row["evidence_ids"],
                    "rationale_origin": row.get(
                        "rationale_origin", "provided_not_causally_verified"
                    ),
                }
            )
        except (ValueError, KeyError, RuntimeError) as error:
            rejected.append({"pair_index": index, "reason": str(error)})
    write_json(output, records)
    audit = {
        "source_pairs_sha256": sha256(pairs),
        "upstream": backend.provenance,
        "accepted": len(records),
        "rejected": rejected,
        "meaning": "Graph edit supervision; chemical synthesis labels require independent route/purchase evidence",
    }
    write_json(output.with_suffix(".audit.json"), audit)
    return audit
