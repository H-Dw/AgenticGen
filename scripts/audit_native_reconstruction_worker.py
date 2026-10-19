"""Cross-version native reconstruction check inside the existing chemistry worker."""

import copy
import importlib.util
import json
from pathlib import Path

from rdkit import Chem, rdBase

spec = importlib.util.spec_from_file_location("native_utils", "/syncraft/src/utils.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = []
for item in json.loads(Path("/job/requests.json").read_text()):
    result = module.reconstruct_from_sequence(item["smiles"], copy.deepcopy(item["edits"]))
    raw = Chem.MolFromSmiles(result["smiles_with_maps"], sanitize=False)
    results.append(
        {
            "case_id": item["case_id"],
            "parseable": Chem.MolFromSmiles(result["smiles_with_maps"]) is not None,
            "problem_types": [p.GetType() for p in Chem.DetectChemistryProblems(raw)]
            if raw is not None
            else [],
        }
    )
Path("/job/cross_version.json").write_text(
    json.dumps({"rdkit": rdBase.rdkitVersion, "cases": results}, indent=2)
)
print(rdBase.rdkitVersion, sum(r["parseable"] for r in results), len(results))
