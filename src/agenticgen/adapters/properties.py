"""Fast graph property evaluation. SA scores are explicitly auxiliary."""

from rdkit import rdBase
from rdkit.Contrib.SA_Score import sascorer

from agenticgen.chemistry import molecule
from agenticgen.contracts import PropertyResult


class RDKitSAEvaluator:
    def evaluate(self, smiles: str) -> PropertyResult:
        try:
            return PropertyResult(
                status="ok",
                value=float(sascorer.calculateScore(molecule(smiles))),
                method=f"rdkit-{rdBase.rdkitVersion}.contrib.SA_Score",
            )
        except (ValueError, RuntimeError) as error:
            return PropertyResult(status="error", reason=str(error))
