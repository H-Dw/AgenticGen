import copy

import pytest

from agenticgen.adapters.loading import load_file
from agenticgen.chemistry import canonical_smiles, graph_hash, mapped_smiles
from agenticgen.contracts import EditConstraints
from agenticgen.editing import SynCraftEditor


@pytest.mark.parametrize(
    "smiles,commands",
    [
        ("CCO", [["DEL_BOND", 1, 3], ["STOP"]]),
        ("CCO", [["ADD_BOND", 1, 2, "DOUBLE"], ["STOP"]]),
        ("CCO", [["MUTATE_ATOM", 99, "N"], ["STOP"]]),
        ("CCO", [["DEL_ATOM", 3], ["DEL_BOND", 2, 3], ["STOP"]]),
        ("CCO", [["DEL_BOND", 1, 2], ["STOP"]]),
        ("F[C@H](Cl)Br", [["STOP"]]),
        ("[13CH3]CO", [["STOP"]]),
        ("F/C=C/F", [["STOP"]]),
    ],
)
def test_adapter_matches_original_syncraft_without_independent_veto(
    editor, syncraft_root, smiles, commands
):
    upstream = load_file(syncraft_root / "src/utils.py", "native_parity")
    source = mapped_smiles(smiles)
    expected = upstream.reconstruct_from_sequence(source, copy.deepcopy(commands))
    result = editor.apply(source, commands, EditConstraints())
    assert result.status in {"accepted", "unchanged"}
    assert result.product_smiles == canonical_smiles(expected["smiles_with_maps"])
    assert result.upstream_mapped_smiles == expected["smiles_with_maps"]
    assert result.was_fragmented == expected["was_fragmented"]
    assert result.reconstruction_policy == "syncraft_native"


def test_largest_fragment_is_kept_and_implicit_atom_loss_is_reported(editor):
    result = editor.apply(mapped_smiles("CCO"), [["DEL_BOND", 1, 2], ["STOP"]], EditConstraints())
    assert result.status == "accepted"
    assert result.product_smiles == "CO"
    assert result.was_fragmented is True
    assert result.deleted_atom_maps == result.discarded_atom_maps == [1]
    assert result.upstream_smiles == "CO"


def test_edit_guidance_is_not_enforced_as_a_second_execution_contract(editor):
    result = editor.apply(
        mapped_smiles("CCO"),
        [["DEL_ATOM", 3], ["STOP"]],
        EditConstraints(max_deleted_atoms=0, min_similarity=1, protected_atom_maps=(2, 3)),
    )
    assert result.status == "accepted"
    assert result.product_smiles == "CC"
    assert result.deleted_atom_maps == [3]
    assert result.was_fragmented is False


def test_unreadable_native_product_retains_raw_output_and_does_not_repair(editor):
    result = editor.apply(
        mapped_smiles("CCO"),
        [["CHANGE_BOND", 1, 2, "TRIPLE"], ["STOP"]],
        EditConstraints(),
    )
    assert result.status == "rejected"
    assert result.error_stage == "product_loading"
    assert result.upstream_smiles and result.upstream_mapped_smiles
    assert result.product_smiles is None


def test_upstream_failure_is_a_recorded_error_not_a_replacement_product():
    class BrokenBackend:
        def reconstruct(self, smiles, commands):
            commands.append(["DEL_ATOM", 1])
            raise ValueError("original reconstruction failed")

    commands = [["STOP"]]
    result = SynCraftEditor(BrokenBackend()).apply(
        mapped_smiles("CCO"), commands, EditConstraints()
    )
    assert commands == [["STOP"]]
    assert result.status == "rejected" and result.error_stage == "upstream_reconstruction"
    assert "original reconstruction failed" in result.reason
    assert result.product_smiles is None


def test_noop_is_not_an_optimization(editor):
    result = editor.apply(mapped_smiles("C[NH3+]"), [["STOP"]], EditConstraints())
    assert result.status == "unchanged"
    assert "[NH3+]" in result.product_smiles


def test_source_identity_includes_stereochemistry_and_microstate():
    assert graph_hash("F[C@H](Cl)Br") != graph_hash("F[C@@H](Cl)Br")
    assert graph_hash("CN") != graph_hash("C[NH3+]")
