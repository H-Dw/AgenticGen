"""A single stateless decision policy, separate from transport and graph execution."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from agenticgen.adapters.syncraft import SynCraftBackend
from agenticgen.contracts import Observation, Proposal
from agenticgen.ports import LLMPort, PromptPort


def proposal_schema() -> dict[str, Any]:
    schema = Proposal.model_json_schema()
    integer = {"type": "integer", "minimum": 1}
    element = {"type": "string", "enum": ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]}
    bond = {"type": "string", "enum": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]}
    signatures = {
        "STOP": [],
        "DEL_ATOM": [integer],
        "MUTATE_ATOM": [integer, element],
        "ADD_ATOM": [{"type": "integer", "minimum": 500}, element],
        "DEL_BOND": [integer, integer],
        "ADD_BOND": [integer, integer, bond],
        "CHANGE_BOND": [integer, integer, bond],
        "SET_CHIRAL": [integer, {"enum": ["CW", "CCW"]}],
        "SET_BOND_STEREO": [integer, integer, {"enum": ["E", "Z"]}, integer, integer],
    }
    schema["properties"]["edits"]["items"] = {
        "oneOf": [
            {
                "type": "array",
                "prefixItems": [{"const": name}, *args],
                "minItems": len(args) + 1,
                "maxItems": len(args) + 1,
            }
            for name, args in signatures.items()
        ]
    }
    return schema


class SynCraftPromptBuilder:
    def __init__(self, backend: SynCraftBackend, few_shot_count: int = 5) -> None:
        self.backend = backend
        self.few_shot_count = few_shot_count
        self.system = (
            files("agenticgen").joinpath("prompts/direction.txt").read_text(encoding="utf-8")
        )

    def build(self, observation: Observation, proposal_index: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": self.backend.format_examples(
                    observation.source_smiles, self.few_shot_count
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "observation": observation.model_dump(mode="json"),
                        "proposal_index": proposal_index,
                        "task": "Propose one local edit for synthetic accessibility, or abstain.",
                    },
                    ensure_ascii=False,
                ),
            },
        ]


class DirectionAgent:
    def __init__(self, llm: LLMPort, prompts: PromptPort) -> None:
        self.llm = llm
        self.prompts = prompts

    def propose(self, observation: Observation, proposal_index: int, output: Path) -> Proposal:
        schema = proposal_schema()
        schema["properties"]["case_id"]["const"] = observation.case_id
        result = self.llm.complete(self.prompts.build(observation, proposal_index), schema, output)
        proposal = Proposal.model_validate(result.payload)
        if proposal.case_id != observation.case_id:
            raise ValueError("Proposal is bound to a different case")
        if proposal.action == "abstain" and proposal.edits != [["STOP"]]:
            raise ValueError("An abstention cannot contain graph edits")
        return proposal
