"""Optional evidence injection; baseline runs never open the evidence file."""

from pathlib import Path

from agenticgen.base_config import EnhancementConfig
from agenticgen.chemistry import graph_hash
from agenticgen.contracts import CaseDataset, EnhancementPacket
from agenticgen.io import read_json, sha256


class EvidenceInjector:
    def __init__(self, settings: EnhancementConfig, path: Path | None, dataset: CaseDataset):
        self.packets: dict[str, list[EnhancementPacket]] = {}
        self.provenance = {"enabled": settings.enabled}
        if not settings.enabled:
            return
        raw = read_json(path)
        if raw.get("schema_version") != "agenticgen.enhancements.v1":
            raise ValueError("Unsupported generation evidence schema")
        cases = {c.case_id: c for c in dataset.cases}
        for item in raw["packets"]:
            packet = EnhancementPacket.model_validate(item)
            if packet.case_id not in cases or packet.source_graph_sha256 != graph_hash(
                cases[packet.case_id].smiles
            ):
                raise ValueError("Generation evidence belongs to a different source graph")
            if packet.kind not in settings.allowed_kinds:
                raise ValueError("Evidence kind is not allowed in this experiment arm")
            self.packets.setdefault(packet.case_id, []).append(packet)
        if set(self.packets) != set(cases):
            raise ValueError("Enhanced arm requires evidence for every source case")
        if any(
            sum(len(p.text) for p in group) > settings.max_characters
            for group in self.packets.values()
        ):
            raise ValueError("Generation evidence exceeds its fixed input budget")
        self.provenance.update(packets_sha256=sha256(path), allowed_kinds=settings.allowed_kinds)

    def for_case(self, case_id: str) -> list[dict]:
        return [p.model_dump(mode="json") for p in self.packets.get(case_id, [])]
