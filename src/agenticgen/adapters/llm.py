"""Use EvoMolSteer's real DeepSeek protocol; replay is an explicit test backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from agenticgen.adapters.loading import load_evomolsteer
from agenticgen.contracts import CallResult
from agenticgen.io import object_hash, read_json, sha256, write_json


class EvoMolSteerDeepSeek:
    """Delegate HTTP, model checks, retries, format repair and receipts to the existing API."""

    def __init__(
        self,
        root: Path,
        *,
        model: str = "deepseek-flash",
        timeout: int = 240,
        attempts: int = 3,
        max_output_tokens: int = 4096,
        max_format_repairs: int = 1,
        transport: Any = None,
    ) -> None:
        if model != "deepseek-flash":
            raise ValueError(
                "The EvoMolSteer adapter supports deepseek-flash without model fallback"
            )
        self.module = load_evomolsteer(root)
        self.parameters = {
            "model": model,
            "timeout": timeout,
            "attempts": attempts,
            "max_output_tokens": max_output_tokens,
            "max_format_repairs": max_format_repairs,
        }
        self.transport = transport

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], output: Path
    ) -> CallResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        result = self.module.call_json(
            messages,
            schema,
            output,
            **self.parameters,
            transport=self.transport,
            input_store=output.parent / "inputs",
        )
        return CallResult(
            payload=result,
            backend="real_api",
            receipt_path=str(output.with_suffix(".receipt.json")),
        )


class ReplayLLM:
    """Engineering fixture, not a proxy model and never selected after API failure."""

    def __init__(self, fixture: Path) -> None:
        self.fixture = fixture
        self.data = read_json(fixture)
        if self.data.get("kind") != "engineering_replay":
            raise ValueError("Replay file must explicitly identify engineering fixtures")

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], output: Path
    ) -> CallResult:
        if output.exists():
            raise FileExistsError("Replay output already exists")
        request = json.loads(messages[-1]["content"])
        case_id = request["observation"]["case_id"]
        result = self.data["responses"][case_id][request["proposal_index"]]
        jsonschema.validate(result, schema)
        write_json(output.with_suffix(".request.json"), {"messages": messages, "schema": schema})
        write_json(output, result)
        receipt = output.with_suffix(".receipt.json")
        write_json(
            receipt,
            {
                "backend": "engineering_replay",
                "fixture_sha256": sha256(self.fixture),
                "response_sha256": sha256(output),
                "request_sha256": object_hash(messages),
                "scientific_result": False,
                "api_calls": 0,
            },
        )
        return CallResult(payload=result, backend="engineering_replay", receipt_path=str(receipt))
