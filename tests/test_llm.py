import json

import httpx
import pytest

from agenticgen.adapters.llm import EvoMolSteerDeepSeek
from agenticgen.io import read_json, sha256

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"const": True}},
    "required": ["ok"],
    "additionalProperties": False,
}


def wire(content, model="deepseek-flash", finish_reason="stop"):
    return httpx.Response(
        200,
        json={
            "model": model,
            "id": "unit-test",
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    )


def test_actual_evo_api_contract_and_audited_cache(evo_root, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-not-a-secret")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0
        return wire('{"ok": true}')

    client = EvoMolSteerDeepSeek(evo_root, transport=httpx.MockTransport(handler))
    output = tmp_path / "call.json"
    messages = [{"role": "user", "content": "Format test"}]
    assert client.complete(messages, SCHEMA, output).payload == {"ok": True}
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    client.complete(messages, SCHEMA, output)
    assert len(calls) == 1
    receipt = read_json(output.with_suffix(".receipt.json"))
    assert receipt["response_sha256"] == sha256(output)
    assert receipt["requested_model"] == receipt["returned_model"] == "deepseek-flash"
    assert "unit-test-not-a-secret" not in "".join(p.read_text() for p in tmp_path.rglob("*.json"))


def test_wrong_model_and_truncation_never_become_success(evo_root, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    for index, response in enumerate(
        (wire('{"ok":true}', "another-model"), wire('{"ok":', finish_reason="length"))
    ):
        client = EvoMolSteerDeepSeek(
            evo_root, transport=httpx.MockTransport(lambda _, r=response: r)
        )
        output = tmp_path / f"failure_{index}.json"
        with pytest.raises(ValueError):
            client.complete([], SCHEMA, output)
        assert not output.exists()
        assert not output.with_suffix(".receipt.json").exists()


def test_neutral_format_repair_is_bounded(evo_root, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return wire('{"ok":true,"extra":1}' if len(calls) == 1 else '{"ok":true}')

    client = EvoMolSteerDeepSeek(evo_root, transport=httpx.MockTransport(handler))
    output = tmp_path / "repair.json"
    assert client.complete([], SCHEMA, output).payload == {"ok": True}
    assert len(calls) == 2
    assert len(read_json(output.with_suffix(".receipt.json"))["interface_repairs"]) == 1
    assert "Do not add research advice" in calls[-1]["messages"][-1]["content"]


def test_missing_key_does_not_fallback(evo_root, tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = EvoMolSteerDeepSeek(evo_root)
    with pytest.raises(ValueError, match="absent"):
        client.complete([], SCHEMA, tmp_path / "missing.json")
