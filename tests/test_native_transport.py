import json

import httpx

from agenticgen.base_config import FoundationLLM
from agenticgen.native import EvoNativeLLM


def test_native_envelope_preserves_syn_craft_text_and_exact_evo_wire_contract(
    evo_root, tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "engineering-test-key")
    text = '[REASONING]\nBrief chemical explanation.\n[EDITS]\n```json\n[["STOP"]]\n```'
    calls = []

    def reply(request):
        data = json.loads(request.content)
        calls.append(data)
        assert data["model"] == "deepseek-flash"
        assert data["thinking"] == {"type": "disabled"}
        assert data["response_format"] == {"type": "json_object"}
        assert "native_output" in data["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "id": "native-fixture",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"native_output": text})},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    provider = EvoNativeLLM(
        evo_root,
        FoundationLLM(evomolsteer_root=str(evo_root)),
        transport=httpx.MockTransport(reply),
    )
    assert (
        provider.generate(
            [{"role": "user", "content": "Mapped graph fixture"}],
            "fixture",
            0,
            tmp_path / "native.json",
        )
        == text
    )
    assert len(calls) == 1
