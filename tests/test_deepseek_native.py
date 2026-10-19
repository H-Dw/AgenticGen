"""Portable real-API transport contract verified entirely with HTTP mocks."""

import json
from types import SimpleNamespace

import httpx
import pytest

from agenticgen.deepseek_native import ENDPOINT, DeepSeekNative
from agenticgen.io import read_json

KEY = "sk-unit-test-do-not-persist"
NATIVE = "[REASONING]\nSimplify the substituent.\n[EDITS]\n[['DEL_ATOM', 1], ['STOP']]"
MESSAGES = [{"role": "user", "content": "Propose a local graph edit."}]


def settings(**kwargs):
    return SimpleNamespace(
        **{
            "model": "deepseek-flash",
            "temperature": 0,
            "reasoning_effort": None,
            "network_attempts": 3,
            "max_output_tokens": 8192,
            "timeout_seconds": 240,
            **kwargs,
        }
    )


def response(content=None, model="deepseek-flash", finish="stop", **kwargs):
    return {
        "model": model,
        "id": "mock-provider-id",
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": content
                    if content is not None
                    else json.dumps({"native_output": NATIVE})
                },
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 17, "total_tokens": 48},
        **kwargs,
    }


def artifacts_text(path):
    return "\n".join(p.read_text(encoding="utf-8") for p in path.rglob("*") if p.is_file())


def test_fixed_wire_contract_and_secret_free_native_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == "Bearer " + KEY
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] == 8192
        assert "native_output" in body["messages"][-1]["content"]
        return httpx.Response(200, json=response())

    output = tmp_path / "proposal_0.json"
    native = DeepSeekNative(settings(), httpx.MockTransport(handler)).generate(
        MESSAGES, "case", 0, output
    )
    assert native == NATIVE
    assert read_json(output) == {"native_output": NATIVE}
    assert output.with_suffix(".native.txt").read_text(encoding="utf-8") == NATIVE
    receipt = read_json(output.with_suffix(".receipt.json"))
    assert receipt["attempts"] == 1
    assert receipt["format_repair_budget"] == 0 and receipt["interface_repairs"] == []
    assert receipt["backend"] == "real_api" and receipt["status"] == "completed"
    assert receipt["usage"]["prompt_tokens"] == 31
    assert KEY not in artifacts_text(tmp_path)
    assert len(calls) == 1


def test_transient_http_retry_is_bounded_and_does_not_change_prompt_or_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    bodies, delays = [], []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) < 3:
            return httpx.Response(503, text=KEY, headers={"secret": KEY})
        return httpx.Response(200, json=response())

    output = tmp_path / "proposal_0.json"
    result = DeepSeekNative(settings(), httpx.MockTransport(handler), sleep=delays.append).generate(
        MESSAGES, "case", 0, output
    )
    assert result == NATIVE and len(set(bodies)) == 1
    assert delays == [5, 10]
    assert read_json(output.with_suffix(".receipt.json"))["attempts"] == 3
    assert KEY not in artifacts_text(tmp_path)


def test_network_error_message_and_headers_are_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("Gateway echoed " + KEY, request=request)

    output = tmp_path / "proposal_0.json"
    with pytest.raises(RuntimeError, match="retries exhausted") as error:
        DeepSeekNative(settings(), httpx.MockTransport(handler), sleep=lambda _: None).generate(
            MESSAGES, "case", 0, output
        )
    assert KEY not in str(error.value)
    assert len(calls) == 3
    assert not output.exists()
    receipt = read_json(output.with_suffix(".receipt.json"))
    assert receipt["attempts"] == 3 and receipt["backend"] == "real_api_failure"
    assert KEY not in artifacts_text(tmp_path)


def test_http_auth_error_never_persists_response_body_or_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"echo_authorization": KEY}, headers={"x-secret": KEY})

    output = tmp_path / "proposal_0.json"
    with pytest.raises(RuntimeError, match="HTTP 401"):
        DeepSeekNative(settings(), httpx.MockTransport(handler)).generate(
            MESSAGES, "case", 0, output
        )
    assert len(calls) == 1 and KEY not in artifacts_text(tmp_path)


@pytest.mark.parametrize("model", ["deepseek-reasoner", {"malformed": KEY}])
def test_returned_model_mismatch_has_no_fallback_or_content_persistence(
    tmp_path, monkeypatch, model
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(model=model))

    output = tmp_path / "proposal_0.json"
    with pytest.raises(ValueError, match="different model"):
        DeepSeekNative(settings(), httpx.MockTransport(handler)).generate(
            MESSAGES, "case", 0, output
        )
    assert len(calls) == 1 and not output.exists()
    assert not output.with_suffix(".raw.txt").exists()
    assert KEY not in artifacts_text(tmp_path)


@pytest.mark.parametrize(
    "content", ["not JSON", "{}", '{"native_output":""}', '{"native_output":"ok","extra":true}']
)
def test_format_failure_is_not_retried(tmp_path, monkeypatch, content):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response(content=content))

    output = tmp_path / "proposal_0.json"
    with pytest.raises(ValueError, match="no format retry"):
        DeepSeekNative(settings(), httpx.MockTransport(handler)).generate(
            MESSAGES, "case", 0, output
        )
    assert len(calls) == 1 and not output.exists()


def test_incomplete_finish_is_rejected_even_with_a_valid_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response(finish="length")))
    output = tmp_path / "proposal_0.json"
    with pytest.raises(ValueError, match="Incomplete"):
        DeepSeekNative(settings(), transport).generate(MESSAGES, "case", 0, output)
    assert read_json(output.with_suffix(".receipt.json"))["finish_reason"] == "length"
    assert not output.exists()


def test_encoded_credentials_in_provider_content_are_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    encoded = "".join(f"\\u{ord(c):04x}" for c in KEY)
    content = '{"native_output":"' + encoded + '"}'
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response(content=content)))
    with pytest.raises(ValueError, match="credentials"):
        DeepSeekNative(settings(), transport).generate(
            MESSAGES, "case", 0, tmp_path / "proposal_0.json"
        )
    assert KEY not in artifacts_text(tmp_path)
    assert encoded not in artifacts_text(tmp_path)
    assert not list(tmp_path.glob("*.txt"))


def test_api_key_in_prompt_is_blocked_before_storage_or_network(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)

    def no_network(_):
        pytest.fail("A credential-bearing prompt must not be sent")

    output = tmp_path / "proposal_0.json"
    with pytest.raises(ValueError, match="Credentials"):
        DeepSeekNative(settings(), httpx.MockTransport(no_network)).generate(
            [{"role": "user", "content": KEY}], "case", 0, output
        )
    assert KEY not in artifacts_text(tmp_path)
    assert not output.with_suffix(".request.json").exists()


@pytest.mark.parametrize(
    "change",
    [{"model": "other"}, {"temperature": 1}, {"reasoning_effort": "high"}, {"network_attempts": 0}],
)
def test_constructor_refuses_silent_protocol_changes(change):
    with pytest.raises(ValueError):
        DeepSeekNative(settings(**change))
