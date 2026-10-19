"""Portable DeepSeek native-text transport using EvoMolSteer's audited wire policy.

There is no model fallback, format retry, chemistry repair or EvoMolSteer import.
Provider error bodies and authorization headers are never persisted or reported.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from agenticgen.io import sha256, utc_now, write_json

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
FORMAT_POLICY = "agenticgen.native-envelope.v1"
ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {"native_output": {"type": "string", "minLength": 1}},
    "required": ["native_output"],
    "additionalProperties": False,
}
ENVELOPE_INSTRUCTION = (
    "Transport envelope only: return a JSON object with one string field native_output. "
    "Its decoded value must follow the SynCraft [REASONING]/[EDITS] format above. "
    "Do not change the chemical task. Return JSON satisfying this schema: "
    + json.dumps(ENVELOPE_SCHEMA, separators=(",", ":"))
)


def _safe_usage(value):
    if not isinstance(value, dict):
        return None
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    return {
        k: value[k]
        for k in keys
        if isinstance(value.get(k), int) and not isinstance(value[k], bool)
    }


class DeepSeekNative:
    """Fixed DeepSeek endpoint/model, JSON envelope, bounded transport-only retries."""

    def __init__(self, settings, transport=None, sleep=time.sleep):
        if settings.model != MODEL:
            raise ValueError("DeepSeekNative requires exactly deepseek-flash; no fallback")
        if (
            getattr(settings, "temperature", 0) != 0
            or getattr(settings, "reasoning_effort", None) is not None
        ):
            raise ValueError("DeepSeekNative requires temperature=0 and disabled thinking")
        if not 1 <= settings.network_attempts <= 5:
            raise ValueError("DeepSeekNative requires between one and five network attempts")
        if not 512 <= settings.max_output_tokens <= 65536:
            raise ValueError("DeepSeekNative requires a bounded output-token budget")
        if not 0 < settings.timeout_seconds <= 1200:
            raise ValueError("DeepSeekNative requires a bounded request timeout")
        self.settings, self.transport, self.sleep = settings, transport, sleep

    def generate(self, messages, case_id: str, index: int, output: Path) -> str:
        output = Path(output)
        request_path = output.with_suffix(".request.json")
        receipt_path = output.with_suffix(".receipt.json")
        raw_path = output.with_suffix(".raw.txt")
        native_path = output.with_suffix(".native.txt")
        if any(
            path.exists() for path in (output, request_path, receipt_path, raw_path, native_path)
        ):
            raise FileExistsError("Use a new output path; prior DeepSeek artifacts are preserved")
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "provider": "DeepSeek",
            "endpoint": ENDPOINT,
            "backend": "real_api_failure",
            "execution_kind": "real_api",
            "status": "started",
            "case_id": case_id,
            "proposal_index": index,
            "requested_model": MODEL,
            "format_policy": FORMAT_POLICY,
            "thinking": "disabled",
            "temperature": 0,
            "max_output_tokens": self.settings.max_output_tokens,
            "network_attempt_budget": self.settings.network_attempts,
            "format_repair_budget": 0,
            "attempts": 0,
            "http_attempts": [],
            "interface_repairs": [],
            "started_at": utc_now(),
        }
        key = os.environ.get("DEEPSEEK_API_KEY")

        def safe(value):
            return value.replace(key, "[REDACTED]") if key and isinstance(value, str) else value

        receipt["case_id"] = safe(receipt["case_id"])

        def fail(reason: str, error_type=ValueError):
            receipt.update(status="failed", error=reason, completed_at=utc_now())
            receipt["case_id"] = safe(receipt["case_id"])
            write_json(receipt_path, receipt)
            raise error_type(reason) from None

        if not key:
            fail("DEEPSEEK_API_KEY is absent from this process")
        if not isinstance(messages, list) or not messages:
            fail("DeepSeek requires a nonempty message list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
            ):
                fail("Unsupported DeepSeek message structure")
            if key in message["content"]:
                fail("Credentials must not appear in persisted prompt content")
        body = {
            "model": MODEL,
            "messages": [*messages, {"role": "user", "content": ENVELOPE_INSTRUCTION}],
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": self.settings.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        write_json(
            request_path,
            {
                "provider": "DeepSeek",
                "endpoint": ENDPOINT,
                "body": body,
                "response_schema": ENVELOPE_SCHEMA,
                "format_policy": FORMAT_POLICY,
            },
        )
        receipt["request_sha256"] = sha256(request_path)
        wire = None
        with httpx.Client(
            timeout=self.settings.timeout_seconds, transport=self.transport, trust_env=True
        ) as client:
            for attempt in range(self.settings.network_attempts):
                receipt["attempts"] += 1
                try:
                    response = client.post(
                        ENDPOINT, json=body, headers={"Authorization": "Bearer " + key}
                    )
                except (httpx.TimeoutException, httpx.NetworkError):
                    receipt["http_attempts"].append(
                        {"attempt": attempt + 1, "status": "network_or_timeout"}
                    )
                    if attempt + 1 == self.settings.network_attempts:
                        fail("DeepSeek network/timeout retries exhausted", RuntimeError)
                    self.sleep(min(30, 5 * 2**attempt))
                    continue
                receipt["http_attempts"].append(
                    {"attempt": attempt + 1, "status_code": response.status_code}
                )
                if (
                    response.status_code in {429, 500, 502, 503, 504}
                    and attempt + 1 < self.settings.network_attempts
                ):
                    self.sleep(min(30, 5 * 2**attempt))
                    continue
                if response.status_code != 200:
                    fail(f"DeepSeek HTTP {response.status_code}", RuntimeError)
                try:
                    wire = response.json()
                except ValueError:
                    fail("DeepSeek returned an invalid provider JSON envelope")
                break
        if not isinstance(wire, dict):
            fail("DeepSeek returned no provider response object")
        choices = wire.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            fail("DeepSeek returned an invalid choice list")
        choice = choices[0]
        receipt.update(
            returned_model=safe(wire.get("model")) if isinstance(wire.get("model"), str) else None,
            provider_request_id=safe(wire.get("id")) if isinstance(wire.get("id"), str) else None,
            finish_reason=safe(choice.get("finish_reason"))
            if isinstance(choice.get("finish_reason"), str)
            else None,
            usage=_safe_usage(wire.get("usage")),
        )
        if wire.get("model") != MODEL:
            fail("Provider returned a different model; fallback is prohibited")
        if choice.get("finish_reason") != "stop":
            fail("Incomplete DeepSeek answer; no format retry or chemical resampling")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            fail("DeepSeek content is not text")
        if key in content:
            fail("Provider content contains credentials and was not persisted")
        try:
            envelope = json.loads(content)
        except ValueError:
            fail("DeepSeek native envelope is invalid JSON; no format retry")
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"native_output"}
            or not isinstance(envelope["native_output"], str)
            or not envelope["native_output"].strip()
        ):
            fail("DeepSeek native envelope violates its schema; no format retry")
        native = envelope["native_output"]
        if key in native:
            fail("Decoded provider content contains credentials and was not persisted")
        raw_path.write_text(content, encoding="utf-8", newline="\n")
        receipt["raw_response_path"] = raw_path.name
        receipt["raw_response_sha256"] = sha256(raw_path)
        write_json(output, envelope)
        native_path.write_text(native, encoding="utf-8", newline="\n")
        receipt.update(
            status="completed",
            backend="real_api",
            completed_at=utc_now(),
            response_sha256=sha256(output),
            native_output_path=native_path.name,
            native_output_sha256=sha256(native_path),
        )
        write_json(receipt_path, receipt)
        return native
