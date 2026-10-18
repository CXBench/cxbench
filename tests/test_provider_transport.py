"""Offline transport regressions; no provider credentials or network are used."""

from __future__ import annotations

import http.client
import io
import json
import ssl
import urllib.error

import pytest

from cxbench import openai_compat as api


def _client(mode, retries=2):
    return api.OpenAICompatibleClient(
        api.OpenAICompatConfig(
            model="fake",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            wire_api="responses" if mode == "responses" else "chat",
            provider="test",
            max_retries=retries,
            retry_backoff_seconds=1.0,
        )
    )


def _request(client, mode):
    if mode == "json":
        return client._post_json("/probe", {})
    if mode == "responses":
        return client._post_responses_stream({})
    return client._post_chat_stream({})


def _body(mode):
    if mode == "json":
        return b'{"ok": true}'
    if mode == "responses":
        event = {"type": "response.completed", "response": {"model": "fake", "output": []}}
    else:
        event = {
            "model": "fake",
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
        }
    return ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_ssl_eof_retries_the_same_request_then_recovers(monkeypatch, mode, wrapped):
    requests = []
    delays = []

    def urlopen(request, **kwargs):
        requests.append(request)
        if len(requests) < 3:
            error = ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING")
            raise urllib.error.URLError(error) if wrapped else error
        return io.BytesIO(_body(mode))

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", delays.append)
    result = _request(_client(mode), mode)
    assert result["_cxbench"]["transport_attempts"] == 3
    assert len(requests) == 3 and all(r is requests[0] for r in requests)
    assert delays == [1.0, 2.0]


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
def test_ssl_eof_during_read_discards_partial_response(monkeypatch, mode):
    class BrokenResponse(io.BytesIO):
        def __iter__(self):
            if mode == "chat":
                yield b'data: {"choices":[{"delta":{"content":"discard-me"}}]}\n'
            else:
                yield b'data: {"type":"response.output_text.delta","delta":"discard-me"}\n'
            raise ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING")

        def read(self, *args):
            raise ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING")

    calls = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        return BrokenResponse() if len(calls) == 1 else io.BytesIO(_body(mode))

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    result = _request(_client(mode), mode)
    assert len(calls) == 2
    assert result["_cxbench"]["transport_attempts"] == 2
    assert "discard-me" not in json.dumps(result)


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
def test_persistent_connection_failure_has_a_finite_request_budget(monkeypatch, mode):
    calls = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError(ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING"))

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    with pytest.raises(api.ProviderTransportError, match="after 3 request attempt") as exc:
        _request(_client(mode), mode)
    assert len(calls) == 3
    assert "resume the same run ID" in str(exc.value)
    assert "test-key" not in str(exc.value)


@pytest.mark.parametrize("mode", ["chat", "responses"])
def test_truncated_stream_is_not_accepted_as_a_completed_response(monkeypatch, mode):
    calls = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n')

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    with pytest.raises(api.ProviderTransportError, match="stream ended"):
        _request(_client(mode), mode)
    assert len(calls) == 3


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_certificate_verification_failure_is_not_retried(monkeypatch, mode, wrapped):
    calls = []

    def urlopen(*args, **kwargs):
        calls.append(1)
        error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        raise urllib.error.URLError(error) if wrapped else error

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    with pytest.raises(api.ProviderTransportError, match="certificate verify failed"):
        _request(_client(mode), mode)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
def test_non_retryable_provider_rejection_stops_immediately(monkeypatch, mode):
    calls = []
    detail = b'{"detail":"model is not supported for this account"}'

    def urlopen(request, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(detail))

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    with pytest.raises(api.ProviderRequestError, match="not supported"):
        _request(_client(mode), mode)
    assert len(calls) == 1


def test_completed_responses_stream_does_not_wait_for_connection_eof(monkeypatch):
    class CompletedResponse(io.BytesIO):
        def __iter__(self):
            yield _body("responses").split(b"\n")[0] + b"\n"
            raise ssl.SSLEOFError(8, "connection closed after completion")

    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: CompletedResponse())
    assert _request(_client("responses"), "responses")["_cxbench"]["transport_attempts"] == 1


@pytest.mark.parametrize("mode", ["chat", "json", "responses"])
@pytest.mark.parametrize("prefix", ["openai", "openrouter"])
def test_default_request_retries_survive_a_brief_outage(
    monkeypatch, tmp_path, caplog, mode, prefix
):
    monkeypatch.chdir(tmp_path)
    for name in ["MAX_RETRIES", "RETRY_BACKOFF_SECONDS"]:
        for namespace in ["OPENAI", "CXBENCH_PROVIDER"]:
            monkeypatch.delenv(f"{namespace}_{name}", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/v1")
    client = api.OpenAICompatibleClient(
        api.OpenAICompatConfig.from_model_string(f"{prefix}:fake", role="test")
    )
    requests, delays = [], []

    def urlopen(request, **kwargs):
        requests.append(request)
        if len(requests) <= 5:
            raise urllib.error.URLError(ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING"))
        return io.BytesIO(_body(mode))

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", delays.append)
    result = _request(client, mode)
    assert result["_cxbench"]["transport_attempts"] == 6
    assert len(requests) == 6 and all(r is requests[0] for r in requests)
    assert len(delays) == 5 and 50 <= sum(delays) <= 90
    assert all(0 < delay <= 30 for delay in delays)
    assert "provider_retry" in caplog.text and "SSLEOFError" in caplog.text
    assert "test-key" not in caplog.text


@pytest.mark.parametrize("with_usage", [False, True])
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("close_error", [ssl.SSLEOFError, http.client.IncompleteRead])
def test_completed_chat_response_survives_connection_close_error(
    monkeypatch, with_usage, tool_call, close_error
):
    delta = {"content": "ok"}
    finish_reason = "stop"
    if tool_call:
        delta = {
            "tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "probe", "arguments": '{"x":1}'}}
            ]
        }
        finish_reason = "tool_calls"
    chunk = {"model": "fake", "choices": [{"delta": delta, "finish_reason": finish_reason}]}

    class CompletedResponse(io.BytesIO):
        def __iter__(self):
            yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            if with_usage:
                yield b'data: {"choices":[],"usage":{"total_tokens":12}}\n\n'
            raise close_error(
                b""
                if close_error is http.client.IncompleteRead
                else "connection closed after completion"
            )

    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        return CompletedResponse()

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    result = _request(_client("chat"), "chat")
    assert len(requests) == 1
    assert result["_cxbench"]["transport_attempts"] == 1
    assert result["choices"][0]["finish_reason"] == finish_reason
    message = result["choices"][0]["message"]
    if tool_call:
        assert message["tool_calls"][0]["function"] == {"name": "probe", "arguments": '{"x":1}'}
    else:
        assert message["content"] == "ok"
    if with_usage:
        assert result["usage"] == {"total_tokens": 12}
    else:
        assert "usage" not in result
