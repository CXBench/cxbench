import pytest

from cxbench.openai_compat import (
    OpenAICompatConfig,
    OpenAICompatError,
    OpenAICompatibleClient,
    _messages_to_responses_input,
    parse_json_object,
)


def test_json_repair_parses_trailing_comma_object():
    assert parse_json_object('{"process_pass": true, "findings": [],}') == {
        "process_pass": True,
        "findings": [],
    }


def test_config_reads_retry_timeout_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "3")
    monkeypatch.setenv("OPENAI_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("OPENAI_VALIDATION_RETRIES", "2")
    monkeypatch.delenv("OPENAI_WIRE_API", raising=False)
    config = OpenAICompatConfig.from_model_string("openai:gpt-5.6-sol", role="test")
    assert config.provider == "openai"
    assert config.wire_api == "responses"
    assert config.timeout_seconds == 7
    assert config.max_retries == 3
    assert config.retry_backoff_seconds == 0
    assert config.validation_retries == 2


def test_responses_input_preserves_tool_call_lifecycle():
    _, items = _messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_session_context", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_session_context",
                "content": '{"ok":true}',
            },
        ]
    )

    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call_1"
    assert items[1]["type"] == "function_call_output"
    assert items[1]["call_id"] == "call_1"


def test_openai_responses_supports_non_stream_without_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_RESPONSES_STREAM", "false")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    client = OpenAICompatibleClient(
        OpenAICompatConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            wire_api="responses",
            provider="openai",
        )
    )
    captured = {}

    def post_json(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
        }

    monkeypatch.setattr(client, "_post_json", post_json)
    monkeypatch.setattr(
        client,
        "_post_responses_stream",
        lambda payload: (_ for _ in ()).throw(AssertionError("streaming was explicitly disabled")),
    )

    response = client.create([{"role": "user", "content": "hello"}])

    assert captured["path"] == "/responses"
    assert captured["payload"]["stream"] is False
    assert "reasoning" not in captured["payload"]
    assert response["choices"][0]["message"]["content"] == "ok"


def test_openai_responses_defaults_to_streaming_and_supports_reasoning(monkeypatch):
    monkeypatch.delenv("OPENAI_RESPONSES_STREAM", raising=False)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    client = OpenAICompatibleClient(
        OpenAICompatConfig(
            model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            wire_api="responses",
            provider="openai",
        )
    )
    captured = {}

    def responses(payload):
        captured.update(payload)
        return {"output_text": "ok"}

    monkeypatch.setattr(client, "_post_responses_stream", responses)
    response = client.create([{"role": "user", "content": "hello"}])

    assert captured["stream"] is True
    assert captured["reasoning"] == {"effort": "medium"}
    assert response["choices"][0]["message"]["content"] == "ok"


def test_create_validated_retries_with_feedback(monkeypatch):
    client = OpenAICompatibleClient(
        OpenAICompatConfig(
            model="fake",
            base_url="http://example.invalid",
            api_key="x",
            wire_api="chat",
            provider="test",
            validation_retries=1,
        )
    )
    calls = []

    def fake_create(messages, *, tools=None, response_format=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "choices": [{"message": {"role": "assistant", "content": "not json"}}],
                "_cxbench": {},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
            "_cxbench": {},
        }

    monkeypatch.setattr(client, "create", fake_create)
    _, parsed = client.create_validated(
        [{"role": "user", "content": "return json"}],
        validator=lambda response: client.validate_json_object(
            response, required_keys=["ok"], context="test"
        ),
        validation_context="test JSON",
    )

    assert parsed == {"ok": True}
    assert len(calls) == 2
    assert "previous test JSON response was invalid" in calls[1][-1]["content"]


def test_chat_payload_uses_configured_max_output_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "2048")
    client = OpenAICompatibleClient(
        OpenAICompatConfig(
            model="fake",
            base_url="http://example.invalid",
            api_key="x",
            wire_api="chat",
            provider="test",
        )
    )
    captured = {}

    def fake_post_chat_stream(payload):
        captured["path"] = "/chat/completions"
        captured["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(client, "_post_chat_stream", fake_post_chat_stream)

    client.create([{"role": "user", "content": "hello"}])

    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["max_completion_tokens"] == 2048


def test_openai_profile_uses_standard_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("OPENAI_WIRE_API", "responses")
    monkeypatch.setenv("OPENROUTER_API_KEY", "should-not-win")
    config = OpenAICompatConfig.from_model_string("openai:gpt-5.6-sol", role="user-sim")
    assert config.provider == "openai"
    assert config.model == "gpt-5.6-sol"
    assert config.api_key == "openai-key"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.wire_api == "responses"


def test_openai_profile_does_not_fall_back_to_another_provider_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unrelated-key")
    with pytest.raises(OpenAICompatError, match="requires OPENAI_API_KEY"):
        OpenAICompatConfig.from_model_string("openai:gpt-5.6-sol", role="user-sim")


def test_explicit_deepseek_provider_prefix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_APIKEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASEURL", "https://deepseek.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-win")

    config = OpenAICompatConfig.from_model_string(
        "deepseek:deepseek-ai/DeepSeek-V4-Flash", role="agent"
    )

    assert config.provider == "deepseek"
    assert config.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert config.api_key == "deepseek-key"
    assert config.base_url == "https://deepseek.example/v1"
    assert config.wire_api == "chat"


def test_explicit_siliconflow_provider_prefix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)
    monkeypatch.setenv("SILICONFLOW_APIKEY", "siliconflow-key")
    monkeypatch.setenv("SILICONFLOW_BASEURL", "https://siliconflow.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-win")

    config = OpenAICompatConfig.from_model_string(
        "siliconflow:deepseek-ai/DeepSeek-V4-Flash", role="agent"
    )

    assert config.provider == "siliconflow"
    assert config.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert config.api_key == "siliconflow-key"
    assert config.base_url == "https://siliconflow.example/v1"
    assert config.wire_api == "chat"


def test_explicit_shubiaobiao_provider_prefix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHUBIAOBIAO_API_KEY", raising=False)
    monkeypatch.delenv("SHUBIAOBIAO_BASE_URL", raising=False)
    monkeypatch.setenv("SHUBIAOBIAO_APIKEY", "shubiaobiao-key")
    monkeypatch.setenv("SHUBIAOBIAO_BASEURL", "https://shubiaobiao.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-win")

    config = OpenAICompatConfig.from_model_string("shubiaobiao:gpt-5.4-mini", role="agent")

    assert config.provider == "shubiaobiao"
    assert config.model == "gpt-5.4-mini"
    assert config.api_key == "shubiaobiao-key"
    assert config.base_url == "https://shubiaobiao.example/v1"
    assert config.wire_api == "chat"
