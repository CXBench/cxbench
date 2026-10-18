from __future__ import annotations

import json
import logging
import os
import re
import ssl
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from json_repair import loads as json_repair_loads


class OpenAICompatError(RuntimeError):
    pass


class ProviderTransportError(OpenAICompatError):
    """Request retries are exhausted; restarting the task cannot repair transport."""


class ProviderRequestError(ProviderTransportError):
    """The provider rejected a request that will not improve on task retry."""


class ProviderOutputError(OpenAICompatError):
    """Raised when a provider response cannot be used by the agent loop."""


T = TypeVar("T")

_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
    ssl.SSLError,
)


@dataclass(frozen=True)
class OpenAICompatConfig:
    model: str
    base_url: str
    api_key: str
    wire_api: str
    provider: str
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    validation_retries: int = 1

    @classmethod
    def from_model_string(cls, model: str, *, role: str) -> "OpenAICompatConfig":
        load_dotenv()
        prefixed = _explicit_provider_model(model)
        if prefixed:
            provider, model_name = prefixed
            api_key = _key_for_explicit_provider(provider, role)
            base_url = _explicit_provider_base_url(provider)
            wire_api = _explicit_provider_wire_api(provider)
            return cls(
                model=model_name,
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                wire_api=wire_api,
                provider=provider,
                timeout_seconds=int(
                    os.environ.get(
                        "OPENAI_TIMEOUT_SECONDS",
                        os.environ.get("CXBENCH_PROVIDER_TIMEOUT_SECONDS", "120"),
                    )
                ),
                max_retries=int(
                    os.environ.get(
                        "OPENAI_MAX_RETRIES", os.environ.get("CXBENCH_PROVIDER_MAX_RETRIES", "5")
                    )
                ),
                retry_backoff_seconds=float(
                    os.environ.get(
                        "OPENAI_RETRY_BACKOFF_SECONDS",
                        os.environ.get("CXBENCH_PROVIDER_RETRY_BACKOFF_SECONDS", "2"),
                    )
                ),
                validation_retries=int(
                    os.environ.get(
                        "OPENAI_VALIDATION_RETRIES",
                        os.environ.get("CXBENCH_PROVIDER_VALIDATION_RETRIES", "1"),
                    )
                ),
            )
        if not model.startswith("openai:"):
            raise OpenAICompatError(
                f"{role} model must use openai:<model>, deepseek:<model>, openrouter:<model>, "
                f"siliconflow:<model>, or shubiaobiao:<model>, got {model!r}"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAICompatError(f"{role} requires OPENAI_API_KEY for openai:<model> calls")
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        wire_api = os.environ.get("OPENAI_WIRE_API") or "responses"
        if wire_api not in {"chat", "responses"}:
            raise OpenAICompatError("OPENAI_WIRE_API must be 'chat' or 'responses'")
        return cls(
            model=model.split(":", 1)[1],
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            wire_api=wire_api,
            provider="openai",
            timeout_seconds=int(
                os.environ.get(
                    "OPENAI_TIMEOUT_SECONDS",
                    os.environ.get("CXBENCH_PROVIDER_TIMEOUT_SECONDS", "120"),
                )
            ),
            max_retries=int(
                os.environ.get(
                    "OPENAI_MAX_RETRIES", os.environ.get("CXBENCH_PROVIDER_MAX_RETRIES", "5")
                )
            ),
            retry_backoff_seconds=float(
                os.environ.get(
                    "OPENAI_RETRY_BACKOFF_SECONDS",
                    os.environ.get("CXBENCH_PROVIDER_RETRY_BACKOFF_SECONDS", "2"),
                )
            ),
            validation_retries=int(
                os.environ.get(
                    "OPENAI_VALIDATION_RETRIES",
                    os.environ.get("CXBENCH_PROVIDER_VALIDATION_RETRIES", "1"),
                )
            ),
        )


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str
    raw_message: Dict[str, Any]


class OpenAICompatibleClient:
    """Reusable OpenAI-compatible client supporting Chat Completions and Responses."""

    def __init__(self, config: OpenAICompatConfig) -> None:
        self.config = config

    def create(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        response = self._create_once(messages, tools=tools, response_format=response_format)
        meta = response.setdefault("_cxbench", {})
        meta.update(
            {
                "provider": self.config.provider,
                "wire_api": self.config.wire_api,
                "model_requested": self.config.model,
                "model_returned": response.get("model"),
                "response_id": response.get("id"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": response.get("usage") or {},
                "cost_usd": _extract_cost_usd(response),
            }
        )
        return response

    def create_validated(
        self,
        messages: List[Dict[str, Any]],
        *,
        validator: Callable[[Dict[str, Any]], T],
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        validation_context: str,
    ) -> tuple[Dict[str, Any], T]:
        attempt_messages = list(messages)
        validation_errors: List[str] = []
        last_error: Optional[ProviderOutputError] = None
        for validation_attempt in range(self.config.validation_retries + 1):
            response = self.create(attempt_messages, tools=tools, response_format=response_format)
            try:
                parsed = validator(response)
            except ProviderOutputError as exc:
                last_error = exc
                validation_errors.append(str(exc))
                if validation_attempt >= self.config.validation_retries:
                    response.setdefault("_cxbench", {})["validation_errors"] = validation_errors
                    raise
                attempt_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": _validation_feedback(validation_context, str(exc)),
                    },
                ]
                continue
            response.setdefault("_cxbench", {})["validation_attempts"] = validation_attempt + 1
            if validation_errors:
                response["_cxbench"]["validation_errors"] = validation_errors
                response["_cxbench"]["recovered_after_validation_error"] = True
            return response, parsed
        raise last_error or ProviderOutputError("provider validation failed")

    def first_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise OpenAICompatError("OpenAI-compatible API response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise OpenAICompatError("OpenAI-compatible API response has no message")
        return message

    def validate_tool_call(
        self,
        response: Dict[str, Any],
        *,
        allowed_tools: List[str],
        tool_schemas: List[Dict[str, Any]],
    ) -> ToolCall:
        message = self.first_message(response)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return ToolCall(
                name="respond",
                arguments={"content": str(message.get("content") or "")},
                call_id="",
                raw_message=message,
            )
        call = tool_calls[0]
        function = call.get("function", {})
        name = str(function.get("name", ""))
        if name not in allowed_tools:
            raise ProviderOutputError(
                f"tool call used unavailable tool {name!r}; allowed tools: {allowed_tools}"
            )
        args_text = function.get("arguments") or "{}"
        args = parse_json_object(str(args_text))
        _validate_tool_args(name, args, tool_schemas)
        call_id = str(call.get("id") or f"call_{int(time.time() * 1000)}")
        raw_message = dict(message)
        raw_message["tool_calls"] = [call]
        return ToolCall(name=name, arguments=args, call_id=call_id, raw_message=raw_message)

    def validate_json_object(
        self, response: Dict[str, Any], *, required_keys: List[str], context: str
    ) -> Dict[str, Any]:
        message = self.first_message(response)
        content = str(message.get("content") or "")
        if not content.strip():
            raise ProviderOutputError(f"{context} returned empty content; expected JSON object")
        obj = parse_json_object(content)
        missing = [key for key in required_keys if key not in obj]
        if missing:
            raise ProviderOutputError(f"{context} JSON missing required keys: {missing}")
        return obj

    def _create_once(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.config.wire_api == "responses":
            return self._create_responses(messages, tools=tools, response_format=response_format)
        return self._create_chat(messages, tools=tools, response_format=response_format)

    def _create_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_completion_tokens": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1024")),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        return self._post_chat_stream(payload)

    def _create_responses(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        instructions, input_items = _messages_to_responses_input(messages)
        if response_format and response_format.get("type") == "json_object":
            instructions = f"{instructions}\nReturn only one valid JSON object."
        stream = _responses_stream_enabled()
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
            "instructions": instructions,
            "stream": stream,
            "max_output_tokens": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1024")),
        }
        reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "low")
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if response_format and response_format.get("type") == "json_schema":
            payload["text"] = {"format": {"type": "json_schema", **response_format["json_schema"]}}
        if tools:
            payload["tools"] = [_chat_tool_to_responses_tool(tool) for tool in tools]
        completed = (
            self._post_responses_stream(payload)
            if stream
            else self._post_json("/responses", payload)
        )
        return _responses_to_chat_compat(completed)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._urlopen_json(request)

    def _post_chat_stream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    try:
                        sock = response.fp.raw._sock
                        sock.settimeout(max(self.config.timeout_seconds, 300))
                    except AttributeError:
                        pass
                    content_parts: List[str] = []
                    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
                    model_name = ""
                    finish_reason = None
                    usage = {}
                    try:
                        for raw_line in response:
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            chunk = json.loads(data)
                            model_name = model_name or chunk.get("model", "")
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                if delta.get("content"):
                                    content_parts.append(delta["content"])
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                                for tc in delta.get("tool_calls", []):
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        tool_calls_acc[idx]["function"]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_acc[idx]["function"]["arguments"] += fn[
                                            "arguments"
                                        ]
                                    if tc.get("id"):
                                        tool_calls_acc[idx]["id"] = tc["id"]
                    except _TRANSPORT_ERRORS:
                        # Completion is authoritative; a closing connection may omit [DONE].
                        if finish_reason is None:
                            raise
                    if finish_reason is None:
                        raise ConnectionError("Chat stream ended before finish_reason")
                    message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": "".join(content_parts) or None,
                    }
                    if tool_calls_acc:
                        message["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                    result: Dict[str, Any] = {
                        "choices": [{"message": message, "finish_reason": finish_reason}],
                        "model": model_name,
                    }
                    if usage:
                        result["usage"] = usage
                    result.setdefault("_cxbench", {})["transport_attempts"] = attempt + 1
                    return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if not _retryable_http_status(exc.code, detail):
                    raise ProviderRequestError(
                        f"OpenAI-compatible API HTTP {exc.code}: {detail}"
                    ) from exc
                if attempt >= self.config.max_retries:
                    raise ProviderTransportError(
                        f"OpenAI-compatible API HTTP {exc.code} after {attempt + 1} request attempts: {detail}"
                    ) from exc
                last_error = exc
            except _TRANSPORT_ERRORS as exc:
                reason = getattr(exc, "reason", exc)
                if (
                    isinstance(reason, ssl.SSLCertVerificationError)
                    or attempt >= self.config.max_retries
                ):
                    raise _transport_error(request, exc, attempt + 1) from exc
                last_error = exc
            _sleep_before_retry(self.config.retry_backoff_seconds, attempt, last_error)
        raise OpenAICompatError(f"OpenAI-compatible API request failed after retries: {last_error}")

    def _post_responses_stream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        return self._urlopen_responses_stream(request)

    def _urlopen_json(self, request: urllib.request.Request) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    payload.setdefault("_cxbench", {})["transport_attempts"] = attempt + 1
                    return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if not _retryable_http_status(exc.code, detail):
                    raise ProviderRequestError(
                        f"OpenAI-compatible API HTTP {exc.code}: {detail}"
                    ) from exc
                if attempt >= self.config.max_retries:
                    raise ProviderTransportError(
                        f"OpenAI-compatible API HTTP {exc.code} after {attempt + 1} request attempts: {detail}"
                    ) from exc
                last_error = exc
            except _TRANSPORT_ERRORS as exc:
                reason = getattr(exc, "reason", exc)
                if (
                    isinstance(reason, ssl.SSLCertVerificationError)
                    or attempt >= self.config.max_retries
                ):
                    raise _transport_error(request, exc, attempt + 1) from exc
                last_error = exc
            _sleep_before_retry(self.config.retry_backoff_seconds, attempt, last_error)
        raise OpenAICompatError(f"OpenAI-compatible API request failed after retries: {last_error}")

    def _urlopen_responses_stream(self, request: urllib.request.Request) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    completed: Optional[Dict[str, Any]] = None
                    output_items: List[Dict[str, Any]] = []
                    text_parts: List[str] = []
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if event.get("type") == "response.output_item.done" and isinstance(
                            event.get("item"), dict
                        ):
                            output_items.append(event["item"])
                        if event.get("type") == "response.output_text.delta" and event.get("delta"):
                            text_parts.append(str(event["delta"]))
                        if event.get("type") == "response.completed":
                            completed = event.get("response")
                            break
                    if not completed:
                        raise ConnectionError("Responses stream ended without response.completed")
                    if not completed.get("output") and output_items:
                        completed["output"] = output_items
                    if text_parts and not completed.get("output"):
                        completed["output"] = [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "".join(text_parts)}],
                            }
                        ]
                    completed.setdefault("_cxbench", {})["transport_attempts"] = attempt + 1
                    return completed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if not _retryable_http_status(exc.code, detail):
                    raise ProviderRequestError(
                        f"OpenAI-compatible Responses API HTTP {exc.code}: {detail}"
                    ) from exc
                if attempt >= self.config.max_retries:
                    raise ProviderTransportError(
                        "OpenAI-compatible Responses API "
                        f"HTTP {exc.code} after {attempt + 1} request attempts: {detail}"
                    ) from exc
                last_error = exc
            except _TRANSPORT_ERRORS as exc:
                reason = getattr(exc, "reason", exc)
                if (
                    isinstance(reason, ssl.SSLCertVerificationError)
                    or attempt >= self.config.max_retries
                ):
                    raise _transport_error(request, exc, attempt + 1) from exc
                last_error = exc
            except OpenAICompatError as exc:
                if attempt >= self.config.max_retries:
                    raise
                last_error = exc
            _sleep_before_retry(self.config.retry_backoff_seconds, attempt, last_error)
        raise OpenAICompatError(
            f"OpenAI-compatible Responses API request failed after retries: {last_error}"
        )


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def apply_provider_env(provider: str, dotenv: Dict[str, str]) -> Dict[str, str | None]:
    provider = provider.lower()
    if provider == "same":
        return {}
    mappings = _provider_env_mappings(dotenv)
    if provider not in mappings:
        raise ValueError(
            f"unknown provider {provider!r}; expected openai, deepseek, openrouter, siliconflow, shubiaobiao, or same"
        )
    managed_keys = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_WIRE_API",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    )
    old = {key: os.environ.get(key) for key in managed_keys}
    for key in managed_keys:
        os.environ.pop(key, None)
    for key, value in mappings[provider].items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    return old


def restore_provider_env(old: Dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _provider_env_mappings(dotenv: Dict[str, str]) -> Dict[str, Dict[str, str | None]]:
    return {
        "openai": {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY")
            or dotenv.get("OPENAI_API_KEY")
            or dotenv.get("OPENAI_APIKEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL")
            or dotenv.get("OPENAI_BASE_URL")
            or dotenv.get("OPENAI_BASEURL")
            or "https://api.openai.com/v1",
            "OPENAI_WIRE_API": os.environ.get("OPENAI_WIRE_API")
            or dotenv.get("OPENAI_WIRE_API")
            or dotenv.get("OPENAI_WIREAPI")
            or "responses",
        },
        "deepseek": {
            "OPENAI_API_KEY": os.environ.get("DEEPSEEK_API_KEY")
            or dotenv.get("DEEPSEEK_API_KEY")
            or dotenv.get("DEEPSEEK_APIKEY"),
            "OPENAI_BASE_URL": os.environ.get("DEEPSEEK_BASE_URL")
            or dotenv.get("DEEPSEEK_BASE_URL")
            or dotenv.get("DEEPSEEK_BASEURL"),
            "OPENAI_WIRE_API": os.environ.get("DEEPSEEK_WIRE_API")
            or dotenv.get("DEEPSEEK_WIRE_API")
            or dotenv.get("DEEPSEEK_WIREAPI")
            or "chat",
        },
        "openrouter": {
            "OPENAI_API_KEY": os.environ.get("OPENROUTER_API_KEY")
            or dotenv.get("OPENROUTER_API_KEY")
            or dotenv.get("OPENROUTER_APIKEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENROUTER_BASE_URL")
            or dotenv.get("OPENROUTER_BASE_URL")
            or dotenv.get("OPENROUTER_BASEURL"),
            "OPENAI_WIRE_API": os.environ.get("OPENROUTER_WIRE_API")
            or dotenv.get("OPENROUTER_WIRE_API")
            or dotenv.get("OPENROUTER_WIREAPI")
            or "chat",
        },
        "siliconflow": {
            "OPENAI_API_KEY": os.environ.get("SILICONFLOW_API_KEY")
            or dotenv.get("SILICONFLOW_API_KEY")
            or dotenv.get("SILICONFLOW_APIKEY"),
            "OPENAI_BASE_URL": os.environ.get("SILICONFLOW_BASE_URL")
            or dotenv.get("SILICONFLOW_BASE_URL")
            or dotenv.get("SILICONFLOW_BASEURL")
            or "https://api.siliconflow.cn/v1",
            "OPENAI_WIRE_API": os.environ.get("SILICONFLOW_WIRE_API")
            or dotenv.get("SILICONFLOW_WIRE_API")
            or dotenv.get("SILICONFLOW_WIREAPI")
            or "chat",
        },
        "shubiaobiao": {
            "OPENAI_API_KEY": os.environ.get("SHUBIAOBIAO_API_KEY")
            or dotenv.get("SHUBIAOBIAO_API_KEY")
            or dotenv.get("SHUBIAOBIAO_APIKEY"),
            "OPENAI_BASE_URL": os.environ.get("SHUBIAOBIAO_BASE_URL")
            or dotenv.get("SHUBIAOBIAO_BASE_URL")
            or dotenv.get("SHUBIAOBIAO_BASEURL")
            or "https://api.shubiaobiao.cn/v1",
            "OPENAI_WIRE_API": os.environ.get("SHUBIAOBIAO_WIRE_API")
            or dotenv.get("SHUBIAOBIAO_WIRE_API")
            or dotenv.get("SHUBIAOBIAO_WIREAPI")
            or "chat",
        },
    }


def _first_present_env(*names: str) -> Optional[tuple[str, str]]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    return None


def _explicit_provider_model(model: str) -> Optional[tuple[str, str]]:
    for provider in ("deepseek", "openrouter", "siliconflow", "shubiaobiao"):
        prefix = f"{provider}:"
        if model.startswith(prefix):
            return provider, model.split(":", 1)[1]
    return None


def _key_for_explicit_provider(provider: str, role: str) -> str:
    aliases = {
        "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_APIKEY"),
        "siliconflow": ("SILICONFLOW_API_KEY", "SILICONFLOW_APIKEY"),
        "shubiaobiao": ("SHUBIAOBIAO_API_KEY", "SHUBIAOBIAO_APIKEY"),
    }
    source = _first_present_env(*aliases[provider])
    if not source:
        keys = ", ".join(aliases[provider])
        raise OpenAICompatError(f"{role} requires one of {keys} for {provider}:<model> calls")
    return source[1]


def _explicit_provider_base_url(provider: str) -> str:
    if provider == "deepseek":
        value = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DEEPSEEK_BASEURL")
        if not value:
            raise OpenAICompatError(
                "deepseek:<model> requires DEEPSEEK_BASE_URL or DEEPSEEK_BASEURL"
            )
        return value
    if provider == "openrouter":
        return (
            os.environ.get("OPENROUTER_BASE_URL")
            or os.environ.get("OPENROUTER_BASEURL")
            or "https://openrouter.ai/api/v1"
        )
    if provider == "siliconflow":
        return (
            os.environ.get("SILICONFLOW_BASE_URL")
            or os.environ.get("SILICONFLOW_BASEURL")
            or "https://api.siliconflow.cn/v1"
        )
    if provider == "shubiaobiao":
        return (
            os.environ.get("SHUBIAOBIAO_BASE_URL")
            or os.environ.get("SHUBIAOBIAO_BASEURL")
            or "https://api.shubiaobiao.cn/v1"
        )
    raise OpenAICompatError(f"unknown provider prefix {provider!r}")


def _explicit_provider_wire_api(provider: str) -> str:
    if provider == "deepseek":
        wire_api = (
            os.environ.get("DEEPSEEK_WIRE_API") or os.environ.get("OPENAI_WIRE_API") or "chat"
        )
    elif provider == "openrouter":
        wire_api = (
            os.environ.get("OPENROUTER_WIRE_API") or os.environ.get("OPENAI_WIRE_API") or "chat"
        )
    elif provider == "siliconflow":
        wire_api = (
            os.environ.get("SILICONFLOW_WIRE_API") or os.environ.get("OPENAI_WIRE_API") or "chat"
        )
    elif provider == "shubiaobiao":
        wire_api = (
            os.environ.get("SHUBIAOBIAO_WIRE_API") or os.environ.get("OPENAI_WIRE_API") or "chat"
        )
    else:
        raise OpenAICompatError(f"unknown provider prefix {provider!r}")
    if wire_api not in {"chat", "responses"}:
        raise OpenAICompatError(f"{provider} wire API must be 'chat' or 'responses'")
    return wire_api


def _messages_to_responses_input(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    instructions = ""
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content") or "")
        if role == "system":
            instructions = f"{instructions}\n{content}".strip()
            continue
        if role == "assistant" and message.get("tool_calls"):
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    }
                )
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": content,
                }
            )
            continue
        if role not in {"user", "assistant", "developer"}:
            role = "user"
        content_type = "output_text" if role == "assistant" else "input_text"
        input_items.append({"role": role, "content": [{"type": content_type, "text": content}]})
    return instructions, input_items


def _chat_tool_to_responses_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function", {})
    return {
        "type": "function",
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object", "properties": {}}),
    }


def _responses_to_chat_compat(response: Dict[str, Any]) -> Dict[str, Any]:
    content_parts: List[str] = []
    if response.get("output_text"):
        content_parts.append(str(response["output_text"]))
    tool_calls: List[Dict[str, Any]] = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    content_parts.append(str(content["text"]))
        if item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )
    message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "model": response.get("model"),
        "id": response.get("id"),
        "usage": response.get("usage"),
        "_cxbench": response.get("_cxbench", {}),
    }


def _responses_stream_enabled() -> bool:
    name = "OPENAI_RESPONSES_STREAM"
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return True
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise OpenAICompatError(f"{name} must be true or false")


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = _JSON_FENCE.sub("", text.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ProviderOutputError("JSON object not found in provider output")
        candidate = cleaned[start : end + 1]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = json_repair_loads(candidate)
            except Exception as exc:
                raise ProviderOutputError(f"JSON parse/repair failed: {exc}") from exc
    if not isinstance(value, dict):
        raise ProviderOutputError("expected a JSON object")
    return value


def _validate_tool_args(
    name: str, args: Dict[str, Any], tool_schemas: List[Dict[str, Any]]
) -> None:
    schema = next(
        (tool for tool in tool_schemas if tool.get("function", {}).get("name") == name), None
    )
    if not schema:
        raise ProviderOutputError(f"tool schema not found for {name!r}")
    params = schema.get("function", {}).get("parameters", {})
    required = list(params.get("required") or [])
    missing = [field for field in required if field not in args]
    if missing:
        raise ProviderOutputError(f"tool {name!r} arguments missing required fields: {missing}")
    if params.get("additionalProperties") is False:
        allowed = set((params.get("properties") or {}).keys())
        extra = [field for field in args if field not in allowed]
        if extra:
            raise ProviderOutputError(
                f"tool {name!r} arguments include unsupported fields: {extra}"
            )


def _validation_feedback(context: str, error: str) -> str:
    return (
        f"Your previous {context} response was invalid and could not be used by the cxbench runtime. "
        f"Error: {error}. Retry once. Return only a valid response in the requested schema. "
        "If you are calling a tool, use exactly one available tool and valid JSON arguments."
    )


_GATEWAY_ROUTING_ERROR = re.compile(
    r"unknown provider for model|no available channel|no channel available",
    re.IGNORECASE,
)


def _retryable_http_status(status: int, detail: str = "") -> bool:
    if status in {408, 409, 425, 429, 500, 502, 503, 504, 525}:
        return True
    # Relay gateways report transient upstream-routing gaps as 400; a genuinely
    # unknown model reports model_not_found instead and must not be retried.
    return status == 400 and bool(_GATEWAY_ROUTING_ERROR.search(detail))


def _extract_cost_usd(response: Dict[str, Any]) -> float | None:
    usage = response.get("usage") or {}
    for key in ("cost_usd", "estimated_cost_usd", "response_cost"):
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, int | float):
            return float(value)
    hidden = response.get("_hidden_params") or {}
    value = hidden.get("response_cost") if isinstance(hidden, dict) else None
    if isinstance(value, int | float):
        return float(value)
    return None


def _transport_error(
    request: urllib.request.Request, exc: Exception, attempts: int
) -> ProviderTransportError:
    host = urllib.parse.urlsplit(request.full_url).hostname
    reason = getattr(exc, "reason", exc)
    return ProviderTransportError(
        f"OpenAI-compatible API transport failed after {attempts} request attempt(s) at {host}: {reason}. "
        "Check the endpoint, network/proxy, and TLS settings, then resume the same run ID."
    )


def _sleep_before_retry(base_delay: float, attempt: int, error: Exception | None = None) -> None:
    if base_delay <= 0:
        return
    delay = min(base_delay * (2 ** min(attempt, 30)), 30.0)
    reason = getattr(error, "reason", error)
    logging.getLogger(__name__).warning(
        "provider_retry attempt=%s error=%s delay_seconds=%s",
        attempt + 1,
        type(reason).__name__,
        delay,
    )
    time.sleep(delay)
