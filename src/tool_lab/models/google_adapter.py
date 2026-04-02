from __future__ import annotations

from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json
import os

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.base import AssistantResponse, BaseModelSession, ToolInvocation


class GoogleModelSession(BaseModelSession):
    provider_name = "google"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
        tools: list[ToolDefinition],
    ) -> None:
        super().__init__(config, system_prompt, initial_user_message, tools)
        api_key_env = config.api_key_env or "GOOGLE_API_KEY"
        api_key = os.environ.get(api_key_env) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {api_key_env} or GEMINI_API_KEY"
            )

        api_version = str(config.extra.get("api_version", "v1beta"))
        api_base = str(
            config.extra.get(
                "api_base",
                f"https://generativelanguage.googleapis.com/{api_version}",
            )
        ).rstrip("/")
        self._endpoint = (
            f"{api_base}/models/{config.model_name}:generateContent?key={quote(api_key)}"
        )

    def _invoke_model(self) -> AssistantResponse:
        payload = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": self._build_contents(),
            "tools": [{"functionDeclarations": self._build_tools()}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        response = self._post_json(payload)
        candidate = (response.get("candidates") or [{}])[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts: list[str] = []
        tool_calls: list[ToolInvocation] = []
        for index, part in enumerate(parts):
            if "text" in part:
                text_parts.append(str(part["text"]))
                continue
            if "functionCall" in part:
                function_call = part["functionCall"]
                tool_calls.append(
                    ToolInvocation(
                        tool_call_id=f"google_call_{index}_{function_call.get('name', 'tool')}",
                        name=str(function_call["name"]),
                        arguments=function_call.get("args", {}) or {},
                    )
                )
        usage = response.get("usageMetadata", {})
        return AssistantResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            raw={"response": response},
        )

    def _build_contents(self) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        pending_tool_parts: list[dict[str, Any]] = []

        def flush_tool_parts() -> None:
            if pending_tool_parts:
                contents.append({"role": "user", "parts": list(pending_tool_parts)})
                pending_tool_parts.clear()

        for entry in self.transcript:
            if entry.role == "user":
                flush_tool_parts()
                contents.append({"role": "user", "parts": [{"text": str(entry.content)}]})
                continue
            if entry.role == "assistant":
                flush_tool_parts()
                parts: list[dict[str, Any]] = []
                if entry.content:
                    parts.append({"text": str(entry.content)})
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in entry.tool_calls
                )
                contents.append({"role": "model", "parts": parts})
                continue
            pending_tool_parts.append(
                {
                    "functionResponse": {
                        "name": entry.tool_name,
                        "response": entry.content,
                    }
                }
            )

        flush_tool_parts()
        return contents

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            for tool in self.tools
        ]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                return json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Google API request failed: {body}") from exc
