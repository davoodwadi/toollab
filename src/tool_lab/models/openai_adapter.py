from __future__ import annotations

from typing import Any
import json
import os

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.base import AssistantResponse, BaseModelSession, ToolInvocation


class OpenAIModelSession(BaseModelSession):
    provider_name = "openai"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
        tools: list[ToolDefinition],
    ) -> None:
        super().__init__(config, system_prompt, initial_user_message, tools)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Run `python -m pip install -e .`."
            ) from exc

        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
        self._client = OpenAI(api_key=api_key)

    def _invoke_model(self) -> AssistantResponse:
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=self._build_messages(),
            tools=self._build_tools(),
            # temperature=self.config.temperature,
            # max_tokens=self.config.max_output_tokens,
        )
        message = response.choices[0].message
        tool_calls = []
        for tool_call in message.tool_calls or []:
            tool_calls.append(
                ToolInvocation(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=_safe_json_loads(tool_call.function.arguments),
                )
            )
        usage = getattr(response, "usage", None)
        return AssistantResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw={"id": getattr(response, "id", None)},
        )

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for entry in self.transcript:
            if entry.role == "user":
                messages.append({"role": "user", "content": str(entry.content)})
                continue
            if entry.role == "assistant":
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": str(entry.content) if entry.content else "",
                }
                if entry.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in entry.tool_calls
                    ]
                messages.append(message)
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry.tool_call_id,
                    "content": json.dumps(entry.content),
                }
            )
        return messages

    def _build_tools(self) -> list[dict[str, Any]]: 
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in self.tools
        ]


def _safe_json_loads(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
