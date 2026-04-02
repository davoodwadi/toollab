from __future__ import annotations

from typing import Any
import json
import os

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.base import AssistantResponse, BaseModelSession, ToolInvocation


class AnthropicModelSession(BaseModelSession):
    provider_name = "anthropic"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
        tools: list[ToolDefinition],
    ) -> None:
        super().__init__(config, system_prompt, initial_user_message, tools)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is not installed. Run `python -m pip install -e .`."
            ) from exc

        api_key_env = config.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
        self._client = Anthropic(api_key=api_key)

    def _invoke_model(self) -> AssistantResponse:
        response = self._client.messages.create(
            model=self.config.model_name,
            system=self.system_prompt,
            messages=self._build_messages(),
            tools=self._build_tools(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        text_parts: list[str] = []
        tool_calls: list[ToolInvocation] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
                continue
            if block_type == "tool_use":
                tool_calls.append(
                    ToolInvocation(
                        tool_call_id=getattr(block, "id"),
                        name=getattr(block, "name"),
                        arguments=getattr(block, "input", {}) or {},
                    )
                )
        usage = getattr(response, "usage", None)
        return AssistantResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw={"id": getattr(response, "id", None)},
        )

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                messages.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for entry in self.transcript:
            if entry.role == "user":
                flush_tool_results()
                messages.append({"role": "user", "content": str(entry.content)})
                continue
            if entry.role == "assistant":
                flush_tool_results()
                content: list[dict[str, Any]] = []
                if entry.content:
                    content.append({"type": "text", "text": str(entry.content)})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.tool_call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in entry.tool_calls
                )
                messages.append({"role": "assistant", "content": content})
                continue
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": entry.tool_call_id,
                    "content": json.dumps(entry.content),
                }
            )

        flush_tool_results()
        return messages

    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in self.tools
        ]
