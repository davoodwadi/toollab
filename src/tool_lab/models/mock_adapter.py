from __future__ import annotations

from typing import Any
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.base import (
    ToolInvocation,
    AssistantResponse,
    _to_serializable,
)

import numpy as np
import json
import time

class MockModelSession:
    provider_name = "mock"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str, 
        tools: list[ToolDefinition],
    ) -> None:
        self._tool_names = {str(tool["name"]) for tool in tools}
        self._behavior = str(config.extra.get("mock_behavior", "submit_after_few_tools"))
        self._max_mock_inspections = int(config.extra.get("mock_inspections", 4))
        self._inspection_count = 0

        self._option_ids = self._parse_section_ids(initial_user_message, "Options:")
        self._attribute_ids = self._parse_section_ids(initial_user_message, "Attributes available in this task:")

        self.messages = [
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':initial_user_message},
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.tools = tools

    @staticmethod
    def _parse_section_ids(prompt: str, section_header: str) -> list[str]:
        ids = []
        in_section = False
        for line in prompt.splitlines():
            if line.strip() == section_header:
                in_section = True
                continue
            if in_section:
                if not line.strip():
                    break
                if line.startswith("- "):
                    ids.append(line[2:].split(":")[0].strip())
        return ids

    def _call_model(self) -> AssistantResponse:
        # submit_choice if we've hit the inspection limit, or randomly (~5% chance)
        should_submit = (
            self._inspection_count >= (self._max_mock_inspections+8)
            or np.random.random() < 0.15
        )
        print('_max_mock_inspections', self._inspection_count, self._max_mock_inspections)

        if should_submit and self._option_ids:
            option_id = self._option_ids[np.random.randint(len(self._option_ids))]
            tool_calls = [
                ToolInvocation(
                    tool_call_id=f"mock_{uuid4().hex[:8]}",
                    name="submit_choice",
                    arguments={"option_id": option_id, "confidence": round(float(np.random.uniform(0.5, 1.0)), 2)},
                ) 
            ]
        else:
            option_id = self._option_ids[np.random.randint(len(self._option_ids))]
            attribute_id = self._attribute_ids[np.random.randint(len(self._attribute_ids))]
            tool_calls = [
                ToolInvocation(
                    tool_call_id=f"mock_{uuid4().hex[:8]}",
                    name="inspect_cell",
                    arguments={"option_id": option_id, "attribute_id": attribute_id},
                )
            ]
            self._inspection_count += 1

        # maybe reasoning
        reasoning = None
        if np.random.random() > 0.2:
            reasoning = 'Let me think...'

        text = "Submitting choice." if tool_calls[0].name == "submit_choice" else "Inspecting a cell."

        input_tokens = estimate_tokens_from_text(self.messages)
        output_tokens = estimate_tokens_from_text(text, tool_calls[0].arguments)
        response_raw = {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": tc.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in tool_calls
                        ] or None,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        self.messages.append(response_raw["choices"][0]["message"])
        tool_calls_response = response_raw['choices'][0]['message']['tool_calls']
        if tool_calls_response:
            tool_name = tool_calls_response[0]['function']['name']
        else:
            tool_name = None
        return AssistantResponse(
            content=text,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=response_raw['choices'][0]['finish_reason']
        )



    def _get_tool_error_one_tool_only(self, tool_call):
        return {
            'role':'tool',
            'tool_call_id': tool_call.tool_call_id,
            'content':'Error: you are allowed to call only one tool per turn'
        }





def estimate_tokens_from_text(*parts: Any) -> int:
    combined = " ".join(json.dumps(_to_serializable(part), sort_keys=True) for part in parts)
    return max(1, len(combined) // 4)