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

        self.messages = [
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':initial_user_message},
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.tools = tools

    def _call_model(self) -> AssistantResponse:
        tool_calls = []
        tool_call = None
        if np.random.random()>0.2:        
            tool_call = ToolInvocation(
                tool_call_id="mock_abc123",
                name="inspect_cell",
                arguments={"option_id": "bronze_plus", "attribute_id": "monthly_premium"},
            )
            tool_calls.append(tool_call)

        if np.random.random()>0.2:        
            tool_call = ToolInvocation(
                tool_call_id="mock_abc124",
                name="inspect_cell",
                arguments={"option_id": "bronze_plus", "attribute_id": "monthly_premium"},
            )
            tool_calls.append(tool_call)

        text = "Using the next available tool." if tool_call else "Stopping."
        input_tokens = estimate_tokens_from_text(self.messages)
        output_tokens=estimate_tokens_from_text(text, tool_call.arguments if tool_call else {})
        # print(input_tokens)
        # print(output_tokens)
        return AssistantResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=dict(role='assistant',content=text, tool_calls=tool_calls)
        )


    def _get_tool_error(self, tool_call):
        return {
            'role':'tool',
            'tool_call_id': tool_call.tool_call_id,
            'content':'Error: you are allowed to call only one tool per turn'
        }





def estimate_tokens_from_text(*parts: Any) -> int:
    combined = " ".join(json.dumps(_to_serializable(part), sort_keys=True) for part in parts)
    return max(1, len(combined) // 4)