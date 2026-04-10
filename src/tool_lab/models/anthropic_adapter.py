from __future__ import annotations

import json
import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.models.base import AssistantResponse, ToolInvocation
 
import anthropic


class SubmitChoiceInput(BaseModel):
    option_id: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    justification: Optional[str] = None

class InspectCellInput(BaseModel):
    option_id: str
    attribute_id: str


submit_choice_tool = dict(
    name="submit_choice",
    description="Record the final decision.",
    # strict= True,
    input_schema=SubmitChoiceInput.model_json_schema(),
)

inspect_cell_tool = dict(
    name="inspect_cell",
    description="Reveal one hidden cell in the fixed information matrix by option and attribute.",
    # strict= True,
    input_schema=InspectCellInput.model_json_schema(),
)

tools = [submit_choice_tool, inspect_cell_tool]
 
if __name__=='__main__':
    api_key_env = "ANTHROPIC_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
    
    client = anthropic.Anthropic(api_key=api_key)
    messages = [
        {'role':'system', 'content':'think before you answer'},
        {'role':'user', 'content':'tell me a story'}
    ]
    response = client.messages.create(
        model='claude-sonnet-4-6',
        # tools=tools,
        messages=messages,
        max_tokens=17
    )
    print(response)

class AnthropicModelSession:
    provider_name = "openai"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
    ) -> None:
        api_key_env = config.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
        self._client = anthropic.Anthropic(api_key=api_key)

        self.messages = [
            {'role':"user", "content":initial_user_message},
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.tools = tools


    def _call_model(self) -> AssistantResponse:
        # for tool in self.tools:
        #     print(tool)
        # exit()
        response = self._client.messages.create(
            model=self.config.model_name,
            tools=self.tools,
            messages=self.messages,
            system=self.system_prompt,
            max_tokens=6_000,
        )
        print(response)

        model_version = response.model
        
        self.messages.append(response)

        # exit()

        text_parts = []
        reasoning_parts = []
        tool_calls = []
    
        for item in response.content:
            if item.type=='text':
                text_parts.append(item.text)
            if item.type == 'tool_use':
                tc_id = item.id if item.id else f"call_{uuid4().hex[:8]}"
                args_dict = item.input

                tool_calls.append(ToolInvocation(
                    tool_call_id=tc_id,
                    name=item.name,
                    arguments=args_dict
                ))
        
        text = "\n".join(text_parts)
        reasoning = "\n".join(reasoning_parts)
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        return AssistantResponse(
            content=text,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            meta={model_version:model_version}
        )

    def add_tool_response(self, tool_response, tool_name):
        # print('tool_response', tool_response)
        if tool_response.get('error'):
            function_call_output = {
                "type": "function_call_output",
                "call_id": tool_response['tool_call_id'],
                "output": tool_response['error']
            }
        elif tool_response.get('content'):
            function_call_output = {
                "type": "function_call_output",
                "call_id": tool_response['tool_call_id'],
                "output": tool_response['content']
            }

        self.messages.append(function_call_output) 


    def _get_tool_error_one_tool_only(self, tool_call: ToolInvocation) -> dict[str, str]:
        return {
            'tool_call_id': tool_call.tool_call_id,
            'error': 'Error: You are allowed to call only one tool per turn'
        }

    def add_force_message(self, force_message):
        self.messages.append(
            {'role':"user", 'content':force_message}
        )
        
    def add_reminder(self, reminder_message):
        self.messages.append(
            {'role':"user", 'content':reminder_message}
        )

