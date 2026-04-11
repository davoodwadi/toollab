from __future__ import annotations

import json
import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.models.base import AssistantResponse, ToolInvocation
 
from openai import OpenAI


class SubmitChoiceInput(BaseModel):
    option_id: str
    # confidence: Optional[float] = Field(default=None, ge=0, le=1)
    # justification: Optional[str] = None

class InspectCellInput(BaseModel):
    option_id: str
    attribute_id: str


submit_choice_tool = dict(
    type= "function",
    name="submit_choice",
    description="Record the final decision.",
    # strict= True,
    parameters=SubmitChoiceInput.model_json_schema(),
)

inspect_cell_tool = dict(
    type= "function",
    name="inspect_cell",
    description="Reveal one hidden cell in the fixed information matrix by option and attribute.",
    # strict= True,
    parameters=InspectCellInput.model_json_schema(),
)

tools = [submit_choice_tool, inspect_cell_tool]
 
if __name__=='__main__':
    client = OpenAI(
        base_url='http://127.0.0.1:8080/v1',
        api_key='none',
    )
    messages = [
        {'role':'user', 'content':'Call one of the tools.'}
    ]
    model_name = client.models.list().data[0].id
    print(model_name)
    response = client.responses.create(
        model=model_name,
        tools=tools,
        input=messages,
        max_output_tokens=1700
    )
    for output in response.output:
        print(output)

class LlamaCPPModelSession:
    provider_name = "llamacpp"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
    ) -> None:
        self._client = OpenAI(
            base_url='http://127.0.0.1:8080/v1',
            api_key='none',
        )
        self.messages = [
            {'role':"user", "content":initial_user_message}
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.tools = tools

    def _call_model(self) -> AssistantResponse:
        # for tool in self.tools:
            # print(tool)
        # exit()
        meta = {}
        assert self._client.models.list().data[0].id==self.config.model_name, f'the hosted model {self._client.models.list().data[0].id} is different {self.config.model_name}'
        # print('self.config.model_name', self.config.model_name)
        response = self._client.responses.create(
            model=self.config.model_name,
            tools=self.tools,
            input=self.messages,
            instructions=self.system_prompt,
            parallel_tool_calls=False,
        )
        # print(response)

        meta["model_version"] = response.model

        if response.output:
            self.messages.extend(response.output)

        # exit()

        text_parts = []
        reasoning_parts = []
        tool_calls = []
    
        for item in response.output:
            # print(item)
            # print('*'*50)
            if item.type == 'reasoning':
                reasoning_parts.append(item.content[0].text)
            elif item.type == 'output_text':
                text_parts.append(item.content[0].text)
            if item.type == "function_call":
                tc_id = item.call_id if item.call_id else f"call_{uuid4().hex[:8]}"
                # print(item.arguments)
                # print(type(item.arguments))
                # print('*'*50)
                
                args_dict = json.loads(item.arguments)

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
            finish_reason=response.status,
            meta=meta
        )

    def add_tool_results(self, tool_results_and_name):
        for tool_response, tool_name in tool_results_and_name:
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

