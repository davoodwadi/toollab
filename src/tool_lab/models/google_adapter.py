from __future__ import annotations

import json
import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.models.base import AssistantResponse, ToolInvocation
 
from google import genai 
from google.genai import types


class SubmitChoiceInput(BaseModel):
    option_id: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    justification: Optional[str] = None

class InspectCellInput(BaseModel):
    option_id: str
    attribute_id: str


submit_choice_tool = types.FunctionDeclaration(
    name="submit_choice",
    description="Record the final decision.",
    parameters=SubmitChoiceInput.model_json_schema(),
)

inspect_cell_tool = types.FunctionDeclaration(
    name="inspect_cell",
    description="Reveal one hidden cell in the fixed information matrix by option and attribute.",
    parameters=InspectCellInput.model_json_schema(),
)

google_tools = types.Tool(function_declarations=[submit_choice_tool, inspect_cell_tool])

if __name__=='__main__':
    print(google_tools)

class GoogleModelSession:
    provider_name = "google"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
    ) -> None:
        api_key_env = config.api_key_env or "GEMINI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
        self._client = genai.Client(api_key=api_key)


        self.contents = [
            types.Content(
                role="user", parts=[types.Part(text=initial_user_message)]
            )
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.google_tools = google_tools


    def _call_model(self) -> AssistantResponse:
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[self.google_tools],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )

        response = self._client.models.generate_content(
            model=self.config.model_name,
            contents=self.contents,
            config=config,
        )
        
        if response.candidates and response.candidates[0].content:
            self.contents.append(response.candidates[0].content)

        # print(response)
        # exit()

        text_parts = []
        reasoning_parts = []
        tool_calls = []
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    if part.thought==True:
                        reasoning_parts.append(part.text)
                    else:
                        text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    tc_id = fc.id if fc.id else f"call_{uuid4().hex[:8]}"
                    args_dict = {}
                    if hasattr(fc.args, 'model_dump'):
                        args_dict = fc.args.model_dump()
                    elif isinstance(fc.args, dict):
                        args_dict = fc.args
                    elif hasattr(fc.args, 'items'):
                        args_dict = dict(fc.args.items())
                        
                    tool_calls.append(ToolInvocation(
                        tool_call_id=tc_id,
                        name=fc.name,
                        arguments=args_dict
                    ))
        
        text = "\n".join(text_parts)
        reasoning = "\n".join(reasoning_parts)
        
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            
        finish_reason = None
        if response.candidates and hasattr(response.candidates[0].finish_reason, 'name'):
            finish_reason = str(response.candidates[0].finish_reason)

        return AssistantResponse(
            content=text,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason
        )
    
    def add_tool_results(self, tool_results_and_name):
        for tool_response, tool_name in tool_results_and_name:
            if tool_response.get('error'):
                function_response_parts = [
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"error": tool_response['error']},
                    )
                ]
            elif tool_response.get('content'):
                function_response_parts = [
                        types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_response['content']},
                    )
                ]

            self.contents.append(types.Content(role="tool", parts=function_response_parts)) 


    def _get_tool_error_one_tool_only(self, tool_call: ToolInvocation) -> dict[str, str]:
        return {
            'tool_call_id': tool_call.tool_call_id,
            'error': 'Error: You are allowed to call only one tool per turn'
        }

    def add_force_message(self, force_message):
        self.contents.append(
            types.Content(
                role="user", parts=[types.Part(text=force_message)]
        ))

    def add_reminder(self, reminder_message):
        self.contents.append(
            types.Content(
                role="user", parts=[types.Part(text=reminder_message)]
        ))

