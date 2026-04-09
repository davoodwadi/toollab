from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.base import AssistantResponse, ToolInvocation

from google import genai
from google.genai import types

class GoogleModelSession:
    provider_name = "google"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
        tools: list[ToolDefinition],
    ) -> None:
        api_key_env = config.api_key_env or "GEMINI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {api_key_env}")
        self._client = genai.Client(api_key=api_key)

        self._tool_names = {str(tool["name"]) for tool in tools}
        self._inspection_count = 0

        self.contents = [
            types.Content(
                role="user", parts=[types.Part(text=initial_user_message)]
            )
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message
        self.tools = tools
        # print('tools', tools)
        # exit()
        self.google_tools = types.Tool(function_declarations=tools)
        self.google_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tools],
        )


    def _build_tools(self) -> list[Any]:
        from google.genai import types
        
        declarations = []
        for tool in self.tools:
            declarations.append(types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool.get("input_schema")
            ))
        
        if not declarations:
            return []
        return [types.Tool(function_declarations=declarations)]

    def _build_contents(self) -> list[Any]:
        from google.genai import types
        contents = []
        
        # skip system prompt at index 0
        current_user_parts = []

        for msg in self.messages[1:]:
            role = msg["role"]
            if role == "user":
                current_user_parts.append(types.Part.from_text(text=str(msg["content"])))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                
                # find the name from previous assistant tool calls
                tool_name = "unknown"
                for prev_msg in reversed(self.messages):
                    if prev_msg["role"] == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc["id"] == tool_call_id:
                                tool_name = tc["function"]["name"]
                                break
                        if tool_name != "unknown":
                            break
                            
                content_str = msg.get("content", "{}")
                try:
                    response_obj = json.loads(content_str)
                except json.JSONDecodeError:
                    response_obj = {"result": content_str}
                    
                current_user_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response=response_obj,
                        id=tool_call_id
                    )
                ))
            elif role == "assistant":
                # flush accumulated user parts to contents
                if current_user_parts:
                    contents.append(types.Content(role="user", parts=current_user_parts))
                    current_user_parts = []
                
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=str(msg["content"])))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        args_dict = {}
                        try:
                            args_dict = json.loads(tc["function"]["arguments"])
                        except Exception:
                            pass
                        
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=tc["function"]["name"],
                                args=args_dict,
                                id=tc.get("id")
                            )
                        ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
        
        # Add any trailing user parts
        if current_user_parts:
            contents.append(types.Content(role="user", parts=current_user_parts))
            
        return contents

    def _call_model(self) -> AssistantResponse:

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self._build_tools(),
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

        response = self._client.models.generate_content(
            model=self.config.model_name,
            contents=self._build_contents(),
            config=config,
        )
        
        text_parts = []
        tool_calls = []
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
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
        
        text = "\n".join(text_parts).strip()
        
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
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
            ] if tool_calls else None,
        }
        self.messages.append(assistant_msg)
        
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            
        finish_reason = None
        if response.candidates and hasattr(response.candidates[0].finish_reason, 'name'):
            finish_reason = response.candidates[0].finish_reason.name

        return AssistantResponse(
            content=text,
            reasoning=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason
        )

    def _get_tool_error_one_tool_only(self, tool_call: ToolInvocation) -> dict[str, str]:
        return {
            'role': 'tool',
            'tool_call_id': tool_call.tool_call_id,
            'content': 'Error: you are allowed to call only one tool per turn'
        }
