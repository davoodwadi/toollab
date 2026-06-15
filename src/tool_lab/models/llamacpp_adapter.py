from __future__ import annotations

import json
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from uuid import uuid4

from tool_lab.config import ModelConfig
from tool_lab.models.base import AssistantResponse, ToolInvocation
 
from openai import OpenAI

import os
import re
import time
import signal
import requests
import subprocess
from pathlib import Path


class LlamaCppClient:
    BASE_URL = "http://127.0.0.1:8080/v1"
    LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
    # LLAMA_CACHE = os.environ.get("LLAMA_CACHE")
    LLAMA_CACHE = os.environ.get("LLAMA_CACHE", os.path.expanduser("~/data/.lcpp_cache"))

    def __init__(self, config):
        self.config = config
        print(f'Cache Dir: {self.LLAMA_CACHE}')

        self._ensure_server()

    # ------------------------------------------------------------------
    # server lifecycle
    # ------------------------------------------------------------------

    def _ensure_server(self):
        """Guarantee the server is running with the correct model."""
        running_model = self._get_running_model()
        print('_ensure_server', 'running_model', running_model)
        print('_ensure_server', 'self.config.model_name', self.config.model_name)
        if running_model is None:
            print("[llama] Server not running – starting...")
            self._start_server()


        elif running_model != self.config.model_name:
            print(f"[llama] Server running wrong model ({running_model!r}) – restarting...")
            self._kill_server()
            self._start_server()

        else:
            print(f"[llama] Server already serving {running_model!r} – reusing.")

    def _get_running_model(self) -> str | None:
        """
        Return the model name currently served, or None if the server is
        unreachable / not responding.
        """
        try:
            resp = requests.get(f"{self.BASE_URL}/models", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible response: {"data": [{"id": "<model>", ...}]}
            return data["data"][0]["id"]
        except Exception:
            return None

    def _kill_server(self):
        """Kill every llama-server process and wait for the port to free."""
        result = subprocess.run(
            ["pkill", "-f", "llama-server"],
            capture_output=True,
        )
        print('Killed the running LlamaCPP', result)
        # give the OS a moment to release the port
        time.sleep(1)


    def _list_available_models(self) -> str:
        """Human-readable list of every .gguf found in LLAMA_CACHE."""
        cache = Path(self.LLAMA_CACHE)
        paths = sorted(cache.rglob("*.gguf"))
        if not paths:
            return "  (none found)"
        return "\n".join(f"  {p.stem}  [{p.parent.name}]" for p in paths)

    def _find_model_path(self, model_name: str) -> Path:
        """
        Recursively search LLAMA_CACHE for a .gguf file whose stem matches
        model_name (case-insensitive).

        Directory structure assumed:
            $LLAMA_CACHE/
                <model-dir>/
                    <model-file>.gguf
        """
        cache = Path(self.LLAMA_CACHE)
        # matches = list(cache.rglob(f"{model_name}"))
        matches = [
            p for p in cache.rglob("*.gguf")
            if p.name.lower() == model_name.lower()
        ]
        # print('matches', matches)        

        if not matches:
            raise FileNotFoundError(
                f"No .gguf file found for model {model_name!r} under {cache}\n"
                f"Available models:\n{self._list_available_models()}"
            )

        if len(matches) > 1:
            # prefer an exact stem match over a glob hit
            exact = [p for p in matches if p.name == model_name]
            if len(exact) == 1:
                return exact[0]
            raise ValueError(
                f"Ambiguous model name {model_name!r}, matches:\n"
                + "\n".join(f"  {p}" for p in matches)
            )

        return matches[0]

    def _build_server_command(self) -> list[str]:
        """Build the llama-server argv list from config."""
        model_path = self._find_model_path(self.config.model_name)
        # print('model_path', model_path)
        parallel = 1 if '31B' in self.config.model_name else 4

        cmd = [
            self.LLAMA_SERVER_BIN,
            "-m", model_path.as_posix(),
            "-ngl", str(999),
            "--parallel", str(parallel),
        ]
        return cmd

    def _start_server(self):
        """Spawn llama-server as a background process and wait until ready."""
        cmd = self._build_server_command()
        print(f"[llama] Starting: {' '.join(cmd)}")

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,   # detach so it survives this process if needed
        )

        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 120, poll_interval: float = 2.0):
        """Block until the server responds or timeout is reached."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._get_running_model() is not None:
                print("[llama] Server is ready.")
                return
            time.sleep(poll_interval)

        raise TimeoutError(
            f"llama-server did not become ready within {timeout}s"
        )





 
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
        environment
    ) -> None:
        # run the server
        LlamaCppClient(config)

        self._client = OpenAI(
            base_url='http://127.0.0.1:8080/v1',
            api_key='none',
        )
        # print('self._client', self._client)
        self.messages = [
            {'role':"user", "content":initial_user_message}
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message

        # see if the endpoint 'http://127.0.0.1:8080/v1' is running

        # if yes, check to see if it is serving self.config.model_name
        # if not serving self.config.model_name -> kill process and serve self.config.model_name

        # if no, serve self.config.model_name

        # self.config.model_name

        if environment.spec.inspect_mode=='cell':
            class InspectInput(BaseModel):
                option_id: str
                attribute_id: str

            inspect_tool = dict(
                type= "function",
                name="inspect",
                description="Reveal one hidden cell in the fixed information matrix by option and attribute.",
                parameters=InspectInput.model_json_schema(),
            )
        elif environment.spec.inspect_mode=='full':
            class InspectInput(BaseModel):
                option_id: str

            inspect_tool = dict(
                type= "function",
                name="inspect",
                description="Reveal the full details of one option in the fixed information matrix.",
                parameters=InspectInput.model_json_schema(),
            )
        else:
            raise ValueError('invalid inspect_mode', environment.spec.inspect_mode)

        class SubmitChoiceInput(BaseModel):
            option_id: str
            most_important_attribute: str = Field(
                description="The attribute that influenced your choice most."
            )
            confidence_score: int = Field(
                description="Your confidence in this decision from 1 (completely guessing) to 5 (absolutely certain)."
            )

        submit_choice_tool = dict(
            type= "function",
            name="submit_choice",
            description="Record the final decision.",
            # strict= True,
            parameters=SubmitChoiceInput.model_json_schema(),
        )
        tools = [submit_choice_tool, inspect_tool]

        self.tools = tools

        assert self._client.models.list().data[0].id==self.config.model_name, f'the hosted model {self._client.models.list().data[0].id} is different {self.config.model_name}'


    def _call_model(self) -> AssistantResponse:
        # for tool in self.tools:
            # print(tool)
        meta = {}
        # print('self.config.model_name', self.config.model_name)
        # print(self.tools)
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

        if reasoning_parts:
            reasoning = "\n".join(reasoning_parts)
        else:
            reasoning = None

        
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

