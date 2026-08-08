from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional, get_args
from pydantic import BaseModel, Field, create_model
from uuid import uuid4
from tqdm import tqdm

from tool_lab.config import ModelConfig
from tool_lab.models.base import AssistantResponse, ToolInvocation
 
from google import genai 
from google.genai import types
from google.genai import errors


class LikertResponse(BaseModel):
    value: int = Field(
                description="Rating from 1 (Strongly disagree) to 7 (Strongly agree)."
    )

class BooleanResponse(BaseModel):
    value: bool


# if __name__=='__main__':
#     print(google_tools)

class GoogleModelSession:
    provider_name = "google"

    def __init__(
        self,
        config: ModelConfig,
        system_prompt: str,
        initial_user_message: str,
        environment
    ) -> None:
        self.clients = []
        if os.environ.get('GEMINI_API_KEY'):
            # print(os.environ.get('GEMINI_API_KEY'))
            # exit()
            self.clients.append(genai.Client(vertexai=False, api_key = os.environ.get('GEMINI_API_KEY')))
        try:
            self.clients.append(genai.Client(vertexai=True, location='global'))
        except Exception:
            pass
        
        # for client in self.clients:
        #     try:
        #         response = client.models.generate_content(
        #                             model='ggemini-3.1-pro-preview',
        #                             contents='Count to 50',
        #                         )
        #     except errors.ClientError as e:
        #         print(client.vertexai, e.code, type(e.code))
        #     # print(response.text)
        #     print(client.vertexai)
        # exit()

        self.current_client_idx = 0
        
        self.contents = [
            types.Content(
                role="user", parts=[types.Part(text=initial_user_message)]
            )
        ]

        self.config = config
        self.system_prompt = system_prompt
        self.initial_user_message = initial_user_message

        if environment.spec.inspect_mode=='cell':
            class InspectInput(BaseModel):
                option_id: str
                attribute_id: str

            inspect_tool = types.FunctionDeclaration(
                name="inspect",
                description="Reveal one hidden cell in the fixed information matrix by option and attribute.",
                parameters=InspectInput.model_json_schema(),
            )
        elif environment.spec.inspect_mode=='full':
            class InspectInput(BaseModel):
                option_id: str

            inspect_tool = types.FunctionDeclaration(
                name="inspect",
                description="Reveal the full details of one option in the fixed information matrix.",
                # strict= True,
                parameters=InspectInput.model_json_schema(),
            )
        else:
            raise ValueError('invalid inspect_mode', environment.spec.inspect_mode)


        self.attributes = environment.attributes

        # submit_choice
        SubmitChoiceInput = create_model(
            "SubmitChoiceInput",
            option_id=(str, ...),
        )

        submit_choice_tool = types.FunctionDeclaration(
            name="submit_choice",
            description="Record the final decision.",
            parameters=SubmitChoiceInput.model_json_schema(),
        )
        
        google_tools = types.Tool(function_declarations=[submit_choice_tool, inspect_tool])

        self.google_tools = google_tools

        self.metadata = environment.spec.metadata

    def _generate_content_with_retry(self, config):
        response = None
        max_attempts = max(1, len(self.clients) * 2)
        
        for attempt in range(max_attempts):
            client = self.clients[self.current_client_idx]
            try:
                response = client.models.generate_content(
                    model=self.config.model_name,
                    contents=self.contents,
                    config=config,
                )
                break
            except errors.ClientError as e:
                if e.code>=400:
                    print(f"Warning: Client {self.current_client_idx} got {e.code}. Switching client.")
                    self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
                    if attempt == max_attempts - 1:
                        raise e
                    continue
                else:
                    raise e
        return response

    def _call_model(self) -> AssistantResponse:
        
        meta = {'thinking_level': None}
        if self.metadata.get('thinking_level'):
            meta['thinking_level'] = self.metadata.get('thinking_level')

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=[self.google_tools],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=meta.get('thinking_level')
            )
        )
        
        response = self._generate_content_with_retry(config)
                    
        if response and response.candidates and response.candidates[0].content:
            self.contents.append(response.candidates[0].content)

        # print(response)
        # exit()

        text_parts = []
        reasoning_parts = []
        tool_calls = []
        
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
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
        if response and hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            
        finish_reason = None
        if response and response.candidates and hasattr(response.candidates[0].finish_reason, 'name'):
            finish_reason = str(response.candidates[0].finish_reason)

        return AssistantResponse(
            content=text,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            meta=meta,
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

    def administer_survey(self):
        survey_data = {}

        meta = {'thinking_level': None}
        if self.metadata.get('thinking_level'):
            meta['thinking_level'] = self.metadata.get('thinking_level')

        # 0. Ask Most Important Attribute
        print("Collecting Most Important Attribute")
        prompt = "Which attribute influenced your choice most?"
        self.contents.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

        if self.attributes:
            attribute_keys = tuple(self.attributes.keys())
            MostImportantAttributeResponse = create_model(
                "MostImportantAttributeResponse",
                value=(Literal[attribute_keys], Field(description="The attribute that influenced your choice most."))
            )
        else:
            MostImportantAttributeResponse = create_model(
                "MostImportantAttributeResponse",
                value=(str, Field(description="The attribute that influenced your choice most."))
            )

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            response_mime_type="application/json",
            response_schema=MostImportantAttributeResponse,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=meta.get('thinking_level')
            )
        )
        
        response = self._generate_content_with_retry(config)

        if response and response.candidates and response.candidates[0].content:
            self.contents.append(response.candidates[0].content)

        try:
            if response and hasattr(response, 'parsed') and response.parsed is not None:
                survey_data['most_important_attribute'] = response.parsed.value
            elif response and response.text:
                survey_data['most_important_attribute'] = MostImportantAttributeResponse.model_validate_json(response.text).value
            else:
                survey_data['most_important_attribute'] = None
        except Exception:
            survey_data['most_important_attribute'] = None
        print(survey_data['most_important_attribute'])

        likert_questions = {
            'ulterior_motive': "At least one reviewer has an ulterior motive.",
            'suspicious_statements': "At least one of the reviews contains suspicious statements.",
            'exaggerate_performance': "At least one reviewer is motivated to exaggerate the performance of their review."
        }

        # 1. Ask Likert Questions
        for key, statement in tqdm(likert_questions.items(), desc="Collecting Likert Responses", leave=False):
            prompt = f'''Based on the reviews you read answer the following question on a 7-point Likert scale: 
"{statement}"
1 (Strongly disagree) to 7 (Strongly agree).
'''
            self.contents.append(
                types.Content(role="user", parts=[types.Part(text=prompt)])
            )
            
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=LikertResponse,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=meta.get('thinking_level')
                )
            )
            
            response = self._generate_content_with_retry(config)
            
            if response and response.candidates and response.candidates[0].content:
                self.contents.append(response.candidates[0].content)
            
            try:
                if response and hasattr(response, 'parsed') and response.parsed is not None:
                    survey_data[key] = response.parsed.value
                elif response and response.text:
                    survey_data[key] = LikertResponse.model_validate_json(response.text).value
                else:
                    survey_data[key] = None
            except Exception:
                survey_data[key] = None

        # 2. Ask True/False Manipulation Checks
        manipulation_checks = {
            'manipulation_sponsored': "At least 1 review was sponsored.",
            'manipulation_positive': "At least 1 review was positive.",
            'manipulation_negative': "At least 1 review was negative."
        }

        for key, statement in tqdm(manipulation_checks.items(), desc="Collecting Manipulation Checks", leave=False):
            prompt = f'''Based on the reviews you read, answer the following True or False question: 
"{statement}"
'''
            self.contents.append(
                types.Content(role="user", parts=[types.Part(text=prompt)])
            )
            
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=BooleanResponse,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=meta.get('thinking_level')
                )
            )
            
            response = self._generate_content_with_retry(config)
            
            if response and response.candidates and response.candidates[0].content:
                self.contents.append(response.candidates[0].content)
            
            try:
                if response and hasattr(response, 'parsed') and response.parsed is not None:
                    survey_data[key] = response.parsed.value
                elif response and response.text:
                    survey_data[key] = BooleanResponse.model_validate_json(response.text).value
                else:
                    survey_data[key] = None
            except Exception:
                survey_data[key] = None

        return survey_data

