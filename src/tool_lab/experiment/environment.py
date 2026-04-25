from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from math import log
from typing import Any
import json

from tool_lab.models.base import (
    AssistantResponse,
    _to_serializable,
)
from tool_lab.config import AttributeSpec, CueSpec, ExperimentSpec, OptionSpec
from tool_lab.models.base import AssistantResponse

import random

def build_environment(spec: ExperimentSpec) -> "ToolLabEnvironment":
    return FixedMatrixEnvironment(spec)



class ToolLabEnvironment(ABC):
    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec
        self.options: dict[str, OptionSpec] = {option.id: option for option in spec.options}
        self.attributes: dict[str, AttributeSpec] = {
            attribute.id: attribute for attribute in spec.attributes
        }
        self.cumulative_cost_usd = 0.0
        self.budget_remaining_usd = spec.budget_usd
        self.last_turn_cost_usd: float = 0.0

        self.cumulative_cost_tools = 0
        self.budget_remaining_tools = spec.budget_tools
        self.last_turn_cost_tool: int = 0

        self.cumulative_cost_tool_usd = 0.
        self.last_turn_cost_tool_usd: float = 0. 

        self.cumulative_cost_tokens = 0
        self.budget_remaining_tokens = spec.budget_tokens
        self.last_turn_cost_tokens: int = 0

        self.cumulative_cost_points = 0
        self.budget_remaining_points = spec.budget_points
        self.last_turn_cost_points: int = 0


        self.cues: dict[str, CueSpec] = {cue.id: cue for cue in spec.cues}
        self.opened_cues: set[str] = set()
        self.choice: str | None = None
        self.choice_justification: str = ""
        self.choice_confidence: float | None = None
        self.vote_status: str = "pending"
        self.stop_reason: str = ""
        self.trace: list[dict[str, Any]] = []
        self.model_turns: list[dict[str, Any]] = []
        self.forced_choice_requested = False
        self.awaiting_forced_choice = False
        self._step_index = 0

        self._last_inspect: dict[str, str] | None = None  # {"option_id": ..., "attribute_id": ...}
        self._last_is_revisit: bool = False
        self._last_transition: str | None = None

    def build_system_prompt(self, ) -> str:
        mode_rules = self._mode_rules()

        return (
            "You are a subject in a Tool-Lab decision experiment. "
            "You can use the available tools to gather information or make a decision. "
            "At any point you can record your final decision by calling submit_choice. "
            "You cannot call tools concurrently. Each time you can call ONLY one tool maximum. "
            f"{mode_rules} "
        )

    def build_user_prompt(self, ) -> str:
        option_lines = []
        for option in self.spec.options:
            s = f'- {option.id}'
            if option.display_name:
                s += f': {option.display_name}'
            if option.description:
                s += f" {option.description}"
            option_lines.append(s.strip())            

        attribute_lines = []
        for attribute in self.spec.attributes:
            attr_str = f"- {attribute.id}: "
            if attribute.display_name:
                attr_str+=f'{attribute.display_name}.'
            if attribute.description:
                attr_str+=f' {attribute.description}'
            if attribute.cost_multiplier and self.spec.budget_type=='points':
                cost = self.spec.inspect_cell_tool_cost * attribute.cost_multiplier
                attr_str+=f'(tool cost: {cost})'
            
            attribute_lines.append(attr_str)
        
        attribute_lines_str = '\n'.join(attribute_lines)

        if self.spec.budget_type == 'usd':
            budget_str = f"Budget: ${self.spec.budget_usd:.4f}"
        elif self.spec.budget_type == 'tokens':
            budget_str = f"Budget: {self.spec.budget_tokens} tokens"
        elif self.spec.budget_type == 'tools' and self.spec.budget_tools>0:
            budget_str = f"Budget: {self.spec.budget_tools} tools"
        elif self.spec.budget_type == 'tools' and self.spec.budget_tools<=0:
            # budget_str = f"REMEMBER: You have to make a decision and submit a choice with minimal number of tool calls."
            budget_str = ''
        elif self.spec.budget_type == 'tool_usd':
            budget_str = f"Each `inspect_cell` tool call you make costs the user ${self.spec.inspect_cell_tool_cost}"
        elif self.spec.budget_type == 'points':
            budget_str = f"Your starting points: {self.spec.budget_points}"
        
        if "{inspect_cell_tool_cost}" in self.spec.participant.profile:
            formatted_profile = self.spec.participant.profile.format(inspect_cell_tool_cost=f"${self.spec.inspect_cell_tool_cost}")
        else:
            formatted_profile = self.spec.participant.profile

        user_prompt = [
            formatted_profile,
            'REMEMBER:',
            budget_str,
            '',
            "Options:",
            *option_lines,
            "",
            "Attributes available in this task:",
            attribute_lines_str,
            "",
            "Critical Note:",
            budget_str,
        ]
        return "\n".join(user_prompt)

    def reminder_message(self) -> str:
        return (
            "You did not call any tools. Use one of the available tools to inspect information "
            "or call submit_choice to end the run. Do not answer in plain text alone."
        )

    def forced_vote_message(self) -> str:
        return (
            "Your budget is exhausted. You may not inspect more information. "
            "You have one final chance to submit a vote by calling submit_choice. "
        )

    def get_model_cost(self, message: AssistantResponse) -> dict:
        input_cost = message.input_tokens * (self.spec.model.pricing.input_per_million / 1e6)
        output_cost = message.output_tokens * (self.spec.model.pricing.output_per_million / 1e6)
        cost_usd = {'input_cost':input_cost, 'output_cost':output_cost}
        cost_tools = 0
        cost_points = 0 
        cost_tool_usd = 0.0
        cost_tokens = {'input_tokens':message.input_tokens, 'output_tokens':message.output_tokens}
        if message.tool_calls:
            # print('+'*40)
            # print(message.tool_calls[0])
            tool_name = message.tool_calls[0].name
            if tool_name=='inspect_cell':
                attribute = message.tool_calls[0].arguments['attribute_id']
                
                cost_multiplier_list = [att.cost_multiplier for att in self.spec.attributes if att.id==attribute]
                if cost_multiplier_list:
                    cost_multiplier = cost_multiplier_list[0]
                else:
                    cost_multiplier = 1.

                cost_tools += (1 * cost_multiplier)
                cost_tool_usd += self.spec.inspect_cell_tool_cost
                cost_points += self.spec.inspect_cell_tool_cost * cost_multiplier

        return cost_usd, cost_tools, cost_tokens, cost_points, cost_tool_usd
    
    def apply_model_cost(self, cost_usd: dict, cost_tools: int, cost_tokens: dict, cost_points: int, cost_tool_usd: float) -> None:
        self.cumulative_cost_usd += (cost_usd['input_cost'] + cost_usd['output_cost'])
        self.budget_remaining_usd = self.spec.budget_usd - self.cumulative_cost_usd

        self.cumulative_cost_tools += cost_tools
        self.budget_remaining_tools = self.spec.budget_tools - self.cumulative_cost_tools

        self.cumulative_cost_tool_usd += cost_tool_usd

        self.cumulative_cost_points += cost_points
        self.budget_remaining_points = self.spec.budget_points - self.cumulative_cost_points

        self.cumulative_cost_tokens += (cost_tokens['input_tokens'] + cost_tokens['output_tokens'])
        self.budget_remaining_tokens = self.spec.budget_tokens - self.cumulative_cost_tokens

    def charge_model_turn(self, message: AssistantResponse) -> None:
        cost_usd, cost_tools, cost_tokens, cost_points, cost_tool_usd = self.get_model_cost(message)
        self.apply_model_cost(cost_usd, cost_tools, cost_tokens, cost_points, cost_tool_usd)
        message.input_cost = cost_usd['input_cost']
        message.output_cost = cost_usd['output_cost']
        message.tool_cost = cost_tools
        message.point_cost = cost_points
        message.tool_usd_cost = cost_tool_usd

        self.last_turn_cost_usd = cost_usd['input_cost'] + cost_usd['output_cost']
        self.last_turn_cost_tool = cost_tools
        self.last_turn_cost_tool_usd = cost_tool_usd
        self.last_turn_cost_points = cost_points
        self.last_turn_cost_tokens = cost_tokens['input_tokens'] + cost_tokens['output_tokens']

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        try:
            if tool_name == "submit_choice":
                payload = self._submit_choice(arguments, tool_call_id)
            elif tool_name == "inspect_cell":
                payload = self._inspect_cell(arguments, tool_call_id)
            else:
                raise ValueError(f"Unsupported tool: {tool_name}")
        except Exception as exc:
            payload = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "error": json.dumps({"error": str(exc)}),
            }

        extra = {}
        if tool_name == "inspect_cell":
            extra["is_revisit"] = self._last_is_revisit
            extra["transition"] = self._last_transition

        # exit()
        payload_data = payload.get("content") if payload.get("content") else payload.get("error")
        self._record_event(
            kind="tool",
            data={
                "tool_name": tool_name, 
                "tool_call_id": tool_call_id, 
                **json.loads(payload_data), 
                **extra
            },
        )
        return payload

    def _submit_choice(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        option_id = str(arguments["option_id"])
        if option_id not in self.options:
            raise ValueError(f"Unknown option_id: {option_id}")

        # --- THIS IS THE CRITICAL FIX ---
        # Map back to the original option id (e.g. "coffee_a") for the final results trace
        original_id = self.options[option_id].metadata.get('original_id', option_id)
        self.choice = original_id 
        # --------------------------------

        payload = {
            "role": "tool",
            "tool_call_id": tool_call_id,
        }
        
        if self.options[option_id].display_name:
            arguments['option_label'] = self.options[option_id].display_name
        
        payload['content'] = json.dumps(arguments)
        
        return payload

    def _mode_rules(self) -> str:
        return (
            "You may reveal any cell with inspect_cell."
        )

    def _record_event(
        self,
        *,
        kind: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        self._step_index += 1
        event = {
            "step_index": self._step_index,
            "kind": kind,

            "cumulative_cost_usd": round(self.cumulative_cost_usd, 8),
            "budget_remaining_usd": round(self.budget_remaining_usd, 8),
            
            "cumulative_cost_tools": self.cumulative_cost_tools,
            "budget_remaining_tools":self.budget_remaining_tools,

            "cumulative_cost_points": self.cumulative_cost_points,
            "budget_remaining_points":self.budget_remaining_points,
            
            "cumulative_cost_tokens": self.cumulative_cost_tokens,
            "budget_remaining_tokens":self.budget_remaining_tokens,
            
            **data,
        }

        self.trace.append(event)
        return event

class FixedMatrixEnvironment(ToolLabEnvironment):
    def _inspect_cell(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        option_id = str(arguments["option_id"])
        attribute_id = str(arguments["attribute_id"])
        cue = self._cue_for(option_id, attribute_id)

        self._last_is_revisit = cue.id in self.opened_cues
        self._last_transition = classify_transition(
            self._last_inspect, cue.option_id, cue.attribute_id
        )
        # print('self._last_transition'.upper(), self._last_transition)
        self.opened_cues.add(cue.id)
        self._last_inspect = {"option_id": cue.option_id, "attribute_id": cue.attribute_id}

        content_dict = {
                "option_id": cue.option_id,
                "attribute_id": cue.attribute_id,
                "result": cue.value,
        }
        if self.spec.budget_type=='usd':
            content_dict['turn_cost_usd'] = round(self.last_turn_cost_usd, 4)
            content_dict['budget_remaining_usd'] = round(self.budget_remaining_usd, 4)
        elif self.spec.budget_type=='tools' and self.spec.budget_tools>=0:
            content_dict['cumulative_cost_tools'] = self.cumulative_cost_tools
            content_dict['budget_remaining_tools'] = self.budget_remaining_tools
        elif self.spec.budget_type=='tools' and self.spec.budget_tools<0:
            content_dict['cost_for_this_tool_call'] = self.last_turn_cost_tool
            content_dict['cumulative_cost_tools'] = self.cumulative_cost_tools
        elif self.spec.budget_type=='tool_usd':
            content_dict['turn_cost_usd'] = self.last_turn_cost_tool_usd
            content_dict['cumulative_cost_tool_usd'] = self.cumulative_cost_tool_usd

        elif self.spec.budget_type=='tokens':
            content_dict['turn_cost_tokens'] = self.last_turn_cost_tokens
            content_dict['budget_remaining_tokens'] = self.budget_remaining_tokens
        elif self.spec.budget_type=='points':
            content_dict['total_points_lost'] = self.cumulative_cost_points
            content_dict['points_lost_for_this_tool'] = self.last_turn_cost_points
            content_dict['points_remaining'] = self.budget_remaining_points
        else: 
            raise ValueError(f'budget_type not recognized: {self.spec.budget_type}')
        payload = {
            'role': 'tool',
            "tool_call_id": tool_call_id,
            "content": json.dumps(content_dict),
        }
        return payload

    def _cue_for(self, option_id: str, attribute_id: str) -> CueSpec:
        for cue in self.spec.cues:
            if cue.option_id == option_id and cue.attribute_id == attribute_id:
                return cue
        raise ValueError(f"No cue exists for option {option_id} and attribute {attribute_id}")


def classify_transition(prev: dict | None, option_id: str, attribute_id: str) -> str:
    if prev is None:
        return "first"
    same_option = prev["option_id"] == option_id
    same_attribute = prev["attribute_id"] == attribute_id
    if same_option and same_attribute:
        return "revisit"
    if same_option:
        return "alternative"   # alternative-based: stay on same option, switch attribute
    if same_attribute:
        return "attribute"  # attribute-based: switch option, same attribute
    return "diagonal"             # switch both
