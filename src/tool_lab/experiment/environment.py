from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from math import log
from random import Random
from typing import Any
import json

from tool_lab.models.base import (
    AssistantResponse,
    _to_serializable,
)
from tool_lab.config import AttributeSpec, CueSpec, ExperimentSpec, OptionSpec
from tool_lab.models.base import AssistantResponse

def build_environment(spec: ExperimentSpec, seed: int) -> "ToolLabEnvironment":
    if spec.matrix_mode == "fixed":
        return FixedMatrixEnvironment(spec, seed)
    return ScrollingMatrixEnvironment(spec, seed)


class ToolLabEnvironment(ABC):
    def __init__(self, spec: ExperimentSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        self.random = Random(seed)
        self.options: dict[str, OptionSpec] = {option.id: option for option in spec.options}
        self.attributes: dict[str, AttributeSpec] = {
            attribute.id: attribute for attribute in spec.attributes
        }
        self.cumulative_cost_usd = 0.0
        self.budget_remaining_usd = spec.budget_usd

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

    def build_system_prompt(self) -> str:
        mode_rules = self._mode_rules()
        return (
            "You are a subject in a Tool-Lab decision experiment. "
            "Use the available tools to inspect information before deciding. "
            "Do not guess hidden information. Record the final decision only by calling submit_choice. "
            f"The total budget is ${self.spec.budget_usd:.4f}. Each token you generate reduces the remaining budget. "
            f"{mode_rules}"
        )

    def build_user_prompt(self) -> str:
        option_lines = [
            f"- {option.id}: {option.display_name}. {option.description}".strip()
            for option in self.spec.options
        ]
        attribute_lines = [
            f"- {attribute.id}: {attribute.display_name}. {attribute.description}".strip()
            for attribute in self.spec.attributes
        ]
        return "\n".join(
            [
                self.spec.participant.profile,
                "",
                self.spec.task_prompt,
                "",
                "Options:",
                *option_lines,
                "",
                "Attributes available in this task:",
                *attribute_lines,
                "",
                f"Budget: ${self.spec.budget_usd:.4f}",
                "Use the tools to inspect information and then call submit_choice when ready.",
            ]
        )

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


    def get_model_cost(self, message: AssistantResponse) -> float:
        input_cost = message.input_tokens * (self.spec.model.pricing.input_per_million / 1e6)
        output_cost = message.output_tokens * (self.spec.model.pricing.output_per_million / 1e6)
        return {'input_cost':input_cost, 'output_cost':output_cost}
    
    def apply_model_cost(self, cost: dict) -> None:
        self.cumulative_cost_usd += (cost['input_cost'] + cost['output_cost'])
        self.budget_remaining_usd = self.spec.budget_usd - self.cumulative_cost_usd

    def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        try:
            if tool_name == "submit_choice":
                return self._submit_choice(arguments, tool_call_id)
            if tool_name == "inspect_cell":
                return self._inspect_cell(arguments, tool_call_id)
            if tool_name == "inspect_item":
                return self._inspect_item(arguments, tool_call_id)
            raise ValueError(f"Unsupported tool: {tool_name}")
        except Exception as exc:
            payload = {
                "status": "error",
                "message": str(exc),
                "tool_call_id": tool_call_id,
            }
            self._charge_and_record(tool_name, payload, kind="tool_error")
            return payload

    def _submit_choice(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        option_id = str(arguments["option_id"])
        if option_id not in self.options:
            raise ValueError(f"Unknown option_id: {option_id}")

        self.choice = option_id

        payload = {
            "tool_call_id": tool_call_id,
            "option_id": option_id,
            "option_label": self.options[option_id].display_name,
        }
        return payload

    def _mode_rules(self) -> str:
        if self.spec.matrix_mode == "fixed":
            return (
                "You may reveal any cell with inspect_cell."
            )
        window_size = int(self.spec.interface.get("window_size", 6))
        step_size = int(self.spec.interface.get("auto_advance_steps", 1))
        return (
            "Only a limited window of cue labels is visible at a time. "
            f"The visible window contains {window_size} labels and advances by {step_size} step(s) after every tool action."
        )


    def _record_event(
        self,
        *,
        kind: str,
        tool_name: str | None,
        cost: dict | None,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        self._step_index += 1
        event = {
            "step_index": self._step_index,
            "kind": kind,
            "tool_name": tool_name,
            **(cost or {}),
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 8),
            "budget_remaining_usd": round(self.budget_remaining_usd, 8),
            **data,
        }

        self.trace.append(event)
        return event

    def _build_inspection_payload(self, cue: CueSpec, tool_call_id: str) -> dict[str, Any]:
        is_revisit = cue.id in self.opened_cues
        self.opened_cues.add(cue.id)
        payload = {
            'role':'tool',
            # "status": "ok",
            "tool_call_id": tool_call_id,
            # "item_id": cue.id,
            # "option_id": cue.option_id,
            # "option_label": self.options[cue.option_id].display_name,
            # "attribute_id": cue.attribute_id,
            # "attribute_label": self.attributes[cue.attribute_id].display_name,
            # "label": cue.label,
            cue.label: cue.value,
            # "is_revisit": is_revisit,
        }
        return payload

class FixedMatrixEnvironment(ToolLabEnvironment):
    def _inspect_cell(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        option_id = str(arguments["option_id"])
        attribute_id = str(arguments["attribute_id"])
        cue = self._cue_for(option_id, attribute_id)
        payload = self._build_inspection_payload(cue, tool_call_id)
        # self._charge_and_record(tool_name="inspect_cell", payload=payload, kind="inspect")
        return payload

    def current_accessible_cues(self) -> list[CueSpec]:
        return list(self.cues.values())

    def _cue_for(self, option_id: str, attribute_id: str) -> CueSpec:
        for cue in self.spec.cues:
            if cue.option_id == option_id and cue.attribute_id == attribute_id:
                return cue
        raise ValueError(f"No cue exists for option {option_id} and attribute {attribute_id}")


# class ScrollingMatrixEnvironment(ToolLabEnvironment):
#     def __init__(self, spec: ExperimentSpec, seed: int) -> None:
#         super().__init__(spec, seed)
#         self.window_size = int(spec.interface.get("window_size", 6))
#         self.auto_advance_steps = int(spec.interface.get("auto_advance_steps", 1))
#         self._ordered_cue_ids = self._weighted_shuffle(spec.cues)
#         self._window_start = 0

#     def _view_visible_items(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
#         payload = {
#             "status": "ok",
#             "tool_call_id": tool_call_id,
#             "mode": "scrolling",
#             "visible_items": self._visible_items_payload(),
#         }
#         self._charge_and_record(tool_name="view_visible_items", payload=payload, kind="observe")
#         return payload

#     def _inspect_item(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
#         item_id = str(arguments["item_id"])
#         visible_map = {item.id: item for item in self.current_accessible_cues()}
#         if item_id not in visible_map:
#             raise ValueError(f"Item {item_id} is not currently visible")
#         cue = visible_map[item_id]
#         payload = self._build_inspection_payload(cue, tool_call_id)
#         payload["visible_items_after"] = []
#         self._charge_and_record(tool_name="inspect_item", payload=payload, kind="inspect")
#         self._auto_advance()
#         payload["visible_items_after"] = self._visible_items_payload()
#         return payload

#     def _advance_window(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
#         steps = int(arguments.get("steps", 1))
#         self._window_start = (self._window_start + steps) % len(self._ordered_cue_ids)
#         payload = {
#             "status": "ok",
#             "tool_call_id": tool_call_id,
#             "advanced_by": steps,
#             "visible_items": self._visible_items_payload(),
#         }
#         self._charge_and_record(tool_name="advance_window", payload=payload, kind="advance")
#         return payload

#     def current_accessible_cues(self) -> list[CueSpec]:
#         cue_ids = []
#         for offset in range(self.window_size):
#             index = (self._window_start + offset) % len(self._ordered_cue_ids)
#             cue_ids.append(self._ordered_cue_ids[index])
#         return [self.cues[cue_id] for cue_id in cue_ids]

#     def _visible_items_payload(self) -> list[dict[str, Any]]:
#         return [
#             {
#                 "item_id": cue.id,
#                 "label": cue.label,
#                 "option_id": cue.option_id,
#                 "option_label": self.options[cue.option_id].display_name,
#                 "attribute_id": cue.attribute_id,
#                 "attribute_label": self.attributes[cue.attribute_id].display_name,
#                 "is_opened": cue.id in self.opened_cues,
#             }
#             for cue in self.current_accessible_cues()
#         ]

#     def _auto_advance(self) -> None:
#         self._window_start = (self._window_start + self.auto_advance_steps) % len(self._ordered_cue_ids)

#     def _weighted_shuffle(self, cues: list[CueSpec]) -> list[str]:
#         keyed = []
#         for cue in cues:
#             weight = max(cue.visibility_weight, 1e-6)
#             key = -log(self.random.random()) / weight
#             keyed.append((key, cue.id))
#         keyed.sort()
#         return [cue_id for _, cue_id in keyed]
