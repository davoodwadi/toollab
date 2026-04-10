from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
import json

from tool_lab.config import ModelConfig


@dataclass(slots=True)
class ToolInvocation:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class AssistantResponse:
    content: str
    reasoning: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    tool_cost: int = 0
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    finish_reason: str | None = None
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    name: str
    content: dict[str, Any]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_serializable(getattr(value, key)) for key in value.__dataclass_fields__}
    return value
