from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

ToolDefinition = dict[str, Any]


class _InlineToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _schema(model_name: str, **fields: Any) -> dict[str, Any]:
    model = create_model(model_name, __base__=_InlineToolSchema, **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


BUILTIN_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "advance_window": {
        "name": "advance_window",
        "description": "Advance the scrolling information window without revealing any hidden value.",
        "input_schema": _schema(
            "AdvanceWindowInput",
            steps=(int, Field(default=1, ge=1, le=5)),
        ),
    },
    "submit_choice": {
        "name": "submit_choice",
        "description": "Record the final decision.",
        "input_schema": _schema(
            "SubmitChoiceInput",
            option_id=(str, ...),
            confidence=(float | None, Field(default=None, ge=0, le=1)),
            justification=(str | None, None),
        ),
    },
    "inspect_cell": {
        "name": "inspect_cell",
        "description": "Reveal one hidden cell in the fixed information matrix by option and attribute.",
        "input_schema": _schema(
            "InspectCellInput",
            option_id=(str, ...),
            attribute_id=(str, ...),
        ),
    },
    "inspect_item": {
        "name": "inspect_item",
        "description": "Reveal the value behind a currently visible cue in the scrolling window.",
        "input_schema": _schema(
            "InspectItemInput",
            item_id=(str, ...),
        ),
    },
    "view_matrix": {
        "name": "view_matrix",
        "description": "Show the full fixed matrix of hidden cells and their labels.",
        "input_schema": _schema("ViewMatrixInput"),
    },
    "view_visible_items": {
        "name": "view_visible_items",
        "description": "Show the labels that are currently visible in the scrolling information window.",
        "input_schema": _schema("ViewVisibleItemsInput"),
    },
}
