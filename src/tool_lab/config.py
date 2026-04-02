from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import json

import yaml

MatrixMode = Literal["fixed", "scrolling"]

DEFAULT_TOOLS_BY_MODE: dict[MatrixMode, list[str]] = {
    "fixed": ["view_matrix", "inspect_cell", "submit_choice"],
    "scrolling": ["view_visible_items", "inspect_item", "advance_window", "submit_choice"],
}

REQUIRED_TOOLS_BY_MODE: dict[MatrixMode, set[str]] = {
    "fixed": {"inspect_cell", "submit_choice"},
    "scrolling": {"inspect_item", "submit_choice"},
}


@dataclass(slots=True)
class PricingConfig:
    input_per_million: float = 0.0
    output_per_million: float = 0.0


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 1.0
    max_output_tokens: int = 1024
    api_key_env: str | None = None
    pricing: PricingConfig = field(default_factory=PricingConfig)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParticipantSpec:
    role: str
    profile: str


@dataclass(slots=True)
class OptionSpec:
    id: str
    display_name: str
    description: str = ""
    base_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttributeSpec:
    id: str
    display_name: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CueSpec:
    id: str
    option_id: str
    attribute_id: str
    label: str
    value: str
    normative_score: float = 0.0
    visibility_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    description: str
    task_prompt: str
    participant: ParticipantSpec
    options: list[OptionSpec]
    attributes: list[AttributeSpec]
    cues: list[CueSpec]
    model: ModelConfig
    matrix_mode: MatrixMode = "scrolling"
    replications: int = 5
    budget_usd: float = 0.10
    max_turns: int = 20
    tools: list[str] = field(default_factory=list)
    default_tool_cost_usd: float = 0.0
    tool_costs: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    interface: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools = list(DEFAULT_TOOLS_BY_MODE[self.matrix_mode])
        self.validate()

    def validate(self) -> None:
        option_ids = {option.id for option in self.options}
        attribute_ids = {attribute.id for attribute in self.attributes}
        cue_ids: set[str] = set()

        if len(option_ids) != len(self.options):
            raise ValueError("Option ids must be unique.")
        if len(attribute_ids) != len(self.attributes):
            raise ValueError("Attribute ids must be unique.")

        for cue in self.cues:
            if cue.id in cue_ids:
                raise ValueError(f"Duplicate cue id: {cue.id}")
            cue_ids.add(cue.id)
            if cue.option_id not in option_ids:
                raise ValueError(f"Cue {cue.id} references unknown option {cue.option_id}")
            if cue.attribute_id not in attribute_ids:
                raise ValueError(f"Cue {cue.id} references unknown attribute {cue.attribute_id}")

        if not REQUIRED_TOOLS_BY_MODE[self.matrix_mode].issubset(set(self.tools)):
            required = ", ".join(sorted(REQUIRED_TOOLS_BY_MODE[self.matrix_mode]))
            raise ValueError(
                f"Tool set for {self.matrix_mode} mode must include: {required}"
            )

        if self.replications < 1:
            raise ValueError("replications must be at least 1")
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    def with_runtime_overrides(
        self,
        *,
        matrix_mode: MatrixMode | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        replications: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        api_key_env: str | None = None,
    ) -> "ExperimentSpec":
        resolved_mode = matrix_mode or self.matrix_mode
        resolved_tools = list(self.tools)
        if matrix_mode and not REQUIRED_TOOLS_BY_MODE[resolved_mode].issubset(set(resolved_tools)):
            resolved_tools = list(DEFAULT_TOOLS_BY_MODE[resolved_mode])

        resolved_model = replace(
            self.model,
            provider=provider or self.model.provider,
            model_name=model_name or self.model.model_name,
            temperature=self.model.temperature if temperature is None else temperature,
            max_output_tokens=(
                self.model.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            api_key_env=api_key_env or self.model.api_key_env,
        )

        updated = replace(
            self,
            matrix_mode=resolved_mode,
            replications=self.replications if replications is None else replications,
            tools=resolved_tools,
            model=resolved_model,
        )
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]: 
        return asdict(self)


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    file_path = Path(path)
    raw = _load_structured_file(file_path)
    experiment = raw.get("experiment", raw)

    participant = ParticipantSpec(**experiment["participant"])
    model_data = experiment["model"]
    pricing = PricingConfig(**model_data.get("pricing", {}))
    model = ModelConfig(
        provider=model_data["provider"],
        model_name=model_data["model_name"], 
        temperature=model_data.get("temperature", 1.0),
        max_output_tokens=model_data.get("max_output_tokens", 1024),
        api_key_env=model_data.get("api_key_env"),
        pricing=pricing,
        extra=model_data.get("extra", {}),
    )

    options = [OptionSpec(**item) for item in experiment["options"]]
    attributes = [AttributeSpec(**item) for item in experiment["attributes"]]
    cues = [CueSpec(**item) for item in experiment["cues"]]

    return ExperimentSpec(
        name=experiment["name"],
        description=experiment["description"],
        task_prompt=experiment["task_prompt"],
        participant=participant,
        options=options,
        attributes=attributes,
        cues=cues,
        model=model,
        matrix_mode=experiment.get("matrix_mode", "scrolling"),
        replications=experiment.get("replications", 5),
        budget_usd=experiment.get("budget_usd", 0.10),
        max_turns=experiment.get("max_turns", 20),
        tools=experiment.get("tools", []),
        default_tool_cost_usd=experiment.get("default_tool_cost_usd", 0.0),
        tool_costs=experiment.get("tool_costs", {}),
        seed=experiment.get("seed"),
        interface=experiment.get("interface", {}),
        analysis=experiment.get("analysis", {}),
    )


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)
