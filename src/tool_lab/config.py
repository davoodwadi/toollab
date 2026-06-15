from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import json

import yaml



@dataclass(slots=True)
class PricingConfig:
    input_per_million: float = 0.0
    output_per_million: float = 0.0


MODEL_CONFIG = {
    'google': {
        'gemini-3.1-flash-lite-preview':  PricingConfig(input_per_million=0.25, output_per_million=1.50),
        'gemini-3-flash-preview':  PricingConfig(input_per_million=0.50, output_per_million=3.00),
        'gemini-3.1-pro-preview':  PricingConfig(input_per_million=2.00, output_per_million=12.00)
    },
    'openai': {
        'gpt-5.4':PricingConfig(input_per_million=2.5, output_per_million=15.),
        'gpt-5.4-mini': PricingConfig(input_per_million=0.75, output_per_million=4.5),
        'gpt-5.4-nano': PricingConfig(input_per_million=0.2, output_per_million=1.25),
    },
    'anthropic': {
        'claude-opus-4-6':PricingConfig(input_per_million=5.0, output_per_million=25.),
        'claude-sonnet-4-6': PricingConfig(input_per_million=3., output_per_million=15.),
        'claude-haiku-4-5': PricingConfig(input_per_million=1., output_per_million=5.),
    },
    'llamacpp': {
        'default': PricingConfig(input_per_million=1, output_per_million=3)
    },

    'mock': {
        'mock-1': PricingConfig(input_per_million=1.0, output_per_million=6.0)
    }
}


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model_name: str
    thinking_type: str | None = None
    effort: str | None = None
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
    cost_multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CueSpec:
    id: str
    option_id: str
    attribute_id: str
    value: str
    normative_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    participant: ParticipantSpec
    randomization: dict
    options: list[OptionSpec]
    attributes: list[AttributeSpec] | None
    cues: list[CueSpec]
    model: ModelConfig
    inspect_mode: Literal['full', 'cell'] = 'cell'
    replications: int = 5
    budget_type: Literal["usd", "tools", "tool_usd", 'tokens', 'points'] = "tool_usd"
    budget_usd: float | None = None
    budget_tools: int | None = None
    budget_tokens: int | None = None
    budget_points: int | None = None
    inspect_tool_cost: float | None = None
    max_turns: int = 20
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        option_ids = {option.id for option in self.options}
        if self.attributes:
            attribute_ids = {attribute.id for attribute in self.attributes}
        else:
            attribute_ids = []
        cue_ids: set[str] = set()

        if len(option_ids) != len(self.options):
            raise ValueError("Option ids must be unique.")
        if self.attributes and len(attribute_ids) != len(self.attributes):
            raise ValueError("Attribute ids must be unique.")

        for cue in self.cues:
            if cue.id in cue_ids:
                raise ValueError(f"Duplicate cue id: {cue.id}")
            cue_ids.add(cue.id)
            if cue.option_id not in option_ids:
                raise ValueError(f"Cue {cue.id} references unknown option {cue.option_id}")
            # if cue.attribute_id not in attribute_ids:
            #     raise ValueError(f"Cue {cue.id} references unknown attribute {cue.attribute_id}")

        if self.replications < 1:
            raise ValueError("replications must be at least 1")
        if self.budget_type=='usd' and self.budget_usd<=0:
            raise ValueError("budget USD must be positive")
        # if self.budget_type=='tools' and self.budget_tools<=0:
        #     raise ValueError("budget Tools must be positive")
        if self.budget_type=='tokens' and self.budget_tokens<=0:
            raise ValueError("budget Tokens must be positive")
        # if self.budget_type=='points' and self.budget_points<=0:
        #     raise ValueError("budget Points must be positive")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    def with_runtime_overrides(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        replications: int | None = None,
        api_key_env: str | None = None,
        **budget_override,
    ) -> "ExperimentSpec":

        resolved_model = replace(
            self.model,
            provider=provider or self.model.provider,
            model_name=model_name or self.model.model_name,
            api_key_env=api_key_env or self.model.api_key_env,
        )

        updated = replace(
            self,
            replications=self.replications if replications is None else replications,
            model=resolved_model,
            **budget_override
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
    pricing = MODEL_CONFIG[model_data['provider']].get(model_data['model_name'])
    if not pricing:
        # for llamacpp models
        pricing = MODEL_CONFIG[model_data['provider']].get('default')
    model = ModelConfig(
        pricing=pricing,
        **model_data,
    )
    inspect_mode = experiment.get('inspect_mode')
    # print('inspect_mode', inspect_mode)

    if inspect_mode=='full':
        attributes = None
        options = []
        cues = []

        for opt_data in experiment['options']:

            opt_cues = opt_data.pop("cues", {})
            original_id = opt_data.get("id")
            display_name = opt_data.get("display_name")

            # 2. Build the OptionSpec
            options.append(OptionSpec(
                id=original_id,
                display_name=display_name if display_name else '', # Placeholder: overwritten in runner
                base_score=opt_data.get("base_score", 0.0),
                metadata={"original_id": original_id}  # Save original ID for the final trace!
            ))
            
            # 3. Automatically expand the dictionary into a flat list of CueSpecs
            for attr_id, val in opt_cues.items():
                cues.append(CueSpec(
                    id=f"{original_id}_{attr_id}",
                    option_id=original_id,
                    attribute_id=attr_id,
                    value=str(val)
                ))

        # print(options)
        # print(cues)
        # exit()
        # pass

    elif inspect_mode=='cell':
        attributes = [AttributeSpec(**item) for item in experiment["attributes"]]

        options = []
        cues = []
        
        # Loop through the simplified options list from the YAML
        for opt_data in experiment["options"]:
            # 1. Pop out the nested 'cues' dictionary
            opt_cues = opt_data.pop("cues", {})
            original_id = opt_data.get("id")
            display_name = opt_data.get("display_name")

            # 2. Build the OptionSpec
            options.append(OptionSpec(
                id=original_id,
                display_name=display_name if display_name else '', # Placeholder: overwritten in runner
                base_score=opt_data.get("base_score", 0.0),
                metadata={"original_id": original_id}  # Save original ID for the final trace!
            ))
            
            # 3. Automatically expand the dictionary into a flat list of CueSpecs
            for attr_id, val in opt_cues.items():
                cues.append(CueSpec(
                    id=f"{original_id}_{attr_id}",
                    option_id=original_id,
                    attribute_id=attr_id,
                    value=str(val)
                ))
    else:
        print(f'inspect_mode: {inspect_mode} invalid.')
        exit()            

    return ExperimentSpec(
        name=experiment["name"],
        inspect_mode=inspect_mode,
        participant=participant,
        randomization=experiment.get('randomization'),
        options=options,
        attributes=attributes,
        cues=cues,
        model=model,
        replications=experiment.get("replications", 5),
        budget_type=experiment.get("budget_type"),
        budget_usd=experiment.get("budget_usd", 0.0),
        budget_tools=experiment.get("budget_tools", 0),
        budget_tokens=experiment.get("budget_tokens", 0),
        budget_points=experiment.get("budget_points", 0),
        inspect_tool_cost=experiment.get("inspect_tool_cost", 0),
        max_turns=experiment.get("max_turns", 20),
        metadata=experiment.get("metadata", {}),
    )


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)
