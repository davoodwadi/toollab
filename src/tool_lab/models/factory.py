from __future__ import annotations

from tool_lab.config import ModelConfig
from tool_lab.experiment.tools import ToolDefinition
from tool_lab.models.anthropic_adapter import AnthropicModelSession
from tool_lab.models.base import BaseModelSession
from tool_lab.models.google_adapter import GoogleModelSession
from tool_lab.models.mock_adapter import MockModelSession
from tool_lab.models.openai_adapter import OpenAIModelSession


def create_model_session(
    config: ModelConfig,
    system_prompt: str,
    initial_user_message: str,
    tools: list[ToolDefinition],
) -> BaseModelSession:
    provider = config.provider.lower()
    if provider == "anthropic":
        return AnthropicModelSession(config, system_prompt, initial_user_message, tools)
    if provider == "google":
        return GoogleModelSession(config, system_prompt, initial_user_message, tools)
    if provider == "mock":
        return MockModelSession(config, system_prompt, initial_user_message, tools)
    if provider == "openai":
        return OpenAIModelSession(config, system_prompt, initial_user_message, tools)
    raise ValueError(f"Unsupported provider: {config.provider}")
