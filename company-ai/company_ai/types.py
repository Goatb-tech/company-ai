from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ModelEndpoint:
    provider: str
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 1200
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    provider_type: str = "cloud"


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: Usage
    raw: Any = None


@dataclass(slots=True)
class SpendRecord:
    timestamp: datetime
    agent: str
    provider: str
    provider_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    amount_usd: float


@dataclass(slots=True)
class ActivityEvent:
    timestamp: datetime
    agent: str
    kind: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentProfile:
    agent_id: str
    name: str
    role: str
    endpoint: ModelEndpoint
    tools: list[str]
    created_at: datetime = field(default_factory=utc_now)
    status: str = "idle"
    last_result: str = ""


@dataclass(slots=True)
class RuntimeSettings:
    budget_limit_usd: float
    ceo_endpoint: ModelEndpoint
    worker_endpoint: ModelEndpoint


@dataclass(slots=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    tool: str
    success: bool
    output: Any
