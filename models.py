from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class DialogueTurn(BaseModel):
    """Represents an incoming dialogue turn with user input, historical context, and assessed risk level."""
    model_config = ConfigDict(frozen=True)

    user_input: str
    history: list[str] = Field(default_factory=list)
    risk_level: str = Field(default="low")


class RoutingDecision(BaseModel):
    """Decision made by the dynamic router between compute paths."""
    model_config = ConfigDict(frozen=True)

    path: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class AgentResponse(BaseModel):
    """Standardized output from an agent or multi-agent orchestration pipeline."""
    model_config = ConfigDict(frozen=True)

    content: str
    strategy: str
    latency_ms: float = Field(ge=0.0)


class SycophancyAuditResult(BaseModel):
    """Structured result of sycophancy detection analysis."""
    model_config = ConfigDict(frozen=True)

    is_sycophantic: bool
    risk_category: str = "none"
    reasons: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class EvaluationReport(BaseModel):
    """Structured evaluation report encompassing latency, empathy, sycophancy, and routing metrics."""
    model_config = ConfigDict(frozen=True)

    pipeline_type: str
    response: str
    empathy_score: float = Field(ge=0.0, le=1.0)
    sycophancy_flag: bool
    latency_ms: float = Field(ge=0.0)
    routing_decision: str
    intrusiveness_score: float = Field(ge=0.0, le=1.0, default=0.0)
