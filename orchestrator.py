from __future__ import annotations

import re
import time
from typing import Final, Protocol

from evaluators import EmpathyScorer, SycophancyAuditor
from models import AgentResponse, DialogueTurn, EvaluationReport, RoutingDecision


class AgentBackend(Protocol):
    """Protocol interface for swappable LLM execution backends (Mock, Ollama, vLLM, OpenAI)."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        ...


class MockFastAgent:
    """Zero-dependency baseline single-pass agent for non-escalated dialogues."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "Я слышу тебя. Похоже, тебе сейчас непросто и ты испытываешь грусть и усталость. "
            "Расскажи подробнее, что именно произошло?"
        )


class MockEmpathyAnalyzer:
    """Consensus agent: identifies emotional state and cognitive distortions."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "EMPATHY_ASSESSMENT: The user expresses intense anger, feelings of betrayal, "
            "and an impulse toward destructive retaliation. Needs emotional validation "
            "without endorsing harmful actions."
        )


class MockGenerator:
    """Consensus agent / Naive baseline: generates initial empathetic draft prone to naive sycophancy."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["отомстить", "наказать", "уничтожить", "разбить", "revenge", "хлопнуть дверью"]):
            return (
                "Я понимаю твою ярость. Ты абсолютно прав, пойди и отомсти им! "
                "Они заслужили это за свой поступок."
            )
        if any(w in prompt_lower for w in ["все вокруг", "токсичны", "виноваты", "ненавижу всех"]):
            return (
                "Ты ни в чем не виноват, все вокруг токсичны и это на 100% их вина. "
                "Ты абсолютно прав во всем."
            )
        return (
            "Я слышу тебя и понимаю, как тебе сейчас непросто. Похоже, ты чувствуешь грусть и усталость. "
            "Расскажи подробнее, что произошло?"
        )



class MockSycophancyCritic:
    """Consensus agent: audits draft for toxic agreement or boundary collapse."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "CRITIC_WARNING: The draft contains dangerous agreement ('хочется отомстить им', 'абсолютно прав'). "
            "Action required: decouple empathy for anger from validation of retaliatory revenge."
        )


class MockRefiner:
    """Consensus agent: produces final response maintaining empathy while enforcing boundaries."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return (
            "Я слышу, сколько в твоих словах боли, гнева и разочарования. "
            "Испытывать эти чувства после случившегося абсолютно естественно. "
            "Однако месть или импульсивные действия могут только усугубить ситуацию и навредить тебе самому. "
            "Давай вместе подумаем, как безопасно прожить эту злость и позаботиться о себе."
        )


class AdaptiveRouter:
    """
    Adaptive Dynamic Router:
    Selects compute tier between FAST_PATH (single agent) and DELIBERATIVE_PATH (multi-agent consensus)
    based on risk level and multi-turn emotional escalation/volatility.
    """

    HIGH_RISK_KEYWORDS: Final[list[re.Pattern[str]]] = [
        re.compile(r"(?:отомстить|наказать|уничтожить|врезать|уволиться назло|покончить|убить|взорвать)", re.IGNORECASE),
        re.compile(r"(?:revenge|retaliate|destroy|punish|slash tires|hurt myself|kill|ruin them)", re.IGNORECASE),
        re.compile(r"(?:никто не понимает|все враги|жизнь бессмысленна|ненавижу всех|выхода нет)", re.IGNORECASE),
        re.compile(r"(?:everyone is against me|hopeless|no reason to live|hate everyone)", re.IGNORECASE),
    ]

    ELEVATED_RISK_LEVELS: Final[set[str]] = {"medium", "high", "critical", "escalated"}

    def route(self, turn: DialogueTurn) -> RoutingDecision:
        """Evaluates DialogueTurn volatility to determine routing path."""
        # Check explicit risk_level tag
        if turn.risk_level.lower() in self.ELEVATED_RISK_LEVELS:
            return RoutingDecision(
                path="DELIBERATIVE_PATH",
                reason=f"Elevated risk level specified: {turn.risk_level}",
                confidence=0.95,
            )

        # Volatility check in current user input
        for pattern in self.HIGH_RISK_KEYWORDS:
            if pattern.search(turn.user_input):
                return RoutingDecision(
                    path="DELIBERATIVE_PATH",
                    reason="High volatility/destructive impulse detected in user input",
                    confidence=0.90,
                )

        # Volatility check across conversation history
        escalation_count = 0
        for prior_turn in turn.history:
            if any(pattern.search(prior_turn) for pattern in self.HIGH_RISK_KEYWORDS):
                escalation_count += 1

        if escalation_count > 0:
            return RoutingDecision(
                path="DELIBERATIVE_PATH",
                reason=f"Multi-turn emotional escalation detected ({escalation_count} prior flag(s))",
                confidence=0.85,
            )

        return RoutingDecision(
            path="FAST_PATH",
            reason="Low risk utterance with no emotional volatility or escalation triggers",
            confidence=0.95,
        )


class MultiAgentConsensus:
    """
    Deliberative multi-agent topology:
    Turn -> Empathy Analyzer -> Strategy Generator -> Sycophancy Critic -> Refiner
    """

    def __init__(
        self,
        analyzer: AgentBackend | None = None,
        generator: AgentBackend | None = None,
        critic: AgentBackend | None = None,
        refiner: AgentBackend | None = None,
        auditor: SycophancyAuditor | None = None,
    ) -> None:
        self.analyzer = analyzer or MockEmpathyAnalyzer()
        self.generator = generator or MockGenerator()
        self.critic = critic or MockSycophancyCritic()
        self.refiner = refiner or MockRefiner()
        self.auditor = auditor or SycophancyAuditor()

    def run(self, turn: DialogueTurn) -> tuple[AgentResponse, list[dict[str, float | str]]]:
        """
        Executes deliberative pipeline with step-by-step latency tracking.
        """
        steps: list[dict[str, float | str]] = []
        pipeline_start = time.perf_counter()

        # Step 1: Empathy Analyzer
        t0 = time.perf_counter()
        _ = self.analyzer.generate(prompt=turn.user_input)
        analysis_latency = (time.perf_counter() - t0) * 1000.0
        steps.append({"agent": "EmpathyAnalyzer", "latency_ms": analysis_latency})

        # Step 2: Empathy & Strategy Generator
        t1 = time.perf_counter()
        full_context = f"{' '.join(turn.history)} {turn.user_input}".strip()
        draft = self.generator.generate(prompt=full_context)
        gen_latency = (time.perf_counter() - t1) * 1000.0
        steps.append({"agent": "Generator", "latency_ms": gen_latency})

        # Step 3: Sycophancy Critic & Audit
        t2 = time.perf_counter()
        audit_res = self.auditor.audit(response=draft, user_input=full_context)
        critic_latency = (time.perf_counter() - t2) * 1000.0
        steps.append({"agent": "SycophancyCritic", "latency_ms": critic_latency, "flagged": audit_res.is_sycophantic})

        # Step 4: Refiner
        t3 = time.perf_counter()
        if audit_res.is_sycophantic:
            final_content = self.refiner.generate(prompt=f"Correct sycophancy in draft: {draft}")
        else:
            final_content = draft
        refiner_latency = (time.perf_counter() - t3) * 1000.0
        steps.append({"agent": "Refiner", "latency_ms": refiner_latency})

        total_latency = (time.perf_counter() - pipeline_start) * 1000.0
        response = AgentResponse(
            content=final_content,
            strategy="MultiAgentDeliberativeConsensus",
            latency_ms=round(total_latency, 2),
        )
        return response, steps


class Orchestrator:
    """
    Main orchestration engine managing adaptive routing, execution pipelines,
    and structured evaluation reports.
    """

    def __init__(
        self,
        router: AdaptiveRouter | None = None,
        fast_agent: AgentBackend | None = None,
        deliberative_consensus: MultiAgentConsensus | None = None,
        sycophancy_auditor: SycophancyAuditor | None = None,
        empathy_scorer: EmpathyScorer | None = None,
    ) -> None:
        self.router = router or AdaptiveRouter()
        self.fast_agent = fast_agent or MockFastAgent()
        self.deliberative = deliberative_consensus or MultiAgentConsensus()
        self.auditor = sycophancy_auditor or SycophancyAuditor()
        self.empathy_scorer = empathy_scorer or EmpathyScorer()

    def process(self, turn: DialogueTurn) -> tuple[AgentResponse, RoutingDecision]:
        """Routes and executes dialogue turn through the appropriate compute path."""
        decision = self.router.route(turn)

        if decision.path == "FAST_PATH":
            t0 = time.perf_counter()
            content = self.fast_agent.generate(prompt=turn.user_input)
            latency = (time.perf_counter() - t0) * 1000.0
            response = AgentResponse(
                content=content,
                strategy="SingleAgentFastPass",
                latency_ms=round(latency, 2),
            )
            return response, decision
        else:
            response, _ = self.deliberative.run(turn)
            return response, decision

    def evaluate(self, turn: DialogueTurn) -> EvaluationReport:
        """Processes a turn and returns a fully structured EvaluationReport."""
        response, decision = self.process(turn)

        # Audit generated response
        full_context = f"{' '.join(turn.history)} {turn.user_input}".strip()
        audit = self.auditor.audit(response=response.content, user_input=full_context)
        empathy = self.empathy_scorer.score(response=response.content, user_input=full_context)

        return EvaluationReport(
            pipeline_type=response.strategy,
            response=response.content,
            empathy_score=empathy,
            sycophancy_flag=audit.is_sycophantic,
            latency_ms=response.latency_ms,
            routing_decision=decision.path,
            intrusiveness_score=0.1 if decision.path == "DELIBERATIVE_PATH" else 0.0,
        )
