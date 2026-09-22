# Multi-Agent Emotional Support & Anti-Sycophancy Harness Protocol

## Project Identity & Research Scope
Autonomous research engineering protocol for evaluating, routing, and distilling multi-agent emotional support architectures. Developed for the Sber AI Lab Master's research track at AI Talent Hub (ITMO).

### Core Research Objectives
1. **Sycophancy Mitigation:** Decouple emotional validation from destructive agreement. Enforce therapeutic boundaries without toxic compliance.
2. **Adaptive Dynamic Routing:** Trade off system latency against reasoning depth (single-agent vs. multi-agent consensus) based on multi-turn user volatility.
3. **Process & Response Distillation:** Compress multi-agent debate/critic traces into a single-pass inference model without quality collapse.
4. **Latency-Quality Benchmarking:** Quantify trade-offs between empathetic depth, sycophancy rate, safety compliance, and response latency ($ms$).

---

## Architecture Standards & Data Models

### 1. Data Contract Rigor
- All state transfers between dialogue turns, routing mechanisms, and evaluators must utilize strict, immutable `pydantic.BaseModel` (v2) instances.
- Evaluation metrics must always output structured reports:
  - `latency_ms: float`
  - `empathy_score: float` (0.0 to 1.0)
  - `sycophancy_flag: bool`
  - `intrusiveness_score: float` (0.0 to 1.0)
  - `routing_decision: str`

### 2. Multi-Agent Topology
- **Turn Context & Volatility Analyzer:** Tracks emotional escalation, cognitive distortions, and boundary violations across conversation history.
- **Adaptive Router:** Selects compute tier (`FAST_PATH` single agent vs. `DELIBERATIVE_PATH` multi-agent debate/critic).
- **Empathy & Strategy Generator:** Applies evidence-based conversational strategies (Questioning, Restatement, Reflection of Feelings).
- **Anti-Sycophancy Auditor / Critic:** Identifies false validation, destructive alignment, and unchecked cognitive distortions.

---

## Code Quality & Engineering Directives

- **Zero-Dependency Core Execution:** Core routing, state machines, and mocking must run on clean Python standard libraries and Pydantic. External LLM backends (Ollama, vLLM, OpenAI-compatible APIs) must sit behind swappable client interfaces (`Protocol` or `ABC`).
- **Python Conventions:**
  - Modern Python 3.10+ typing (`tuple[int, ...]`, `A | B`, `typing.Final`, `typing.Protocol`).
  - No untyped dictionaries for agent state; use explicit Pydantic models.
  - Strict exception handling: fail fast on corrupt state, record timing via `time.perf_counter()`.
- **Reproducible Evaluation:**
  - Deterministic evaluation pipelines for synthetic emotional escalation scenarios and standard datasets (ESConv, EmpatheticDialogues).
  - Green pytest suite for all critical paths: routing logic, sycophancy detection rules, and latency benchmarking.

---

## Sub-Agent Delegation Protocol (Antigravity)

When performing complex multi-file refactoring, benchmark scaling, or dataset ingestion:

1. **Sub-Agent 1: Dataset & Evaluator Specialist (`@evaluator-subagent`)**
   - Role: Handle synthetic scenario generation, ESConv/EmpatheticDialogues parsers, and automated LLM-as-a-judge rubrics.
   - Constraint: Must not modify orchestrator execution logic; strictly generate data and evaluation suites.

2. **Sub-Agent 2: Core Orchestrator & Router Specialist (`@orchestrator-subagent`)**
   - Role: Maintain the state machine, latency optimization, and router decision trees.
   - Constraint: Must ensure zero network dependencies in test mode; verify all state models are frozen.

3. **Sub-Agent 3: Test & Benchmark Runner (`@runner-subagent`)**
   - Role: Run isolated `pytest` suites, track latency regressions, and format markdown benchmark tables.
   - Constraint: Always verify all unit tests pass before reporting task completion.

---

## Repository Structure Baseline

```text
sycophancy-mitigation-harness/
├── GEMINI.md                       # Engineering protocol and research guidelines
├── requirements.txt                # pydantic>=2.0.0, pytest>=8.0.0
├── models.py                       # Pydantic schemas: DialogueTurn, EvaluationReport, RouterDecision
├── orchestrator.py                 # Multi-agent consensus, router, and latency tracking
├── evaluators.py                   # Sycophancy auditor, empathy scorer, LLM-as-a-judge
├── benchmark.py                    # Multi-scenario latency vs. sycophancy test bench
└── tests/
    ├── test_router.py              # Verification of dynamic routing thresholds
    └── test_sycophancy_auditor.py  # Validation of anti-sycophancy detection edge cases