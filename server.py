import base64
import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import fcntl

from mcp.server import MCPServer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b",
)

OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE",
    "30m",
)

OLLAMA_REQUEST_TIMEOUT = int(
    os.getenv(
        "OLLAMA_REQUEST_TIMEOUT",
        "240",
    )
)

OLLAMA_STATUS_TIMEOUT = int(
    os.getenv(
        "OLLAMA_STATUS_TIMEOUT",
        "10",
    )
)

OLLAMA_TEMPERATURE = float(
    os.getenv(
        "OLLAMA_TEMPERATURE",
        "0.1",
    )
)

OLLAMA_NUM_CTX = int(
    os.getenv(
        "OLLAMA_NUM_CTX",
        "32768",
    )
)

LOG_FILE = Path(
    os.getenv(
        "OLLAMA_WORKER_LOG",
        str(Path.home() / "tools/ollama-worker-mcp/ollama-worker.log"),
    )
)

LOG_FULL_CONTEXT = (
    os.getenv("OLLAMA_LOG_FULL_CONTEXT", "1").lower()
    not in {"0", "false", "no"}
)

LOG_MAX_BYTES = int(
    os.getenv(
        "OLLAMA_WORKER_LOG_MAX_BYTES",
        str(10 * 1024 * 1024),  # 10 MB
    )
)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = MCPServer(
    "ollama-worker",
    version="0.1.0",
    instructions=(
        "Local secondary software-engineering reviewer powered by a local "
        "Ollama model. "
        "Use these tools for bounded independent analysis, plan review, "
        "code review, debugging, test design, edge-case discovery and "
        "second opinions. "
        "The worker has no direct repository access. "
        "The caller must provide all relevant requirements, code, diffs, "
        "logs and test output. "
        "The primary agent remains responsible for repository inspection, "
        "implementation, testing and final decisions. "
        "review_screenshot accepts a caller-supplied base64 image "
        "(e.g. a UI screenshot or design export) for visual review. "
        "Every tool accepts an optional per-call 'model' override (any tag "
        "installed in the local Ollama) and an optional 'think' flag to "
        "enable the model's reasoning mode; both default to the server's "
        "configured values. status reports the worker's configuration "
        "and live Ollama connectivity."
    ),
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    """
    Append an entry to the worker log.

    Never print operational logs to stdout because stdout belongs to the
    MCP stdio protocol.
    """

    try:
        LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Best-effort single-file rotation. This log is shared across every
        # project session using this server, so it grows unbounded without
        # this — a small race between concurrent sessions rotating at the
        # same time is possible but harmless (worst case, one entry lands
        # in the rotated file instead of the fresh one).
        if (
            LOG_FILE.exists()
            and LOG_FILE.stat().st_size > LOG_MAX_BYTES
        ):
            try:
                LOG_FILE.replace(
                    LOG_FILE.with_name(LOG_FILE.name + ".1")
                )
            except Exception:
                pass

        with LOG_FILE.open(
            "a",
            encoding="utf-8",
        ) as file:
            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_EX,
            )

            file.write(message)
            file.write("\n")
            file.flush()

            fcntl.flock(
                file.fileno(),
                fcntl.LOCK_UN,
            )

    except Exception:
        # Logging must never break an MCP tool call.
        pass


def _log_request(
    *,
    request_id: str,
    tool_name: str,
    model: str,
    role: str,
    task: str,
    context: str,
    max_tokens: int,
    think: bool | str,
    num_ctx: int,
    image_count: int,
    image_base64_chars: int,
) -> None:

    timestamp = datetime.now().astimezone().isoformat()

    if LOG_FULL_CONTEXT:
        context_for_log = context
    else:
        context_for_log = (
            f"[context logging disabled; "
            f"{len(context)} characters supplied]"
        )

    _log(
        f"""
======================================================================
[{timestamp}]
REQUEST ID: {request_id}
TOOL: {tool_name}
MODEL: {model}
ROLE: {role}
MAX TOKENS: {max_tokens}
THINKING: {think}
NUM CTX: {num_ctx}
IMAGES: {image_count} attached (~{image_base64_chars // 1024} KB base64)

TASK
----------------------------------------------------------------------
{task}

CONTEXT
----------------------------------------------------------------------
{context_for_log}

STATUS
----------------------------------------------------------------------
SENT TO MODEL
======================================================================
""".strip()
    )


def _log_response(
    *,
    request_id: str,
    tool_name: str,
    answer: str,
    done_reason: str,
    prompt_count: int,
    output_count: int,
    prompt_speed: float,
    generation_speed: float,
    load_seconds: float,
    total_seconds: float,
) -> None:

    timestamp = datetime.now().astimezone().isoformat()

    _log(
        f"""
======================================================================
[{timestamp}]
REQUEST ID: {request_id}
TOOL: {tool_name}

RESPONSE
----------------------------------------------------------------------
{answer}

METRICS
----------------------------------------------------------------------
Done reason: {done_reason}
Prompt tokens: {prompt_count}
Output tokens: {output_count}
Prompt processing: {prompt_speed:.1f} tok/s
Generation: {generation_speed:.1f} tok/s
Load time: {load_seconds:.2f}s
Total time: {total_seconds:.2f}s
======================================================================
""".strip()
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama_get(path: str) -> dict | str:
    """
    GET a JSON endpoint from the local Ollama instance.

    Used only for lightweight introspection (status), never for a
    model call, so it uses a short timeout and never touches the worker
    log.
    """

    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_STATUS_TIMEOUT,
        ) as response:
            return json.load(response)

    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} from Ollama at {path}."

    except urllib.error.URLError as exc:
        return f"Failed to contact Ollama at {path}: {exc}"

    except TimeoutError:
        return f"Timed out after {OLLAMA_STATUS_TIMEOUT}s contacting {path}."

    except Exception as exc:
        return f"Failed to read {path}: {type(exc).__name__}: {exc}"


def _resolve_think(model: str, think: bool) -> bool | str:
    """
    Translate the tool-facing boolean `think` flag into whatever a given
    model actually expects.

    gpt-oss models ignore a plain true/false `think` value and silently
    fall back to "medium" reasoning effort, which is far more verbose
    (and slower) than this worker's tools are tuned for. They require an
    explicit "low"/"medium"/"high" string instead. Every other model
    installed here (the Qwen and Gemma families) accepts the plain
    boolean, so only gpt-oss needs translation.
    """

    if model.startswith("gpt-oss"):
        return "high" if think else "low"

    return think


def _pad_max_tokens_for_thinking(
    *,
    model: str,
    resolved_think: bool | str,
    max_tokens: int,
) -> int:
    """
    Reasoning models spend part of their output budget on hidden thinking
    tokens before producing a final answer. If num_predict is too small,
    a model can exhaust the entire budget on reasoning and return empty
    content — observed directly: gpt-oss with think=True and the default
    1400-token budget returned "no final answer" every time, because
    "high" reasoning effort consumed the whole budget before any answer
    token was emitted. Pad the budget whenever reasoning is active so
    there is still room left for the actual answer.

    gpt-oss always reasons at some effort level (see _resolve_think — it
    has no true "off" state), so it always gets at least a 2x pad here.
    """

    if not resolved_think:
        return max_tokens

    if model.startswith("gpt-oss") and resolved_think == "high":
        multiplier = 3
    else:
        multiplier = 2

    return max(
        max_tokens,
        max_tokens * multiplier,
        1024,
    )


def _call_model(
    *,
    tool_name: str,
    role: str,
    task: str,
    context: str,
    additional_rules: str,
    max_tokens: int,
    think: bool,
    model: str = "",
    num_ctx: int = 0,
    images: list[str] | None = None,
) -> str:
    """
    Internal helper used by every MCP tool.
    """

    request_id = uuid4().hex[:10]
    resolved_model = model or OLLAMA_MODEL
    resolved_think = _resolve_think(resolved_model, think)
    resolved_num_ctx = num_ctx or OLLAMA_NUM_CTX
    effective_max_tokens = _pad_max_tokens_for_thinking(
        model=resolved_model,
        resolved_think=resolved_think,
        max_tokens=max_tokens,
    )

    system_prompt = f"""
You are a secondary software-engineering agent working for a primary
AI coding agent.

ROLE
{role}

GENERAL RULES

- Focus only on the delegated task.
- Be precise, technical and actionable.
- Keep the response proportional to the task.
- Prefer findings over long explanations.
- Do not restate the supplied requirements, code, diff or plan.
- Never claim that a file, API, entity, service, table, endpoint, or behavior
  already exists unless the supplied context establishes that it exists.
- You may recommend creating new files, entities, services, migrations,
  endpoints, or other implementation components when justified by the
  supplied requirements and architecture.
- Clearly label newly proposed components as proposed or new.
- Clearly distinguish verified facts from assumptions.
- Explicitly state when evidence is insufficient.
- Prefer minimal, maintainable changes.
- Do not introduce new requirements.
- Optional improvements must be explicitly labeled OPTIONAL.
- Do not report style preferences as defects.
- Do not claim code compiles or tests pass unless evidence was supplied.
- Do not make unrelated recommendations.
- The primary agent makes all final decisions.
- The primary agent performs all repository modifications and verification.

SPECIALIZED RULES

{additional_rules}
""".strip()

    user_prompt = f"""
TASK
----------------------------------------------------------------------
{task}

CONTEXT
----------------------------------------------------------------------
{context if context.strip() else "(No additional context supplied)"}
""".strip()

    _log_request(
        request_id=request_id,
        tool_name=tool_name,
        model=resolved_model,
        role=role,
        task=task,
        context=context,
        max_tokens=effective_max_tokens,
        think=resolved_think,
        num_ctx=resolved_num_ctx,
        image_count=len(images) if images else 0,
        image_base64_chars=sum(len(s) for s in images) if images else 0,
    )

    user_message = {
        "role": "user",
        "content": user_prompt,
    }

    if images:
        user_message["images"] = images

    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            user_message,
        ],
        "stream": False,
        "think": resolved_think,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": effective_max_tokens,
            "num_ctx": resolved_num_ctx,
        },
    }

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        ) as response:
            result = json.load(response)

    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            body = "(could not read error body)"

        error = (
            f"Worker received HTTP {exc.code} "
            f"from Ollama: {body}"
        )

        _log(
            f"[{request_id}] ERROR: {error}"
        )

        return error

    except urllib.error.URLError as exc:
        error = (
            f"Worker failed to contact Ollama: "
            f"{exc}"
        )

        _log(
            f"[{request_id}] ERROR: {error}"
        )

        return error

    except TimeoutError:
        error = (
            f"Worker timed out after "
            f"{OLLAMA_REQUEST_TIMEOUT}s."
        )

        _log(
            f"[{request_id}] ERROR: {error}"
        )

        return error

    except Exception as exc:
        error = (
            f"Worker failed: "
            f"{type(exc).__name__}: {exc}"
        )

        _log(
            f"[{request_id}] ERROR: {error}"
        )

        return error

    message = result.get(
        "message",
        {},
    )

    answer = (
        message.get(
            "content",
            "",
        )
        or ""
    ).strip()

    if not answer:
        # A reasoning model can exhaust its entire token budget on hidden
        # thinking before emitting any content, especially with a small
        # max_tokens and think=True (see _pad_max_tokens_for_thinking).
        # Surface the reasoning trace instead of a bare "no answer",
        # since it may still contain a usable partial answer.
        thinking = (
            message.get(
                "thinking",
                "",
            )
            or ""
        ).strip()

        if thinking:
            answer = (
                "(No final answer content — falling back to the model's "
                "reasoning trace, which may be incomplete or unpolished:)\n\n"
                + thinking
            )
        else:
            answer = (
                "Model returned no final answer."
            )

    done_reason = result.get(
        "done_reason",
        "unknown",
    )

    prompt_count = result.get(
        "prompt_eval_count",
        0,
    )

    prompt_duration = result.get(
        "prompt_eval_duration",
        0,
    )

    output_count = result.get(
        "eval_count",
        0,
    )

    output_duration = result.get(
        "eval_duration",
        0,
    )

    load_duration = result.get(
        "load_duration",
        0,
    )

    total_duration = result.get(
        "total_duration",
        0,
    )

    prompt_speed = 0.0

    if prompt_count and prompt_duration:
        prompt_speed = (
            prompt_count
            / (prompt_duration / 1_000_000_000)
        )

    generation_speed = 0.0

    if output_count and output_duration:
        generation_speed = (
            output_count
            / (output_duration / 1_000_000_000)
        )

    load_seconds = (
        load_duration
        / 1_000_000_000
    )

    total_seconds = (
        total_duration
        / 1_000_000_000
    )

    _log_response(
        request_id=request_id,
        tool_name=tool_name,
        answer=answer,
        done_reason=done_reason,
        prompt_count=prompt_count,
        output_count=output_count,
        prompt_speed=prompt_speed,
        generation_speed=generation_speed,
        load_seconds=load_seconds,
        total_seconds=total_seconds,
    )

    truncation_warning = ""

    if done_reason in {
        "length",
        "max_tokens",
    }:
        truncation_warning = (
            "\n\n"
            "WARNING: Model reached its generation limit. "
            "The response may be incomplete."
        )

    # Keep metrics compact because this result is fed back into
    # Claude/Codex context.
    metrics = (
        "\n\n---\n"
        f"[ollama-worker "
        f"id={request_id} "
        f"model={resolved_model} "
        f"prompt={prompt_count} "
        f"output={output_count} "
        f"prompt_speed={prompt_speed:.1f}tok/s "
        f"generation={generation_speed:.1f}tok/s "
        f"total={total_seconds:.2f}s "
        f"done={done_reason}]"
    )

    return (
        answer
        + truncation_warning
        + metrics
    )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

def _review_plan_context(plan: str, requirements: str, constraints: str) -> str:
    return f"""
REQUIREMENTS
{requirements or "(not supplied)"}

CONSTRAINTS
{constraints or "(not supplied)"}

PROPOSED PLAN
{plan}
""".strip()


@mcp.tool()
def review_plan(
    plan: str,
    requirements: str = "",
    constraints: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """Independently review an implementation plan before coding starts."""
    return _call_model(
        tool_name="review_plan",
        role="senior software engineer performing an independent implementation-plan review",
        task="Review the proposed implementation plan and identify only material issues that could cause an incorrect, incomplete, unsafe or unnecessarily complex implementation.",
        context=_review_plan_context(plan, requirements, constraints),
        max_tokens=1024,
        think=think,
        model=model,
        additional_rules="""
Return:

VERDICT: APPROVE | APPROVE WITH CHANGES | REWORK

Then include only:

1. Material findings
2. Missing requirements or steps
3. Important edge cases
4. Testing gaps
5. Recommended plan changes

Maximum 5 material findings.

If the plan is already sound, say so briefly.

Do not rewrite a good plan simply to make it different.
Do not introduce requirements not present in the supplied context.
""".strip(),
    )


@mcp.tool()
def final_plan_check(
    plan: str,
    requirements: str = "",
    constraints: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """Perform a concise GO/NO-GO check on a revised implementation plan."""
    context = f"""
REQUIREMENTS
{requirements or "(not supplied)"}

CONSTRAINTS
{constraints or "(not supplied)"}

FINAL REVISED PLAN
{plan}
""".strip()
    return _call_model(
        tool_name="final_plan_check",
        role="senior engineer performing a final implementation-readiness review",
        task=(
            "Determine whether this revised plan is ready for implementation. "
            "Report only genuine blockers or missing stated requirements."
        ),
        context=context,
        max_tokens=640,
        think=think,
        model=model,
        additional_rules="""
A blocker is something that prevents a correct implementation from proceeding
using the supplied requirements, architecture, and constraints.

IMPORTANT:

- The fact that a required entity, repository, service, migration, endpoint,
  table, or file does not exist yet is NOT a blocker by itself.
- An implementation plan is allowed to propose new components.
- Never claim a proposed component already exists unless the context proves it.
- Do not interpret "do not invent" as "do not create".
- Do not require an additional architecture decision merely because the plan
  introduces a new component, unless the supplied architecture or requirements
  genuinely leave its design unresolved.
- Do not turn optional improvements into blockers.
- Do not introduce new business rules.
- Do not require automation when the requirements specify a manual process.
- If a requirement depends on an unresolved business-policy decision, that CAN
  be a blocker.

If there are no genuine blockers or missing stated requirements, respond exactly:

GO — NO BLOCKERS

Nothing else.

If blockers exist, respond exactly in this format:

NO-GO

BLOCKERS:

1. <short blocker title>
   - Evidence: <specific supplied requirement or constraint>
   - Why it blocks: <one concise explanation>
   - Minimal plan change: <one concise correction>

Rules:

- Maximum 4 blockers.
- Only include blockers supported by supplied evidence.
- Maximum 3 lines per blocker.
- Do not restate the entire plan.
- Do not restate all requirements.
- Do not report optional improvements.
- Do not report style suggestions.
- Do not speculate about undocumented policy.
""".strip(),
    )


@mcp.tool()
def review_diff(
    diff: str,
    requirements: str = "",
    test_results: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """Perform an independent code review of a proposed git diff."""
    context = f"""
REQUIREMENTS
{requirements or "(not supplied)"}

TEST RESULTS
{test_results or "(not supplied)"}

DIFF
{diff}
""".strip()
    return _call_model(
        tool_name="review_diff",
        role="strict senior code reviewer",
        task="Review this change for real correctness problems, regressions, security issues, broken assumptions, validation gaps, data integrity problems, concurrency issues, error-handling problems, API compatibility problems and missing high-value tests.",
        context=context,
        max_tokens=2400,
        think=think,
        model=model,
        additional_rules="""
Only report findings supported by the supplied evidence.

For each finding provide:

- Severity: Critical | High | Medium | Low
- Location
- Problem
- Why it matters
- Minimal recommended fix

Maximum 8 findings.

End with:

BLOCKING FINDINGS:
...

NON-BLOCKING FINDINGS:
...

VERDICT: APPROVE | CHANGES REQUESTED

Do not report formatting or stylistic preferences unless they materially
affect correctness or maintainability.
""".strip(),
    )


@mcp.tool()
def find_edge_cases(
    requirements: str,
    implementation_context: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """Identify realistic edge cases and failure scenarios for a feature."""
    context = f"""
REQUIREMENTS
{requirements}

IMPLEMENTATION CONTEXT
{implementation_context or "(not supplied)"}
""".strip()
    return _call_model(
        tool_name="find_edge_cases",
        role="software quality engineer specializing in edge-case and failure analysis",
        task="Identify the realistic edge cases and failure scenarios most likely to expose defects in this feature.",
        context=context,
        max_tokens=1800,
        think=think,
        model=model,
        additional_rules="""
Identify only realistic edge cases that could materially affect correctness.

Consider only relevant categories:

- Input and validation
- State transitions
- Data and persistence
- Concurrency and idempotency
- External dependencies
- Authorization and security
- Compatibility and regression
- Operational failures

Return at most 10 edge cases.

For each edge case use exactly this compact format:

### <short title>
- Scenario: <one concise sentence>
- Expected: <one concise sentence>
- Risk: <one concise sentence>
- Test: <one concise sentence>

Rules:

- Maximum 10 edge cases.
- Prioritize Critical and High-value cases first.
- Do not restate requirements.
- Do not summarize the implementation.
- Do not propose new business rules.
- Do not invent authorization requirements.
- Do not invent ledger/accounting behavior.
- If behavior depends on an undefined business rule, say:
  POLICY DECISION REQUIRED: <what must be clarified>
- Clearly distinguish confirmed behavior from assumptions.
- Do not include theoretical or low-value cases.
""".strip(),
    )


@mcp.tool()
def generate_tests(
    code_or_change: str,
    requirements: str = "",
    testing_context: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """
    Design a focused, high-value test suite for supplied code or a proposed change.

    Use this after requirements and implementation context are sufficiently known.
    """
    context = f"""
REQUIREMENTS
{requirements or "(not supplied)"}

TESTING CONTEXT
{testing_context or "(not supplied)"}

CODE OR CHANGE
{code_or_change}
""".strip()
    return _call_model(
        tool_name="generate_tests",
        role="senior software test engineer",
        task=(
            "Design the smallest high-value test suite that provides strong "
            "confidence in this implementation while avoiding assumptions that "
            "are not supported by the supplied requirements or code."
        ),
        context=context,
        max_tokens=2200,
        think=think,
        model=model,
        additional_rules="""
Return at most 10 tests.

Use exactly this compact format for each test:

### <test name>
- Purpose: <one sentence>
- Setup: <one sentence>
- Action: <one sentence>
- Assert: <one or two sentences>

Group tests only when useful under:

MUST-HAVE
EDGE CASES
INTEGRATION

Rules:

- Maximum 10 tests total.
- Prioritize correctness, data integrity, authorization, idempotency,
  state transitions, and regressions.
- Do not restate requirements.
- Do not summarize the implementation.
- Do not invent HTTP status codes.
- Do not invent business rules.
- Do not assume unresolved policy decisions.
- If a test depends on undefined behavior, write:

  BLOCKED BY POLICY: <missing decision>

- Clearly distinguish confirmed behavior from unresolved behavior.
- Do not propose frontend tests when no stable frontend contract exists.
- Do not generate full test source code unless explicitly requested.
- Prefer integration tests when repository behavior spans persistence,
  transactions, authorization, or external-import logic.
- Avoid low-value duplication.
""".strip(),
    )


@mcp.tool()
def debug(
    problem: str,
    evidence: str = "",
    relevant_code: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """
    Independently analyze a bug, failure, stack trace, log, or unexpected
    runtime behavior.
    """
    context = f"""
OBSERVED PROBLEM
{problem}

EVIDENCE
{evidence or "(not supplied)"}

RELEVANT CODE
{relevant_code or "(not supplied)"}
""".strip()
    return _call_model(
        tool_name="debug",
        role="senior debugging engineer",
        task=(
            "Analyze the supplied failure evidence and identify the most "
            "plausible root cause. Recommend the smallest evidence-driven "
            "next steps to verify the diagnosis."
        ),
        context=context,
        max_tokens=2200,
        think=think,
        model=model,
        additional_rules="""
Return exactly this structure:

MOST LIKELY ROOT CAUSE
<concise explanation>

SUPPORTING EVIDENCE
- ...
- ...

UNCERTAINTY
- ...

OTHER PLAUSIBLE HYPOTHESES
1. ...
2. ...

NEXT DIAGNOSTIC STEPS
1. ...
2. ...
3. ...

MINIMAL FIX
<only if sufficiently supported by evidence>

CONFIDENCE: High | Medium | Low

Rules:

- Maximum 3 alternative hypotheses.
- Maximum 5 diagnostic steps.
- Base every conclusion on supplied evidence.
- Do not invent repository behavior.
- Do not invent logs, APIs, entities, or test results.
- If evidence is insufficient, say so explicitly.
- Do not restate the entire stack trace or supplied code.
- Do not implement the fix.
- Be concise.
""".strip(),
    )


@mcp.tool()
def second_opinion(
    question: str,
    context: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """Ask the configured Ollama model for an independent technical second opinion."""
    return _call_model(
        tool_name="second_opinion",
        role="independent senior software engineer",
        task=question,
        context=context,
        max_tokens=1000,
        think=think,
        model=model,
        additional_rules="""
Do not automatically agree with the primary agent.

Return:

FACTS
...

ASSUMPTIONS
...

RECOMMENDATION
...

RISKS / TRADEOFFS
...

CONFIDENCE: High | Medium | Low

Keep the answer concise.
""".strip(),
    )


@mcp.tool()
def delegate(
    task: str,
    context: str = "",
    role: str = "software engineer",
    model: str = "",
    think: bool = False,
    max_tokens: int = 1400,
    num_ctx: int = 0,
) -> str:
    """
    Generic fallback for delegating a bounded engineering task to a local
    Ollama model.

    max_tokens (default 1400) caps the response length. Raise it (e.g.
    4096-8192) when the task must return full source files rather than a
    short analysis — the default is tuned for concise findings, not
    code generation, and a task that needs more room will otherwise be
    silently truncated mid-file.

    num_ctx (default: the server's OLLAMA_NUM_CTX, currently 32768) caps
    how much of the supplied context the model can actually see. Raise it
    if you're passing a large amount of context (e.g. multiple full files)
    and suspect the model is missing content from earlier in the prompt.
    """
    return _call_model(
        tool_name="delegate",
        role=role,
        task=task,
        context=context,
        max_tokens=max_tokens,
        think=think,
        model=model,
        num_ctx=num_ctx,
        additional_rules="""
Return a concise, actionable answer appropriate to the assigned role.

Prefer verified findings over general advice.
""".strip(),
    )


@mcp.tool()
def review_screenshot(
    question: str,
    image_path: str = "",
    image_base64: str = "",
    context: str = "",
    model: str = "",
    think: bool = False,
) -> str:
    """
    Review a UI screenshot, mockup, or design export (e.g. a Figma frame)
    and answer a focused visual question about it.

    Prefer image_path: a local file path the server reads and encodes
    itself. Passing a large image_base64 string directly as a tool-call
    argument has been observed to get corrupted in MCP transport (an LLM
    regenerating a multi-thousand-character blob token by token is not a
    reliable copy) — image_path is a short string with no such risk.
    image_base64 (no data:image/...;base64, prefix) remains available for
    callers that cannot expose a local file path.
    """
    if image_path:
        try:
            image_base64 = base64.b64encode(
                Path(image_path).read_bytes()
            ).decode("ascii")
        except Exception as exc:
            return (
                f"Worker could not read image_path '{image_path}': "
                f"{type(exc).__name__}: {exc}"
            )
    elif not image_base64:
        return "Worker: supply either image_path or image_base64."

    return _call_model(
        tool_name="review_screenshot",
        role="UI/visual design reviewer",
        task=question,
        context=context,
        images=[image_base64],
        max_tokens=1200,
        think=think,
        model=model,
        additional_rules="""
Only describe what is visibly present in the image; do not invent elements,
text, or colors that are not clearly visible.

Do not claim exact pixel measurements, hex color codes, or contrast ratios
you cannot verify from the image alone — describe them qualitatively
instead.

Focus on concrete, actionable observations: layout, hierarchy, spacing,
consistency, readability, and alignment with any design-system rules
supplied in context.

Do not invent a design-system rule that was not supplied in context.
""".strip(),
    )


@mcp.tool()
def status() -> str:
    """Report the ollama-worker's configuration and live Ollama connectivity."""

    tags = _ollama_get("/api/tags")
    ps = _ollama_get("/api/ps")

    lines = [
        f"OLLAMA_URL: {OLLAMA_URL}",
        f"OLLAMA_MODEL (default): {OLLAMA_MODEL}",
        "",
        "INSTALLED MODELS",
        "----------------------------------------------------------------------",
    ]

    if isinstance(tags, str):
        lines.append(f"ERROR: {tags}")
    else:
        for entry in tags.get("models", []):
            details = entry.get("details", {})
            size_gb = entry.get("size", 0) / 1_000_000_000
            capabilities = ", ".join(entry.get("capabilities", [])) or "?"
            lines.append(
                f"- {entry.get('name', '?')} "
                f"({details.get('parameter_size', '?')}, "
                f"{details.get('quantization_level', '?')}, "
                f"{size_gb:.1f} GB on disk, "
                f"capabilities: {capabilities})"
            )

    lines.append("")
    lines.append("CURRENTLY LOADED")
    lines.append("----------------------------------------------------------------------")

    if isinstance(ps, str):
        lines.append(f"ERROR: {ps}")
    else:
        loaded = ps.get("models", [])
        if not loaded:
            lines.append("(none loaded)")
        for entry in loaded:
            vram_gb = entry.get("size_vram", 0) / 1_000_000_000
            lines.append(
                f"- {entry.get('name', '?')} "
                f"({vram_gb:.1f} GB VRAM, expires_at={entry.get('expires_at', '?')})"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )
