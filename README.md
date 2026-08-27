# ollama-worker

A local MCP server that gives a primary coding agent (Claude Code, Codex, etc.)
a **second, independent opinion** — plan review, code review, edge-case
discovery, test design, debugging, and a screenshot/UI reviewer — powered by
any model already installed in a local [Ollama](https://ollama.com) instance.

It is not tied to any one model family. Every tool accepts a per-call `model`
override (any tag installed in Ollama) and a `think` flag; both fall back to
the server's configured defaults when omitted. It has been exercised with
Qwen, gpt-oss, and Gemma models — see [Models tested](#models-tested) below.

The worker has no repository access of its own. The caller supplies all
relevant requirements, code, diffs, logs, and test output; the worker returns
a structured, bounded analysis. The primary agent remains responsible for
repository inspection, implementation, testing, and final decisions.

**Origin:** this server started life as a Qwen-specific tool, but the
project's own `_call_model`/`_resolve_think`/`_pad_max_tokens_for_thinking`
parameters were themselves iteratively tuned with the help of `gemma4:12b`
before this generic, multi-model version — and its benchmark suite — existed.

## Tools

| Tool | Purpose |
|---|---|
| `review_plan` | Independent review of an implementation plan before coding starts. Returns `APPROVE \| APPROVE WITH CHANGES \| REWORK` plus up to 5 material findings. |
| `final_plan_check` | Concise `GO \| NO-GO` readiness check on a revised plan. |
| `review_diff` | Independent code review of a git diff — correctness, security, regressions, missing tests. Returns severities, a fix per finding, and `APPROVE \| CHANGES REQUESTED`. |
| `find_edge_cases` | Realistic edge cases and failure scenarios for a feature (input/validation, state, concurrency, security, etc.), up to 10. |
| `generate_tests` | A focused, high-value test suite for supplied code or a proposed change. |
| `debug` | Independent analysis of a bug, stack trace, log, or unexpected runtime behavior. |
| `second_opinion` | An independent technical opinion on a specific question — explicitly instructed not to auto-agree with the primary agent. |
| `delegate` | Generic fallback for delegating a bounded engineering task not covered by a more specific tool. `max_tokens` and `num_ctx` are caller-adjustable here since task shape varies widely. |
| `review_screenshot` | Visual review of a UI screenshot, mockup, or design export against a focused question. Prefer `image_path` over `image_base64` — a large base64 blob passed as a literal tool-call argument has been observed to get corrupted in MCP transport when an LLM regenerates it token by token. |
| `status` | Reports the worker's configuration, every model installed in Ollama, and what's currently loaded in VRAM. No model call — safe, cheap health check. |

All review-style tools return a common structure: a verdict line, findings
with severity/location/why-it-matters/minimal-fix, and a compact
`[ollama-worker id=... model=... prompt=N output=N prompt_speed=X generation=Y total=Zs done=reason]`
telemetry line appended to every response.

## Requirements

- A local [Ollama](https://ollama.com) instance with at least one model
  pulled (`ollama pull <model>`).
- Python 3.10+ and the official [`mcp`](https://pypi.org/project/mcp/) SDK.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Configuration

Everything is an environment variable with a sane default:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server address. |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Default model when a tool call omits `model`. |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after a call. |
| `OLLAMA_REQUEST_TIMEOUT` | `240` (s) | Timeout for a model call. |
| `OLLAMA_STATUS_TIMEOUT` | `10` (s) | Timeout for `status`'s lightweight Ollama queries. |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature — low by design for a reviewer role. |
| `OLLAMA_NUM_CTX` | `32768` | Default context window; per-call override via `delegate`'s `num_ctx`. |
| `OLLAMA_WORKER_LOG` | `~/tools/ollama-worker-mcp/ollama-worker.log` | Full request/response log path. |
| `OLLAMA_LOG_FULL_CONTEXT` | `1` | Set to `0`/`false` to log only context length, not full content. |
| `OLLAMA_WORKER_LOG_MAX_BYTES` | `10485760` (10 MB) | Log rotation threshold. |

### Registering with Claude Code

Add to `~/.claude.json` (or the project's MCP config):

```json
{
  "mcpServers": {
    "ollama-worker": {
      "type": "stdio",
      "command": "/path/to/ollama-worker/.venv/bin/python",
      "args": ["/path/to/ollama-worker/server.py"],
      "env": {}
    }
  }
}
```

## A `gpt-oss` quirk this server works around

`gpt-oss` models ignore a plain boolean `think` value and silently fall back
to "medium" reasoning effort regardless of what's requested — they require an
explicit `"low"`/`"medium"`/`"high"` string instead. Every other model tested
here (Qwen, Gemma) accepts the plain boolean. `_resolve_think()` translates
`think=True/False` into whatever the resolved model actually expects, and
`_pad_max_tokens_for_thinking()` pads `num_predict` when reasoning is active —
`gpt-oss` at `"high"` effort was observed to exhaust an unpadded 1400-token
budget on hidden thinking and return no answer at all.

## Hardware tested

- **GPU:** AMD Radeon RX 9070 XT, 16 GB VRAM, RDNA4 (`gfx1201`)
- **OS:** Kubuntu 26.04 LTS
- **ROCm:** 7.1.1 (Ubuntu packages) — the GPU is on ROCm's known-issue list
  for HSA-discovery hangs on `gfx1200`/`gfx1201` in 7.1.1
  ([ROCm/ROCm#5812](https://github.com/ROCm/ROCm/issues/5812)); on this
  machine `rocminfo` and Ollama's ROCm backend worked without hitting that
  hang, but it wasn't hit under sustained load either — treat as "works so
  far," not "confirmed unaffected."
- **Backend used:** Ollama's ROCm backend (default). Vulkan (`OLLAMA_VULKAN=1`)
  is available as a fallback; note that on the Ollama version tested,
  `OLLAMA_VULKAN=0` does not reliably disable Vulkan once compiled in
  ([ollama/ollama#13212](https://github.com/ollama/ollama/issues/13212)) —
  confirm the active backend from server logs, not the env var alone.
- **Ollama:** v0.32.14

## Models tested

| Model | Quant | Size | GPU @ 32K ctx | Gen speed | Multimodal | Ollama-reported capabilities | Reasoning control |
|---|---|---|---|---|---|---|---|
| `qwen3.8-27b-q3` (`unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL`) | Q3_K_L (3-bit) | 14.1 GB | 100% GPU | ~29.3 tok/s | Yes | `completion, vision` | Boolean via the worker; `low`/`medium`/`high` via Ollama's native `--think` flag (verified directly — see note below) |
| `qwen3.8:27b` (official Ollama library tag) | Q4_K_M (4-bit) | ~18 GB | Partial CPU offload | ~14–15 tok/s | Yes | `completion, vision` (same base model) | Same as above |
| `gpt-oss:20b` (server default) | MXFP4 | 12.0 GB | 100% GPU | ~97–100 tok/s | No | `completion, tools, thinking` | Always reasons, no true "off" state — the worker's boolean maps to `low`/`high` effort |
| `gemma4:latest` | Q4_K_M | 3.3 GB | 100% GPU | ~74 tok/s | No | `completion, tools, thinking` | Boolean |
| `gemma4:12b` | Q4_K_M | 8.1 GB | 100% GPU | ~52 tok/s | Yes | `completion, tools, thinking, vision` | Boolean |

**Note on the `qwen3.8-27b-q3`/`qwen3.8:27b` capability gap:** Ollama's
reported capabilities for these two (both sourced from a community GGUF
rather than an Ollama-library-native template) omit `thinking` and `tools`
entirely, despite `think=low/medium/high` working correctly when tested
directly against the Ollama CLI earlier in this project. Treat the
`Ollama-reported capabilities` column as what Ollama itself claims to
know about a model, not a ground-truth functional test — it can undersell
a model sourced outside the official library. Native tool-calling was not
functionally tested through this worker for any model listed here; the
worker doesn't itself invoke tool-calling on the underlying model.

All four stay 100% GPU-resident on this 16 GB card at 32K context — only
the 4-bit `qwen3.8:27b` quant (not shown as a separate row: same model as
`qwen3.8-27b-q3`, just Q4_K_M instead of Q3_K_L) needs partial CPU offload,
at roughly 2x slower generation and ~7.5x slower prompt processing than its
3-bit counterpart. Generation speed otherwise tracks model size, as
expected: `gemma4:latest` (8B) and `gpt-oss:20b` (20B) both substantially
outrun `qwen3.8-27b-q3` (27B); `gemma4:12b` sits in between, presumably the
cost of its added vision encoder and larger param count relative to
`gemma4:latest`. A `gemma4-131k:latest` tag (same 11.9B vision-capable
model, extended context) is also installed but untested here.

Two tools were run against all four models — `review_diff` against a diff
with a planted SQL-injection vulnerability and an off-by-one bug, and
`second_opinion` on an architecture question. All four correctly caught
both planted bugs with a `CHANGES REQUESTED` verdict, though not with
identical framing: `qwen3.8-27b-q3`, `gpt-oss:20b`, and `gemma4:latest`
called the wrong divisor Critical/High directly, while `gemma4:12b` framed
the `len(values)==1` case as a High-severity division-by-zero bug and the
general wrong-formula issue as a separate Medium finding — different
organization, same underlying defect correctly identified either way.
`gemma4:latest` additionally flagged the empty-list `ZeroDivisionError`
edge case unprompted, matching `qwen3.8-27b-q3`; `gemma4:12b` did not call
that case out separately. `find_edge_cases` and `generate_tests` were only
exercised against `qwen3.8-27b-q3` (on a Minimum-Window-Substring
implementation), not against the other three.

Qualitatively, `gpt-oss:20b`'s free-form `second_opinion` answers were
noticeably faster but occasionally less precise in their technical framing
than `qwen3.8-27b-q3`'s — spot-check free-form (non-structured) answers
before trusting them unread, regardless of which model you point the
worker at.

`review_screenshot` was tested against `gemma4:12b` on a synthetic login-form
mockup with two deliberately planted visual defects: a "Password" label
rendered in near-white text on a white field (effectively invisible), and a
"Log In" button positioned so it overflows past the card's bottom edge. The
model clearly processed the actual image — its findings referenced specific,
correct details of the mockup, not generic boilerplate — but it **missed the
more obvious of the two bugs**: it never noted that the button overflows the
card boundary, and only vaguely characterized the label issue as "lower
visual weight" rather than flagging that the label is effectively unreadable.
Unlike every text-based tool call tested above, this is a case where the
worker's output should not be trusted at face value — treat `review_screenshot`
results as a starting point, not a substitute for actually looking at the
image yourself.

For a much deeper dive into these same models' actual code/design output
quality — not just whether a tool call succeeds — see
[driftnote-bakeoff](https://github.com/auxmedrano/driftnote-bakeoff)
([live site](https://auxmedrano.github.io/driftnote-bakeoff/)): four models
given an identical realistic front-end brief (a self-contained landing
page), scored against a 23-item checklist, plus a follow-up experiment on
whether internal reasoning or an explicit plan-then-build workflow produces
better results.

## Known limitations

- `server_info.version` is now populated (`0.1.0`) but not wired to any
  automated bump — update it by hand in `server.py` on real changes.
- The worker log (`OLLAMA_WORKER_LOG`) includes full task/context/response
  content by default (`OLLAMA_LOG_FULL_CONTEXT=1`) — treat the log file as
  sensitive if the tasks you delegate include proprietary code or secrets.
