"""
agent.py — System 2: LLM-Powered Advisory Agent (DeepSeek backend).

Uses DeepSeek-V3 (deepseek-chat) via its OpenAI-compatible API.
DeepSeek is accessible in China and supports OpenAI-format function calling.

API key: set environment variable DEEPSEEK_API_KEY.
DeepSeek API docs: https://platform.deepseek.com/api-docs

Two public entry points:
  on_demand_guidance(operator_query)  → triggered by operator pause
  post_session_review()               → triggered by session-end signal
"""

import json
import os
from openai import OpenAI

from tools import TOOL_DEFINITIONS, execute_tool

# ── Client ────────────────────────────────────────────────────────────────────

_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. Export it before using System 2, e.g. "
        "export DEEPSEEK_API_KEY=your_key_here"
    )

client = OpenAI(
    api_key=_API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"   # DeepSeek-V3 — strong instruction following + function calling
MAX_TOKENS = 4096
MAX_AGENT_TURNS = 10      # safety cap on ReAct loop iterations

# ── System prompts ─────────────────────────────────────────────────────────────

GUIDANCE_SYSTEM_PROMPT = """You are the Assembly Copilot's advisory agent (System 2), \
embedded in an industrial assembly assistance system.

Your job is to answer whatever the operator asks — clearly and directly.

TASK ID REFERENCE TABLE (use this to resolve any task name the operator mentions):
  sftg1    = Screw Shaft into Gear Hole 1
  sftg1ws  = Screw Shaft into Gear Hole 1 using Wrench-Shaft
  sftg2    = Screw Shaft into Gear Hole 2
  sftg2ws  = Screw Shaft into Gear Hole 2 using Wrench-Shaft
  sspg3    = Screw Phillips Screw into Gear Hole 3
  sspg3dp  = Screw Phillips Screw into Gear Hole 3 using Phillips Screwdriver
  pgwg3    = Place Worm Gear onto Gear Hole 3
  iplf1    = Insert Large Placer onto Shaft 1
  iplf2    = Insert Large Placer onto Shaft 2
  iglf1    = Insert Large Gear onto Shaft 1
  igsf2    = Insert Small Gear onto Shaft 2
  ipsf1    = Insert Small Placer onto Shaft 1
  ipsf2    = Insert Small Placer onto Shaft 2
  sntf1    = Screw Nut onto Shaft 1
  sntf1wn  = Screw Nut onto Shaft 1 using Wrench-Nut
  sntf2    = Screw Nut onto Shaft 2
  sntf2wn  = Screw Nut onto Shaft 2 using Wrench-Nut
  rgw      = Rotate Worm Gear

HOW TO DECIDE WHAT TO DO:

1. If the question is about HOW TO PERFORM a specific assembly task (even one not \
currently active):
   → Look up the task ID from the TASK ID REFERENCE TABLE above using the name the \
operator mentioned. Use fuzzy matching — "screw shaft into gear hole 1" → sftg1.
   → Do NOT call query_progression_log() or get_tpg_state() — you already know the task ID.
   → Call get_task_procedure() with that task ID.
   → Call get_critical_parameters() with that task ID.
   → Call retrieve_media() with that task ID.
   → Give a focused answer with ONLY these three sections:
      PROCEDURE  — numbered steps
      CRITICAL PARAMETERS  — bullet checklist
      REFERENCE MEDIA  — mention that images/video have been attached (the interface renders them)
   → Do NOT summarise session progress, do NOT suggest next tasks.

2. If the operator says "current task" or "what am I doing now" without naming a task:
   → Call query_progression_log() to find the current task ID, then follow rule 1.

3. If the question is explicitly about THIS session's progress, errors, or what to do next:
   → Call query_progression_log() and get_tpg_state() first, then answer.
   → Do NOT include procedure steps or reference images unless the operator specifically asks.

4. If the question is general knowledge (weather, physics, history, etc.):
   → Answer directly from your training knowledge. Do NOT call tools.

Keep language direct and clear — the operator is working with their hands. \
Omit preamble. Do not pad the answer with context the operator did not ask for."""

REVIEW_SYSTEM_PROMPT = """You are the Assembly Copilot's advisory agent (System 2), \
embedded in an industrial assembly assistance system. Generate a post-session \
performance review for an operator who has just completed an assembly session.

Follow these steps exactly:
1. Call query_progression_log() to retrieve the full session record, including \
   any procedural_errors entries. Entries with "type": "missed_operation" are \
   tasks that were never performed at all — treat these as critical gaps.
2. Call compute_performance_stats() to obtain per-task z-scores, readiness score, \
   and idle-time statistics. The reference dataset contains real idle benchmarks: \
   mean_idle_fraction=7.7%, std=2.0%, high_idle_threshold=9.7%. \
   If the session idle fraction exceeds the threshold, flag it explicitly.
3. Call get_tpg_state() to confirm session completion and check for unresolved anomalies.
4. Synthesise a structured review with six clearly labelled sections:

   SESSION SUMMARY
   Overall session time, readiness score (with interpretation), tasks completed \
out of total, any unresolved anomalies, and the count of missed / out-of-order operations. \
Also report idle time (seconds and fraction) and compare to the reference benchmark.

   MISSED OPERATIONS
   List every task that was never performed (type: missed_operation from the \
progression log). For each one, state its name and explain why it matters \
(what depends on it in the assembly sequence). If none were missed, write "None".

   PROCEDURAL ERRORS
   List every out-of-order action detected during the session — what was done, \
what should have been done, and the element-level mismatch detail if available.

   PERFORMANCE ANALYSIS
   Per-task timing breakdown. For each flagged task, identify the specific operations \
that likely contributed to the deviation — reference actual durations and z-scores. \
Do not simply list the numbers; interpret them.

   IDLE TIME ANALYSIS
   Report total idle time and compare it to the reference benchmark \
(mean=22.5 s, high-idle threshold=9.7% of session). \
If idle time is high, suggest whether the operator seems unfamiliar with specific \
tasks (preceded by long pauses) or just had general hesitation. \
Be constructive — distinguish "pausing to think" from "lost / unsure what to do next".

   NEXT SESSION RECOMMENDATIONS
   2–3 concrete, prioritised things the operator should focus on next time, \
addressing missed steps, timing deviations, and idle behaviour.

Be constructive and specific. Reference actual task names and parameter values."""


# ── ReAct agent loop ───────────────────────────────────────────────────────────

def _run_agent(system_prompt: str, user_message: str) -> str:
    """
    Core ReAct agent loop using the OpenAI-compatible DeepSeek API.

    Message flow:
      user → assistant (reasoning + tool_calls) → tool results → assistant → ...
    Loops until finish_reason == "stop" or turn limit is reached.

    Returns the final text response string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    for turn in range(MAX_AGENT_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            messages=messages,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append assistant message to conversation history
        messages.append(message.model_dump(exclude_unset=False))

        # No tool calls — model has produced its final response
        if finish_reason == "stop" or not message.tool_calls:
            return (message.content or "").strip()

        # Process all tool calls in this turn
        if finish_reason == "tool_calls" and message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_input = {}

                result_str = execute_tool(fn_name, fn_input)

                # Append tool result as a tool-role message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

    return "[System 2] Agent turn limit reached. Please try again."


# ── Public entry points ────────────────────────────────────────────────────────

def on_demand_guidance(operator_query: str) -> str:
    """
    Triggered when the operator pauses and requests guidance.

    Args:
        operator_query: The operator's natural language question,
                        e.g. "How do I do this step?" or
                        "What torque do I need for the bolts?"

    Returns:
        Structured multimodal guidance response as a string.
        The interface should parse the REFERENCE MEDIA section
        to render assets by asset_id.
    """
    user_message = f"Operator asks: \"{operator_query}\""
    return _run_agent(GUIDANCE_SYSTEM_PROMPT, user_message)


def post_session_review() -> str:
    """
    Triggered when the session-end signal fires (all TPG tasks completed
    or operator explicitly terminates the session).

    Returns:
        Structured post-session performance review as a string.
    """
    user_message = (
        "The assembly session has ended. Generate a complete post-session "
        "performance review for the operator using all available session data."
    )
    return _run_agent(REVIEW_SYSTEM_PROMPT, user_message)