"""
tools.py — System 2 agent tool implementations.

Tool schemas use the OpenAI function-calling format, which DeepSeek's API
is fully compatible with. No other changes from the Anthropic version.

Each function implements one tool the LLM agent can call.
TOOL_DEFINITIONS contains the OpenAI-format tool schemas.
execute_tool() dispatches tool names to functions.

To adapt to your real system:
  - Replace JSON file paths with your real data sources.
  - Replace SESSION_STATE with your real System 1 runtime state object.
"""
from __future__ import annotations   # dict[str,Any] syntax requires Python 3.9+ without this

import json
from pathlib import Path
from typing import Any

# ── Data paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str) -> dict:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# ── Session state (injected at runtime by System 1) ───────────────────────────
# Replace this with your real runtime state object.
# System 1 should update this dict continuously as the session progresses.

SESSION_STATE: dict[str, Any] = {
    # Task progression log L: list of completed task records
    # Each record: {task_id, hand, start_time, end_time, duration_seconds,
    #               mean_confidence, error_flags}
    "progression_log": [],

    # TPG traversal record D_t: set of completed task IDs
    "completed_tasks": set(),

    # Current in-progress tasks per hand (None if hand is idle/null)
    "current_task": {"left": None, "right": None},

    # Eligibility set F_t: list of currently eligible task IDs
    "eligible_tasks": [],

    # Active anomaly flags
    "anomaly_flags": [],

    # Procedural errors logged by System 1 (out-of-order actions)
    "procedural_errors": [],

    # Tasks that were never performed before the session ended
    # Each entry: {"id": str, "label": str}
    "missed_tasks": [],

    # Idle / total time derived from LiveSessionLog (seconds)
    "idle_seconds": None,
    "total_session_seconds": None,

    # Session timestamps (Unix seconds)
    "session_start_time": None,
    "session_end_time": None,
}


# ── Tool implementations ──────────────────────────────────────────────────────

def get_task_procedure(task_id: str) -> dict:
    """
    Retrieve step-by-step procedure for a task from knowledge base K.
    """
    kb = _load("knowledge_base.json")
    if task_id not in kb["tasks"]:
        return {"error": f"Task '{task_id}' not found in knowledge base."}
    entry = kb["tasks"][task_id]
    return {
        "task_id": task_id,
        "task_name": entry["task_name"],
        "procedure": entry["procedure"],
        "common_errors": entry["common_errors"],
        "estimated_duration_seconds": entry["estimated_duration_seconds"],
    }


# Tracks which task IDs retrieve_media was called for during one agent run.
# Reset by calling clear_media_log() at the start of each run_system2 invocation.
MEDIA_RETRIEVED_TASK_IDS: list = []


def clear_media_log() -> None:
    """Reset the per-run media retrieval log. Call before each agent invocation."""
    MEDIA_RETRIEVED_TASK_IDS.clear()


def retrieve_media(task_id: str) -> dict:
    """
    Retrieve reference images and video clips for a task from asset database B.

    NOTE: Currently uses exact task_id lookup. To enable semantic (RAG-style)
    retrieval, replace the filter with a vector similarity search over asset
    embeddings indexed by task description and tags.
    """
    db = _load("multimedia_db.json")
    assets = [a for a in db["assets"] if a["task_id"] == task_id]
    if not assets:
        return {"error": f"No media assets found for task '{task_id}'."}
    # Record this call so run_system2.py can include it in the output JSON
    if task_id not in MEDIA_RETRIEVED_TASK_IDS:
        MEDIA_RETRIEVED_TASK_IDS.append(task_id)
    return {
        "task_id": task_id,
        "asset_count": len(assets),
        "assets": assets,
    }


def get_critical_parameters(task_id: str) -> dict:
    """
    Retrieve critical parameters (torque, force limits, part numbers, etc.)
    for a task from knowledge base K.
    """
    kb = _load("knowledge_base.json")
    if task_id not in kb["tasks"]:
        return {"error": f"Task '{task_id}' not found in knowledge base."}
    entry = kb["tasks"][task_id]
    return {
        "task_id": task_id,
        "task_name": entry["task_name"],
        "critical_parameters": entry["critical_parameters"],
        "common_errors": entry["common_errors"],
    }


def query_progression_log() -> dict:
    """
    Return the full task progression log L for the current session.
    Derives current in-progress tasks and completed task set from the log.
    """
    return {
        "session_start_time": SESSION_STATE["session_start_time"],
        "session_end_time": SESSION_STATE["session_end_time"],
        "current_task": SESSION_STATE["current_task"],
        "completed_tasks": list(SESSION_STATE["completed_tasks"]),
        "completed_count": len(SESSION_STATE["completed_tasks"]),
        "progression_log": SESSION_STATE["progression_log"],
        "procedural_errors": SESSION_STATE["procedural_errors"],
        "anomaly_flags": SESSION_STATE["anomaly_flags"],
    }


def compute_performance_stats() -> dict:
    """
    Compute per-task z-scores comparing observed durations against
    the reference performance dataset R.
    Z_THRESHOLD controls flagging sensitivity (default 1.0 sigma).
    """
    Z_THRESHOLD = 1.0

    ref = _load("reference_performance.json")
    kb = _load("knowledge_base.json")
    log = SESSION_STATE["progression_log"]

    if not log:
        return {"error": "No completed tasks in session log yet."}

    stats = []
    slower_than_average = []
    faster_than_average = []

    for record in log:
        task_id = record.get("task_id")
        if task_id not in ref["tasks"]:
            continue
        observed = record.get("duration_seconds")
        if observed is None:
            continue

        r = ref["tasks"][task_id]
        mu, sigma = r["mean_seconds"], r["std_seconds"]
        z = (observed - mu) / sigma if sigma > 0 else 0.0

        flag_type = (
            "slower_than_average" if z > Z_THRESHOLD
            else "faster_than_average" if z < -Z_THRESHOLD
            else "within_normal_range"
        )
        entry = {
            "task_id": task_id,
            "task_name": kb["tasks"].get(task_id, {}).get("task_name", task_id),
            "observed_seconds": round(observed, 1),
            "reference_mean_seconds": mu,
            "reference_std_seconds": sigma,
            "z_score": round(z, 2),
            "flagged": abs(z) > Z_THRESHOLD,
            "flag_type": flag_type,
        }
        stats.append(entry)
        if z > Z_THRESHOLD:
            slower_than_average.append(task_id)
        elif z < -Z_THRESHOLD:
            faster_than_average.append(task_id)

    completed = list(SESSION_STATE["completed_tasks"])
    readiness_score = (
        sum(1 for s in stats if not s["flagged"]) / len(completed)
        if completed else 0.0
    )

    # ── Idle-time analysis ────────────────────────────────────────────────────
    idle_ref = ref.get("idle_reference", {})
    ref_idle_mean   = idle_ref.get("mean_idle_seconds",  22.5)
    ref_idle_std    = idle_ref.get("std_idle_seconds",    6.6)
    ref_idle_mean_f = idle_ref.get("mean_idle_fraction",  0.077)
    ref_idle_std_f  = idle_ref.get("std_idle_fraction",   0.020)
    ref_high_thresh = idle_ref.get("high_idle_threshold_fraction", 0.097)

    session_idle_s = SESSION_STATE.get("idle_seconds", None)
    session_total_s = SESSION_STATE.get("total_session_seconds", None)
    idle_analysis: dict = {}
    if session_idle_s is not None and session_total_s and session_total_s > 0:
        session_idle_frac = session_idle_s / session_total_s
        idle_z = ((session_idle_s - ref_idle_mean) / ref_idle_std
                  if ref_idle_std > 0 else 0.0)
        idle_analysis = {
            "session_idle_seconds":    round(session_idle_s, 1),
            "session_total_seconds":   round(session_total_s, 1),
            "session_idle_fraction":   round(session_idle_frac, 3),
            "reference_mean_idle_seconds": ref_idle_mean,
            "reference_mean_idle_fraction": ref_idle_mean_f,
            "reference_high_idle_threshold_fraction": ref_high_thresh,
            "idle_z_score": round(idle_z, 2),
            "high_idle_flag": session_idle_frac > ref_high_thresh,
            "idle_interpretation": (
                "High idle time — operator may be unfamiliar with certain steps or "
                "lost track of the assembly sequence."
                if session_idle_frac > ref_high_thresh else
                "Normal idle time — within reference range."
            ),
        }
    else:
        idle_analysis = {
            "session_idle_seconds": None,
            "reference_mean_idle_seconds": ref_idle_mean,
            "reference_mean_idle_fraction": ref_idle_mean_f,
            "note": "Idle time not recorded in current session log.",
        }

    return {
        "z_threshold": Z_THRESHOLD,
        "task_stats": stats,
        "slower_than_average": slower_than_average,
        "faster_than_average": faster_than_average,
        "readiness_score": round(readiness_score, 3),
        "readiness_interpretation": (
            "Excellent" if readiness_score >= 0.9
            else "Good" if readiness_score >= 0.75
            else "Needs improvement" if readiness_score >= 0.5
            else "Significant gaps identified"
        ),
        "idle_analysis": idle_analysis,
    }


def get_tpg_state() -> dict:
    """
    Return current TPG traversal state: D_t, F_t, progress, and error flags.
    """
    tpg = _load("tpg.json")
    all_tasks = list(tpg["tasks"].keys())
    completed = list(SESSION_STATE["completed_tasks"])
    remaining = [t for t in all_tasks if t not in completed]
    progress_pct = round(100 * len(completed) / len(all_tasks), 1) if all_tasks else 0.0

    return {
        "total_tasks": len(all_tasks),
        "completed_tasks": completed,
        "completed_count": len(completed),
        "remaining_tasks": remaining,
        "eligible_next_tasks": SESSION_STATE["eligible_tasks"],
        "progress_percent": progress_pct,
        "procedural_errors": SESSION_STATE["procedural_errors"],
        "anomaly_flags": SESSION_STATE["anomaly_flags"],
        "session_complete": len(remaining) == 0,
    }


# ── OpenAI-format tool schema definitions ─────────────────────────────────────
# DeepSeek's API is OpenAI-compatible and uses this exact format.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_task_procedure",
            "description": (
                "Retrieve the step-by-step procedure for an assembly task from "
                "the knowledge base. Returns numbered steps, common errors, and "
                "estimated duration. Use when the operator asks how to perform "
                "a specific task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task identifier, e.g. 'T001', 'T008'.",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_media",
            "description": (
                "Retrieve reference images and video clips for an assembly task "
                "from the multimedia asset database. Returns assets with URIs, "
                "types (image/video), and captions. Use to provide visual guidance "
                "alongside procedural narration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task identifier, e.g. 'T002'.",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_critical_parameters",
            "description": (
                "Retrieve critical technical parameters for an assembly task: "
                "torque values, force limits, part numbers, orientation constraints, "
                "and safety precautions. Use to generate the 'what to watch out for' "
                "checklist in the guidance response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task identifier, e.g. 'T008'.",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_progression_log",
            "description": (
                "Return the full task progression log for the current session, "
                "including completed tasks, current in-progress tasks per hand, "
                "timing records, confidence scores, and any logged errors. "
                "Always call this first in both guidance and review workflows."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_performance_stats",
            "description": (
                "Compute per-task performance z-scores by comparing observed task "
                "durations against the reference worker population dataset. Returns "
                "flagged slow/fast tasks, per-task statistics, and a readiness score. "
                "Use in the post-session review workflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tpg_state",
            "description": (
                "Return the current TPG traversal state: completed tasks (D_t), "
                "eligible next tasks (F_t), overall progress percentage, and any "
                "active procedural error or anomaly flags. Use to understand where "
                "the operator is in the assembly procedure."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch a tool call from the LLM agent to the appropriate function.
    Returns a JSON string passed back as the tool message content.
    """
    dispatch = {
        "get_task_procedure":      lambda i: get_task_procedure(i["task_id"]),
        "retrieve_media":          lambda i: retrieve_media(i["task_id"]),
        "get_critical_parameters": lambda i: get_critical_parameters(i["task_id"]),
        "query_progression_log":   lambda i: query_progression_log(),
        "compute_performance_stats": lambda i: compute_performance_stats(),
        "get_tpg_state":           lambda i: get_tpg_state(),
    }

    if tool_name not in dispatch:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        result = dispatch[tool_name](tool_input)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})