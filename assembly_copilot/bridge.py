"""
bridge.py — Adapter between System 1 (realtime_system1.py LiveSessionLog)
            and System 2 (Assembly_copilot SESSION_STATE).

System 1 writes a JSON file (LiveSessionLog) after every window.
This module:
  1. Loads that file from disk.
  2. Translates it into the SESSION_STATE format that System 2 tools expect.
  3. Injects the translated state into tools.SESSION_STATE in-place.

Usage from realtime_system1.py (or any driver script):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "Assembly_copilot"))

    from bridge import sync_session_state
    from agent  import on_demand_guidance, post_session_review

    # Update state from the latest live log, then call System 2:
    sync_session_state(log_path)
    answer = on_demand_guidance("What should I do next?")

LiveSessionLog schema (System 1 output)
---------------------------------------
{
  "meta":    { "video", "fps", "total_frames", "session_start_wall" },
  "status":  "in_progress" | "complete",
  "progress": { "completed", "total", "pct" },
  "current_task": {
      "id", "label", "t_start",
      "lh": { "t_start", "t_last" } | null,
      "rh": { "t_start", "t_last" } | null
  } | null,
  "eligible_next": [ {"id", "label"}, ... ],
  "completed_tasks": [
      { "id", "label", "t_start", "t_end",
        "lh": { "t_start", "t_end" } | null,
        "rh": { "t_start", "t_end" } | null }
  ],
  "errors": [
      { "t_start", "t_end",
        "wrong_action":  { "id", "label" },
        "right_action":  { "id", "label" },
        "element_mismatch": { elem: { pred, expected, match } } }
  ],
  "side_activities": [ { "type", "label", "t_start", "t_end" } ],
  "missed_tasks":    [ { "id", "label" } ]   -- filled by finalize()
}

SESSION_STATE schema (System 2 input — tools.py)
-------------------------------------------------
{
  "progression_log": [
      { task_id, hand, start_time, end_time,
        duration_seconds, mean_confidence, error_flags }
  ],
  "completed_tasks":   set[str],
  "current_task":      { "left": str|None, "right": str|None },
  "eligible_tasks":    list[str],
  "anomaly_flags":     list,
  "procedural_errors": list[dict],
  "session_start_time": float | None,   # Unix seconds
  "session_end_time":   float | None,
}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Import tools.SESSION_STATE from within this package ──────────────────────
_AC_DIR = Path(__file__).parent
sys.path.insert(0, str(_AC_DIR))
import tools                                        # noqa: E402  (path manipulation above)


# ── Public entry points ───────────────────────────────────────────────────────

def load_log(log_path: str) -> dict:
    """Load the latest LiveSessionLog JSON written by System 1."""
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


def sync_session_state(log_path: str) -> None:
    """
    One-shot sync: load System 1's live JSON log and update SESSION_STATE
    so that System 2 tools see the current assembly state.
    """
    log = load_log(log_path)
    _translate(log)


def sync_from_dict(log: dict) -> None:
    """
    Same as sync_session_state, but accepts an already-loaded dict
    (useful when the caller already holds the in-memory LiveSessionLog).
    """
    _translate(log)


# ── Internal translation logic ────────────────────────────────────────────────

def _parse_wall_time(iso_str: Optional[str]) -> Optional[float]:
    """Convert ISO-8601 wall-clock string to Unix timestamp."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def _hand_label(lh: Optional[dict], rh: Optional[dict]) -> str:
    """Determine 'left' / 'right' / 'both' from per-hand activity dicts."""
    if lh and rh:
        return "both"
    if lh:
        return "left"
    if rh:
        return "right"
    return "both"   # default when System 1 didn't record hand detail


def _translate(log: dict) -> None:
    """
    Core translation: LiveSessionLog dict → tools.SESSION_STATE (in-place).
    """
    meta = log.get("meta", {})
    session_start_time = _parse_wall_time(meta.get("session_start_wall"))

    # ── Build progression_log ─────────────────────────────────────────────────
    progression_log = []
    completed_ids: set = set()

    for entry in log.get("completed_tasks", []):
        tid = entry.get("id")
        if not tid:
            continue

        t_start_vid = entry.get("t_start", 0.0)   # seconds into video
        t_end_vid   = entry.get("t_end")

        # Convert video-relative times to Unix timestamps for System 2
        unix_start = (session_start_time or 0.0) + t_start_vid
        unix_end   = (session_start_time or 0.0) + t_end_vid if t_end_vid is not None else None

        duration = (t_end_vid - t_start_vid) if t_end_vid is not None else None

        # Collect any errors whose wrong_action matches this task
        error_flags: list = []
        for err in log.get("errors", []):
            wa = err.get("wrong_action") or {}   # None-safe: missed_prerequisite entries have null wrong_action
            if wa.get("id") == tid:
                ra = err.get("right_action") or {}
                error_flags.append(
                    f"out_of_order: performed '{tid}' "
                    f"but expected '{ra.get('id', 'unknown')}' "
                    f"({ra.get('label', '')})"
                )

        record = {
            "task_id":          tid,
            "hand":             _hand_label(entry.get("lh"), entry.get("rh")),
            "start_time":       round(unix_start, 2),
            "end_time":         round(unix_end, 2) if unix_end is not None else None,
            "duration_seconds": round(duration, 1) if duration is not None else None,
            "mean_confidence":  None,   # LiveSessionLog does not carry per-task confidence
            "error_flags":      error_flags,
        }
        progression_log.append(record)

        if t_end_vid is not None:
            completed_ids.add(tid)   # only tasks with a closed t_end are "completed"

    # ── current_task per hand ─────────────────────────────────────────────────
    ct_raw = log.get("current_task")
    if ct_raw:
        ct_id = ct_raw.get("id")
        lh_active = ct_raw.get("lh") is not None
        rh_active = ct_raw.get("rh") is not None
        if lh_active and rh_active:
            current_task = {"left": ct_id, "right": ct_id}
        elif lh_active:
            current_task = {"left": ct_id, "right": None}
        elif rh_active:
            current_task = {"left": None, "right": ct_id}
        else:
            # Hand detail unavailable — assume both hands active
            current_task = {"left": ct_id, "right": ct_id}
    else:
        current_task = {"left": None, "right": None}

    # ── eligible_tasks ────────────────────────────────────────────────────────
    eligible_tasks = [e["id"] for e in log.get("eligible_next", [])]

    # ── procedural_errors (rich format for System 2 LLM) ─────────────────────
    procedural_errors = []
    for err in log.get("errors", []):
        error_type = err.get("error_type", "wrong_action")

        if error_type == "missed_prerequisite":
            # Worker performed the correct downstream task but skipped required steps.
            missed = err.get("missed_steps", [])
            for ms in missed:
                procedural_errors.append({
                    "t_start":            err.get("t_start"),
                    "t_end":              err.get("t_end"),
                    "type":               "missed_prerequisite",
                    "wrong_action_id":    None,
                    "wrong_action_label": None,
                    "right_action_id":    ms.get("id"),
                    "right_action_label": ms.get("label"),
                    "element_mismatch":   [],
                    "description": (
                        f"Task '{ms.get('label')}' ({ms.get('id')}) was skipped "
                        f"at t={err.get('t_start')}s — required prerequisite that "
                        "the worker bypassed before continuing to the next step."
                    ),
                })
        else:
            wrong = err.get("wrong_action") or {}
            right = err.get("right_action") or {}
            mm = err.get("element_mismatch", {})
            mismatch_summary = [
                f"{elem}: predicted '{info.get('pred')}' "
                f"but expected '{info.get('expected')}'"
                for elem, info in mm.items()
                if not info.get("match", True)
            ]
            procedural_errors.append({
                "t_start":            err.get("t_start"),
                "t_end":              err.get("t_end"),
                "type":               "wrong_action",
                "wrong_action_id":    wrong.get("id"),
                "wrong_action_label": wrong.get("label"),
                "right_action_id":    right.get("id"),
                "right_action_label": right.get("label"),
                "element_mismatch":   mismatch_summary,
                "description": (
                    f"Worker performed '{wrong.get('label', wrong.get('id'))}' "
                    f"at t={err.get('t_start')}s but the expected next task was "
                    f"'{right.get('label', right.get('id'))}'. "
                    + ("Mismatch: " + "; ".join(mismatch_summary)
                       if mismatch_summary else "")
                ),
            })

    # ── missed_tasks ──────────────────────────────────────────────────────────
    # Tasks that were in the TPG but never performed before the session ended.
    # Also appended to procedural_errors so the LLM sees them in one place.
    missed_tasks = log.get("missed_tasks", [])
    for t in missed_tasks:
        procedural_errors.append({
            "t_start":            None,
            "t_end":              None,
            "wrong_action_id":    None,
            "wrong_action_label": None,
            "right_action_id":    t["id"],
            "right_action_label": t["label"],
            "element_mismatch":   [],
            "type":               "missed_operation",
            "description": (
                f"Task '{t['label']}' ({t['id']}) was never performed "
                "during the session (missed operation)."
            ),
        })

    # ── session_end_time ──────────────────────────────────────────────────────
    session_end_time: Optional[float] = None
    if log.get("status") == "complete" and session_start_time is not None:
        all_ends = [e.get("t_end") for e in log.get("completed_tasks", [])
                    if e.get("t_end") is not None]
        if all_ends:
            session_end_time = session_start_time + max(all_ends)

    # ── Timing summary (idle time for System 2 advisory) ─────────────────────
    timing = log.get("timing_summary", {})
    tools.SESSION_STATE["idle_seconds"]          = timing.get("idle_seconds")
    tools.SESSION_STATE["total_session_seconds"] = timing.get("total_seconds")

    # ── Inject into SESSION_STATE ─────────────────────────────────────────────
    tools.SESSION_STATE["progression_log"]    = progression_log
    tools.SESSION_STATE["completed_tasks"]    = completed_ids
    tools.SESSION_STATE["current_task"]       = current_task
    tools.SESSION_STATE["eligible_tasks"]     = eligible_tasks
    tools.SESSION_STATE["procedural_errors"]  = procedural_errors
    tools.SESSION_STATE["missed_tasks"]       = missed_tasks
    tools.SESSION_STATE["anomaly_flags"]      = []   # reserved for future sensor data
    tools.SESSION_STATE["session_start_time"] = session_start_time
    tools.SESSION_STATE["session_end_time"]   = session_end_time
