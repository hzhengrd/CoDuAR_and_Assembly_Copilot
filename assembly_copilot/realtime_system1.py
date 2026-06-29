"""
realtime_system1.py
===================
Assembly Copilot — System 1: Real-Time Task Monitoring.

Implements the full System 1 pipeline from Section 3.2 of the paper:
  • Sliding-window compositional inference (CoDuAR model)
  • Compositional-to-task mapping  (Ψ)
  • Per-hand majority-vote task confirmation
  • Task Progression Tracking (TPT)
  • TPG-constrained next-task recommendation
  • Operational error detection with element-level mismatch report

Works with a pre-recorded video file (offline evaluation) or can be adapted
for a live camera stream by replacing the frame source.

Mock Task Precedence Graph
--------------------------
A representative penlight-assembly TPG is hard-coded below for testing.
Replace MOCK_TPG_TASKS and MOCK_TPG_EDGES with your actual assembly procedure.
Each task entry needs:
    id         (int)              unique task identifier
    label      (str)              short readable name
    signature  (tuple[int,4])     (verb, manip_obj, target_obj, tool) indices

Usage
-----
    python realtime_system1.py \\
        --video  path/to/video.mp4 \\
        --checkpoint  output/compositional_transformer_single_stream_finetuned/checkpoint-best.pth \\
        --composition_file  /path/to/case_study/pt_mapping_list_havid.txt \\
        [--window 16] [--stride 4] [--vote_window 5] [--n_crops 1]
        [--device cuda:0]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from timm.models import create_model

# ── Register the single-stream model ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: F401
import models.modeling_finetune_compositional_dual_transformer_single_stream  # noqa: F401

ELEMENTS = ("verb", "manip_obj", "target_obj", "tool")

# ---------------------------------------------------------------------------
# ANSI colours for terminal display
# ---------------------------------------------------------------------------
class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    CYAN   = '\033[96m'
    WHITE  = '\033[97m'
    GREY   = '\033[90m'

def _c(text, *codes):
    return ''.join(codes) + str(text) + C.RESET


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve an inference device, falling back cleanly on non-CUDA machines."""
    requested = (device_arg or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")

    if requested == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            print("[warn] MPS requested but unavailable; using CPU.")
            return torch.device("cpu")

    return torch.device(requested)


def _find_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path

    env_bin = Path(sys.executable).resolve().parent / name
    if env_bin.is_file():
        return str(env_bin)
    return None


# ===========================================================================
# MOCK TASK PRECEDENCE GRAPH
# ===========================================================================
# Built from:
#   case_study_labels/pt_mapping_list_havid.txt       (signatures)
#   case_study_labels/pt_mapping_list_havid_semantics.txt (labels)
#
# Task IDs are the compositional label strings produced by Ψ.
# Replace MOCK_TPG_EDGES with your actual procedural dependencies.
#
# Verb IDs   (case-study):  0=null 1=i(insert) 2=p(place) 3=r(rotate) 4=s(screw)
# Manip IDs  (case-study):  0=null 1=gl 2=gs 3=pl 4=ps 5=dp 6=gw 7=wn 8=ws
#                           9=ft  10=nt 11=sp
# Target IDs (case-study):  0=null 1=f1 2=f2 3=g3 4=g1 5=g2
# Tool IDs   (case-study):  0=null 1=dp 2=wn 3=ws
# ─────────────────────────────────────────────────────────────────────────────
MOCK_TPG_TASKS = [
    # ── null ──────────────────────────────────────────────────────────────────
    {"id": "null",      "label": "null",                                           "signature": (0,  0,  0,  0)},

    # ── Stage 1: screw shafts into gear holes (no prerequisites) ─────────────
    {"id": "sftg1",     "label": "Screw Shaft into Gear Hole 1",                     "signature": (4,  9,  4,  0)},
    {"id": "sftg1ws",   "label": "Screw Shaft into Gear Hole 1 using Wrench-Shaft",      "signature": (4,  9,  4,  3)},
    {"id": "sftg2",     "label": "Screw Shaft into Gear Hole 2",                     "signature": (4,  9,  5,  0)},
    {"id": "sftg2ws",   "label": "Screw Shaft into Gear Hole 2 using Wrench-Shaft",      "signature": (4,  9,  5,  3)},

    # ── Stage 1: screw Phillips screw into gear hole 3 (no prerequisites) ────
    {"id": "sspg3",     "label": "Screw Phillips Screw into Gear Hole 3",            "signature": (4, 11,  3,  0)},
    {"id": "sspg3dp",   "label": "Screw Phillips Screw into Gear Hole 3 using Phillips Screwdriver",   "signature": (4, 11,  3,  1)},

    # ── Stage 2: insert gears onto shafts ────────────────────────────────────
    {"id": "iglf1",     "label": "Insert Large Gear onto Shaft 1",                   "signature": (1,  1,  1,  0)},
    {"id": "igsf2",     "label": "Insert Small Gear onto Shaft 2",                   "signature": (1,  2,  2,  0)},

    # ── Stage 2: place worm gear (after screw in g3) ─────────────────────────
    {"id": "pgwg3",     "label": "Place Worm Gear onto Gear Hole 3",                 "signature": (2,  6,  3,  0)},

    # ── Stage 3: insert large placers ─────────────────────────────────────────
    {"id": "iplf1",     "label": "Insert Large Placer onto Shaft 1",                 "signature": (1,  3,  1,  0)},
    {"id": "iplf2",     "label": "Insert Large Placer onto Shaft 2",                 "signature": (1,  3,  2,  0)},

    # ── Stage 3: rotate worm gear ─────────────────────────────────────────────
    {"id": "rgw",       "label": "Rotate Worm Gear",                              "signature": (3,  6,  0,  0)},

    # ── Stage 4: insert small placers ─────────────────────────────────────────
    {"id": "ipsf1",     "label": "Insert Small Placer onto Shaft 1",                 "signature": (1,  4,  1,  0)},
    {"id": "ipsf2",     "label": "Insert Small Placer onto Shaft 2",                 "signature": (1,  4,  2,  0)},

    # ── Stage 5: screw nuts onto shafts ──────────────────────────────────────
    {"id": "sntf1",     "label": "Screw Nut onto Shaft 1",                           "signature": (4, 10,  1,  0)},
    {"id": "sntf1wn",   "label": "Screw Nut onto Shaft 1 using Wrench-Nut",              "signature": (4, 10,  1,  2)},
    {"id": "sntf2",     "label": "Screw Nut onto Shaft 2",                           "signature": (4, 10,  2,  0)},
    {"id": "sntf2wn",   "label": "Screw Nut onto Shaft 2 using Wrench-Nut",              "signature": (4, 10,  2,  2)},

    # ── Stage 6: place tools back (cleanup) ───────────────────────────────────
    {"id": "pwn",       "label": "Place back Wrench-Nut",                         "signature": (2,  7,  0,  0)},
    {"id": "pws",       "label": "Place back Wrench-Shaft",                       "signature": (2,  8,  0,  0)},
    {"id": "pdp",       "label": "Place back Phillips Screwdriver",               "signature": (2,  5,  0,  0)},
]

# Directed edges: (prerequisite_id, task_id)
# ─ replace / extend with your actual assembly procedure ─
MOCK_TPG_EDGES = [
    # Stage 1 → Stage 2: shaft screwing unlocks gear insertion
    ("sftg1",   "sftg1ws"), ("sftg1ws", "iplf1"), ("iplf1", "iglf1"), ("iglf1",  "ipsf1"), ("ipsf1", "sntf1"), ("sntf1", "sntf1wn"), ("sntf1wn", "rgw"),
    ("sftg2",   "sftg2ws"), ("sftg2ws", "iplf2"), ("iplf2", "igsf2"), ("igsf2", "ipsf2"), ("ipsf2", "sntf2"), ("sntf2", "sntf2wn"), ("sntf2wn", "rgw"),
    ("pgwg3", "sspg3"), ("sspg3", "sspg3dp"), ("sspg3dp", "rgw"),  
]
# ===========================================================================


# ---------------------------------------------------------------------------
# Task Precedence Graph
# ---------------------------------------------------------------------------

class TaskPrecedenceGraph:
    """Encodes assembly task dependencies as a DAG.  Task IDs are strings."""

    # Task IDs that are NEVER counted as progress even if they appear in
    # MOCK_TPG_TASKS.  Add side / cleanup activities here so they are silently
    # ignored during tracking and never shown in eligibility recommendations.
    # Tasks simply absent from MOCK_TPG_TASKS are also ignored automatically.
    _IGNORE = {"null", "unknown", "pwn", "pws", "pdp"}

    def __init__(self, tasks: List[dict], edges: List[Tuple[str, str]]):
        self.tasks: Dict[str, dict] = {t["id"]: t for t in tasks}
        # predecessors[task_id] = set of prerequisite task ids (strings)
        self.predecessors: Dict[str, set] = {t["id"]: set() for t in tasks}
        for pre, task in edges:
            if task in self.predecessors:          # only register edges for known tasks
                self.predecessors[task].add(pre)
        # Precompute DAG depth (longest path from any root) for each tracked task.
        # Used to sort the eligible list so later-stage tasks appear first.
        self._depth: Dict[str, int] = self._compute_depths()

    def _compute_depths(self) -> Dict[str, int]:
        """Longest path from any root node to each task (topological DP)."""
        depth: Dict[str, int] = {tid: 0 for tid in self.tasks}
        # Iteratively relax depths until convergence (handles any DAG).
        changed = True
        while changed:
            changed = False
            for tid, preds in self.predecessors.items():
                for p in preds:
                    if p in depth and depth[p] + 1 > depth[tid]:
                        depth[tid] = depth[p] + 1
                        changed = True
        return depth

    def is_tracked(self, tid: str) -> bool:
        """True if tid is a proper TPG task (not null/unknown and exists in the graph)."""
        return tid not in self._IGNORE and tid in self.tasks

    def eligible(self, completed: set) -> List[str]:
        """
        Tasks whose ALL prerequisites are satisfied (completed) and that are
        themselves not yet completed.

        Results are sorted by DAG depth descending so that later-stage tasks
        (e.g. the final convergence step) appear first even when earlier missed
        tasks also become eligible at the same time.

        Only tasks present in the TPG are ever returned — side activities
        such as 'null', 'unknown', or any task absent from MOCK_TPG_TASKS are
        silently excluded, even if they somehow ended up in *completed*.
        """
        candidates = [
            tid for tid, preds in self.predecessors.items()
            if self.is_tracked(tid)
            and tid not in completed
            and preds.issubset(completed)
        ]
        return sorted(candidates, key=lambda t: self._depth.get(t, 0), reverse=True)

    def ancestors(self, tid: str) -> set:
        """Return the set of all transitive predecessors of *tid* in the DAG.

        Used by error detection to find which eligible tasks were bypassed
        on the way to a confirmed out-of-order task.
        """
        visited: set = set()
        stack = list(self.predecessors.get(tid, set()))
        while stack:
            t = stack.pop()
            if t not in visited:
                visited.add(t)
                stack.extend(self.predecessors.get(t, set()))
        return visited

    def label(self, tid: str) -> str:
        return self.tasks[tid]["label"] if tid in self.tasks else tid


# ---------------------------------------------------------------------------
# Compositional-to-Task Mapper  (Ψ)
# ---------------------------------------------------------------------------

class CompositionToTaskMapper:
    """
    Maps (verb_id, manip_obj_id, target_obj_id, tool_id) to a task label
    string by concatenating the non-null element abbreviations:

        Ψ(v, m, t, o) = v_str + m_str + t_str* + o_str*
                       (* omitted when the value is "null")

    If the verb is "null" the result is always "null".
    If the formed string is not in the known label set, returns "unknown".

    Label maps are dicts {int_id -> abbreviation_str} loaded from the
    label_map_*.txt files.
    """

    NULL_LABEL = "null"
    UNKNOWN_LABEL = "unknown"

    def __init__(self, label_maps: Dict[str, Dict[int, str]],
                 composition_file: Optional[str] = None):
        self.label_maps = label_maps
        # Build a set of valid label strings from the composition file
        self.valid_labels: set = {self.NULL_LABEL}
        if composition_file and Path(composition_file).is_file():
            with open(composition_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        self.valid_labels.add(parts[1])

    def map(self, verb: int, manip: int, target: int, tool: int) -> str:
        """
        Concatenate non-null element abbreviations to form the task label.
        Rules:
          1. If verb abbreviation == "null"  → return "null"
          2. Concatenate: verb + manip + target (if not null) + tool (if not null)
          3. If the result is not in the known label set → return "unknown"
        """
        def _name(elem: str, idx: int) -> str:
            return self.label_maps.get(elem, {}).get(idx, str(idx))

        v_str = _name("verb", verb)
        if v_str == "null":
            return self.NULL_LABEL

        m_str = _name("manip_obj", manip)
        t_str = _name("target_obj", target)
        o_str = _name("tool", tool)

        label = v_str + m_str
        if t_str != "null":
            label += t_str
        if o_str != "null":
            label += o_str

        return label if label in self.valid_labels else self.UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Per-Hand Majority Voting Confirmer
# ---------------------------------------------------------------------------

class MajorityVotingConfirmer:
    """
    Sliding window majority vote for task confirmation (Eq. 4 in paper).
    Retains previous confirmed task when no majority is reached.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._buffer: deque = deque(maxlen=window)
        self._confirmed: str = "null"

    def push(self, task_id: str) -> Tuple[str, bool]:
        """
        Add a new task candidate.  Returns (confirmed_task_id, changed).
        """
        prev = self._confirmed
        self._buffer.append(task_id)

        # Find majority
        counts: Dict[str, int] = {}
        for t in self._buffer:
            counts[t] = counts.get(t, 0) + 1
        majority_id = max(counts, key=lambda t: counts[t])
        threshold = math.ceil(len(self._buffer) / 2)

        if counts[majority_id] >= threshold:
            self._confirmed = majority_id

        return self._confirmed, (self._confirmed != prev)

    @property
    def confirmed(self) -> str:
        return self._confirmed


# ---------------------------------------------------------------------------
# Holistic task fusion  (mirrors evaluate_task_sequence.py)
# ---------------------------------------------------------------------------

def normalize_task(t: str) -> str:
    """Canonicalise a parallel-task label so 'A|B' == 'B|A'."""
    if "|" in t:
        return "|".join(sorted(t.split("|")))
    return t


def fuse_holistic(lh: str, rh: str) -> str:
    """
    Combine per-hand task labels into one holistic task label for this window.

    Rules (in priority order):
      null + null   → "null"
      null + task   → task          (active hand drives the estimate)
      task + null   → task
      A    + A      → A
      A    + B      → "A|B"  (parallel bimanual work; label order normalised)
    """
    if lh == "null" and rh == "null":
        return "null"
    if lh == "null":
        return rh
    if rh == "null":
        return lh
    if lh == rh:
        return lh
    return normalize_task(f"{lh}|{rh}")


# ---------------------------------------------------------------------------
# System 1 Monitor
# ---------------------------------------------------------------------------

class System1Monitor:
    """
    Real-time assembly process monitor.

    Pipeline per window
    -------------------
    1. Map LH/RH element tuples → per-hand task labels  (Ψ)
    2. Fuse both labels into a single holistic task label
       (mirrors fuse_holistic() used in evaluate_task_sequence.py)
    3. Majority-vote on the HOLISTIC stream  → confirmed holistic task
    4. Parse the confirmed label (may be "taskA|taskB" for parallel work)
    5. Update Task Progression Tracking (TPT) for each parsed task
    6. Recommend next eligible tasks from the TPG
    7. Detect out-of-order errors

    Maintains
    ---------
    • holistic_confirmer  – single majority-vote buffer on the fused stream
    • completed  (D_n)    – set of confirmed TPG task IDs
    • log  (L)            – ordered list of completed-task entries
    """

    def __init__(self, tpg: TaskPrecedenceGraph, mapper: CompositionToTaskMapper,
                 vote_window: int = 5, dwell_required: int = 3):
        self.tpg = tpg
        self.mapper = mapper
        # One confirmer on the fused holistic stream (not per-hand)
        self.holistic_confirmer = MajorityVotingConfirmer(vote_window)

        self.completed: set = set()          # D_n
        self.log: List[dict] = []            # L
        self._task_start: Dict[str, float] = {}
        self._frame_time: float = 0.0

        # Dwell gate: task must be the confirmed label for this many consecutive
        # windows before it is accepted into self.completed.
        # Prevents brief false-positive predictions from permanently marking a
        # task as done (e.g. model sees "rgw" for 1–2 windows near session end).
        self._dwell_required: int = max(1, dwell_required)
        self._dwell_count: Dict[str, int] = {}   # confirmed_task → consecutive windows

        # Accumulates every task ever identified as a "missed step" in an error.
        # Used by the GUI to label eligible-but-previously-skipped tasks
        # differently in the NEXT TASKS panel.
        self.missed_ever: set = set()

    # ── Main entry point ────────────────────────────────────────────────────

    def step(self, frame_time: float,
             lh_pred: Tuple[int, int, int, int],
             rh_pred: Tuple[int, int, int, int],
             lh_conf: Dict[str, float],
             rh_conf: Dict[str, float]) -> dict:
        """
        Process one sliding-window prediction.
        Returns a full state dict consumed by display_state().
        """
        self._frame_time = frame_time

        # Step 1 — Ψ: element tuples → per-hand task labels
        lh_task = self.mapper.map(*lh_pred)
        rh_task = self.mapper.map(*rh_pred)

        # Step 2 — Fuse into holistic task label
        holistic_raw = fuse_holistic(lh_task, rh_task)

        # Step 3 — Majority vote on the holistic stream
        confirmed_holistic, changed = self.holistic_confirmer.push(holistic_raw)

        # Step 4 — Parse confirmed label (handles "taskA|taskB" parallel work)
        confirmed_tasks = self._parse_holistic(confirmed_holistic)

        # Step 5 — Snapshot state BEFORE updating progression.
        #   Error detection must use the pre-update eligible/completed sets,
        #   otherwise a task that is just being confirmed is already in
        #   self.completed when we check, masking out-of-order errors.
        eligible_pre   = self.tpg.eligible(self.completed)
        completed_pre  = set(self.completed)

        # Step 6 — Dwell gate: maintain per-task consecutive-window counters.
        #   Tasks absent from the current confirmed set have their counter reset
        #   so a later re-appearance starts the dwell from scratch.
        confirmed_set = set(confirmed_tasks)
        for tid in list(self._dwell_count):
            if tid not in confirmed_set:
                del self._dwell_count[tid]

        # Update TPT for each atomic task in the confirmed label
        for tid in confirmed_tasks:
            self._update_progression(tid, changed, frame_time)

        # Step 7 — Eligible after update (used for recommendation display)
        eligible = self.tpg.eligible(self.completed)

        # Step 8 — Error detection against the PRE-UPDATE state.
        #   eligible_pre / completed_pre → used to decide whether an error occurred.
        #   eligible (post-update)       → used to pick the "closest recommended task"
        #                                  so it matches what is shown in Next eligible.
        error = self._detect_error(confirmed_tasks, changed, lh_pred, rh_pred,
                                   eligible_pre, completed_pre, eligible)

        n_tracked = sum(1 for tid in self.tpg.tasks if self.tpg.is_tracked(tid))
        return {
            "frame_time":          frame_time,
            "lh_raw_pred":         lh_pred,
            "rh_raw_pred":         rh_pred,
            "lh_raw_task":         lh_task,
            "rh_raw_task":         rh_task,
            "holistic_raw":        holistic_raw,
            "confirmed_holistic":  confirmed_holistic,
            "confirmed_tasks":     confirmed_tasks,
            "changed":             changed,
            "lh_conf":             lh_conf,
            "rh_conf":             rh_conf,
            "completed":           set(self.completed),
            "eligible":            eligible,
            "missed_ever":         set(self.missed_ever),
            "error":               error,
            "progress_pct":        len(self.completed) / max(1, n_tracked) * 100,
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_holistic(holistic: str) -> List[str]:
        """Split 'taskA|taskB' → ['taskA','taskB']; 'taskA' → ['taskA']."""
        return holistic.split("|") if "|" in holistic else [holistic]

    def _update_progression(self, tid: str, changed: bool, t: float):
        """Add a newly confirmed TPG task to D_n and the log.

        Side tasks (null, unknown, pwn, pws, pdp …) are silently ignored —
        they must never pollute the completed set or affect TPG eligibility.
        Tasks absent from MOCK_TPG_TASKS are also ignored automatically.

        Dwell gate: the task must be the confirmed label for _dwell_required
        consecutive windows.  A single-window blip (common near the end of a
        session when the model briefly mis-predicts the final task) will NOT
        reach the threshold and thus will NOT enter self.completed.

        Error detection fires on the FIRST window (changed=True) regardless of
        dwell, so warnings are always immediate.
        """
        if not self.tpg.is_tracked(tid):
            return
        if tid in self.completed:   # do not double-count
            return

        # ── Dwell gate ────────────────────────────────────────────────────────
        if changed:
            # First window this task has been confirmed — start / restart dwell
            self._dwell_count[tid] = 1
        else:
            # Task still confirmed — advance dwell counter
            self._dwell_count[tid] = self._dwell_count.get(tid, 0) + 1

        if self._dwell_count[tid] < self._dwell_required:
            return   # not yet sustained — skip progression update

        # ── Dwell threshold met: accept task into completed ───────────────────
        self._task_start.setdefault(tid, t)
        self.completed.add(tid)
        self.log.append({
            "task_id":  tid,
            "label":    self.tpg.label(tid),
            "t_start":  t,
            "t_end":    None,
            "error":    None,
        })
        del self._dwell_count[tid]   # clear gate for this task

    def _detect_error(self, confirmed_tasks: List[str], changed: bool,
                      lh_pred: Tuple, rh_pred: Tuple,
                      eligible_pre: List[str],
                      completed_pre: set,
                      eligible_current: Optional[List[str]] = None) -> Optional[dict]:
        """Detect out-of-order TPG tasks per Eq. 6 in the paper.

        Uses the state BEFORE the current step's progression update so that
        a newly confirmed task cannot hide its own eligibility violation
        (the previous bug: task added to self.completed before the check).

        Two error types are detected, in priority order:

        1. SKIPPED STEP  — tasks that were eligible (recommended) and are
           transitive prerequisites of the confirmed task, but were never done.
           These are *missed* actions the worker bypassed on the way to tid.

        2. WRONG ORDER   — the confirmed task itself was not eligible.

        Both can be active at the same time (e.g. skip sftg1ws, jump to iplf1).
        The returned dict carries both "missed_steps" (list) and the
        "closest_eligible" / "mismatch" fields for the wrong-order part.

        Logic per task:
          - Not a tracked TPG task → ignore (side activity)
          - Was already completed before this window → re-confirmation, not an error
          - Was in the pre-update eligible set → correct expected step;
            still check for parallel eligible tasks that were silently skipped
            only if they are ancestors of tid AND tid IS eligible (that would
            be a contradiction, so this branch is always clean — skip)
          - Otherwise → out-of-order error; also compute missed prerequisites
        """
        if not changed:
            return None
        for tid in confirmed_tasks:
            if not self.tpg.is_tracked(tid):
                continue                          # side activity — not an error
            if tid in completed_pre:
                continue                          # previously completed, re-confirmed
            if tid in eligible_pre:
                continue                          # was a valid next step — OK

            # ── tid was NOT eligible → out-of-order confirmed ─────────────────

            # Step A: find SKIPPED STEPS — eligible tasks that are ancestors
            # of tid (i.e. should have been done before tid) but were not.
            ancestors = self.tpg.ancestors(tid)
            missed_steps = [
                {"id": t, "label": self.tpg.label(t)}
                for t in eligible_pre
                if t in ancestors and t not in completed_pre
            ]
            # Accumulate into the session-wide set so the GUI can label
            # these tasks differently in the NEXT TASKS panel forever after.
            for ms in missed_steps:
                self.missed_ever.add(ms["id"])

            # Step B: classify the error type.
            #
            # MISSED-PREREQUISITE ("jumped ahead in the right chain"):
            #   Every DIRECT predecessor of tid that is still incomplete was
            #   already in the eligible set (= accessible, just not done yet).
            #   The worker is doing the correct downstream task but skipped a
            #   required preparatory step.  → report ONLY the missed steps;
            #   the confirmed task itself is not wrong.
            #
            # WRONG-ACTION ("doing something unrelated / genuinely mis-ordered"):
            #   At least one direct predecessor was NOT eligible, meaning the
            #   worker has left the expected sequence entirely.
            #   → report both the missed steps AND the wrong-action detail.
            blocking_preds = set(self.tpg.predecessors.get(tid, set())) - completed_pre
            missed_step_ids = {ms["id"] for ms in missed_steps}
            is_missed_prereq_only = (
                len(blocking_preds) > 0
                and blocking_preds.issubset(missed_step_ids)
            )

            if is_missed_prereq_only:
                # Worker is on the correct chain — only flag the skipped step(s)
                error = {
                    "confirmed_task":   tid,
                    "confirmed_label":  self.tpg.label(tid),
                    "missed_steps":     missed_steps,
                    "closest_eligible": None,
                    "closest_label":    None,
                    "mismatch":         {},
                    "error_type":       "missed_prerequisite",
                }
            else:
                # Genuine wrong action — also compute closest eligible + element diff
                candidates = eligible_current if eligible_current is not None else eligible_pre
                best_task, _ = self._closest_eligible(lh_pred, rh_pred, candidates)
                mismatch = self._element_mismatch_both(lh_pred, rh_pred, best_task)
                error = {
                    "confirmed_task":   tid,
                    "confirmed_label":  self.tpg.label(tid),
                    "missed_steps":     missed_steps,
                    "closest_eligible": best_task,
                    "closest_label":    self.tpg.label(best_task) if best_task else "none",
                    "mismatch":         mismatch,
                    "error_type":       "wrong_action",
                }

            if self.log:
                self.log[-1]["error"] = error
            return error
        return None

    def _closest_eligible(self, lh_pred: Tuple, rh_pred: Tuple,
                          eligible: List[str]):
        """Return the eligible task whose signature best matches either hand."""
        if not eligible:
            return None, 0
        best_id, best_score = eligible[0], -1
        for tid in eligible:
            if tid not in self.tpg.tasks:
                continue
            sig = self.tpg.tasks[tid]["signature"]
            score = max(
                sum(int(lh_pred[i] == sig[i]) for i in range(4)),
                sum(int(rh_pred[i] == sig[i]) for i in range(4)),
            )
            if score > best_score:
                best_score = score
                best_id = tid
        return best_id, best_score

    def _element_mismatch_both(self, lh_pred: Tuple, rh_pred: Tuple,
                                tid: Optional[str]) -> dict:
        """Element-level mismatch report for the hand that is closer to tid."""
        if tid is None or tid not in self.tpg.tasks:
            return {}
        sig = self.tpg.tasks[tid]["signature"]
        lh_score = sum(int(lh_pred[i] == sig[i]) for i in range(4))
        rh_score = sum(int(rh_pred[i] == sig[i]) for i in range(4))
        pred = lh_pred if lh_score >= rh_score else rh_pred
        names = ["verb", "manip_obj", "target_obj", "tool"]
        return {
            names[i]: {"pred": pred[i], "expected": sig[i],
                       "match": pred[i] == sig[i]}
            for i in range(4)
        }


# ---------------------------------------------------------------------------
# Live session JSON log  (feeds System 2)
# ---------------------------------------------------------------------------

class LiveSessionLog:
    """
    Maintains a continuously-updated JSON document that System 2 can poll for
    on-demand guidance and post-session review.

    JSON schema
    -----------
    {
      "meta": { video, fps, total_frames, session_start_wall },
      "status": "in_progress" | "complete",
      "progress": { completed, total, pct },
      "current_task": {
          id, label, t_start,
          lh: { t_start, t_last } | null,
          rh: { t_start, t_last } | null
      } | null,
      "eligible_next": [ {id, label}, ... ],
      "completed_tasks": [
          { id, label, t_start, t_end,
            lh: { t_start, t_end } | null,
            rh: { t_start, t_end } | null }
      ],
      "errors": [
          { t_start, t_end,
            wrong_action:  { id, label },
            right_action:  { id, label },
            element_mismatch: { elem: { pred, expected, match } } }
      ],
      "side_activities": [
          { type, label, t_start, t_end | null }
      ]
    }
    """

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 output_path: str, n_tracked: int,
                 label_maps: Optional[dict] = None,
                 tpg: Optional["TaskPrecedenceGraph"] = None):
        self.output_path = Path(output_path)
        self.label_maps  = label_maps or {}
        self._tpg        = tpg

        self._data: dict = {
            "meta": {
                "video":              str(video_path),
                "fps":                round(fps, 3),
                "total_frames":       total_frames,
                "session_start_wall": datetime.now(timezone.utc).isoformat(),
            },
            "status":         "in_progress",
            "progress":       {"completed": 0, "total": n_tracked, "pct": 0.0},
            "current_task":   None,
            "eligible_next":  [],
            "completed_tasks": [],
            "errors":          [],
            "side_activities": [],
            "missed_tasks":    [],   # filled by finalize()
        }

        # ── Internal bookkeeping ──────────────────────────────────────────────
        self._active_tid:      Optional[str]   = None   # currently performing
        self._active_t_start:  Optional[float] = None

        # per-task hand activity windows
        self._lh_t_start: Dict[str, float] = {}
        self._lh_t_end:   Dict[str, float] = {}
        self._rh_t_start: Dict[str, float] = {}
        self._rh_t_end:   Dict[str, float] = {}

        # side activities
        self._side_type:    Optional[str]   = None
        self._side_t_start: Optional[float] = None

        # open error entry awaiting t_end
        self._open_error: Optional[dict] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, state: dict):
        """Call after every monitor.step(). Updates the JSON file atomically."""
        tpg        = self._tpg
        frame_time = state["frame_time"]
        changed    = state["changed"]
        lh_task    = state["lh_raw_task"]
        rh_task    = state["rh_raw_task"]
        eligible   = state["eligible"]
        completed  = state["completed"]
        error      = state["error"]

        # Parse holistic confirmed label into atomic tasks
        confirmed_tasks = [t.strip()
                           for t in state["confirmed_holistic"].split("|")
                           if t.strip()]
        tracked = [t for t in confirmed_tasks if tpg.is_tracked(t)]
        primary_tid = tracked[0] if tracked else None
        is_side     = primary_tid is None

        # ── 1. Update live hand-activity windows for current task ─────────────
        if primary_tid:
            if lh_task == primary_tid:
                self._lh_t_start.setdefault(primary_tid, frame_time)
                self._lh_t_end[primary_tid] = frame_time
            if rh_task == primary_tid:
                self._rh_t_start.setdefault(primary_tid, frame_time)
                self._rh_t_end[primary_tid] = frame_time

        # ── 2. Handle transitions (changed=True means a new vote was confirmed) ─
        if changed:
            # Close the previous active task if transitioning away
            if self._active_tid and self._active_tid != primary_tid:
                self._close_task(self._active_tid, frame_time)
                self._active_tid     = None
                self._active_t_start = None

            # Close any open error when a new task decision is made
            if self._open_error is not None:
                self._open_error["t_end"] = round(frame_time, 2)
                self._open_error = None

            # Close previous side activity if switching
            if self._side_type is not None:
                new_side = state["confirmed_holistic"] if is_side else None
                if new_side != self._side_type:
                    self._close_side(frame_time)

            # Open new tracked task
            if primary_tid and primary_tid != self._active_tid:
                self._active_tid     = primary_tid
                self._active_t_start = frame_time
                # Create an open entry in completed_tasks (t_end=null until closed)
                if self._get_entry(primary_tid) is None:
                    self._data["completed_tasks"].append({
                        "id":      primary_tid,
                        "label":   tpg.label(primary_tid),
                        "t_start": round(frame_time, 2),
                        "t_end":   None,
                        "lh":      None,
                        "rh":      None,
                    })

            # Open new side activity
            if is_side:
                side_type = state["confirmed_holistic"]
                if side_type != self._side_type:
                    self._side_type    = side_type
                    self._side_t_start = frame_time
                    self._data["side_activities"].append({
                        "type":    side_type,
                        "label":   (tpg.label(side_type)
                                    if tpg and side_type in tpg.tasks
                                    else side_type),
                        "t_start": round(frame_time, 2),
                        "t_end":   None,
                    })

        # ── 3. Keep live hand info fresh on the open completed_tasks entry ─────
        if primary_tid:
            self._refresh_hands(primary_tid, frame_time)

        # ── 4. Record errors ──────────────────────────────────────────────────
        if error and changed:
            lm = self.label_maps
            error_type = error.get("error_type", "wrong_action")
            mismatch_out: dict = {}
            for elem, info in (error.get("mismatch") or {}).items():
                elem_lm = lm.get(elem, {})
                mismatch_out[elem] = {
                    "pred":     elem_lm.get(info["pred"],     str(info["pred"])),
                    "expected": elem_lm.get(info["expected"], str(info["expected"])),
                    "match":    info["match"],
                }

            if error_type == "missed_prerequisite":
                # The confirmed task is correct; only the skipped predecessors are wrong.
                err_entry: dict = {
                    "t_start":          round(frame_time, 2),
                    "t_end":            None,
                    "error_type":       "missed_prerequisite",
                    "wrong_action":     None,   # no wrong action — task itself is valid
                    "right_action":     None,
                    "missed_steps":     error.get("missed_steps", []),
                    "element_mismatch": {},
                }
            else:
                err_entry = {
                    "t_start":          round(frame_time, 2),
                    "t_end":            None,
                    "error_type":       "wrong_action",
                    "wrong_action":     {"id": error["confirmed_task"],
                                         "label": error["confirmed_label"]},
                    "right_action":     {"id":    error["closest_eligible"],
                                         "label": error["closest_label"]},
                    "missed_steps":     error.get("missed_steps", []),
                    "element_mismatch": mismatch_out,
                }
            self._data["errors"].append(err_entry)
            self._open_error = err_entry   # keep reference to close it later

        # ── 5. Update progress + eligible snapshot ────────────────────────────
        self._data["progress"] = {
            "completed": len(completed),
            "total":     self._data["progress"]["total"],
            "pct":       round(state["progress_pct"], 1),
        }
        self._data["eligible_next"] = [
            {"id": tid, "label": tpg.label(tid)} for tid in eligible
        ]

        # ── 6. Update current_task snapshot ──────────────────────────────────
        if primary_tid and self._active_tid == primary_tid:
            self._data["current_task"] = {
                "id":      primary_tid,
                "label":   tpg.label(primary_tid),
                "t_start": round(self._active_t_start or frame_time, 2),
                "lh":      ({"t_start": round(self._lh_t_start[primary_tid], 2),
                             "t_last":  round(self._lh_t_end.get(primary_tid,
                                                                  frame_time), 2)}
                            if primary_tid in self._lh_t_start else None),
                "rh":      ({"t_start": round(self._rh_t_start[primary_tid], 2),
                             "t_last":  round(self._rh_t_end.get(primary_tid,
                                                                  frame_time), 2)}
                            if primary_tid in self._rh_t_start else None),
            }
        else:
            self._data["current_task"] = None

        # ── 7. Atomic write ───────────────────────────────────────────────────
        self._write()

    def finalize(self, frame_time: float):
        """Call once after the frame loop ends to seal all open entries."""
        if self._active_tid:
            self._close_task(self._active_tid, frame_time)
        if self._side_type is not None:
            self._close_side(frame_time)
        if self._open_error is not None:
            self._open_error["t_end"] = round(frame_time, 2)
            self._open_error = None

        # Compute missed tasks: tracked TPG tasks never confirmed this session
        if self._tpg:
            completed_ids = {e["id"] for e in self._data["completed_tasks"]}
            self._data["missed_tasks"] = [
                {"id": tid, "label": self._tpg.label(tid)}
                for tid in self._tpg.tasks
                if self._tpg.is_tracked(tid) and tid not in completed_ids
            ]

        # Compute idle time: total session - active tasks - side activities
        total_s = frame_time  # last frame timestamp ≈ total session length
        active_s = sum(
            (e["t_end"] - e["t_start"])
            for e in self._data["completed_tasks"]
            if e.get("t_start") is not None and e.get("t_end") is not None
        )
        side_s = sum(
            (sa["t_end"] - sa["t_start"])
            for sa in self._data["side_activities"]
            if sa.get("t_start") is not None and sa.get("t_end") is not None
        )
        idle_s = max(0.0, total_s - active_s - side_s)
        self._data["timing_summary"] = {
            "total_seconds":         round(total_s,  1),
            "active_task_seconds":   round(active_s, 1),
            "side_activity_seconds": round(side_s,   1),
            "idle_seconds":          round(idle_s,   1),
            "idle_fraction":         round(idle_s / total_s, 3) if total_s > 0 else 0.0,
        }

        self._data["status"]       = "complete"
        self._data["current_task"] = None
        self._write()
        missed = self._data.get("missed_tasks", [])
        if missed:
            print(_c(f"\n  Missed tasks ({len(missed)}):", C.YELLOW, C.BOLD))
            for t in missed:
                print(_c(f"    - {t['label']}", C.YELLOW))
        print(_c(f"\n  Session log written → {self.output_path}", C.GREEN))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_entry(self, tid: str) -> Optional[dict]:
        for e in self._data["completed_tasks"]:
            if e["id"] == tid:
                return e
        return None

    def _refresh_hands(self, tid: str, frame_time: float):
        e = self._get_entry(tid)
        if e is None:
            return
        if tid in self._lh_t_start:
            e["lh"] = {"t_start": round(self._lh_t_start[tid], 2),
                       "t_end":   round(self._lh_t_end.get(tid, frame_time), 2)}
        if tid in self._rh_t_start:
            e["rh"] = {"t_start": round(self._rh_t_start[tid], 2),
                       "t_end":   round(self._rh_t_end.get(tid, frame_time), 2)}

    def _close_task(self, tid: str, t_end: float):
        e = self._get_entry(tid)
        if e and e["t_end"] is None:
            e["t_end"] = round(t_end, 2)
        self._refresh_hands(tid, t_end)

    def _close_side(self, t_end: float):
        for sa in reversed(self._data["side_activities"]):
            if sa["t_end"] is None:
                sa["t_end"] = round(t_end, 2)
                break
        self._side_type    = None
        self._side_t_start = None

    def _write(self):
        tmp = self.output_path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(str(tmp), str(self.output_path))   # atomic on POSIX


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate_video_sliding_window.py)
# ---------------------------------------------------------------------------

def build_model(args, device):
    model = create_model(
        "vit_base_patch16_224_compositional_dual_transformer_single_stream",
        pretrained=False,
        lh_num_verbs=args.lh_num_verbs,
        lh_num_manip_objs=args.lh_num_manip_objs,
        lh_num_target_objs=args.lh_num_target_objs,
        lh_num_tools=args.lh_num_tools,
        rh_num_verbs=args.rh_num_verbs,
        rh_num_manip_objs=args.rh_num_manip_objs,
        rh_num_target_objs=args.rh_num_target_objs,
        rh_num_tools=args.rh_num_tools,
        use_hand_adapters=args.use_hand_adapters,
        adapter_dim=args.adapter_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_dropout=args.decoder_dropout,
        head_dropout=args.head_dropout,
    )
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    sd = ckpt.get("model", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 3): {missing[:3]}")
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing  (mirrors evaluate_video_sliding_window.py)
# ---------------------------------------------------------------------------

def _short_side_scale(frames, size=224):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        out.append(cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR))
    return out


def _center_crop(frames, crop=224):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        out.append(f[y0:y0+crop, x0:x0+crop])
    return out


def _to_tensor(frames) -> torch.Tensor:
    """(N,H,W,C) uint8 BGR → (1,C,T,H,W) float32 normalised."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    ts = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ts.append((rgb - mean) / std)
    arr = np.stack(ts, axis=0)               # T,H,W,C
    arr = arr.transpose(3, 0, 1, 2)          # C,T,H,W
    return torch.from_numpy(arr).unsqueeze(0) # 1,C,T,H,W


def preprocess(frames, input_size=224):
    frames = _short_side_scale(frames, input_size)
    frames = _center_crop(frames, input_size)
    return _to_tensor(frames)


# ---------------------------------------------------------------------------
# Robust video loader  (mirrors evaluate_video_sliding_window.py)
# ---------------------------------------------------------------------------

def _video_fps_and_count(video_path: Path):
    """Return (fps, total_frame_count) using cv2 for metadata only."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 15.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, count, w, h


def _load_all_frames_ffmpeg(video_path: Path, w: int, h: int) -> List[np.ndarray]:
    """
    Pipe every frame out of ffmpeg as raw BGR.

    This tolerates corrupt mpeg4 streams that make cv2.VideoCapture silently
    stop delivering frames mid-video.  ffmpeg's -err_detect ignore_err keeps
    decoding past corrupted macroblocks, so we get all recoverable frames.
    """
    ffmpeg = _find_executable("ffmpeg")
    ffprobe = _find_executable("ffprobe")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH; install ffmpeg or use a video that "
            "OpenCV can decode directly.")

    if w <= 0 or h <= 0:
        if ffprobe is None:
            raise RuntimeError(
                "ffprobe was not found on PATH, and OpenCV could not read the "
                "video dimensions. Install ffmpeg/ffprobe or re-encode the video.")
        # Use ffprobe to get dimensions when cv2 can't
        try:
            probe = subprocess.run(
                [ffprobe, "-hide_banner", "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x",
                 str(video_path)],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            detail = f" ffprobe said: {detail}" if detail else ""
            raise RuntimeError(
                f"ffprobe could not read video metadata for {video_path}.{detail}"
            ) from exc
        w, h = (int(x) for x in probe.stdout.strip().split("x"))

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    frame_bytes = w * h * 3
    frames: List[np.ndarray] = []
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(
                np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy())
    finally:
        proc.stdout.close()
        proc.wait(timeout=30)
    return frames


def load_all_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """
    Load every frame from *video_path* and return (frames, fps).

    Strategy:
      1. Get metadata (fps, count, w, h) from cv2  (fast).
      2. Try cv2 first — if it returns all expected frames, use it.
      3. Fall back to ffmpeg-pipe which handles corrupt mpeg4 streams.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}. Set VIDEO=/path/to/video.mp4 "
            "or pass --video with a local file on this machine.")

    fps, expected, w, h = _video_fps_and_count(video_path)

    # Try cv2
    cap = cv2.VideoCapture(str(video_path))
    frames_cv2: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_cv2.append(frame)
    cap.release()

    if expected > 0 and len(frames_cv2) >= 0.95 * expected:
        return frames_cv2, fps   # cv2 recovered everything

    # cv2 fell short — use ffmpeg-pipe
    print(_c(f"  [warn] cv2 decoded only {len(frames_cv2)}/{expected} frames; "
             f"switching to ffmpeg-pipe.", C.YELLOW))
    try:
        frames_ff = _load_all_frames_ffmpeg(video_path, w, h)
    except RuntimeError as exc:
        if frames_cv2:
            print(_c(f"  [warn] {exc} Using {len(frames_cv2)} OpenCV frames.", C.YELLOW))
            return frames_cv2, fps
        raise
    if len(frames_ff) > len(frames_cv2):
        print(_c(f"  [info] ffmpeg-pipe recovered {len(frames_ff)} frames.", C.GREEN))
        return frames_ff, fps

    # ffmpeg also fell short — use whichever got more
    best = frames_ff if len(frames_ff) >= len(frames_cv2) else frames_cv2
    print(_c(f"  [warn] best backend recovered only {len(best)}/{expected} frames.", C.YELLOW))
    return best, fps


# ---------------------------------------------------------------------------
# Inference on one window
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_window(model, frames, device, input_size=224):
    """
    Returns (lh_pred_tuple, rh_pred_tuple, lh_conf_dict, rh_conf_dict).
    """
    t = preprocess(frames, input_size).to(device)
    outputs = model({"frames": t})

    result = {}
    for hand in ("lh", "rh"):
        pred_idx, conf = {}, {}
        for elem in ELEMENTS:
            key = f"{hand}_{elem}"
            probs = torch.softmax(outputs[key], dim=-1)[0]
            idx = int(probs.argmax().item())
            pred_idx[elem] = idx
            conf[elem] = float(probs[idx].item())
        result[hand] = (pred_idx, conf)

    lh_pred = tuple(result["lh"][0][e] for e in ELEMENTS)
    rh_pred = tuple(result["rh"][0][e] for e in ELEMENTS)
    return lh_pred, rh_pred, result["lh"][1], result["rh"][1]


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def _bar(pct, width=20):
    filled = int(pct / 100 * width)
    return '[' + '█' * filled + '░' * (width - filled) + f'] {pct:5.1f}%'


def _fmt_tuple(t, label_maps=None):
    if label_maps is None:
        return str(t)
    elems = []
    for val, key in zip(t, ["verb", "manip_obj", "target_obj", "tool"]):
        name = label_maps[key].get(val, str(val)) if label_maps else str(val)
        elems.append(name)
    return f"({', '.join(elems)})"


def display_state(state: dict, tpg: TaskPrecedenceGraph,
                  window_idx: int, fps: float, label_maps=None):
    sep = '─' * 72
    print(f"\n{_c(sep, C.GREY)}")
    print(_c(f"  Window #{window_idx:4d}  |  t={state['frame_time']:.1f}s  |  "
             f"inference {fps:.1f} fps", C.BOLD, C.WHITE))
    print(_c(sep, C.GREY))

    # ── Predictions ─────────────────────────────────────────────────────────
    lh_tuple_str = _fmt_tuple(state["lh_raw_pred"], label_maps)
    rh_tuple_str = _fmt_tuple(state["rh_raw_pred"], label_maps)
    lh_task_lbl  = state["lh_raw_task"]
    rh_task_lbl  = state["rh_raw_task"]
    conf_lh = min(state["lh_conf"].values())
    conf_rh = min(state["rh_conf"].values())

    print(f"  {_c('RAW PRED', C.CYAN, C.BOLD)}")
    print(f"    LH  {lh_tuple_str:<38} → task: {_c(lh_task_lbl, C.YELLOW)}  "
          f"(conf_min={conf_lh:.2f})")
    print(f"    RH  {rh_tuple_str:<38} → task: {_c(rh_task_lbl, C.YELLOW)}  "
          f"(conf_min={conf_rh:.2f})")

    # ── Holistic confirmed task ───────────────────────────────────────────────
    confirmed = state["confirmed_holistic"]
    new_mark  = _c("✔ NEW", C.GREEN, C.BOLD) if state["changed"] else ""
    # Build readable label: look up each component in the TPG
    parts = state["confirmed_tasks"]
    conf_display = "  |  ".join(
        tpg.label(t) if t in tpg.tasks else t for t in parts)
    print(f"\n  {_c('HOLISTIC TASK  (LH⊕RH fused, majority vote)', C.CYAN, C.BOLD)}")
    print(f"    raw      LH:{_c(state['lh_raw_task'], C.YELLOW)}  "
          f"RH:{_c(state['rh_raw_task'], C.YELLOW)}  "
          f"→ fused: {_c(state['holistic_raw'], C.YELLOW)}")
    print(f"    confirmed  {_c(conf_display, C.WHITE, C.BOLD):<45}  {new_mark}")

    # ── Progress ─────────────────────────────────────────────────────────────
    done  = sorted(state["completed"])
    elig  = state["eligible"]
    pct   = state["progress_pct"]
    print(f"\n  {_c('PROGRESS', C.CYAN, C.BOLD)}  {_bar(pct)}")
    print(f"    Completed ({len(done)}): "
          + ", ".join(_c(tpg.label(t), C.GREEN) for t in done) if done
          else "    Completed: —")
    print(f"    Next eligible ({len(elig)}): "
          + ", ".join(_c(tpg.label(t), C.YELLOW) for t in elig) if elig
          else _c("    Next eligible: — (all done or no eligible tasks)", C.GREY))

    # ── Errors ───────────────────────────────────────────────────────────────
    err = state["error"]
    if err:
        print(f"\n  {_c('⚠  OPERATIONAL ERROR', C.RED, C.BOLD)}")
        print(f"    Performed : {_c(err['confirmed_label'], C.RED)}")
        print(f"    Closest   : {_c(err['closest_label'],   C.YELLOW)}")
        if err["mismatch"]:
            for elem, info in err["mismatch"].items():
                lm = (label_maps or {}).get(elem, {})
                pred_name = lm.get(info["pred"],     str(info["pred"]))
                exp_name  = lm.get(info["expected"], str(info["expected"]))
                status = _c("✔", C.GREEN) if info["match"] else \
                         _c(f"✗  pred={pred_name} ≠ expected={exp_name}", C.RED)
                print(f"      {elem:<14} {status}")
        elif err["closest_label"] == "none":
            print(_c("    (no eligible task at this point — all predecessors still pending)", C.GREY))

    print(_c(sep, C.GREY))


# ---------------------------------------------------------------------------
# System 2 integration helpers
# ---------------------------------------------------------------------------

def _run_interactive_qa(session_log: "LiveSessionLog") -> None:
    """
    Post-session interactive Q&A loop powered by System 2.
    Runs after the frame loop (and optional auto-review) until the user
    presses Enter with no text or hits Ctrl-C.
    """
    print(_c("\n  ━━  System 2 Q&A  ━━  Type a question and press Enter.  "
             "Press Enter with no text to exit.", C.CYAN))
    while True:
        try:
            raw = input(_c("  Query → ", C.WHITE))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        query = raw.strip()
        if not query:
            break
        _invoke_system2_guidance(session_log, query)
    print(_c("  System 2 session ended.", C.GREY))


def _invoke_system2_guidance(session_log: "LiveSessionLog", query: str) -> None:
    """
    Pause System 1, bridge live session state into System 2, ask DeepSeek,
    print the response, then return to the caller so System 1 resumes.
    """
    print(_c("\n" + "═" * 72, C.CYAN))
    print(_c("  ⏸  PAUSED — System 2 Advisory (DeepSeek)", C.CYAN, C.BOLD))
    print(_c(f"  Query: \"{query}\"", C.WHITE))
    print(_c("  Thinking…", C.GREY))

    try:
        _ac_dir = str(Path(__file__).parent / "Assembly_copilot")
        if _ac_dir not in sys.path:
            sys.path.insert(0, _ac_dir)

        from bridge import sync_session_state   # noqa: PLC0415
        from agent  import on_demand_guidance   # noqa: PLC0415

        sync_session_state(str(session_log.output_path))
        response = on_demand_guidance(query)

        print(_c("\n  SYSTEM 2 RESPONSE:", C.CYAN, C.BOLD))
        print(response)
    except ImportError as exc:
        print(_c(f"  [System 2 unavailable] {exc}", C.RED))
        print(_c("  Install openai and set DEEPSEEK_API_KEY to enable System 2.", C.YELLOW))
    except Exception as exc:
        print(_c(f"  [System 2 error] {exc}", C.RED))

    print(_c("▶  RESUMING System 1…", C.GREEN))
    print(_c("═" * 72, C.CYAN))


def _invoke_system2_review(session_log: "LiveSessionLog") -> None:
    """
    Trigger the post-session review after the frame loop ends.
    Bridges the final sealed JSON → SESSION_STATE, then calls post_session_review().
    """
    print(_c("\n" + "═" * 72, C.CYAN))
    print(_c("  📋  POST-SESSION REVIEW  (System 2 / DeepSeek)", C.CYAN, C.BOLD))
    print(_c("  Generating review — this may take ~30 s…", C.GREY))

    try:
        _ac_dir = str(Path(__file__).parent / "Assembly_copilot")
        if _ac_dir not in sys.path:
            sys.path.insert(0, _ac_dir)

        from bridge import sync_session_state   # noqa: PLC0415
        from agent  import post_session_review  # noqa: PLC0415

        sync_session_state(str(session_log.output_path))
        response = post_session_review()

        print(response)
    except ImportError as exc:
        print(_c(f"  [System 2 unavailable] {exc}", C.RED))
        print(_c("  Install openai and set DEEPSEEK_API_KEY to enable System 2.", C.YELLOW))
    except Exception as exc:
        print(_c(f"  [System 2 error] {exc}", C.RED))

    print(_c("═" * 72, C.CYAN))


# ---------------------------------------------------------------------------
# Main real-time loop
# ---------------------------------------------------------------------------

def run(args):
    device = resolve_device(args.device)

    # ── Build TPG + mapper ──────────────────────────────────────────────────
    tpg = TaskPrecedenceGraph(MOCK_TPG_TASKS, MOCK_TPG_EDGES)
    # ── Load label maps (needed by mapper AND display) ──────────────────────
    def _lm(dirpath, name):
        m = {}
        p = Path(dirpath) / name
        if p.is_file():
            with open(p) as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        m[int(parts[0])] = parts[1]
        return m

    ldir = args.label_dir or ""
    label_maps = {
        "verb":       _lm(ldir, "label_map_verbs.txt"),
        "manip_obj":  _lm(ldir, "label_map_manip_objs.txt"),
        "target_obj": _lm(ldir, "label_map_target_objs.txt"),
        "tool":       _lm(ldir, "label_map_tools.txt"),
    }

    mapper  = CompositionToTaskMapper(label_maps, args.composition_file)
    monitor = System1Monitor(tpg, mapper,
                             vote_window=args.vote_window,
                             dwell_required=args.dwell_required)

    # ── Load model ──────────────────────────────────────────────────────────
    print(_c("\n  Loading CoDuAR model ...", C.CYAN))
    model = build_model(args, device)
    print(_c(f"  Model loaded on {device}.", C.GREEN))

    # ── Load video (robust: ffmpeg-pipe fallback for corrupt mpeg4) ──────────
    print(_c(f"  Loading video frames from {args.video} …", C.CYAN))
    all_frames, fps_video = load_all_frames(Path(args.video))
    total_frames = len(all_frames)
    print(_c(f"  Video: {args.video}  |  {total_frames} frames @ {fps_video:.1f} fps", C.WHITE))
    print(_c(f"  Window={args.window}  Stride={args.stride}  Vote={args.vote_window}\n", C.GREY))

    # ── Print TPG ────────────────────────────────────────────────────────────
    print(_c("  TASK PRECEDENCE GRAPH", C.CYAN, C.BOLD))
    for t in MOCK_TPG_TASKS:
        preds = tpg.predecessors[t["id"]]
        pred_str = " ← " + ", ".join(tpg.label(p) for p in preds) if preds else "  [start]"
        print(f"    {t['id']:<12}  {t['label']:<28} {pred_str}")
    print()

    # ── Live session log ─────────────────────────────────────────────────────
    n_tracked = sum(1 for tid in tpg.tasks if tpg.is_tracked(tid))
    log_path  = args.session_log or (
        str(Path(args.video).with_suffix("")) + "_session_log.json")
    session_log = LiveSessionLog(
        video_path=args.video, fps=fps_video, total_frames=total_frames,
        output_path=log_path,  n_tracked=n_tracked,
        label_maps=label_maps, tpg=tpg,
    )
    print(_c(f"  Session log → {log_path}", C.CYAN))

    print(_c("  System 2 Q&A available after processing completes.", C.GREY))

    # ── Sliding-window loop over pre-loaded frames ────────────────────────────
    frame_buffer: deque = deque(maxlen=args.window)
    window_idx = 0
    frame_time = 0.0

    t_start = time.perf_counter()
    fps_inf  = 0.0

    for frame_idx, frame in enumerate(all_frames):
        frame_buffer.append(frame)

        # Only run inference every `stride` frames
        if (frame_idx + 1) % args.stride != 0:
            continue
        if len(frame_buffer) < args.window:
            continue

        frames = list(frame_buffer)

        # ── Inference ───────────────────────────────────────────────────────
        t_inf = time.perf_counter()
        lh_pred, rh_pred, lh_conf, rh_conf = infer_window(
            model, frames, device, input_size=224)
        fps_inf = 1.0 / max(1e-6, time.perf_counter() - t_inf)

        frame_time = (frame_idx + 1) / fps_video
        state = monitor.step(frame_time, lh_pred, rh_pred, lh_conf, rh_conf)

        display_state(state, tpg, window_idx, fps_inf, label_maps)
        session_log.update(state)
        window_idx += 1

    # ── Seal the session log ──────────────────────────────────────────────────
    session_log.finalize(frame_time)

    # ── System 2: auto review + interactive Q&A ──────────────────────────────
    if args.post_session_review:
        _invoke_system2_review(session_log)   # full DeepSeek review first
        _run_interactive_qa(session_log)      # then open Q&A loop

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(_c("\n" + "═" * 72, C.CYAN))
    print(_c("  SESSION COMPLETE", C.BOLD, C.WHITE))
    print(_c("═" * 72, C.CYAN))
    print(f"  Total windows   : {window_idx}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    n_tracked = sum(1 for tid in tpg.tasks if tpg.is_tracked(tid))
    print(f"  Tasks completed : {len(monitor.completed)} / {n_tracked}")
    final_pct = len(monitor.completed) / max(1, n_tracked) * 100
    print(f"  Progress bar    : {_bar(final_pct)}")
    if monitor.log:
        print(f"\n  {_c('Task Progression Log', C.CYAN, C.BOLD)}")
        print(f"  {'Task':<32}  {'t_start':>8}  {'Status / Detail'}")
        print(f"  {'─'*72}")
        for entry in monitor.log:
            err = entry["error"]
            if err:
                err_str = _c("ERROR", C.RED)
                print(f"  {entry['label']:<32}  {entry['t_start']:>7.1f}s  {err_str}")
                print(f"    {'Performed':<12}: {_c(err['confirmed_label'], C.RED)}")
                print(f"    {'Closest':<12}: {_c(err['closest_label'],   C.YELLOW)}")
                if err.get("mismatch"):
                    for elem, info in err["mismatch"].items():
                        lm = label_maps.get(elem, {})
                        pred_name = lm.get(info["pred"],     str(info["pred"]))
                        exp_name  = lm.get(info["expected"], str(info["expected"]))
                        sym = "✔" if info["match"] else f"✗  pred={pred_name} ≠ expected={exp_name}"
                        print(f"      {elem:<14} {sym}")
                elif err["closest_label"] == "none":
                    print(_c("      (no eligible task at that point)", C.GREY))
            else:
                err_str = _c("OK", C.GREEN)
                print(f"  {entry['label']:<32}  {entry['t_start']:>7.1f}s  {err_str}")
    print(_c("═" * 72, C.CYAN))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Assembly Copilot — System 1 Real-Time Monitor")

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--video",       required=True, help="Input video file")
    p.add_argument("--checkpoint",  required=True, help="Model checkpoint (.pth)")

    # ── Task mapping ──────────────────────────────────────────────────────────
    p.add_argument("--composition_file", default=None,
                   help="Optional compositional mapping file to augment Ψ")
    p.add_argument("--label_dir", default=None,
                   help="Directory with label_map_*.txt for readable display")

    # ── Sliding window ────────────────────────────────────────────────────────
    p.add_argument("--window",      type=int, default=16,
                   help="Frames per window (default: 16)")
    p.add_argument("--stride",      type=int, default=4,
                   help="Frame stride between windows (default: 4)")
    p.add_argument("--vote_window", type=int, default=5,
                   help="Majority-vote window W (default: 5 windows)")
    p.add_argument("--dwell_required", type=int, default=3,
                   help="Min consecutive windows a task must be confirmed before "
                        "entering completed (dwell gate, default: 3). "
                        "Prevents brief false-positive predictions from "
                        "permanently marking a task as done.")

    # ── Model architecture (must match checkpoint) ────────────────────────────
    p.add_argument("--lh_num_verbs",      type=int, default=5)
    p.add_argument("--lh_num_manip_objs", type=int, default=12)
    p.add_argument("--lh_num_target_objs",type=int, default=6)
    p.add_argument("--lh_num_tools",      type=int, default=4)
    p.add_argument("--rh_num_verbs",      type=int, default=5)
    p.add_argument("--rh_num_manip_objs", type=int, default=12)
    p.add_argument("--rh_num_target_objs",type=int, default=6)
    p.add_argument("--rh_num_tools",      type=int, default=4)
    p.add_argument("--decoder_layers",    type=int, default=3)
    p.add_argument("--decoder_heads",     type=int, default=8)
    p.add_argument("--decoder_dim",       type=int, default=2048)
    p.add_argument("--decoder_dropout",   type=float, default=0.1)
    p.add_argument("--head_dropout",      type=float, default=0.1)
    p.add_argument("--adapter_dim",       type=int, default=128)
    p.add_argument("--use_hand_adapters", action="store_true", default=True)

    # ── Device ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--device", default="auto",
        help="Inference device: auto, cpu, mps, cuda, or cuda:N (default: auto)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--session_log", default=None,
                   help="Path for the live session JSON log "
                        "(default: <video_stem>_session_log.json next to the video)")

    # ── System 2 ──────────────────────────────────────────────────────────────
    p.add_argument("--post_session_review", action="store_true", default=False,
                   help="After the session ends, automatically call System 2 to "
                        "generate a post-session performance review "
                        "(requires DEEPSEEK_API_KEY to be set)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
