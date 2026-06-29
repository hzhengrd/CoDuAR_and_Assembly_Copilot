"""
example_usage.py — Two complete System 2 use cases (DeepSeek backend).

Use Case 1: Operator pauses mid-session and asks for guidance on T008
            (torque step). System 2 provides procedure, media, and
            critical parameters.

Use Case 2: Operator completes the full session. System 2 generates
            a post-session performance review with z-score analysis.

Setup:
    pip install openai
    export DEEPSEEK_API_KEY=your_key_here
    python example_usage.py

Get your DeepSeek API key at: https://platform.deepseek.com
"""

import subprocess
import sys
import time
from pathlib import Path

from tools import SESSION_STATE, retrieve_media
from agent import on_demand_guidance, post_session_review


def _open_file(path: Path) -> None:
    """Open a local file with the OS default viewer."""
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["start", "", str(path)], check=False, shell=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def show_reference_images(task_id: str) -> None:
    """
    Open any local reference images available for the given task.
    """
    media = retrieve_media(task_id)
    assets = media.get("assets", [])
    image_assets = [asset for asset in assets if asset.get("type") == "image"]

    if not image_assets:
        print(f"[Media] No reference images listed for task {task_id}.")
        return

    project_root = Path(__file__).resolve().parent
    opened_any = False

    for asset in image_assets:
        image_path = (project_root / asset["uri"]).resolve()
        if image_path.exists():
            print(f"[Media] Opening image {asset['asset_id']}: {image_path}")
            _open_file(image_path)
            opened_any = True
        else:
            print(f"[Media] Image not found for {asset['asset_id']}: {image_path}")

    if not opened_any:
        print(f"[Media] No local image files were available for task {task_id}.")


# ── Session state simulators ──────────────────────────────────────────────────

def simulate_mid_session():
    """
    Populate SESSION_STATE as a mid-session snapshot: T001–T007 done,
    T008 and T009 currently in progress (parallel torque step).
    T002 was notably slow (threadlocker application difficulties).
    """
    now = time.time()
    t0 = now - 620  # session started ~10 min ago

    SESSION_STATE["session_start_time"] = t0
    SESSION_STATE["session_end_time"] = None

    SESSION_STATE["progression_log"] = [
        {"task_id": "T001", "hand": "both",  "start_time": t0,      "end_time": t0+35,  "duration_seconds": 35, "mean_confidence": 0.91, "error_flags": []},
        {"task_id": "T002", "hand": "left",  "start_time": t0+38,   "end_time": t0+98,  "duration_seconds": 60, "mean_confidence": 0.87, "error_flags": []},
        {"task_id": "T003", "hand": "right", "start_time": t0+38,   "end_time": t0+88,  "duration_seconds": 50, "mean_confidence": 0.89, "error_flags": []},
        {"task_id": "T004", "hand": "left",  "start_time": t0+100,  "end_time": t0+122, "duration_seconds": 22, "mean_confidence": 0.93, "error_flags": []},
        {"task_id": "T005", "hand": "right", "start_time": t0+100,  "end_time": t0+121, "duration_seconds": 21, "mean_confidence": 0.92, "error_flags": []},
        {"task_id": "T006", "hand": "left",  "start_time": t0+125,  "end_time": t0+162, "duration_seconds": 37, "mean_confidence": 0.88, "error_flags": []},
        {"task_id": "T007", "hand": "right", "start_time": t0+125,  "end_time": t0+158, "duration_seconds": 33, "mean_confidence": 0.90, "error_flags": []},
    ]
    SESSION_STATE["completed_tasks"] = {
        "T001", "T002", "T003", "T004", "T005", "T006", "T007"
    }
    SESSION_STATE["current_task"]    = {"left": "T008", "right": "T009"}
    SESSION_STATE["eligible_tasks"]  = ["T008", "T009"]
    SESSION_STATE["procedural_errors"] = []
    SESSION_STATE["anomaly_flags"]     = []


def simulate_completed_session():
    """
    Full session: all 10 tasks done.
    Notable deviations:
      T002 slow  (60 s vs reference mean 43.1 s, z ≈ +1.8)
      T010 fast  (40 s vs reference mean 57.6 s, z ≈ -1.4) — possibly rushed
    """
    simulate_mid_session()

    t0 = SESSION_STATE["session_start_time"]
    SESSION_STATE["progression_log"] += [
        {"task_id": "T008", "hand": "left",  "start_time": t0+165, "end_time": t0+210, "duration_seconds": 45, "mean_confidence": 0.85, "error_flags": []},
        {"task_id": "T009", "hand": "right", "start_time": t0+165, "end_time": t0+207, "duration_seconds": 42, "mean_confidence": 0.86, "error_flags": []},
        {"task_id": "T010", "hand": "both",  "start_time": t0+213, "end_time": t0+253, "duration_seconds": 40, "mean_confidence": 0.82, "error_flags": ["lower_confidence_than_average"]},
    ]
    SESSION_STATE["completed_tasks"] = {
        "T001","T002","T003","T004","T005","T006","T007","T008","T009","T010"
    }
    SESSION_STATE["current_task"]   = {"left": None, "right": None}
    SESSION_STATE["eligible_tasks"] = []
    SESSION_STATE["session_end_time"] = t0 + 253


# ── Use Case 1: On-demand guidance ────────────────────────────────────────────

def use_case_1_guidance():
    print("=" * 70)
    print("USE CASE 1: On-Demand Guidance")
    print("Scenario: Operator pauses before T008 and asks about torque.")
    print("=" * 70)

    simulate_mid_session()

    query = "What torque setting do I need for the bolt, and how do I apply it correctly?"
    print(f"\nOperator query: \"{query}\"\n")
    print("System 2 agent running (DeepSeek-V3)...\n")
    print("-" * 70)

    response = on_demand_guidance(query)
    print(response)
    show_reference_images("T008")

    print("-" * 70)
    print("Use Case 1 complete.\n")


# ── Use Case 2: Post-session review ───────────────────────────────────────────

def use_case_2_review():
    print("=" * 70)
    print("USE CASE 2: Post-Session Performance Review")
    print("Scenario: All 10 tasks done — session-end signal fired.")
    print("  T002 was notably slow; T010 was completed faster than average.")
    print("=" * 70)

    simulate_completed_session()

    print("\nGenerating post-session review (DeepSeek-V3)...\n")
    print("-" * 70)

    response = post_session_review()
    print(response)

    print("-" * 70)
    print("Use Case 2 complete.\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nAssembly Copilot — System 2 (DeepSeek backend)\n")
    use_case_1_guidance()
    use_case_2_review()