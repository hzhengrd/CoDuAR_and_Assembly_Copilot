#!/usr/bin/env python3
"""
run_system2.py — Subprocess entrypoint for System 2 (LLM advisory).

Invoked by assembly_copilot_gui.py via subprocess using the Python environment
that has `openai` installed (Qwen25VL env on this machine).

Usage:
    python run_system2.py --log_path PATH --mode {guidance|review} [--query TEXT]

Output:
    Response text → stdout
    Errors        → stderr
    Exit code:    0 = success, 1 = failure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # Assembly_copilot/
sys.path.insert(0, str(_HERE.parent))   # project root (for bridge.py path)


def main() -> None:
    p = argparse.ArgumentParser(description="System 2 LLM runner")
    p.add_argument("--log_path", required=True,
                   help="Path to the LiveSessionLog JSON file")
    p.add_argument("--mode", required=True, choices=["guidance", "review"],
                   help="'guidance' for on-demand query, 'review' for post-session")
    p.add_argument("--query", default="",
                   help="User query (required for mode=guidance)")
    args = p.parse_args()

    bridge_path = Path(args.log_path).parent.parent  # try to find bridge.py
    for candidate in [_HERE.parent, _HERE, bridge_path]:
        if (candidate / "bridge.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    import json as _json

    from bridge import sync_session_state   # noqa: PLC0415
    import tools as _tools                  # noqa: PLC0415
    _tools.clear_media_log()               # reset per-run media tracking
    sync_session_state(args.log_path)

    if args.mode == "guidance":
        from agent import on_demand_guidance   # noqa: PLC0415
        resp = on_demand_guidance(args.query)
    else:
        from agent import post_session_review  # noqa: PLC0415
        resp = post_session_review()

    # Output JSON so the GUI can distinguish the text from media metadata
    out = {
        "response": resp,
        "media_task_ids": list(_tools.MEDIA_RETRIEVED_TASK_IDS),
    }
    print(_json.dumps(out, ensure_ascii=False), end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
