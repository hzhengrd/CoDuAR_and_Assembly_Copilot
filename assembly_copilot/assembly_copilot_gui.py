"""
assembly_copilot_gui.py — Assembly Copilot unified GUI
=======================================================
Layout:
  LEFT  (video + System 1)          RIGHT (System 2 LLM)
  ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │  Video playback (live)       │  │  System 2 Advisory           │
  ├──────────────────────────────┤  │  ┌──────────────────────┐    │
  │  CURRENT TASK  ████  83%     │  │  │  Response / Review   │    │
  │  [⏸ Pause]  [FPS display]   │  │  │  (text + images)     │    │
  │  Next tasks  │  Event log    │  │  └──────────────────────┘    │
  │  ⚠ ERROR BOX (when active)  │  │  [Query input]               │
  └──────────────────────────────┘  │  [▶ Ask]  [📋 Review]        │
                                    └──────────────────────────────┘

Requirements:  pip install Pillow  (tkinter built-in)
               openai in Qwen25VL conda env (used via subprocess)
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
import contextlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")
warnings.filterwarnings("ignore", message=r"Overwriting vit_.* in registry.*")

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
except ImportError as exc:
    print(f"Missing GUI dependency: {exc}\nRun:  pip install Pillow")
    sys.exit(1)

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from realtime_system1 import (          # noqa: E402
    MOCK_TPG_EDGES, MOCK_TPG_TASKS,
    CompositionToTaskMapper, LiveSessionLog,
    System1Monitor, TaskPrecedenceGraph,
    build_model, infer_window, load_all_frames, resolve_device,
)

# ── System 2 subprocess configuration ────────────────────────────────────────
# Use the same Python that launched the GUI by default. Override via env var if
# System 2 dependencies live in a separate environment.
_S2_PYTHON = os.environ.get("SYSTEM2_PYTHON", sys.executable)
_S2_RUNNER = str(_ROOT / "Assembly_copilot" / "run_system2.py")

# ── Light mode palette ────────────────────────────────────────────────────────
BG       = "#f1f5f9"   # window / outer background
BG_PANEL = "#ffffff"   # main panel fills
BG_CARD  = "#f8fafc"   # inner card / textbox fills
BG_INPUT = "#ffffff"   # input box
FG       = "#0f172a"   # primary text (near-black)
FG_DIM   = "#64748b"   # secondary / placeholder text
FG_TITLE = "#1e293b"   # strong title text
COL_OK   = "#15803d"   # green
COL_ERR  = "#dc2626"   # red
COL_WARN = "#b45309"   # amber
COL_INFO = "#1d4ed8"   # blue
COL_TASK = "#6d28d9"   # violet (current task)
COL_SEP  = "#e2e8f0"   # divider lines
HDR_BG   = "#1e3a5f"   # dark blue header bars
# Error box
ERR_BG   = "#fff1f2"   # rose tint background
ERR_BDR  = "#fda4af"   # rose border
ERR_HEAD = "#be123c"   # dark red header text

# Font family — updated at runtime after Tk initialises (prefers Roboto)
_FF  = "TkDefaultFont"    # overwritten in AssemblyCopilotGUI._build_root
_FFM = "TkFixedFont"

# Convenience tuples — redefined in _build_root once _FF is known
FONT_SM  = (_FF,  10)
FONT_MD  = (_FF,  11)
FONT_LG  = (_FF,  12, "bold")
FONT_TTL = (_FF,  11, "bold")
FONT_MON = (_FFM, 10)

# ── TPG label cache ───────────────────────────────────────────────────────────
_TPG_LABELS: Dict[str, str] = {t["id"]: t["label"] for t in MOCK_TPG_TASKS}

# Full human-readable names for the four compositional elements.
# Keys are the short abbreviation strings stored in label_map_*.txt files.
# Used ONLY in the error detail panel — the short codes are still required by
# CompositionToTaskMapper for task-ID construction.
_ELEM_FULL: Dict[str, Dict[str, str]] = {
    "verb": {
        "null": "null", "i": "insert", "p": "place", "r": "rotate", "s": "screw",
    },
    "manip_obj": {
        "null": "null",
        "gl":   "large gear",     "gs": "small gear",
        "pl":   "large placer",   "ps": "small placer",
        "dp":   "Phillips screwdriver", "gw": "worm gear",
        "wn":   "wrench-nut",     "ws": "wrench-shaft",
        "ft":   "shaft",          "nt": "nut",   "sp": "Phillips screw",
    },
    "target_obj": {
        "null": "null",
        "f1": "Shaft 1",  "f2": "Shaft 2",
        "g1": "Gear Hole 1", "g2": "Gear Hole 2", "g3": "Gear Hole 3",
    },
    "tool": {
        "null": "null",
        "dp": "Phillips screwdriver", "wn": "wrench-nut", "ws": "wrench-shaft",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class InferenceWorker(threading.Thread):
    """
    System 1 – loads all video frames, then runs sliding-window inference.

    Pause/Resume:
      worker.pause()  → sets  _pause_event  (loop blocks at wait())
      worker.resume() → clears _pause_event (loop unblocks)
    """

    def __init__(self, args: Any, ui_q: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self._args  = args
        self._q     = ui_q
        self._stop  = threading.Event()
        # _pause_event is SET when running, CLEARED when paused
        self._pause_event = threading.Event()
        self._pause_event.set()

    # ── Public controls ───────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()
        self._pause_event.set()   # unblock any wait so thread can exit

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: C901
        args = self._args
        try:
            self._put("status", "Loading model…")
            device = resolve_device(args.device)
            tpg    = TaskPrecedenceGraph(MOCK_TPG_TASKS, MOCK_TPG_EDGES)

            ldir = args.label_dir or ""
            label_maps = {
                k: self._load_lm(ldir, f"label_map_{v}.txt")
                for k, v in [
                    ("verb", "verbs"), ("manip_obj", "manip_objs"),
                    ("target_obj", "target_objs"), ("tool", "tools"),
                ]
            }
            mapper  = CompositionToTaskMapper(label_maps, args.composition_file)
            monitor = System1Monitor(tpg, mapper,
                                     vote_window=args.vote_window,
                                     dwell_required=getattr(args, "dwell_required", 3))
            model   = build_model(args, device)
            self._put("model_ok", None)

            self._put("status", f"Loading video: {Path(args.video).name}…")
            all_frames, fps = load_all_frames(Path(args.video))
            self._put("video_ok", (len(all_frames), fps))

            n_tracked = sum(1 for tid in tpg.tasks if tpg.is_tracked(tid))
            log_path  = args.session_log or (
                str(Path(args.video).with_suffix("")) + "_session_log.json")
            slog = LiveSessionLog(
                video_path=args.video, fps=fps, total_frames=len(all_frames),
                output_path=log_path, n_tracked=n_tracked,
                label_maps=label_maps, tpg=tpg,
            )
            self._put("log_path", log_path)

            buf   = deque(maxlen=args.window)
            win   = 0
            ftime = 0.0
            for fi, frame in enumerate(all_frames):
                # ── Pause checkpoint ──────────────────────────────────────────
                self._pause_event.wait()   # blocks when paused
                if self._stop.is_set():
                    break

                buf.append(frame)
                if (fi + 1) % args.stride != 0:
                    continue
                if len(buf) < args.window:
                    continue

                t0 = time.perf_counter()
                lh_p, rh_p, lh_c, rh_c = infer_window(
                    model, list(buf), device)
                fps_inf = 1.0 / max(1e-9, time.perf_counter() - t0)

                ftime = (fi + 1) / fps
                state = monitor.step(ftime, lh_p, rh_p, lh_c, rh_c)
                slog.update(state)

                # Resolve mismatch integer IDs → human-readable labels for display.
                # Step 1: short code from label_map (e.g. "s", "ft", "g1")
                # Step 2: expand short code → full name via _ELEM_FULL
                if state.get("error") and state["error"].get("mismatch"):
                    def _full(elem: str, idx: int) -> str:
                        short = label_maps.get(elem, {}).get(idx, str(idx))
                        return _ELEM_FULL.get(elem, {}).get(short, short)
                    state["error"]["mismatch_labels"] = {
                        elem: {
                            "pred_label":     _full(elem, info["pred"]),
                            "expected_label": _full(elem, info["expected"]),
                            "match": info["match"],
                        }
                        for elem, info in state["error"]["mismatch"].items()
                    }

                self._put("window", {
                    "frame":   frame.copy(),
                    "state":   state,
                    "win":     win,
                    "fps_inf": fps_inf,
                })
                win += 1

            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                slog.finalize(ftime)
            self._put("done", log_path)

        except Exception:
            import traceback
            self._put("error", traceback.format_exc())

    @staticmethod
    def _load_lm(dirpath: str, filename: str) -> Dict[int, str]:
        m: Dict[int, str] = {}
        p = Path(dirpath) / filename if dirpath else Path(filename)
        if p.is_file():
            with open(p) as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        try:
                            m[int(parts[0])] = parts[1]
                        except ValueError:
                            pass
        return m

    def _put(self, kind: str, payload: Any) -> None:
        self._q.put((kind, payload))


# ─────────────────────────────────────────────────────────────────────────────
# Screen recorder
# ─────────────────────────────────────────────────────────────────────────────

class ScreenRecorder:
    """
    Records the Assembly Copilot window to a video file.

    Backends:
      - macOS: ffmpeg avfoundation screen capture.
      - Linux/X11: ffmpeg x11grab.

    Usage:
        rec = ScreenRecorder(output_path, root_widget, fps=10)
        rec.start()   # call after the Tk window is visible and positioned
        ...
        rec.stop()    # call before root.destroy()
    """

    def __init__(self, output_path: str, root: "tk.Tk", fps: int = 10) -> None:
        self._output = output_path
        self._root   = root
        self._fps    = fps
        self._proc: Optional[subprocess.Popen] = None
        self._backend = "macos" if sys.platform == "darwin" else "x11"
        self._capture_output = output_path

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the platform recorder subprocess."""
        self._root.update_idletasks()
        self._root.update()

        # Client-area geometry (excludes OS title bar — gives clean app content)
        x  = self._root.winfo_rootx()
        y  = self._root.winfo_rooty()
        w  = self._root.winfo_width()
        h  = self._root.winfo_height()

        # Round width / height down to nearest even number (H.264 requirement)
        w = w - (w % 2)
        h = h - (h % 2)
        if w < 2 or h < 2:
            print("[recording] Window too small; skipping recorder.", flush=True)
            return

        if self._backend == "macos":
            self._start_macos(x, y, w, h)
        else:
            self._start_x11(x, y, w, h)

    def _macos_display_scale(self) -> float:
        raw = os.environ.get("MACOS_RECORD_SCALE")
        if raw:
            try:
                return max(0.1, float(raw))
            except ValueError:
                pass
        try:
            # Tk reports geometry in logical points on Retina displays, while
            # avfoundation/ffmpeg crop uses physical display pixels.
            return max(0.1, float(self._root.winfo_fpixels("1i")) / 72.0)
        except Exception:
            return 1.0

    def _start_macos(self, x: int, y: int, w: int, h: int) -> None:
        """Start macOS screen recording via ffmpeg avfoundation."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[recording] ffmpeg not found — cannot record on macOS.",
                  flush=True)
            return

        # Device index is machine-dependent. On this Mac, ffmpeg lists
        # "Capture screen 0" as device 2. Override if needed:
        #   MACOS_SCREEN_DEVICE="3:none" bash ... --recording out.mp4
        #
        # By default we crop to the settled Tk geometry. The crop rectangle is
        # static, so do not move the GUI window after recording starts.
        device = os.environ.get("MACOS_SCREEN_DEVICE", "2:none")
        region = os.environ.get("MACOS_RECORD_REGION", "window").strip().lower()
        Path(self._output).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y",
            "-f", "avfoundation",
            "-framerate", str(self._fps),
            "-i", device,
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            self._output,
        ]
        if region in {"window", "crop"}:
            scale = self._macos_display_scale()
            sx, sy = int(round(x * scale)), int(round(y * scale))
            sw, sh = int(round(w * scale)), int(round(h * scale))
            sw -= sw % 2
            sh -= sh % 2
            cmd[cmd.index("-vcodec"):cmd.index("-vcodec")] = [
                "-vf", f"crop={sw}:{sh}:{sx}:{sy}",
            ]
            print(f"[recording] macOS window crop: logical={w}x{h}+{x},{y}, "
                  f"scale={scale:.2f}, physical={sw}x{sh}+{sx},{sy}",
                  flush=True)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.75)
            if self._proc.poll() is not None:
                err = ""
                if self._proc.stderr is not None:
                    err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                print(f"[recording] macOS ffmpeg capture exited immediately; "
                      f"recording disabled."
                      f"{' Error: ' + err if err else ''}", flush=True)
                self._proc = None
                return
            print(f"[recording] ● Started  →  {self._output}  "
                  f"({w}x{h} @ {self._fps} fps)", flush=True)
        except Exception as exc:
            print(f"[recording] Failed to start macOS recorder: {exc}", flush=True)
            self._proc = None

    def _start_x11(self, x: int, y: int, w: int, h: int) -> None:
        """Start Linux/X11 ffmpeg recording for the GUI rectangle."""
        display = os.environ.get("DISPLAY", ":0")
        cmd = [
            "ffmpeg", "-y",
            "-f",         "x11grab",
            "-framerate", str(self._fps),
            "-video_size", f"{w}x{h}",
            "-i",         f"{display}+{x},{y}",
            # H.264, fast encode, web-compatible pixel format
            "-vcodec",  "libx264",
            "-preset",  "fast",
            "-crf",     "18",
            "-pix_fmt", "yuv420p",
            self._output,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.25)
            if self._proc.poll() is not None:
                err = ""
                if self._proc.stderr is not None:
                    err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                print(f"[recording] ffmpeg exited immediately; recording disabled."
                      f"{' Error: ' + err if err else ''}", flush=True)
                self._proc = None
                return
            print(f"[recording] ● Started  →  {self._output}  "
                  f"({w}×{h} @ {self._fps} fps)", flush=True)
        except FileNotFoundError:
            print("[recording] ffmpeg not found — install ffmpeg to enable recording.",
                  flush=True)
            self._proc = None
        except Exception as exc:
            print(f"[recording] Failed to start: {exc}", flush=True)
            self._proc = None

    def stop(self) -> None:
        """Gracefully terminate the recorder and finalise the output file."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            print("[recording] Process already exited.", flush=True)
            return
        if self._backend == "macos":
            self._stop_macos()
        else:
            self._stop_x11()

    def _stop_x11(self) -> None:
        try:
            # Send 'q' to ffmpeg's stdin — clean shutdown
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(b"q")
                self._proc.stdin.flush()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=15)
            print(f"[recording] ■ Stopped  →  {self._output}", flush=True)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            print(f"[recording] Force-killed; partial file at {self._output}", flush=True)

    def _stop_macos(self) -> None:
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(b"q")
                self._proc.stdin.flush()
            elif self._proc:
                self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=15)
            print(f"[recording] ■ Stopped  →  {self._output}", flush=True)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print(f"[recording] Force-stopped; partial file at {self._output}",
                  flush=True)
        else:
            return

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class System2Worker(threading.Thread):
    """
    Calls System 2 (DeepSeek LLM) via a subprocess that uses the Qwen25VL
    conda environment, which has openai installed.
    """

    def __init__(self, log_path: str, mode: str, query: str,
                 ui_q: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self._log_path = log_path
        self._mode     = mode    # "guidance" | "review"
        self._query    = query
        self._q        = ui_q

    def run(self) -> None:
        cmd = [
            _S2_PYTHON, _S2_RUNNER,
            "--log_path", self._log_path,
            "--mode",     self._mode,
            "--query",    self._query,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or "System 2 returned non-zero exit code"
                self._q.put(("s2_err", err))
            else:
                raw = result.stdout.strip()
                # run_system2.py now outputs JSON: {"response": "...", "media_task_ids": [...]}
                try:
                    parsed = json.loads(raw)
                    resp           = parsed.get("response", raw)
                    media_task_ids = parsed.get("media_task_ids", [])
                except (json.JSONDecodeError, ValueError):
                    resp           = raw   # fallback: treat entire stdout as text
                    media_task_ids = []
                self._q.put(("s2_ok", (self._mode, self._query, resp, media_task_ids)))
        except subprocess.TimeoutExpired:
            self._q.put(("s2_err", "System 2 timed out after 180 s."))
        except FileNotFoundError:
            self._q.put((
                "s2_err",
                f"Python interpreter not found:\n  {_S2_PYTHON}\n\n"
                "Set the SYSTEM2_PYTHON environment variable to the python "
                "executable that has 'openai' installed.",
            ))
        except Exception:
            import traceback
            self._q.put(("s2_err", traceback.format_exc()))


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyCopilotGUI:

    def __init__(self, args: Any) -> None:
        self._args         = args
        self._ui_q: "queue.Queue" = queue.Queue()
        self._worker: Optional[InferenceWorker] = None
        self._log_path: Optional[str] = None
        self._session_done = False
        self._paused       = False
        self._photo: Any   = None          # keep video frame ref
        self._s2_photos: List[Any] = []    # keep System 2 inline image refs
        self._canvas_w = 1
        self._canvas_h = 1

        # Session-wide set of task IDs ever flagged as a missed step
        # (populated from state["missed_ever"] on each window update)
        self._missed_ever: set = set()

        # Previous window's completed set — used to detect newly completed tasks
        # so they are always logged even when an error was active at the same window.
        self._prev_completed: set = set()

        # Pre-compute total tracked tasks (avoids per-frame TPG rebuild)
        _tpg = TaskPrecedenceGraph(MOCK_TPG_TASKS, MOCK_TPG_EDGES)
        self._n_tracked = sum(1 for tid in _tpg.tasks if _tpg.is_tracked(tid))

        self._recorder: Optional[ScreenRecorder] = None

        self._build_root()
        self._build_ui()
        self._start_inference()
        self._root.after(30, self._poll)

        # Start screen recorder after the window opens (gives Tk time to
        # settle and the window manager to place/decorate the window).
        if getattr(args, "recording", None):
            self._root.after(2000, self._start_recorder)

    # ── Window construction ───────────────────────────────────────────────────

    def _build_root(self) -> None:
        self._root = tk.Tk()
        self._root.title("Assembly Copilot")
        self._root.geometry("1640x960")
        self._root.configure(bg=BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Detect best available font — prefers Roboto, falls back gracefully
        import tkinter.font as tkfont
        global _FF, FONT_SM, FONT_MD, FONT_LG, FONT_TTL
        available = set(tkfont.families())
        _FF = next(
            (f for f in [
                "Roboto",
                "Inter", "Segoe UI",
                "DejaVu Sans", "Liberation Sans",
            ] if f in available),
            "TkDefaultFont",
        )
        FONT_SM  = (_FF, 10)
        FONT_MD  = (_FF, 11)
        FONT_LG  = (_FF, 12, "bold")
        FONT_TTL = (_FF, 11, "bold")
        # Tags are reconfigured after _build_right creates _resp_txt, so _FF
        # is guaranteed to be the resolved font name by then.

    def _reconfigure_resp_tags(self) -> None:
        """Apply all response-text tags using the currently resolved _FF font.
        Called from _build_right (after _build_root has set _FF).
        """
        t = self._resp_txt
        t.tag_config("user",        foreground=COL_INFO, font=(_FF, 11, "bold"))
        t.tag_config("sys2",        foreground=COL_OK,   font=(_FF, 11, "bold"))
        t.tag_config("rev_hd",      foreground=COL_WARN, font=(_FF, 11, "bold"))
        t.tag_config("h1",          foreground=FG_TITLE, font=(_FF, 13, "bold"))
        t.tag_config("h2",          foreground=COL_TASK, font=(_FF, 12, "bold"))
        t.tag_config("body",        foreground=FG,       font=(_FF, 11))
        t.tag_config("bullet",      foreground=FG,       font=(_FF, 11))
        t.tag_config("bullet_mark", foreground=COL_TASK, font=(_FF, 11, "bold"))
        t.tag_config("bold_inline", foreground=FG_TITLE, font=(_FF, 11, "bold"))
        t.tag_config("label_inline",foreground=COL_INFO, font=(_FF, 11, "bold"))
        t.tag_config("code_inline", foreground=COL_TASK, font=(_FFM, 10))
        t.tag_config("dim",         foreground=FG_DIM,   font=(_FF, 10))
        t.tag_config("err",         foreground=COL_ERR,  font=(_FF, 11))
        t.tag_config("sep",         foreground=COL_SEP,  font=(_FF, 10))

    def _build_ui(self) -> None:
        pane = tk.PanedWindow(self._root, orient=tk.HORIZONTAL,
                              bg=BG, bd=0, sashwidth=5, sashrelief="flat")
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left  = tk.Frame(pane, bg=BG_PANEL)
        right = tk.Frame(pane, bg=BG_PANEL)
        pane.add(left,  minsize=600, stretch="always")
        pane.add(right, minsize=460, stretch="always")
        pane.paneconfigure(left,  width=1060)
        pane.paneconfigure(right, width=540)

        self._build_left(left)
        self._build_right(right)

    # ── LEFT panel ────────────────────────────────────────────────────────────

    def _build_left(self, parent: tk.Frame) -> None:
        self._mk_header(parent, "SYSTEM 1  ·  TASK PROGRESS TRACKING")

        vsplit = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                bg=BG_PANEL, bd=0, sashwidth=5)
        vsplit.pack(fill=tk.BOTH, expand=True)

        vframe = tk.Frame(vsplit, bg="#000000")
        sframe = tk.Frame(vsplit, bg=BG_PANEL)
        vsplit.add(vframe, minsize=120, stretch="always")
        vsplit.add(sframe, minsize=380, stretch="never")
        vsplit.paneconfigure(vframe, height=280)

        # Video canvas — click toggles pause
        self._video_canvas = tk.Canvas(
            vframe, bg="#000000", bd=0, highlightthickness=0,
            cursor="hand2")
        self._video_canvas.pack(fill=tk.BOTH, expand=True)
        self._video_canvas.bind("<Configure>", self._on_canvas_resize)
        self._video_canvas.bind("<Button-1>", lambda _: self._on_pause_resume())

        # Status section
        self._build_status(sframe)

    def _build_status(self, parent: tk.Frame) -> None:
        outer = tk.Frame(parent, bg=BG_PANEL)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # ── Row 0: CURRENT TASK header ──
        tk.Label(outer, text="CURRENT TASK", bg=BG_PANEL, fg=FG_TITLE,
                 font=FONT_TTL).pack(anchor="w")
        self._task_lbl = tk.Label(
            outer, text="—", bg=BG_CARD, fg=COL_TASK,
            font=(_FF, 13, "bold"), anchor="w", padx=10, pady=8,
            relief="flat", bd=0)
        self._task_lbl.pack(fill=tk.X, pady=(2, 8))

        # ── Row 1: progress bar ──
        prog_hdr = tk.Frame(outer, bg=BG_PANEL)
        prog_hdr.pack(fill=tk.X, pady=(0, 2))
        tk.Label(prog_hdr, text="PROGRESS", bg=BG_PANEL, fg=FG_TITLE,
                 font=FONT_TTL).pack(side=tk.LEFT)
        self._pct_lbl = tk.Label(prog_hdr, text="0.0%  (0 / 0)",
                                  bg=BG_PANEL, fg=COL_INFO, font=FONT_TTL)
        self._pct_lbl.pack(side=tk.RIGHT)

        self._prog_bar = tk.Canvas(outer, bg=BG_CARD, height=20,
                                   bd=0, highlightthickness=1,
                                   highlightbackground=COL_SEP)
        self._prog_bar.pack(fill=tk.X, pady=(0, 6))

        # FPS label kept as a no-op attribute so existing update calls don't crash
        self._fps_lbl = tk.Label(outer, text="", bg=BG_PANEL, fg=FG_DIM,
                                 font=FONT_SM)
        # Not packed — intentionally hidden

        # ── Row 3: ERROR box — built here so it sits ABOVE the task columns ──
        # Starts hidden; _show_error / _clear_error manage pack visibility.
        # Outer frame (red border + fixed header), then a scrollable canvas
        # that holds all content (A / B / C) so nothing is ever clipped.
        self._err_frame = tk.Frame(outer, bg=ERR_BG,
                                   highlightbackground=ERR_BDR,
                                   highlightthickness=2)

        # ── Fixed header (always visible, not scrolled) ───────────────────────
        self._err_hdr = tk.Label(
            self._err_frame,
            text="[!]  TASK ORDER ERROR",
            bg=ERR_BG, fg=ERR_HEAD,
            font=(_FF, 15, "bold"),
            anchor="w", padx=14, pady=10,
        )
        self._err_hdr.pack(fill=tk.X)
        tk.Frame(self._err_frame, bg=ERR_BDR, height=1).pack(fill=tk.X)

        # ── Scrollable content area ───────────────────────────────────────────
        _ecvs_row = tk.Frame(self._err_frame, bg=ERR_BG)
        _ecvs_row.pack(fill=tk.BOTH, expand=True)

        _err_scroll = tk.Scrollbar(_ecvs_row, orient="vertical",
                                   bg=ERR_BG, troughcolor=ERR_BG, width=8)
        _err_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._err_canvas = tk.Canvas(_ecvs_row, bg=ERR_BG, bd=0,
                                     highlightthickness=0,
                                     yscrollcommand=_err_scroll.set)
        self._err_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _err_scroll.config(command=self._err_canvas.yview)

        # All sections live in this inner frame placed on the canvas
        _ec = tk.Frame(self._err_canvas, bg=ERR_BG)
        self._err_canvas_win = self._err_canvas.create_window(
            (0, 0), window=_ec, anchor="nw")

        def _ec_resize(e: tk.Event) -> None:
            self._err_canvas.configure(
                scrollregion=self._err_canvas.bbox("all"))
        def _ecvs_resize(e: tk.Event) -> None:
            self._err_canvas.itemconfig(self._err_canvas_win,
                                        width=self._err_canvas.winfo_width())
        _ec.bind("<Configure>", _ec_resize)
        self._err_canvas.bind("<Configure>", _ecvs_resize)

        # ── Section A: Missed Steps ────────────────────────────────────────────
        self._missed_section = tk.Frame(_ec, bg=ERR_BG)
        self._missed_section.pack(fill=tk.X, padx=14, pady=(8, 4))

        miss_hdr = tk.Frame(self._missed_section, bg=ERR_BG)
        miss_hdr.pack(fill=tk.X)
        tk.Label(miss_hdr, text="MISSED STEP(S)",
                 bg=ERR_BG, fg=COL_ERR, font=(_FF, 11, "bold")).pack(side=tk.LEFT)
        tk.Label(miss_hdr, text="  — required steps that were skipped",
                 bg=ERR_BG, fg=FG_DIM, font=(_FF, 10)).pack(side=tk.LEFT)

        self._missed_txt = tk.Text(
            self._missed_section, bg="#fff8f8", fg=COL_ERR,
            font=(_FF, 12, "bold"),
            height=2, bd=0, relief="flat",
            padx=6, pady=4, state="disabled", wrap=tk.WORD,
            highlightthickness=1, highlightbackground=ERR_BDR,
        )
        self._missed_txt.pack(fill=tk.X, pady=(4, 0))

        # ── Sections B + C wrapped in a frame that can be hidden for
        #    "missed_prerequisite" errors where no wrong action occurred.
        self._err_bc_frame = tk.Frame(_ec, bg=ERR_BG)
        self._err_bc_frame.pack(fill=tk.X)

        tk.Frame(self._err_bc_frame, bg=ERR_BDR, height=1).pack(
            fill=tk.X, pady=(8, 0))

        # ── Section B: Wrong Action + Do This Instead ─────────────────────────
        tk.Label(self._err_bc_frame, text="WRONG ACTION  /  DO THIS INSTEAD",
                 bg=ERR_BG, fg=COL_ERR, font=(_FF, 11, "bold"),
                 anchor="w", padx=14).pack(fill=tk.X, pady=(8, 2))

        err_top = tk.Frame(self._err_bc_frame, bg=ERR_BG)
        err_top.pack(fill=tk.X, padx=14, pady=(0, 6))
        tk.Label(err_top, text="Performed:", bg=ERR_BG, fg=FG_DIM,
                 font=(_FF, 11, "bold"), anchor="w", width=12
                 ).grid(row=0, column=0, sticky="w", pady=3)
        self._err_performed = tk.Label(
            err_top, text="", bg=ERR_BG, fg=COL_ERR,
            font=(_FF, 13, "bold"), anchor="w", wraplength=600)
        self._err_performed.grid(row=0, column=1, sticky="w")
        tk.Label(err_top, text="Do Instead:", bg=ERR_BG, fg=FG_DIM,
                 font=(_FF, 11, "bold"), anchor="w", width=12
                 ).grid(row=1, column=0, sticky="w", pady=3)
        self._err_expected = tk.Label(
            err_top, text="", bg=ERR_BG, fg=COL_WARN,
            font=(_FF, 13, "bold"), anchor="w", wraplength=600)
        self._err_expected.grid(row=1, column=1, sticky="w")
        err_top.columnconfigure(1, weight=1)

        tk.Frame(self._err_bc_frame, bg=ERR_BDR, height=1).pack(
            fill=tk.X, pady=(6, 0))

        # ── Section C: Element-level mismatch grid ────────────────────────────
        tk.Label(self._err_bc_frame, text="ELEMENT-WISE DETAILS",
                 bg=ERR_BG, fg=COL_ERR, font=(_FF, 11, "bold"),
                 anchor="w", padx=14).pack(fill=tk.X, pady=(8, 4))

        hdr_row = tk.Frame(self._err_bc_frame, bg=ERR_BG)
        hdr_row.pack(fill=tk.X, padx=20, pady=(0, 2))
        for col_txt, col_w in [("Element", 8), ("Performed", 18),
                                ("", 4), ("Expected", 18), ("", 4)]:
            tk.Label(hdr_row, text=col_txt, bg=ERR_BG, fg=FG_DIM,
                     font=(_FF, 10, "bold"), anchor="w",
                     width=col_w).pack(side=tk.LEFT)

        self._mismatch_grid = tk.Frame(self._err_bc_frame, bg=ERR_BG)
        self._mismatch_grid.pack(fill=tk.X, padx=20, pady=(0, 12))

        elem_names = ["Verb", "Object", "Target", "Tool"]
        elem_keys  = ["verb", "manip_obj", "target_obj", "tool"]
        self._mismatch_rows: Dict[str, dict] = {}
        for row_i, (ename, ekey) in enumerate(zip(elem_names, elem_keys)):
            bg_row = "#fff8f8" if row_i % 2 == 0 else ERR_BG
            tk.Label(self._mismatch_grid, text=f"{ename}",
                     bg=bg_row, fg=FG_DIM, font=(_FF, 12, "bold"),
                     anchor="w", width=8, padx=4, pady=4).grid(
                         row=row_i, column=0, sticky="nsew")
            pred_lbl = tk.Label(self._mismatch_grid, text="—",
                                bg=bg_row, font=(_FF, 12, "bold"),
                                anchor="w", width=18, padx=4, pady=4)
            pred_lbl.grid(row=row_i, column=1, sticky="nsew")
            tk.Label(self._mismatch_grid, text="-->",
                     bg=bg_row, fg=FG_DIM, font=(_FF, 11),
                     anchor="center", width=4, pady=4).grid(
                         row=row_i, column=2, sticky="nsew")
            exp_lbl = tk.Label(self._mismatch_grid, text="—",
                               bg=bg_row, font=(_FF, 12, "bold"),
                               anchor="w", width=18, padx=4, pady=4)
            exp_lbl.grid(row=row_i, column=3, sticky="nsew")
            icon_lbl = tk.Label(self._mismatch_grid, text="",
                                bg=bg_row, font=(_FF, 12, "bold"),
                                anchor="center", width=4, pady=4)
            icon_lbl.grid(row=row_i, column=4, sticky="nsew")
            self._mismatch_rows[ekey] = {
                "pred": pred_lbl, "exp": exp_lbl, "icon": icon_lbl,
                "bg": bg_row,
            }
        for c in range(5):
            self._mismatch_grid.columnconfigure(c, weight=1 if c in (1, 3) else 0)

        # ── Row 4: Next tasks | Event log ──
        # Saved as self._cols_frame so _show_error / _clear_error can swap it
        # with the error box (they share the same slot / expand space).
        cols = tk.Frame(outer, bg=BG_PANEL)
        cols.pack(fill=tk.BOTH, expand=True)
        self._cols_frame = cols
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        tk.Label(cols, text="NEXT TASKS", bg=BG_PANEL, fg=FG_TITLE,
                 font=FONT_TTL).grid(row=0, column=0, sticky="w", pady=(0, 2))
        next_outer = tk.Frame(cols, bg=BG_CARD)
        next_outer.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self._next_txt = tk.Text(
            next_outer, bg=BG_PANEL, fg=COL_TASK,
            font=(_FF, 12),
            height=5, state="disabled", bd=0, relief="flat",
            padx=10, pady=8, wrap=tk.WORD,
            spacing1=6, spacing2=2, spacing3=6,
        )
        next_scroll = tk.Scrollbar(next_outer, command=self._next_txt.yview,
                                   bg=BG_CARD, troughcolor=BG_PANEL, width=8)
        self._next_txt.configure(yscrollcommand=next_scroll.set)
        # "missed" tag: red + bold to match the error message colour
        self._next_txt.tag_config("missed", foreground=COL_ERR,
                                  font=(_FF, 12, "bold"))
        self._next_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        next_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(cols, text="EVENT LOG", bg=BG_PANEL, fg=FG_TITLE,
                 font=FONT_TTL).grid(row=0, column=1, sticky="w", pady=(0, 2))
        log_outer = tk.Frame(cols, bg=BG_CARD)
        log_outer.grid(row=1, column=1, sticky="nsew")
        self._log_txt = tk.Text(
            log_outer, bg=BG_PANEL, fg=FG,
            font=FONT_SM,
            height=5, state="disabled", bd=0, relief="flat",
            padx=10, pady=8, wrap=tk.WORD,
            spacing1=3, spacing2=1, spacing3=3,
        )
        log_scroll = tk.Scrollbar(log_outer, command=self._log_txt.yview,
                                  bg=BG_CARD, troughcolor=BG_PANEL, width=8)
        self._log_txt.configure(yscrollcommand=log_scroll.set)
        self._log_txt.tag_config("ok",   foreground=COL_OK)
        self._log_txt.tag_config("err",  foreground=COL_ERR)
        self._log_txt.tag_config("warn", foreground=COL_WARN)
        self._log_txt.tag_config("info", foreground=COL_INFO)
        self._log_txt.tag_config("dim",  foreground=FG_DIM)
        self._log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        cols.rowconfigure(1, weight=1)

    # ── RIGHT panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent: tk.Frame) -> None:
        self._mk_header(parent, "SYSTEM 2  ·  LLM ADVISORY  (DeepSeek)")

        # ── Response display ──
        resp_outer = tk.Frame(parent, bg=BG_CARD)
        resp_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 4))

        self._resp_txt = tk.Text(
            resp_outer, bg=BG_PANEL, fg=FG, font=(_FF, 11),
            wrap=tk.WORD, state="disabled", bd=0,
            padx=14, pady=10, spacing1=4, spacing2=2,
            insertbackground=FG,
        )
        resp_scroll = tk.Scrollbar(resp_outer, command=self._resp_txt.yview,
                                   bg=BG_CARD, troughcolor=BG_PANEL, width=10)
        self._resp_txt.configure(yscrollcommand=resp_scroll.set)
        self._resp_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        resp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Text tags — configured once here; _reconfigure_resp_tags() re-applies
        # them after _build_root() finishes so _FF is definitely correct.
        self._reconfigure_resp_tags()

        self._resp_append(
            "System 2 Advisory\n\n"
            "You can ask me anything during the assembly. After the session, "
            "I will review the whole session and provide you suggestions to "
            "improve your skills.\n",
            "dim",
        )

        # ── Separator ──
        tk.Frame(parent, bg=COL_SEP, height=1).pack(fill=tk.X, padx=8)

        # ── Input area ──
        input_area = tk.Frame(parent, bg=BG_PANEL)
        input_area.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(input_area, text="YOUR QUESTION",
                 bg=BG_PANEL, fg=FG_TITLE, font=FONT_TTL).pack(anchor="w")

        qbox = tk.Frame(input_area, bg=BG_CARD)
        qbox.pack(fill=tk.X, pady=(2, 6))
        self._query_entry = tk.Text(
            qbox, bg=BG_PANEL, fg=FG, font=(_FF, 11),
            height=3, bd=0, relief="flat",
            padx=10, pady=8, spacing1=3,
            insertbackground=FG, wrap=tk.WORD,
        )
        self._query_entry.pack(fill=tk.X)
        self._query_entry.bind("<Control-Return>", lambda _: self._on_ask())

        btn_row = tk.Frame(input_area, bg=BG_PANEL)
        btn_row.pack(fill=tk.X)

        self._ask_btn = tk.Button(
            btn_row, text="Ask System 2",
            bg=COL_INFO, fg="white",
            font=(_FF, 12, "bold"),
            bd=0, padx=16, pady=8, cursor="hand2",
            relief="flat", state="normal",
            command=self._on_ask,
            activebackground="#1e40af", activeforeground="white",
        )
        self._ask_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._review_btn = tk.Button(
            btn_row, text="Post-Session Review",
            bg=BG_PANEL, fg=COL_TASK,
            font=(_FF, 12, "bold"),
            bd=2, relief="solid", padx=16, pady=8, cursor="hand2",
            state="normal",
            command=self._on_review,
            activebackground=BG_CARD, activeforeground=COL_TASK,
        )
        self._review_btn.pack(side=tk.LEFT)


    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mk_header(parent: tk.Frame, title: str) -> None:
        bar = tk.Frame(parent, bg=HDR_BG, height=38)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text=f"  {title}", bg=HDR_BG, fg="#e2e8f0",
                 font=(_FF, 13, "bold")).pack(side=tk.LEFT, padx=8)

    @staticmethod
    def _mk_textbox(parent: tk.Frame, height: int = 6,
                    fg: str = FG) -> tk.Text:
        return tk.Text(parent, bg=BG_CARD, fg=fg, font=FONT_SM,
                       height=height, state="disabled", bd=1, relief="solid",
                       highlightbackground=COL_SEP,
                       padx=8, pady=6, wrap=tk.WORD,
                       spacing1=3, spacing2=1, spacing3=3)

    def _text_set(self, widget: tk.Text, content: str, tag: str = "") -> None:
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content, tag) if tag else widget.insert(tk.END, content)
        widget.config(state="disabled")

    def _text_append(self, widget: tk.Text, text: str, tag: str = "") -> None:
        widget.config(state="normal")
        widget.insert(tk.END, text, tag) if tag else widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state="disabled")

    def _resp_append(self, text: str, tag: str = "body") -> None:
        self._text_append(self._resp_txt, text, tag)

    def _log(self, tag: str, msg: str) -> None:
        pfx = {"ok": "[OK]  ", "err": "[X]   ", "warn": "[!]   ",
               "info": "-->   ", "dim": "      "}.get(tag, "      ")
        self._text_append(self._log_txt, f"{pfx}{msg}\n", tag)

    # ── Error box ─────────────────────────────────────────────────────────────

    def _show_error(self, performed: str, expected: str,
                    missed_steps: Optional[List] = None,
                    mismatch_labels: Optional[Dict] = None,
                    error_type: str = "wrong_action") -> None:
        # ── Header text reflects error type ──────────────────────────────────
        if error_type == "missed_prerequisite":
            self._err_hdr.config(text="[!]  MISSED PREREQUISITE STEP(S)")
        else:
            self._err_hdr.config(text="[!]  TASK ORDER ERROR")

        self._err_performed.config(text=performed)
        self._err_expected.config(text=expected)

        # ── Section A: Missed Steps ──────────────────────────────────────────
        self._missed_txt.config(state="normal")
        self._missed_txt.delete("1.0", tk.END)
        if missed_steps:
            for step in missed_steps:
                self._missed_txt.insert(tk.END, f"  [SKIP]  {step['label']}\n")
            # Resize height to fit (max 4 lines)
            self._missed_txt.config(height=min(len(missed_steps) + 1, 4),
                                    state="disabled")
            self._missed_section.pack(fill=tk.X, padx=14, pady=(8, 4))
        else:
            self._missed_txt.config(state="disabled")
            self._missed_section.pack_forget()

        # ── Section C: Element-level detail rows ─────────────────────────────
        for ekey, row in self._mismatch_rows.items():
            bg = row["bg"]
            info = (mismatch_labels or {}).get(ekey)
            if not info:
                row["pred"].config(text="—", fg=FG_DIM, bg=bg)
                row["exp"].config(text="—",  fg=FG_DIM, bg=bg)
                row["icon"].config(text="",  fg=FG_DIM, bg=bg)
            else:
                pred_txt = info.get("pred_label", "?")
                exp_txt  = info.get("expected_label", "?")
                match    = info.get("match", True)
                if match:
                    row["pred"].config(text=pred_txt, fg=COL_OK,   bg=bg)
                    row["exp"].config(text=exp_txt,   fg=COL_OK,   bg=bg)
                    row["icon"].config(text="OK",     fg=COL_OK,   bg=bg)
                else:
                    row["pred"].config(text=pred_txt, fg=COL_ERR,  bg=bg)
                    row["exp"].config(text=exp_txt,   fg=COL_WARN, bg=bg)
                    row["icon"].config(text="X",      fg=COL_ERR,  bg=bg)

        # Show / hide Sections B + C based on error type
        if error_type == "missed_prerequisite":
            self._err_bc_frame.pack_forget()
        else:
            if not self._err_bc_frame.winfo_ismapped():
                self._err_bc_frame.pack(fill=tk.X)

        # Hide the Next Tasks / Event Log columns and expand error into that space
        if self._cols_frame.winfo_ismapped():
            self._cols_frame.pack_forget()
        if not self._err_frame.winfo_ismapped():
            self._err_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

    def _clear_error(self) -> None:
        # Restore the Next Tasks / Event Log columns
        if self._err_frame.winfo_ismapped():
            self._err_frame.pack_forget()
        if not self._cols_frame.winfo_ismapped():
            self._cols_frame.pack(fill=tk.BOTH, expand=True)

    # ── Video display ─────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas_w = max(1, event.width)
        self._canvas_h = max(1, event.height)

    def _show_frame(self, bgr: np.ndarray) -> None:
        try:
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            fw, fh = pil.size
            scale  = min(self._canvas_w / fw, self._canvas_h / fh)
            nw = max(1, int(fw * scale))
            nh = max(1, int(fh * scale))
            pil    = pil.resize((nw, nh), Image.LANCZOS)
            photo  = ImageTk.PhotoImage(pil)
            x = (self._canvas_w - nw) // 2
            y = (self._canvas_h - nh) // 2
            self._video_canvas.create_image(x, y, anchor="nw", image=photo)
            self._photo = photo      # prevent GC
        except Exception:
            pass

    # ── Pause / Resume ────────────────────────────────────────────────────────

    def _on_pause_resume(self) -> None:
        if self._session_done or not self._worker:
            return
        if self._paused:
            # Resume
            self._paused = False
            self._worker.resume()
            self._log("dim", "Inference resumed.")
        else:
            # Pause
            self._paused = True
            self._worker.pause()
            self._log("dim", "Paused — ask System 2 a question, then click video to resume.")
            if self._log_path:
                self._ask_btn.config(state="normal")

    # ── System 1 state update ─────────────────────────────────────────────────

    def _update_state(self, data: dict) -> None:
        state   = data["state"]
        win_idx = data["win"]
        fps_inf = data["fps_inf"]
        frame   = data["frame"]

        self._show_frame(frame)

        confirmed = state.get("confirmed_holistic", "—")
        completed = state.get("completed", set())
        eligible  = state.get("eligible", [])
        pct       = state.get("progress_pct", 0.0)
        changed   = state.get("changed", False)
        error     = state.get("error")
        ftime     = state.get("frame_time", 0.0)
        # Accumulate missed_ever from backend
        self._missed_ever.update(state.get("missed_ever", set()))

        # Current task label
        label = _TPG_LABELS.get(confirmed, confirmed)
        self._task_lbl.config(text=label)

        # Progress
        self._pct_lbl.config(
            text=f"{pct:.1f}%  ({len(completed)} / {self._n_tracked})")
        self._draw_progress(pct)

        # FPS display
        self._fps_lbl.config(
            text=f"t = {ftime:.1f} s  |  {fps_inf:.1f} fps")

        # Next tasks — missed tasks are labelled differently
        self._next_txt.configure(state="normal")
        self._next_txt.delete("1.0", tk.END)
        if not eligible:
            self._next_txt.insert(tk.END, "  (none)")
        else:
            for t in eligible:
                lbl = _TPG_LABELS.get(t, t)
                if t in self._missed_ever:
                    self._next_txt.insert(tk.END, f"  [MISSING]  {lbl}\n", "missed")
                else:
                    self._next_txt.insert(tk.END, f"  •  {lbl}\n")
        self._next_txt.configure(state="disabled")

        # ── Always log newly completed tasks ─────────────────────────────────
        # Done BEFORE error/normal branching so a task that passes the dwell gate
        # while an error is active is never silently dropped from the Event Log.
        newly_done = completed - self._prev_completed
        self._prev_completed = set(completed)
        for tid in sorted(newly_done):  # stable order
            lbl = _TPG_LABELS.get(tid, tid)
            if tid in self._missed_ever:
                self._log("warn", f"t={ftime:.1f}s  [after missing prerequisite]  {lbl}")
            else:
                self._log("info", f"t={ftime:.1f}s  {lbl}")

        # ── Error box ─────────────────────────────────────────────────────────
        if error:
            performed    = error.get("confirmed_label", confirmed)
            closest      = error.get("closest_label", "")
            missed_steps = error.get("missed_steps", [])
            error_type   = error.get("error_type", "wrong_action")
            self._show_error(
                performed=performed,
                expected=closest or "—",
                missed_steps=missed_steps,
                mismatch_labels=error.get("mismatch_labels"),
                error_type=error_type,
            )
            if changed:
                if error_type == "missed_prerequisite":
                    for ms in missed_steps:
                        self._log("err",
                                  f"t={ftime:.1f}s  Missed prerequisite: "
                                  f"'{ms['label']}' — complete this before continuing")
                else:
                    self._log("err",
                              f"t={ftime:.1f}s  Wrong action: '{performed}'"
                              f" — do instead: '{closest}'")
                    for ms in missed_steps:
                        self._log("warn",
                                  f"t={ftime:.1f}s  Missed step: '{ms['label']}'")
        elif changed:
            # Correct task — clear any prior error box (completion already logged above)
            self._clear_error()

    def _draw_progress(self, pct: float) -> None:
        w = self._prog_bar.winfo_width()
        h = self._prog_bar.winfo_height()
        if w < 2:
            return
        filled = int(w * pct / 100)
        self._prog_bar.delete("all")
        self._prog_bar.create_rectangle(0, 0, w, h, fill=BG_CARD, outline="")
        self._prog_bar.create_rectangle(
            0, 0, filled, h, fill=COL_INFO, outline="")
        self._prog_bar.create_text(
            w // 2, h // 2, text=f"{pct:.1f}%", fill=FG, font=FONT_SM)

    # ── System 2 interaction ──────────────────────────────────────────────────

    def _on_ask(self) -> None:
        query = self._query_entry.get("1.0", tk.END).strip()
        if not query or not self._log_path:
            return
        self._query_entry.delete("1.0", tk.END)
        self._set_s2_busy(True)
        self._resp_append("\n────────────────────────────────────────\n", "sep")
        self._resp_append(f"You: {query}\n", "user")
        self._resp_append("System 2 is thinking…\n", "dim")
        System2Worker(self._log_path, "guidance", query, self._ui_q).start()

    def _on_review(self) -> None:
        if not self._log_path:
            return
        self._set_s2_busy(True)
        self._resp_append("\n────────────────────────────────────────\n", "sep")
        self._resp_append("POST-SESSION REVIEW\nGenerating analysis...\n",
                          "rev_hd")
        System2Worker(self._log_path, "review", "", self._ui_q).start()

    def _set_s2_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        # Ask button: available when paused (mid-session) or session done
        if busy:
            self._ask_btn.config(state="disabled")
        elif self._log_path and (self._paused or self._session_done):
            self._ask_btn.config(state="normal")

        # Review button: only after session is done
        if self._session_done:
            self._review_btn.config(state=state)

    # ── Markdown rendering ────────────────────────────────────────────────────

    # Compiled patterns used by _insert_md_spans
    _RE_BOLD   = __import__("re").compile(r"\*\*(.+?)\*\*")
    _RE_ITALIC = __import__("re").compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
    _RE_CODE   = __import__("re").compile(r"`([^`]+)`")

    # Map common emoji codepoints to short ASCII labels (or empty string to drop)
    _EMOJI_MAP: Dict[str, str] = {
        "\U0001f4cb": "",   # 📋 clipboard
        "\U0001f7e1": "",   # 🟡 yellow circle
        "\U0001f7e2": "",   # 🟢 green circle
        "\U0001f534": "",   # 🔴 red circle
        "\U0001f7e0": "",   # 🟠 orange circle
        "\U0001f535": "",   # 🔵 blue circle
        "\U00002705": "",   # ✅ check mark button
        "\U0000274c": "",   # ❌ cross mark
        "\U000026a0": "",   # ⚠  warning (text-range, but still ugly in some fonts)
        "\U0001f6a7": "",   # 🚧
        "\U0001f527": "",   # 🔧
        "\U0001f4a1": "",   # 💡
        "\U0001f3af": "",   # 🎯
        "\U00002139": "",   # ℹ info
        "\U0001f4dd": "",   # 📝
    }

    @classmethod
    def _strip_emoji(cls, text: str) -> str:
        """Replace known emoji with empty string (or short label) so they
        don't render as squares in Tkinter."""
        import re
        for ch, replacement in cls._EMOJI_MAP.items():
            text = text.replace(ch, replacement)
        # Catch any remaining emoji/symbol in the range U+1F000-U+1FFFF and
        # common misc-symbols blocks that Roboto/DejaVu can't render.
        text = re.sub(
            r"[\U0001F000-\U0001FFFF"    # emoticons / misc symbols
            r"\U00002600-\U000027BF"     # misc symbols & dingbats (keep basic ones below)
            r"\U0001F900-\U0001FA9F]",   # supplemental symbols
            "", text,
        )
        return text

    def _insert_md_spans(self, text: str, default_tag: str) -> None:
        """
        Insert text into the response box, rendering inline markdown:
          **bold**   → bold_inline tag
          *italic*   → rendered without asterisks (same style as body)
          `code`     → code_inline tag (monospace)
          Label: …   → label_inline + rest
        Raw markers are never shown in the output.
        """
        import re

        # Remove emoji that Tkinter cannot display
        text = self._strip_emoji(text)

        # Check for "Label: rest" at the start (common in LLM structured output)
        label_m = re.match(r"^(\s*)([\w /\-]+):\s+", text)
        if label_m and "**" not in text:
            prefix = label_m.group(1)
            lbl    = label_m.group(2)
            rest   = text[label_m.end():]
            if prefix:
                self._resp_append(prefix, default_tag)
            self._resp_append(lbl + ": ", "label_inline")
            self._insert_md_spans(rest, default_tag)
            return

        # Tokenise the line into bold / italic / code / plain segments.
        # We build a combined pattern and walk the matches in order.
        combined = re.compile(
            r"\*\*(.+?)\*\*"             # group 1: **bold**
            r"|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"  # group 2: *italic*
            r"|`([^`]+)`"                # group 3: `code`
        )
        pos = 0
        for m in combined.finditer(text):
            # Plain text before this match
            if m.start() > pos:
                self._resp_append(text[pos:m.start()], default_tag)
            if m.group(1) is not None:
                self._resp_append(m.group(1), "bold_inline")
            elif m.group(2) is not None:
                # Italic — render without asterisks, same weight as body
                self._resp_append(m.group(2), default_tag)
            elif m.group(3) is not None:
                self._resp_append(m.group(3), "code_inline")
            pos = m.end()
        # Remaining plain text
        if pos < len(text):
            self._resp_append(text[pos:], default_tag)

    def _display_s2_response(self, mode: str, _query: str, resp: str,
                              media_task_ids: "list[str] | None" = None) -> None:
        import re
        hd_tag = "sys2" if mode == "guidance" else "rev_hd"
        label  = "System 2:" if mode == "guidance" else "Post-Session Review:"
        self._resp_append(f"\n{label}\n", hd_tag)

        # Strip emoji from the whole response before line-by-line parsing
        resp  = self._strip_emoji(resp)
        lines = resp.split("\n")
        i = 0
        while i < len(lines):
            raw  = lines[i]
            line = raw.rstrip()
            s    = line.lstrip()
            i   += 1

            # ── Blank line ──
            if not s:
                self._resp_append("\n")
                continue

            # ── Horizontal rule  ---  or  ===  ──
            if re.fullmatch(r"[-=]{3,}", s):
                self._resp_append("─" * 52 + "\n", "sep")
                continue

            # ── Headings  # / ##  ──
            if s.startswith("### "):
                self._resp_append("\n" + s[4:] + "\n", "h2")
                continue
            if s.startswith("## "):
                self._resp_append("\n" + s[3:] + "\n", "h2")
                continue
            if s.startswith("# "):
                self._resp_append("\n" + s[2:] + "\n", "h1")
                continue

            # ── ALL-CAPS section headers (e.g. PROCEDURE, SKILL GAPS) ──
            if (re.fullmatch(r"[A-Z][A-Z0-9 /\-]{2,}", s)
                    and not s[0].isdigit()):
                self._resp_append("\n" + s + "\n", "h1")
                continue

            # ── Bullet list  - / * / • ──
            bullet_m = re.match(r"^[\-\*•]\s+(.+)", s)
            if bullet_m:
                self._resp_append("  •  ", "bullet_mark")
                self._insert_md_spans(bullet_m.group(1), "bullet")
                self._resp_append("\n")
                continue

            # ── Numbered list  1. / 1) ──
            num_m = re.match(r"^(\d+[.)]\s+)(.+)", s)
            if num_m:
                self._resp_append("  " + num_m.group(1), "bullet_mark")
                self._insert_md_spans(num_m.group(2), "body")
                self._resp_append("\n")
                continue

            # ── Block-quote  > ──
            if s.startswith("> "):
                self._resp_append("│  ", "sep")
                self._insert_md_spans(s[2:], "dim")
                self._resp_append("\n")
                continue

            # ── Plain paragraph line ──
            self._insert_md_spans(line, "body")
            self._resp_append("\n")

        self._embed_media(media_task_ids or [])
        self._resp_txt.see(tk.END)

    def _embed_media(self, task_ids: "list[str]") -> None:
        """Embed reference images for the given task IDs (explicit list from the agent)."""
        if not task_ids:
            return
        db_path = _ROOT / "Assembly_copilot" / "data" / "multimedia_db.json"
        if not db_path.exists():
            return
        with open(db_path) as f:
            db = json.load(f)

        for asset in db.get("assets", []):
            if asset.get("task_id") not in task_ids:
                continue
            uri  = asset.get("uri", "")
            cap  = asset.get("caption", uri)
            atype = asset.get("type", "")

            if atype == "image":
                img_path = (_ROOT / "Assembly_copilot" / uri).resolve()
                if img_path.exists():
                    try:
                        pil = Image.open(str(img_path))
                        pil.thumbnail((420, 280), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(pil)
                        self._s2_photos.append(photo)
                        self._resp_txt.config(state="normal")
                        self._resp_append(f"\n📷  {cap}\n", "dim")
                        self._resp_txt.image_create(tk.END, image=photo)
                        self._resp_append("\n\n")
                    except Exception:
                        pass
            elif atype == "video":
                vid_path = (_ROOT / "Assembly_copilot" / uri).resolve()
                if vid_path.exists():
                    self._resp_append(
                        f"\n🎬  {cap}\n"
                        f"   Path: {vid_path}\n", "dim")

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._ui_q.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self._root.after(30, self._poll)

    def _handle(self, kind: str, payload: Any) -> None:  # noqa: C901
        if kind == "status":
            self._log("dim", payload)

        elif kind == "model_ok":
            self._log("dim", "Model loaded. Loading video…")

        elif kind == "video_ok":
            total, fps = payload
            self._log("info", f"Video: {total} frames @ {fps:.1f} fps")

        elif kind == "log_path":
            self._log_path = payload

        elif kind == "window":
            self._update_state(payload)

        elif kind == "done":
            self._log_path     = payload
            self._session_done = True
            self._paused       = False
            self._log("ok", "Session complete. System 2 ready.")
            # Report missed tasks in the event log
            try:
                import json as _json
                with open(payload, encoding="utf-8") as _f:
                    _slog = _json.load(_f)
                missed = _slog.get("missed_tasks", [])
                if missed:
                    self._log("warn", f"MISSED OPERATIONS ({len(missed)}):")
                    for t in missed:
                        self._log("warn", f"  -- {t['label']}")
            except Exception:
                pass
            self._ask_btn.config(state="normal")
            self._review_btn.config(state="normal")
            if getattr(self._args, "post_session_review", False):
                self._on_review()

        elif kind == "error":
            self._log("err", f"Inference error:\n{payload}")

        elif kind == "s2_ok":
            mode, query, resp, *_media = payload
            media_task_ids = _media[0] if _media else []
            self._display_s2_response(mode, query, resp, media_task_ids)
            self._set_s2_busy(False)

        elif kind == "s2_err":
            self._resp_append(f"\n[System 2 error]\n{payload}\n", "err")
            self._set_s2_busy(False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _start_inference(self) -> None:
        self._worker = InferenceWorker(self._args, self._ui_q)
        self._worker.start()

    def _start_recorder(self) -> None:
        """Determine the output path and launch the ScreenRecorder."""
        raw = self._args.recording  # "" (auto) or an explicit path
        if not raw:
            # Auto-name: same stem as the input video
            stem = Path(self._args.video).stem
            out_path = str(Path(self._args.video).parent / f"{stem}_recording.mp4")
        else:
            out_path = raw

        self._recorder = ScreenRecorder(out_path, self._root, fps=10)
        self._recorder.start()

    def _stop_recorder(self) -> None:
        if self._recorder and self._recorder.active:
            self._root.update()          # flush any pending redraws first
            self._recorder.stop()

    def _on_close(self) -> None:
        self._stop_recorder()
        if self._worker:
            self._worker.stop()
        self._root.destroy()

    def run(self) -> None:
        previous_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(_signum, _frame):
            self._root.after(0, self._on_close)

        signal.signal(signal.SIGINT, _handle_sigint)
        try:
            self._root.mainloop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self._stop_recorder()


# ── CLI argument parser ───────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Assembly Copilot GUI")

    p.add_argument("--video",            required=True)
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--composition_file", default=None)
    p.add_argument("--label_dir",        default=None)
    p.add_argument("--window",           type=int,   default=16)
    p.add_argument("--stride",           type=int,   default=4)
    p.add_argument("--vote_window",      type=int,   default=5)
    p.add_argument("--lh_num_verbs",       type=int, default=5)
    p.add_argument("--lh_num_manip_objs",  type=int, default=12)
    p.add_argument("--lh_num_target_objs", type=int, default=6)
    p.add_argument("--lh_num_tools",       type=int, default=4)
    p.add_argument("--rh_num_verbs",       type=int, default=5)
    p.add_argument("--rh_num_manip_objs",  type=int, default=12)
    p.add_argument("--rh_num_target_objs", type=int, default=6)
    p.add_argument("--rh_num_tools",       type=int, default=4)
    p.add_argument("--decoder_layers",   type=int,   default=3)
    p.add_argument("--decoder_heads",    type=int,   default=8)
    p.add_argument("--decoder_dim",      type=int,   default=2048)
    p.add_argument("--decoder_dropout",  type=float, default=0.1)
    p.add_argument("--head_dropout",     type=float, default=0.1)
    p.add_argument("--adapter_dim",      type=int,   default=128)
    p.add_argument("--use_hand_adapters", action="store_true", default=True)
    p.add_argument("--device",           default="auto")
    p.add_argument("--session_log",      default=None)
    p.add_argument("--post_session_review", action="store_true", default=False)
    p.add_argument(
        "--recording", nargs="?", const="", default=None,
        metavar="OUTPUT.mp4",
        help=(
            "Record the GUI session to a video file using ffmpeg x11grab (Linux).\n"
            "Pass without a value to auto-name from the input video stem.\n"
            "Example: --recording demo.mp4"
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    AssemblyCopilotGUI(_parse_args()).run()
