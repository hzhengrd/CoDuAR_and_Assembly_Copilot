# Assembly Copilot

This folder contains the implementation of **Assembly Copilot**, the LLM-driven assembly assistance demonstration built on top of CoDuAR predictions. The system uses CoDuAR as a perception module to convert video observations into compositional action elements, maps those elements to symbolic assembly tasks, tracks task progress against a task precedence graph, and invokes an LLM-based advisory agent for operator guidance and post-session review.

The code in this folder is intended for reproducing the demonstration workflow described in the paper. The CoDuAR model should be trained and evaluated first; this folder then uses a trained CoDuAR checkpoint, a demonstration video, and task-specific symbolic assets to run the interactive assistance system.

## Folder Layout

```text
assembly_copilot/
|-- assets/
|   `-- images/                    # Reference images rendered by the advisory interface
|-- data/
|   |-- tpg.json                    # Task precedence graph for the assembly procedure
|   |-- knowledge_base.json         # Step-level procedural knowledge base
|   |-- multimedia_db.json          # Mapping from task IDs to reference media assets
|   `-- reference_performance.json  # Reference timing statistics for session review
|-- script/
|   `-- launch.sh                   # Local launch wrapper to adapt for a target video/checkpoint
|-- assembly_copilot_gui.py         # Unified GUI: video, System 1, and System 2 interface
|-- realtime_system1.py             # Sliding-window inference and task-progress monitoring
|-- run_system2.py                  # Subprocess entry point for LLM guidance/review
|-- agent.py                        # System 2 LLM agent and prompts
|-- tools.py                        # Tool functions exposed to the LLM agent
`-- bridge.py                       # Adapter from System 1 session logs to System 2 state
```

## System Overview

Assembly Copilot has two interacting components:

- **System 1: real-time task monitoring.** `realtime_system1.py` loads a trained CoDuAR checkpoint, performs sliding-window inference over an input video, maps compositional predictions to task IDs, tracks completed tasks on the task precedence graph, recommends eligible next tasks, and logs procedural anomalies.
- **System 2: LLM advisory reasoning.** `agent.py`, `tools.py`, and `run_system2.py` use the live System 1 session log to answer operator queries, retrieve task procedures and media, and generate post-session performance reviews.

The unified GUI in `assembly_copilot_gui.py` combines video playback, System 1 progress monitoring, error display, next-task recommendation, on-demand LLM guidance, reference media rendering, and post-session review.

## Environment

Install the shared repository dependencies from the repository root:

```bash
conda create -n coduar python=3.10 -y
conda activate coduar
pip install -r requirements.txt
```

The GUI uses `tkinter`, `Pillow`, `opencv-python`, PyTorch, timm, and the OpenAI-compatible client used by the System 2 agent. `tkinter` is usually included with Python, but on some Linux systems it must be installed separately through the system package manager.

System 2 currently uses the DeepSeek OpenAI-compatible API. Set the API key before launching guidance or review functions:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

If System 2 dependencies are installed in a separate environment, set `SYSTEM2_PYTHON` to the Python executable for that environment:

```bash
export SYSTEM2_PYTHON=/path/to/env/bin/python
```

## Required Inputs

To reproduce the Assembly Copilot demonstration, prepare the following inputs:

- A trained CoDuAR checkpoint, preferably trained on the custom assembly dataset described in `../coduar/README.md`.
- A demonstration video for the assembly procedure.
- Label maps for the four compositional elements: verbs, manipulated objects, target objects, and tools.
- A compositional mapping file, if additional task-ID mappings are needed beyond the built-in case-study mapping.
- The task assets in `data/` and `assets/images/`, including the task precedence graph, knowledge base, multimedia database, and reference performance statistics.

The default class-count settings in the GUI and launch script correspond to the custom assembly dataset:

```text
left/right verbs:              5
left/right manipulated objects: 12
left/right target objects:      6
left/right tools:               4
```

These counts must match the checkpoint used for inference.

## Launching the GUI

The maintained GUI entry point is `assembly_copilot_gui.py`. A typical launch command is:

```bash
cd assembly_copilot
python assembly_copilot_gui.py \
  --video /path/to/demo_video.mp4 \
  --checkpoint /path/to/coduar_checkpoint.pth \
  --label_dir /path/to/label_maps \
  --composition_file /path/to/composition_mapping.txt \
  --device auto \
  --session_log output/session_log.json
```

Use `--device cuda:0` for a specific CUDA GPU, `--device mps` on supported Apple Silicon environments, or `--device cpu` for CPU-only testing. The default `--device auto` selects CUDA when available, then MPS, then CPU.

The local wrapper `script/launch.sh` can be used as a template. Before running it, edit the video path, checkpoint path, GPU setting, and class counts:

```bash
cd assembly_copilot
bash script/launch.sh
```

## Running System 1 Only

For debugging perception, task mapping, and task-progress tracking without the GUI, run:

```bash
cd assembly_copilot
python realtime_system1.py \
  --video /path/to/demo_video.mp4 \
  --checkpoint /path/to/coduar_checkpoint.pth \
  --label_dir /path/to/label_maps \
  --composition_file /path/to/composition_mapping.txt \
  --session_log output/session_log.json \
  --device auto
```

System 1 writes a live JSON session log containing current task state, completed tasks, eligible next tasks, detected errors, side activities, and missed tasks after finalization.

## Running System 2 Only

System 2 can be tested from an existing session log. For on-demand guidance:

```bash
cd assembly_copilot
python run_system2.py \
  --log_path output/session_log.json \
  --mode guidance \
  --query "What should I do next?"
```

For a post-session review:

```bash
python run_system2.py \
  --log_path output/session_log.json \
  --mode review
```

`bridge.py` translates the live System 1 log into the `SESSION_STATE` structure used by the System 2 tools before each agent call.

## Task Assets

The demonstration is configured around a penlight gear-box assembly case study:

- `data/tpg.json` defines the task precedence graph.
- `data/knowledge_base.json` stores procedures, critical parameters, common errors, and estimated durations.
- `data/multimedia_db.json` maps task IDs to reference images and media assets.
- `data/reference_performance.json` stores timing references used for post-session performance analysis.
- `assets/images/` contains task-level reference images rendered in the GUI when System 2 retrieves media.

To adapt Assembly Copilot to another assembly process, replace these assets with a new task graph, task knowledge base, reference performance statistics, multimedia records, and label-to-task mappings consistent with the CoDuAR label space.

## Outputs

Typical outputs include:

- A live session log JSON file, either set by `--session_log` or generated next to the input video.
- GUI-displayed progress state, next-task recommendations, and procedural error messages.
- System 2 guidance responses for operator questions.
- System 2 post-session review text summarizing progress, missed operations, procedural errors, timing deviations, idle time, and recommendations.
- Optional GUI recording when `--recording` is used.

## Reproducibility Notes

For paper reproduction, record the CoDuAR checkpoint, input video, class-count settings, label maps, composition mapping file, task precedence graph, LLM backend, API model name, and System 1 parameters such as window length, stride, vote window, and dwell threshold. The demonstration depends on both recognition outputs and the symbolic task assets, so these files should be versioned together for a reproducible run.

Before public release, verify that the launch wrapper points to the maintained GUI entry point and that the model definitions required by `realtime_system1.py` are available on the Python path. Also verify any local path assumptions in `assembly_copilot_gui.py` if the folder name or package layout changes.
