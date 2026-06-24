# Code Migration Plan

Current research code:

```text
/Users/hz4426/projects/compositional_dual_hand
```

The folder is useful but not yet release-clean. Migrate it in small, reviewable steps.

## Step 1: Core Package

Move model definitions into:

```text
src/coduar/models/
```

Candidate source files:

- `models/modeling_finetune_compositional_dual.py`
- `models/modeling_finetune_compositional_dual_with_action_loss.py`
- `models/modeling_finetune_compositional_dual_adaptive.py`
- `models/modeling_finetune_compositional_dual_transformer*.py`
- `models/uniformerv2_standalone.py`

## Step 2: Data Pipeline

Move dataset code into:

```text
src/coduar/data/
```

Candidate source files:

- `dataset/build_compositional.py`
- `dataset/build_dual_hand_whole_action.py`
- `dataset/datasets.py`
- `dataset/transforms.py`
- `dataset/video_transforms.py`
- `dataset/volume_transforms.py`
- `dataset/loader.py`

## Step 3: Training and Evaluation

Create cleaned modules:

```text
src/coduar/training/
src/coduar/evaluation/
src/coduar/refinement/
```

Candidate source files:

- `engine_for_compositional_*.py`
- `run_compositional_*.py`
- `evaluate_*.py`
- `refinement.py`
- `advanced_temporal_refinement.py`
- `smart_hand_coordination.py`

## Step 4: Assembly Copilot

Move the LLM-driven assistant into:

```text
src/assembly_copilot/
```

Candidate source files:

- `Assembly_copilot/agent.py`
- `Assembly_copilot/bridge.py`
- `Assembly_copilot/tools.py`
- `Assembly_copilot/run_system2.py`
- `assembly_copilot_gui.py`
- `realtime_inference_gui.py`

## Step 5: Public Scripts

Keep shell scripts thin and stable:

```text
scripts/train_coduar.sh
scripts/evaluate_coduar.sh
scripts/run_assembly_copilot_demo.sh
```

The scripts should call package entry points instead of containing hard-coded research-machine paths.

## Cleanup Rules

- Remove absolute private paths.
- Remove duplicate, backup, and "old" script variants.
- Move large outputs to releases or external storage.
- Replace ad hoc constants with YAML config fields.
- Add smoke tests for import, config parsing, and one tiny inference path.
