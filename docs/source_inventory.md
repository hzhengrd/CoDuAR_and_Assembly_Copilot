# Source Inventory

This inventory was drafted from:

```text
/Users/hz4426/projects/compositional_dual_hand
```

It is meant to guide cleanup, not to imply that every file should be released.

## Main Areas Found

| Area | Examples | Suggested action |
| --- | --- | --- |
| Models | `models/modeling_finetune_compositional_dual*.py`, `models/uniformerv2_standalone.py` | Keep the final architectures only; remove old variants or move them to an archive branch. |
| Dataset pipeline | `dataset/build_compositional.py`, `dataset/datasets.py`, transforms/loaders | Clean imports and path handling, then migrate to `src/coduar/data/`. |
| Training | `run_compositional_*.py`, `engine_for_compositional_*.py`, many `scripts/train_*.sh` | Collapse into one or two public entry points plus config files. |
| Evaluation | `evaluate_*.py`, `benchmark_*.py`, plotting scripts | Keep reproducible evaluation scripts and move figure-only utilities into `scripts/analysis/` if needed. |
| Temporal refinement | `refinement.py`, `advanced_temporal_refinement.py`, `smart_hand_coordination.py` | Document the final method used in the paper and remove exploratory versions. |
| Assembly Copilot | `Assembly_copilot/agent.py`, `bridge.py`, `tools.py`, `run_system2.py` | Migrate to `src/assembly_copilot/` with provider-agnostic config. |
| GUI/demo | `assembly_copilot_gui.py`, `realtime_inference_gui.py`, `prediction_viewer_gui.py` | Release only the demo path that supports the paper. |
| Labels | `havid_labels/`, `case_study_labels/` | Move public label maps to `data/label_maps/`; document private labels separately. |
| Outputs | `output/`, `figures/`, logs, generated CSVs | Do not commit raw outputs unless they are curated examples. |
| Notes | many `*_GUIDE.md`, `*_SUMMARY.md`, `*_UPDATE.md` | Fold useful content into `docs/`; omit implementation diary files. |

## Files Likely Worth Migrating First

- `models/modeling_finetune_compositional_dual_with_action_loss.py`
- `models/modeling_finetune_compositional_dual_transformer_v2.py`
- `dataset/build_compositional.py`
- `dataset/datasets.py`
- `engine_for_compositional_with_action_loss.py`
- `run_compositional_three_view_with_action_loss.py`
- `evaluate_dual_transformer_with_refinement.py`
- `refinement.py`
- `Assembly_copilot/agent.py`
- `Assembly_copilot/bridge.py`
- `Assembly_copilot/tools.py`

## Files to Avoid Releasing Directly

- `__pycache__/`
- `.DS_Store`
- `test_output.log`
- duplicate scripts containing `old`, `backup`, `wrong`, or `copy`
- machine-specific shell scripts with absolute private paths
- large checkpoints under `scripts/` or `output/`
- raw GUI recordings unless cleared for public release
