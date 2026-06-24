# CoDuAR and Assembly Copilot

Official repository for the paper:

**A Compositional Dual-Hand Action Recognition Method for an LLM-Driven Assembly Assistance**

## Overview

The project has two connected components:

- **CoDuAR**: compositional dual-hand action recognition, recognizing dual-hand actions simultaneously. Each hand is represented through action elements, including verb, manipulated object, target object, and tool.
- **Assembly Copilot**: a demonstration system showing how CoDuAR's compositional action representation can bridge real-time perception and language-based reasoning. A deterministic System 1, driven by CoDuAR, maps predictions to a symbolic task state for task tracking, next-step recommendation, and real-time error detection. An LLM-powered System 2 reasons over the symbolic state to answer operator queries and generate post-session performance reviews.

The central research contribution of this paper is **CoDuAR**. Assembly Copilot is included as an application-level demonstration of how compositional dual-hand action recognition can support language-assisted assembly workflows.

### CoDuAR Architecture



*CoDuAR decomposes dual-hand assembly actions into hand-specific compositional elements and refines their combinations for action recognition.*

## System Demonstration

Assembly Copilot demonstrates how CoDuAR predictions can be converted into symbolic task states and used by downstream assistance modules.

### Assembly Copilot Architecture



*Assembly Copilot uses CoDuAR predictions as a symbolic bridge between real-time task monitoring and LLM-based advisory reasoning.*

## Repository Status


| Area             | Status       | Notes                                                                                   |
| ---------------- | ------------ | --------------------------------------------------------------------------------------- |
| Model code       | Placeholder  | Migrate cleaned files from `compositional_dual_hand/models/` and training engines.      |
| Dataset loaders  | Placeholder  | Migrate cleaned files from `compositional_dual_hand/dataset/`.                          |
| Assembly Copilot | Placeholder  | Migrate cleaned files from `compositional_dual_hand/Assembly_copilot/` and GUI scripts. |
| Demo video       | Included     | See [assets/demo/demo480_24fps.mp4](assets/demo/demo480_24fps.mp4).                     |
| Checkpoints      | Not released | Users should train/evaluate with their own local checkpoints.                           |
| Results          | Placeholder  | Fill metrics after final manuscript tables are frozen.                                  |


## Planned Layout

```text
.
├── assets/
│   ├── demo/                 # Demo video or link placeholder
│   └── figures/              # Paper/repository figures
├── configs/                  # Training/evaluation/demo config templates
├── data/
│   ├── annotations/          # Annotation format examples
│   └── label_maps/           # Label map templates
├── docs/                     # Dataset, demo, migration, and reproduction docs
├── scripts/                  # Thin command wrappers
├── src/
│   ├── coduar/               # CoDuAR package placeholder
│   └── assembly_copilot/     # Assembly Copilot package placeholder
└── tests/                    # Smoke tests for cleaned release code
```

## Quick Start

Create an environment:

```bash
conda create -n coduar python=3.10 -y
conda activate coduar
pip install -r requirements.txt
```

Prepare local paths by copying the template:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Then edit `configs/paths.local.yaml` with your dataset, local checkpoint, and demo paths.

Train a model:

```bash
bash scripts/train_coduar.sh configs/train_coduar.example.yaml
```

Evaluate a local checkpoint:

```bash
bash scripts/evaluate_coduar.sh configs/evaluate_coduar.example.yaml
```

Run the Assembly Copilot demo:

```bash
bash scripts/run_assembly_copilot_demo.sh configs/demo_assembly_copilot.example.yaml
```

These scripts are intentionally thin placeholders. They document the expected public commands and should be connected to the cleaned implementation later.

## Data

See [docs/dataset.md](docs/dataset.md) for the expected dual-hand compositional annotation format.

Data and videos should not be committed directly unless they are small examples with redistribution permission. Use `data/annotations/` and `data/label_maps/` for templates, and document private/local dataset roots in `configs/paths.local.yaml`.

## Demo

The repository includes a compressed demonstration video.

Your browser does not support embedded video. You can download or view the demo at assets/demo/demo480_24fps.mp4.

### Assembly Copilot Interface



*The interface combines real-time task monitoring, progress tracking, next-task recommendation, error messages, and LLM-powered query/review support.*

See [docs/demo.md](docs/demo.md) for demo notes.

## Migration Map

The messy research folder contains useful code that can be cleaned into this structure:


| Current source                                         | Public destination                                    |
| ------------------------------------------------------ | ----------------------------------------------------- |
| `models/`                                              | `src/coduar/models/`                                  |
| `dataset/`                                             | `src/coduar/data/`                                    |
| `engine_for_*.py`                                      | `src/coduar/training/`                                |
| `run_compositional_*.py`                               | `scripts/` plus `src/coduar/cli.py`                   |
| `evaluate_*.py`, `refinement.py`                       | `src/coduar/evaluation/` and `src/coduar/refinement/` |
| `Assembly_copilot/`                                    | `src/assembly_copilot/`                               |
| `assembly_copilot_gui.py`, `realtime_inference_gui.py` | `src/assembly_copilot/gui/` or `scripts/`             |
| `havid_labels/`, `case_study_labels/`                  | `data/label_maps/`                                    |


More detail is in [docs/code_migration_plan.md](docs/code_migration_plan.md).
The current research-code inventory is summarized in [docs/source_inventory.md](docs/source_inventory.md).

## Citation

If you use this repository, please cite the paper. The citation metadata in [CITATION.cff](CITATION.cff) contains placeholders for authors, venue, DOI, and publication date.

## License

This repository currently uses the MIT License. Confirm that all migrated third-party code and pretrained model assets are compatible before public release.