# CoDuAR

This folder contains the implementation of **CoDuAR**, the compositional dual-hand action recognition model used in the paper *A Compositional Dual-Hand Action Recognition Method for an LLM-Driven Assembly Assistance*. CoDuAR predicts action elements for the left and right hands simultaneously, including verb, manipulated object, target object, and tool. 

The code in this folder is intended for reproducing the model training and evaluation pipeline. The repository-level README provides the overall project context, while this README describes the CoDuAR-specific workflow.

## Folder Layout

```text
coduar/
|-- dataset/                                  # Video loading, augmentation, and dataset builders
|-- data/                                     # Data of HA-ViD and the custom dataset, their compositional mapping files and scripts to prepare the data
|-- models/                                   # CoDuAR model
|-- script/
|   `-- train.sh                              # Example training script
|   `-- evaluate.sh                           # Example evaluation script
|-- utils/
|   `-- validate_compositional_annotations.py # Annotation validation utility
|-- engine_for_compositional_transformer.py   # Training, validation, and test loops
|-- run_compositional_transformer.py          # Main entry point
`-- utils.py                                  # Distributed, logging, checkpoint, and metric utilities
```

## Data Organization

CoDuAR uses separate left-hand and right-hand video clips from HA-ViD action recognition dataset (primitive task level). Please request HA-ViD from the [official HA-ViD website](https://iai-hrc.github.io/ha-vid).

The custom data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1SaAPRiS4NoaDxFXge_9AJpc5d379c9im?usp=sharing). 

To process the custom data, using the provided script:
```bash
cd data
bash prepare_custom_dataset.sh
```

The compositional mapping files are provided in `data/`.

The default example script assumes the following structure under `coduar/`:

```text
data/
`-- havid/
    |-- lh_v0/
    |   |-- videos_train/
    |   |-- videos_val/
    |   |-- train_list_compositional.txt
    |   `-- val_list_compositional.txt
    `-- rh_v0/
        |-- videos_train/
        |-- videos_val/
        |-- train_list_compositional.txt
        `-- val_list_compositional.txt
`-- custom_dataset/
    |-- videos_train/
    |-- videos_val/
    |-- lh_train_list_compositional.txt
    |-- rh_train_list_compositional.txt
    |-- lh_val_list_compositional.txt
    `-- rh_val_list_compositional.txt
```

Each annotation file should contain one sample per line:

```text
video_name verb_id manipulated_object_id target_object_id tool_id
```

For example:

```text
sample_000001.mp4 2 14 7 3
sample_000002.mp4 1 8 5 0
```

Class IDs are expected to be zero-indexed non-negative integers. The class counts passed to `run_compositional_transformer.py` must be greater than the maximum ID in the corresponding annotation file.

## Annotation Validation

Before training, validate the left-hand and right-hand annotation files:

```bash
cd coduar
python utils/validate_compositional_annotations.py \
  --lh_train data/havid/lh_v0/train_list_compositional.txt \
  --lh_val data/havid/lh_v0/val_list_compositional.txt \
  --rh_train data/havid/rh_v0/train_list_compositional.txt \
  --rh_val data/havid/rh_v0/val_list_compositional.txt \
  --lh_video_dir data/havid/lh_v0 \
  --rh_video_dir data/havid/rh_v0 \
  --check_videos
```

The validator checks line format, label ranges, class statistics, imbalance warnings, and optional video-file existence. It also prints recommended class-count parameters for training.

## Training

The main training entry point is `run_compositional_transformer.py`. The example script `script/train.sh` provides the default command for the CoDuAR transformer-decoder model:

```bash
bash script/train.sh
```

For traning the CoDuAR on the custom dataset, run:

```bash
bash script/train_custom_dataset.sh
```

Before running the script, update the following variables in `script/train.sh` for the local environment:

- `CUDA_VISIBLE_DEVICES`: GPU index or indices.
- `MODEL_PATH`: pretrained VideoMAE checkpoint path.
- `LH_DATA_DIR` and `RH_DATA_DIR`: left-hand and right-hand data roots.
- `LH_NUM_*` and `RH_NUM_*`: class counts for verbs, manipulated objects, target objects, and tools.
- `OUTPUT_DIR`: directory for checkpoints, logs, and evaluation summaries.

The script trains `vit_base_patch16_224_compositional_dual_transformer` with hand-specific adapters and a transformer decoder for element communication. It saves checkpoints, validation metrics, TensorBoard logs, and a final test summary under `OUTPUT_DIR`.

## Evaluation

To run evaluation only, pass `--eval` and resume from a trained checkpoint:

```bash
bash script/evaluate.sh
```

```bash
cd coduar
python run_compositional_transformer.py \
  --eval \
  --resume output/coduar/checkpoint-best.pth \
  --model vit_base_patch16_224_compositional_dual_transformer \
  --lh_data_dir data/havid/lh_v0 \
  --rh_data_dir data/havid/rh_v0 \
  --lh_train_ann data/havid/lh_v0/train_list_compositional.txt \
  --rh_train_ann data/havid/rh_v0/train_list_compositional.txt \
  --lh_val_ann data/havid/lh_v0/val_list_compositional.txt \
  --rh_val_ann data/havid/rh_v0/val_list_compositional.txt \
  --lh_num_verbs 7 \
  --lh_num_manip_objs 26 \
  --lh_num_target_objs 26 \
  --lh_num_tools 6 \
  --rh_num_verbs 7 \
  --rh_num_manip_objs 26 \
  --rh_num_target_objs 26 \
  --rh_num_tools 6 \
  --output_dir output/coduar
```

Use the class counts and paths that correspond to the dataset split being evaluated.

## Reported Metrics

The training and evaluation code reports element-level and whole-action recognition metrics for both hands:

- Left-hand and right-hand verb accuracy.
- Left-hand and right-hand manipulated-object accuracy.
- Left-hand and right-hand target-object accuracy.
- Left-hand and right-hand tool accuracy.
- Left-hand and right-hand whole-action accuracy, computed from the joint correctness of the four compositional elements.

The best checkpoint is selected by the average of the left-hand and right-hand whole-action top-1 validation accuracies.

## Outputs

Typical outputs under `OUTPUT_DIR` include:

- `checkpoint-*.pth`: periodic training checkpoints.
- `checkpoint-best.pth`: best checkpoint according to validation whole-action accuracy.
- `log.txt`: epoch-level training and validation logs in JSON-lines format.
- `best_model_val_metrics.json`: validation metrics for the best checkpoint.
- `final_test_summary.json`: final test metrics and training summary.
- TensorBoard event files when `--log_dir` is provided.

