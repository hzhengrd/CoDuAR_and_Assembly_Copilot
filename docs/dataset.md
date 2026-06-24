# Dataset Format

CoDuAR expects dual-hand video inputs and compositional action labels.

## Directory Template

```text
dataset_root/
├── lh_v0/
│   ├── videos_train/
│   ├── videos_val/
│   ├── videos_test/
│   ├── train_list_compositional.txt
│   ├── val_list_compositional.txt
│   └── test_list_compositional.txt
└── rh_v0/
    ├── videos_train/
    ├── videos_val/
    ├── videos_test/
    ├── train_list_compositional.txt
    ├── val_list_compositional.txt
    └── test_list_compositional.txt
```

## Annotation Format

Each line should contain:

```text
video_filename verb_id manipulated_object_id target_object_id tool_id
```

Example:

```text
clip_000001.mp4 5 12 8 3
clip_000002.mp4 2 15 0 0
```

All IDs are zero-indexed. Keep label names in:

- `data/label_maps/label_map_verbs.txt`
- `data/label_maps/label_map_manipulated_objects.txt`
- `data/label_maps/label_map_target_objects.txt`
- `data/label_maps/label_map_tools.txt`

## Release Checklist

- Fill final class counts in `configs/train_coduar.example.yaml`.
- Add dataset license and access instructions.
- Document whether raw videos can be redistributed.
- Provide checksums for released annotation files.
- Add a tiny public sample if licensing permits.
