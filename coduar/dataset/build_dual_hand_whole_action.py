# Dual-Hand Whole-Action Dataset Builder
#
# Loads left-hand and right-hand clips from the STANDARD annotation format
#   (train_list_video.txt):  video_path  whole_action_class_id
# and pairs them sample-by-sample so the model sees both hands simultaneously.
#
# The whole-action label is taken from the LH annotation; both LH and RH
# clips carry the same task-level label.

import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from .datasets import VideoClsDataset


class DualHandWholeActionDataset(Dataset):
    """Pairs LH and RH clips with a shared flat whole-action label.

    Each sample returns:
        lh_frames : Tensor [C, T, H, W]  (or list of tensors for multi-aug)
        rh_frames : Tensor [C, T, H, W]  (or list of tensors for multi-aug)
        label     : int  – whole-action class id (same for LH and RH)
    """

    def __init__(self, lh_dataset: VideoClsDataset, rh_dataset: VideoClsDataset):
        self.lh_dataset = lh_dataset
        self.rh_dataset = rh_dataset
        self.length = min(len(lh_dataset), len(rh_dataset))
        print(f"DualHandWholeAction dataset: LH={len(lh_dataset)}, "
              f"RH={len(rh_dataset)}, using min={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        lh_sample = self.lh_dataset[index % len(self.lh_dataset)]
        rh_sample = self.rh_dataset[index % len(self.rh_dataset)]

        # ---------- Training mode ----------
        # VideoClsDataset returns (frames, label, index, {}) in train mode.
        # With num_sample > 1 it returns (list[frames], list[label], list[idx], {}).
        if len(lh_sample) == 4:
            lh_frames, lh_label, lh_idx, _ = lh_sample
            rh_frames, _rh_label, rh_idx, _ = rh_sample

            if isinstance(lh_frames, list):   # repeated-augmentation
                label = lh_label[0]           # same label for every augmentation
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'label': label,
                    'lh_index': lh_idx,
                    'rh_index': rh_idx,
                    'multiple_samples': True,
                }
            else:
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'label': lh_label,
                    'lh_index': lh_idx,
                    'rh_index': rh_idx,
                    'multiple_samples': False,
                }

        # ---------- Test mode (5-tuple) ----------
        if len(lh_sample) == 5:
            lh_frames, lh_label, lh_name, lh_chunk, lh_split = lh_sample
            rh_frames, _rh_label, rh_name, rh_chunk, rh_split = rh_sample
            return {
                'lh_frames': lh_frames,
                'rh_frames': rh_frames,
                'label': lh_label,
                'lh_name': lh_name,
                'rh_name': rh_name,
                'multiple_samples': False,
            }

        # ---------- Validation mode (3-tuple) ----------
        lh_frames, lh_label, lh_name = lh_sample
        rh_frames, _rh_label, rh_name = rh_sample
        return {
            'lh_frames': lh_frames,
            'rh_frames': rh_frames,
            'label': lh_label,
            'lh_name': lh_name,
            'rh_name': rh_name,
            'multiple_samples': False,
        }


def dual_hand_whole_action_collate(batch):
    """Collate function that handles repeated-augmentation (num_sample > 1)."""
    if batch[0].get('multiple_samples', False):
        lh_flat, rh_flat, label_flat = [], [], []
        for sample in batch:
            num_augs = len(sample['lh_frames'])
            lh_flat.extend(sample['lh_frames'])
            rh_flat.extend(sample['rh_frames'])
            label_flat.extend([sample['label']] * num_augs)
        return {
            'lh_frames': default_collate(lh_flat),
            'rh_frames': default_collate(rh_flat),
            'label': default_collate(label_flat),
        }

    # Single sample per instance
    collated = {}
    for key in batch[0].keys():
        if key in ('multiple_samples', 'lh_name', 'rh_name', 'lh_index', 'rh_index'):
            collated[key] = [d[key] for d in batch]
        else:
            collated[key] = default_collate([d[key] for d in batch])
    return collated


def build_dual_hand_whole_action_datasets(is_train: bool, test_mode: bool, args):
    """Build DualHandWholeActionDataset from standard video annotation files.

    Expects annotation files in the format produced by train_list_video.txt:
        video_relative_path  whole_action_class_id

    Args:
        is_train  : True for training split, False for validation/test.
        test_mode : When False, uses validation mode; when True, uses test mode.
        args      : Namespace with:
                      lh_data_dir, rh_data_dir
                      lh_train_ann, rh_train_ann  (video annotation files)
                      lh_val_ann,   rh_val_ann
                      num_frames, sampling_rate, input_size, short_side_size
                      test_num_segment, test_num_crop, num_sample
    """
    if is_train:
        mode = 'train'
        lh_anno = args.lh_train_ann
        rh_anno = args.rh_train_ann
    elif test_mode:
        mode = 'test'
        lh_anno = args.lh_val_ann
        rh_anno = args.rh_val_ann
    else:
        mode = 'validation'
        lh_anno = args.lh_val_ann
        rh_anno = args.rh_val_ann

    # Disable compositional mode so VideoClsDataset reads flat labels
    args.compositional_mode = False

    common_kwargs = dict(
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        sparse_sample=False,
        args=args,
    )

    lh_dataset = VideoClsDataset(
        anno_path=lh_anno,
        data_root=args.lh_data_dir,
        **common_kwargs,
    )

    rh_dataset = VideoClsDataset(
        anno_path=rh_anno,
        data_root=args.rh_data_dir,
        **common_kwargs,
    )

    return DualHandWholeActionDataset(lh_dataset, rh_dataset), args.nb_classes
