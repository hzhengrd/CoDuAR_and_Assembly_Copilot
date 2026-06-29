# Compositional Dual-Hand Dataset Builder
# Handles datasets with compositional action labels (verb, manip_obj, target_obj, tool)
import os
from torch.utils.data import Dataset
from .datasets import VideoClsDataset


class CompositionalDualHandDataset(Dataset):
    """Dataset wrapper for compositional dual-hand action recognition.
    
    Each sample includes:
    - Left hand video frames
    - Right hand video frames
    - 4 labels per hand: verb, manipulated object, target object, tool
    """
    
    def __init__(self, lh_dataset, rh_dataset):
        """
        Args:
            lh_dataset: Left-hand VideoClsDataset with compositional labels
            rh_dataset: Right-hand VideoClsDataset with compositional labels
        """
        self.lh_dataset = lh_dataset
        self.rh_dataset = rh_dataset
        
        # Use minimum length
        self.min_length = min(len(lh_dataset), len(rh_dataset))
        print(f"Compositional Dataset lengths: LH={len(lh_dataset)}, RH={len(rh_dataset)}, using min={self.min_length}")
    
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, index):
        # Ensure index is within bounds
        lh_index = index % len(self.lh_dataset)
        rh_index = index % len(self.rh_dataset)
        
        # Get samples from both datasets
        lh_sample = self.lh_dataset[lh_index]
        rh_sample = self.rh_dataset[rh_index]
        
        # Handle different return formats (train vs validation/test)
        # In compositional mode, labels are tuples: (verb, manip_obj, target_obj, tool)
        if len(lh_sample) == 4:  # Training mode: (frames, labels_tuple, index, {})
            lh_frames, lh_labels, lh_index, lh_extra = lh_sample
            rh_frames, rh_labels, rh_index, rh_extra = rh_sample
            
            # Handle repeated augmentation (num_sample > 1)
            if isinstance(lh_frames, list):  # Multiple samples - labels are also lists
                # lh_labels and rh_labels are lists of tuples
                # For simplicity, take the first sample's labels (all samples have same label)
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels[0]
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels[0]
                
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'lh_index': lh_index,
                    'rh_index': rh_index,
                    'multiple_samples': True
                }
            else:  # Single sample
                # Unpack compositional labels
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
                
                return {
                    'lh_frames': lh_frames,
                    'rh_frames': rh_frames,
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'lh_index': lh_index,
                    'rh_index': rh_index,
                    'multiple_samples': False
                }
        else:  # Validation/test mode
            # Validation mode: (frames, labels_tuple, video_name) - 3 values
            # Test mode: (frames, labels_tuple, video_name, chunk_nb, split_nb) - 5 values
            if len(lh_sample) == 5:  # Test mode
                lh_frames, lh_labels, lh_name, lh_chunk, lh_split = lh_sample
                rh_frames, rh_labels, rh_name, rh_chunk, rh_split = rh_sample
            else:  # Validation mode (3 values)
                lh_frames, lh_labels, lh_name = lh_sample
                rh_frames, rh_labels, rh_name = rh_sample
            
            # Unpack compositional labels
            lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
            rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
            
            return {
                'lh_frames': lh_frames,
                'rh_frames': rh_frames,
                'lh_verb': lh_verb,
                'lh_manip_obj': lh_manip_obj,
                'lh_target_obj': lh_target_obj,
                'lh_tool': lh_tool,
                'rh_verb': rh_verb,
                'rh_manip_obj': rh_manip_obj,
                'rh_target_obj': rh_target_obj,
                'rh_tool': rh_tool,
                'lh_name': lh_name,
                'rh_name': rh_name,
                'multiple_samples': False
            }


class CompositionalDualHandSingleStreamDataset(Dataset):
    """Dataset wrapper for single-stream dual-hand action recognition.
    
    In this setup, the SAME video is used as input for both left and right hand predictions.
    The model uses a shared encoder and separate classification heads for each hand.
    
    Each sample includes:
    - Single video frames (used for both hands)
    - 4 labels for left hand: verb, manipulated object, target object, tool
    - 4 labels for right hand: verb, manipulated object, target object, tool
    """
    
    def __init__(self, lh_dataset, rh_dataset):
        """
        Args:
            lh_dataset: VideoClsDataset with left-hand compositional labels
            rh_dataset: VideoClsDataset with right-hand compositional labels
            
        Note: Both datasets should point to the SAME videos, but with different labels.
        """
        self.lh_dataset = lh_dataset
        self.rh_dataset = rh_dataset
        
        # Verify both datasets have the same length (they should, since same videos)
        if len(lh_dataset) != len(rh_dataset):
            print(f"WARNING: LH and RH datasets have different lengths: {len(lh_dataset)} vs {len(rh_dataset)}")
            print("Using minimum length for safety.")
        
        self.length = min(len(lh_dataset), len(rh_dataset))
        print(f"Single-Stream Compositional Dataset: {self.length} samples")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Get samples from both datasets (same video, different labels)
        lh_sample = self.lh_dataset[index]
        rh_sample = self.rh_dataset[index]
        
        # Handle different return formats (train vs validation/test)
        if len(lh_sample) == 4:  # Training mode: (frames, labels_tuple, index, {})
            lh_frames, lh_labels, lh_index, lh_extra = lh_sample
            rh_frames, rh_labels, rh_index, rh_extra = rh_sample
            
            # In single-stream mode, frames should be identical
            # We'll use lh_frames as the shared input
            frames = lh_frames
            
            # Handle repeated augmentation (num_sample > 1)
            if isinstance(frames, list):  # Multiple samples
                # Labels are also lists of tuples
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels[0]
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels[0]
                
                return {
                    'frames': frames,  # Single shared video input
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'index': lh_index,
                    'multiple_samples': True
                }
            else:  # Single sample
                # Unpack compositional labels
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
                
                return {
                    'frames': frames,  # Single shared video input
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'index': lh_index,
                    'multiple_samples': False
                }
        else:  # Validation/test mode
            # Test mode: (frames, labels_tuple, video_name, chunk_nb, split_nb) - 5 values
            # Validation mode: (frames, labels_tuple, video_name) - 3 values
            if len(lh_sample) == 5:  # Test mode
                lh_frames, lh_labels, lh_name, lh_chunk, lh_split = lh_sample
                rh_frames, rh_labels, rh_name, rh_chunk, rh_split = rh_sample
            else:  # Validation mode (3 values)
                lh_frames, lh_labels, lh_name = lh_sample
                rh_frames, rh_labels, rh_name = rh_sample
            
            # Use lh_frames as the shared input
            frames = lh_frames
            
            # Unpack compositional labels
            lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
            rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
            
            return {
                'frames': frames,  # Single shared video input
                'lh_verb': lh_verb,
                'lh_manip_obj': lh_manip_obj,
                'lh_target_obj': lh_target_obj,
                'lh_tool': lh_tool,
                'rh_verb': rh_verb,
                'rh_manip_obj': rh_manip_obj,
                'rh_target_obj': rh_target_obj,
                'rh_tool': rh_tool,
                'video_name': lh_name,
                'multiple_samples': False
            }


class CompositionalDualHandThreeViewDataset(Dataset):
    """Dataset wrapper for compositional dual-hand action recognition with three views.
    
    Each sample includes:
    - Left hand video frames from 3 views (v0, v1, v2)
    - Right hand video frames from 3 views (v0, v1, v2)
    - 4 labels per hand: verb, manipulated object, target object, tool
    """
    
    def __init__(self, lh_datasets, rh_datasets):
        """
        Args:
            lh_datasets: List of 3 Left-hand VideoClsDatasets (v0, v1, v2)
            rh_datasets: List of 3 Right-hand VideoClsDatasets (v0, v1, v2)
        """
        self.lh_datasets = lh_datasets
        self.rh_datasets = rh_datasets
        
        # Use minimum length across all datasets
        all_lengths = [len(ds) for ds in lh_datasets + rh_datasets]
        self.min_length = min(all_lengths)
        print(f"Three-View Dataset lengths: LH_v0={len(lh_datasets[0])}, LH_v1={len(lh_datasets[1])}, LH_v2={len(lh_datasets[2])}, "
              f"RH_v0={len(rh_datasets[0])}, RH_v1={len(rh_datasets[1])}, RH_v2={len(rh_datasets[2])}, using min={self.min_length}")
    
    def __len__(self):
        return self.min_length
    
    def __getitem__(self, index):
        # Ensure index is within bounds for all datasets
        samples = {}
        
        # Get samples from all left-hand views
        for i, lh_ds in enumerate(self.lh_datasets):
            idx = index % len(lh_ds)
            samples[f'lh_v{i}'] = lh_ds[idx]
        
        # Get samples from all right-hand views
        for i, rh_ds in enumerate(self.rh_datasets):
            idx = index % len(rh_ds)
            samples[f'rh_v{i}'] = rh_ds[idx]
        
        # All views should have the same labels, so use v0 labels
        lh_sample = samples['lh_v0']
        rh_sample = samples['rh_v0']
        
        # Handle different return formats (train vs validation/test)
        if len(lh_sample) == 4:  # Training mode: (frames, labels_tuple, index, {})
            _, lh_labels, _, _ = lh_sample
            _, rh_labels, _, _ = rh_sample
            
            # Check if we have repeated augmentation
            if isinstance(samples['lh_v0'][0], list):  # Multiple samples
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels[0]
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels[0]
                
                result = {
                    'lh_v0_frames': samples['lh_v0'][0],
                    'lh_v1_frames': samples['lh_v1'][0],
                    'lh_v2_frames': samples['lh_v2'][0],
                    'rh_v0_frames': samples['rh_v0'][0],
                    'rh_v1_frames': samples['rh_v1'][0],
                    'rh_v2_frames': samples['rh_v2'][0],
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'multiple_samples': True
                }
            else:  # Single sample
                lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
                rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
                
                result = {
                    'lh_v0_frames': samples['lh_v0'][0],
                    'lh_v1_frames': samples['lh_v1'][0],
                    'lh_v2_frames': samples['lh_v2'][0],
                    'rh_v0_frames': samples['rh_v0'][0],
                    'rh_v1_frames': samples['rh_v1'][0],
                    'rh_v2_frames': samples['rh_v2'][0],
                    'lh_verb': lh_verb,
                    'lh_manip_obj': lh_manip_obj,
                    'lh_target_obj': lh_target_obj,
                    'lh_tool': lh_tool,
                    'rh_verb': rh_verb,
                    'rh_manip_obj': rh_manip_obj,
                    'rh_target_obj': rh_target_obj,
                    'rh_tool': rh_tool,
                    'multiple_samples': False
                }
        else:  # Validation/test mode
            # Validation mode: (frames, labels_tuple, video_name) - 3 values
            # Test mode: (frames, labels_tuple, video_name, chunk_nb, split_nb) - 5 values
            if len(lh_sample) == 5:  # Test mode
                _, lh_labels, lh_name, _, _ = lh_sample  # Ignore chunk_nb and split_nb
                _, rh_labels, rh_name, _, _ = rh_sample
            else:  # Validation mode (3 values)
                _, lh_labels, lh_name = lh_sample
                _, rh_labels, rh_name = rh_sample
            
            lh_verb, lh_manip_obj, lh_target_obj, lh_tool = lh_labels
            rh_verb, rh_manip_obj, rh_target_obj, rh_tool = rh_labels
            
            result = {
                'lh_v0_frames': samples['lh_v0'][0],
                'lh_v1_frames': samples['lh_v1'][0],
                'lh_v2_frames': samples['lh_v2'][0],
                'rh_v0_frames': samples['rh_v0'][0],
                'rh_v1_frames': samples['rh_v1'][0],
                'rh_v2_frames': samples['rh_v2'][0],
                'lh_verb': lh_verb,
                'lh_manip_obj': lh_manip_obj,
                'lh_target_obj': lh_target_obj,
                'lh_tool': lh_tool,
                'rh_verb': rh_verb,
                'rh_manip_obj': rh_manip_obj,
                'rh_target_obj': rh_target_obj,
                'rh_tool': rh_tool,
                'lh_name': lh_name,
                'rh_name': rh_name,
                'multiple_samples': False
            }
        
        return result


def build_compositional_dual_hand_datasets(is_train, test_mode, args):
    """Build compositional dual-hand datasets.
    
    The datasets expect annotations in compositional format where each line contains:
    video_path verb_id manip_obj_id target_obj_id tool_id
    
    Example annotation line:
    /path/to/video.mp4 5 12 8 3
    
    This means: verb class 5, manipulated object class 12, target object class 8, tool class 3
    """
    if is_train:
        mode = 'train'
        lh_anno_path = args.lh_train_ann
        rh_anno_path = args.rh_train_ann
    elif test_mode:
        mode = 'test'
        lh_anno_path = args.lh_val_ann
        rh_anno_path = args.rh_val_ann
    else:
        mode = 'validation'
        lh_anno_path = args.lh_val_ann
        rh_anno_path = args.rh_val_ann
    
    # Set compositional mode flag
    args.compositional_mode = True
    
    # Create left-hand dataset
    lh_dataset = VideoClsDataset(
        anno_path=lh_anno_path,
        data_root=args.lh_data_dir,
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
        args=args)
    
    # Create right-hand dataset
    rh_dataset = VideoClsDataset(
        anno_path=rh_anno_path,
        data_root=args.rh_data_dir,
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
        args=args)
    
    # Wrap in compositional dataset
    dual_dataset = CompositionalDualHandDataset(lh_dataset, rh_dataset)
    
    return dual_dataset, args.lh_num_verbs  # Return dataset and a sample num_classes


def build_compositional_dual_hand_single_stream_datasets(is_train, test_mode, args):
    """Build compositional dual-hand single-stream datasets.
    
    In this setup, the SAME video is used for both left and right hand predictions.
    The model uses a shared encoder and separate classification heads.
    
    The datasets expect annotations in compositional format where each line contains:
    video_path verb_id manip_obj_id target_obj_id tool_id
    
    Example annotation line:
    clips/S0T0V0_ws32_0237.mp4 4 11 3 1
    
    Args:
        is_train: Whether this is training mode
        test_mode: Whether this is test mode (vs validation)
        args: Arguments containing:
            - data_dir: Base directory containing train/ and test/ subdirectories
            - lh_train_ann, rh_train_ann: Paths to LH/RH training annotations
            - lh_val_ann, rh_val_ann: Paths to LH/RH validation annotations
            - lh_num_verbs, lh_num_manip_objs, etc.: Number of classes per element
            - rh_num_verbs, rh_num_manip_objs, etc.: Number of classes per element
    
    Returns:
        dual_dataset: CompositionalDualHandSingleStreamDataset instance
        num_classes: Sample number of classes (for compatibility)
    """
    if is_train:
        mode = 'train'
        lh_anno_path = args.lh_train_ann
        rh_anno_path = args.rh_train_ann
        data_root = args.train_data_dir
    elif test_mode:
        mode = 'test'
        lh_anno_path = args.lh_test_ann if hasattr(args, 'lh_test_ann') else args.lh_val_ann
        rh_anno_path = args.rh_test_ann if hasattr(args, 'rh_test_ann') else args.rh_val_ann
        data_root = args.test_data_dir if hasattr(args, 'test_data_dir') else args.val_data_dir
    else:
        mode = 'validation'
        lh_anno_path = args.lh_val_ann
        rh_anno_path = args.rh_val_ann
        data_root = args.val_data_dir
    
    # Set compositional mode flag
    args.compositional_mode = True
    
    print(f"\n{'='*60}")
    print(f"Building Single-Stream Compositional Dual-Hand Dataset")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Data root: {data_root}")
    print(f"LH annotation: {lh_anno_path}")
    print(f"RH annotation: {rh_anno_path}")
    print(f"LH classes: V={args.lh_num_verbs}, M={args.lh_num_manip_objs}, T={args.lh_num_target_objs}, Tool={args.lh_num_tools}")
    print(f"RH classes: V={args.rh_num_verbs}, M={args.rh_num_manip_objs}, T={args.rh_num_target_objs}, Tool={args.rh_num_tools}")
    print(f"{'='*60}\n")
    
    # Create left-hand dataset (for labels)
    lh_dataset = VideoClsDataset(
        anno_path=lh_anno_path,
        data_root=data_root,
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
        args=args)
    
    # Create right-hand dataset (for labels)
    # Note: This loads the SAME videos as lh_dataset, but with different labels
    rh_dataset = VideoClsDataset(
        anno_path=rh_anno_path,
        data_root=data_root,
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
        args=args)
    
    # Wrap in single-stream compositional dataset
    dual_dataset = CompositionalDualHandSingleStreamDataset(lh_dataset, rh_dataset)
    
    return dual_dataset, args.lh_num_verbs  # Return dataset and a sample num_classes


def build_compositional_dual_hand_three_view_datasets(is_train, test_mode, args):
    """Build compositional dual-hand datasets with three views.
    
    The datasets expect annotations in compositional format where each line contains:
    video_path verb_id manip_obj_id target_obj_id tool_id
    
    This function loads 3 views for each hand (v0, v1, v2).
    """
    if is_train:
        mode = 'train'
        val_suffix = ''
    elif test_mode:
        mode = 'test'
        val_suffix = '_v2' if hasattr(args, 'use_val_v2') and args.use_val_v2 else ''
    else:
        mode = 'validation'
        val_suffix = '_v2' if hasattr(args, 'use_val_v2') and args.use_val_v2 else ''
    
    # Set compositional mode flag
    args.compositional_mode = True
    
    # Base data directory
    base_data_dir = args.data_dir
    
    # Create datasets for all three views of left hand
    lh_datasets = []
    for view_idx in range(3):
        if is_train:
            anno_path = f"{base_data_dir}/lh_v{view_idx}/train_list_compositional.txt"
        else:
            anno_path = f"{base_data_dir}/lh_v{view_idx}/val_list_compositional{val_suffix}.txt"
        
        data_root = f"{base_data_dir}/lh_v{view_idx}"
        
        lh_ds = VideoClsDataset(
            anno_path=anno_path,
            data_root=data_root,
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
            args=args)
        lh_datasets.append(lh_ds)
        print(f"Created left hand view {view_idx} dataset with {len(lh_ds)} samples")
    
    # Create datasets for all three views of right hand
    rh_datasets = []
    for view_idx in range(3):
        if is_train:
            anno_path = f"{base_data_dir}/rh_v{view_idx}/train_list_compositional.txt"
        else:
            anno_path = f"{base_data_dir}/rh_v{view_idx}/val_list_compositional{val_suffix}.txt"
        
        data_root = f"{base_data_dir}/rh_v{view_idx}"
        
        rh_ds = VideoClsDataset(
            anno_path=anno_path,
            data_root=data_root,
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
            args=args)
        rh_datasets.append(rh_ds)
        print(f"Created right hand view {view_idx} dataset with {len(rh_ds)} samples")
    
    # Wrap in three-view compositional dataset
    three_view_dataset = CompositionalDualHandThreeViewDataset(lh_datasets, rh_datasets)
    
    return three_view_dataset, args.lh_num_verbs  # Return dataset and a sample num_classes

