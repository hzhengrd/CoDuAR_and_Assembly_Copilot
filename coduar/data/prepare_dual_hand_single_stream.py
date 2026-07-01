"""
Dual-Hand Single-Stream Dataset Generator

CORRECT APPROACH:
- Extract video clips ONCE from single video stream
- Generate TWO index files (LH and RH) pointing to same clips with different labels

This matches real-world usage:
- Single video input
- Dual-hand prediction output
"""

import os
import cv2
import random
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

def load_label_mapping(mapping_file: Path) -> Dict[str, int]:
    """Load label string -> numeric ID mapping from file"""
    label_to_id: Dict[str, int] = {}
    
    if not mapping_file.exists():
        print(f"Warning: Mapping file not found: {mapping_file}")
        return label_to_id
    
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.replace('\t', ' ').replace(',', ' ').split(' ') if p.strip()]
            if len(parts) < 2:
                continue
            
            a, b = parts[0], parts[1]
            
            def is_int(s: str) -> bool:
                try:
                    int(s)
                    return True
                except ValueError:
                    return False
            
            if is_int(a) and not is_int(b):
                label_to_id[b] = int(a)
            elif is_int(b) and not is_int(a):
                label_to_id[a] = int(b)
    
    return label_to_id


def read_bundle_list(bundle_path: Path) -> List[str]:
    """Read video names from bundle file"""
    names: List[str] = []
    
    if not bundle_path.exists():
        print(f"Warning: Bundle file not found: {bundle_path}")
        return names
    
    with open(bundle_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith('.txt'):
                base = line[:-4]
            else:
                base = Path(line).stem
            
            names.append(base)
    
    return names


def read_frame_labels(annotation_path: Path) -> List[str]:
    """Load frame-wise annotations from text file"""
    labels: List[str] = []
    
    if not annotation_path.exists():
        return labels
    
    with open(annotation_path, 'r') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            
            tokens = raw.split()
            labels.extend(tokens)
    
    return labels


def get_majority_label(labels: List[str]) -> str:
    """Get majority label from a list using voting"""
    if not labels:
        return "null"
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def save_video_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: Optional[float] = None
) -> bool:
    """Extract and save a video clip"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_fps = fps if fps is not None else (video_fps if video_fps > 0 else 25.0)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
    
    if not writer.isOpened():
        cap.release()
        return False
    
    try:
        frames_extracted = 0
        num_frames_to_extract = end_frame - start_frame + 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if abs(current_pos - start_frame) < 10:
            for _ in range(num_frames_to_extract):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frames_extracted += 1
        
        if frames_extracted < num_frames_to_extract * 0.9:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for _ in range(start_frame):
                ret = cap.grab()
                if not ret:
                    break
            
            frames_extracted = 0
            for _ in range(num_frames_to_extract):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frames_extracted += 1
        
        if frames_extracted < num_frames_to_extract:
            writer.release()
            cap.release()
            return False
    finally:
        writer.release()
        cap.release()
    
    return out_path.exists() and out_path.stat().st_size > 0


# ============================================================================
# Clip Metadata
# ============================================================================

@dataclass
class SingleStreamClipMetadata:
    """Metadata for a single-stream clip with dual-hand labels"""
    video_base: str
    clip_index: int
    clip_name: str
    start_frame: int
    end_frame: int
    window_size: int
    lh_label: str
    lh_label_id: int
    rh_label: str
    rh_label_id: int
    is_null: bool
    is_longtail: bool
    clip_path: str  # Single path to video clip
    fps: float


# ============================================================================
# Long-Tail Analysis
# ============================================================================

class LongTailAnalyzer:
    """Analyze dataset to identify long-tail actions"""
    
    def __init__(self, longtail_threshold: float = 0.3):
        self.longtail_threshold = longtail_threshold
        self.lh_action_counts: Counter = Counter()
        self.rh_action_counts: Counter = Counter()
        self.longtail_lh_actions: Set[str] = set()
        self.longtail_rh_actions: Set[str] = set()
        self.null_label = "null"
    
    def analyze_annotations(
        self,
        lh_annotation_dir: Path,
        rh_annotation_dir: Path,
        video_bases: List[str]
    ):
        """Analyze all annotations to find action distribution"""
        print("Analyzing action distribution...")
        
        lh_found = 0
        rh_found = 0
        
        for video_base in tqdm(video_bases, desc="Analyzing videos"):
            lh_ann_path = lh_annotation_dir / f"{video_base}.txt"
            rh_ann_path = rh_annotation_dir / f"{video_base}.txt"
            
            if lh_ann_path.exists():
                lh_labels = read_frame_labels(lh_ann_path)
                self.lh_action_counts.update(lh_labels)
                lh_found += 1
            else:
                print(f"\n⚠️  LH annotation not found: {lh_ann_path}")
            
            if rh_ann_path.exists():
                rh_labels = read_frame_labels(rh_ann_path)
                self.rh_action_counts.update(rh_labels)
                rh_found += 1
            else:
                print(f"\n⚠️  RH annotation not found: {rh_ann_path}")
        
        print(f"\nAnnotation files found:")
        print(f"  LH: {lh_found}/{len(video_bases)}")
        print(f"  RH: {rh_found}/{len(video_bases)}")
        
        self._identify_longtail_actions()
        self._print_analysis()
    
    def _identify_longtail_actions(self):
        """Identify long-tail actions based on frequency threshold"""
        lh_non_null = {k: v for k, v in self.lh_action_counts.items() 
                       if k.lower() != self.null_label}
        rh_non_null = {k: v for k, v in self.rh_action_counts.items() 
                       if k.lower() != self.null_label}
        
        lh_sorted = sorted(lh_non_null.items(), key=lambda x: x[1])
        rh_sorted = sorted(rh_non_null.items(), key=lambda x: x[1])
        
        lh_longtail_count = int(len(lh_sorted) * self.longtail_threshold)
        rh_longtail_count = int(len(rh_sorted) * self.longtail_threshold)
        
        self.longtail_lh_actions = set(action for action, _ in lh_sorted[:lh_longtail_count])
        self.longtail_rh_actions = set(action for action, _ in rh_sorted[:rh_longtail_count])
    
    def _print_analysis(self):
        """Print analysis results"""
        print("\n" + "=" * 80)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        total_lh = sum(self.lh_action_counts.values())
        null_lh = self.lh_action_counts.get(self.null_label, 0)
        active_lh = total_lh - null_lh
        
        print(f"\nLeft Hand:")
        print(f"  Total frames: {total_lh:,}")
        if total_lh > 0:
            print(f"  Null frames:  {null_lh:,} ({100*null_lh/total_lh:.1f}%)")
            print(f"  Active frames: {active_lh:,} ({100*active_lh/total_lh:.1f}%)")
            print(f"  Long-tail actions: {len(self.longtail_lh_actions)}")
        else:
            print(f"  ⚠️  WARNING: No frames found in LH annotations!")
            print(f"  Check that annotation files exist and are not empty")
        
        total_rh = sum(self.rh_action_counts.values())
        null_rh = self.rh_action_counts.get(self.null_label, 0)
        active_rh = total_rh - null_rh
        
        print(f"\nRight Hand:")
        print(f"  Total frames: {total_rh:,}")
        if total_rh > 0:
            print(f"  Null frames:  {null_rh:,} ({100*null_rh/total_rh:.1f}%)")
            print(f"  Active frames: {active_rh:,} ({100*active_rh/total_rh:.1f}%)")
            print(f"  Long-tail actions: {len(self.longtail_rh_actions)}")
        else:
            print(f"  ⚠️  WARNING: No frames found in RH annotations!")
            print(f"  Check that annotation files exist and are not empty")
        
        if total_lh > 0:
            print(f"\nTop 10 Most Frequent Actions (LH):")
            for action, count in self.lh_action_counts.most_common(10):
                pct = 100 * count / total_lh
                print(f"  {action:<20}: {count:>8,} frames ({pct:>5.1f}%)")
        
        print("=" * 80)
    
    def is_longtail(self, lh_label: str, rh_label: str) -> bool:
        """Check if either hand has a long-tail action"""
        return (lh_label in self.longtail_lh_actions or 
                rh_label in self.longtail_rh_actions)
    
    def is_null(self, lh_label: str, rh_label: str) -> bool:
        """Check if both hands are null"""
        return (lh_label.lower() == self.null_label and 
                rh_label.lower() == self.null_label)


# ============================================================================
# Single-Stream Extractor with Long-Tail Awareness
# ============================================================================

class SingleStreamDualHandExtractor:
    """Extract clips ONCE with dual-hand labels"""
    
    def __init__(
        self,
        video_dir: Path,
        lh_annotation_dir: Path,
        rh_annotation_dir: Path,
        output_dir: Path,
        label_mapping: Dict[str, int],
        analyzer: LongTailAnalyzer,
        window_sizes: List[int] = [16],
        stride_ratio: float = 0.5,
        label_strategy: str = 'majority',
        null_ratio: float = 0.15,
        longtail_upsample_factor: float = 3.0,
        temporal_jitter: int = 0,
        save_clips: bool = True
    ):
        self.video_dir = video_dir
        self.lh_annotation_dir = lh_annotation_dir
        self.rh_annotation_dir = rh_annotation_dir
        self.output_dir = output_dir
        self.label_mapping = label_mapping
        self.analyzer = analyzer
        self.window_sizes = sorted(window_sizes)
        self.stride_ratio = stride_ratio
        self.temporal_jitter = temporal_jitter
        self.save_clips = save_clips
        self.null_ratio = null_ratio
        self.longtail_upsample_factor = longtail_upsample_factor
        
        self.label_strategy = label_strategy
        
        # Statistics
        self.total_clips = 0
        self.null_clips = 0
        self.longtail_clips = 0
        self.regular_clips = 0
        self.lh_label_counts: Dict[str, int] = defaultdict(int)
        self.rh_label_counts: Dict[str, int] = defaultdict(int)
    
    def extract_from_video(self, video_base: str) -> List[SingleStreamClipMetadata]:
        """Extract clips with dual-hand labels from single video stream"""
        
        video_path = self.video_dir / f"{video_base}.mp4"
        lh_ann_path = self.lh_annotation_dir / f"{video_base}.txt"
        rh_ann_path = self.rh_annotation_dir / f"{video_base}.txt"
        
        if not video_path.exists() or not lh_ann_path.exists() or not rh_ann_path.exists():
            return []
        
        # Load annotations
        lh_labels = read_frame_labels(lh_ann_path)
        rh_labels = read_frame_labels(rh_ann_path)
        
        if len(lh_labels) == 0 or len(rh_labels) == 0:
            return []
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        total_frames = min(total_frames, len(lh_labels), len(rh_labels))
        
        # Generate candidates
        all_candidates = []
        for window_size in self.window_sizes:
            stride = max(1, int(window_size * self.stride_ratio))
            candidates = self._generate_clip_candidates(
                video_base, lh_labels, rh_labels, total_frames, window_size, stride, fps
            )
            all_candidates.extend(candidates)
        
        # Apply intelligent sampling
        selected_clips = self._intelligent_sample(all_candidates)
        
        # Save clips (ONCE per clip, not twice!)
        final_metadata = []
        for metadata in selected_clips:
            if self.save_clips:
                clip_output_path = self.output_dir / f"{metadata.clip_name}.mp4"
                success = save_video_clip(
                    video_path, clip_output_path, 
                    metadata.start_frame, metadata.end_frame, metadata.fps
                )
                if not success:
                    continue
            
            final_metadata.append(metadata)
            
            # Update statistics
            self.total_clips += 1
            self.lh_label_counts[metadata.lh_label] += 1
            self.rh_label_counts[metadata.rh_label] += 1
            
            if metadata.is_null:
                self.null_clips += 1
            elif metadata.is_longtail:
                self.longtail_clips += 1
            else:
                self.regular_clips += 1
        
        return final_metadata
    
    def _generate_clip_candidates(
        self, video_base: str, lh_labels: List[str], rh_labels: List[str],
        total_frames: int, window_size: int, stride: int, fps: float
    ) -> List[SingleStreamClipMetadata]:
        """Generate candidate clips"""
        candidates = []
        clip_idx = 0
        
        max_start = total_frames - window_size
        if max_start < 0:
            return candidates
        
        for start_frame in range(0, max_start + 1, stride):
            if self.temporal_jitter > 0:
                jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
                jittered_start = max(0, min(start_frame + jitter, max_start))
            else:
                jittered_start = start_frame
            
            end_frame = jittered_start + window_size - 1
            
            # Get labels for BOTH hands from SAME temporal window
            lh_label = get_majority_label(lh_labels[jittered_start:end_frame+1])
            rh_label = get_majority_label(rh_labels[jittered_start:end_frame+1])
            
            lh_label_id = self.label_mapping.get(lh_label)
            rh_label_id = self.label_mapping.get(rh_label)
            
            if lh_label_id is None or rh_label_id is None:
                continue
            
            clip_name = f"{video_base}_ws{window_size}_{clip_idx:04d}"
            # Use only the last directory name (e.g., "videos_train" from "dataset/single_stream_aa/videos_train")
            output_dir_name = Path(self.output_dir).name
            clip_path = f"{output_dir_name}/{clip_name}.mp4"  # Single path!
            
            is_null = self.analyzer.is_null(lh_label, rh_label)
            is_longtail = self.analyzer.is_longtail(lh_label, rh_label)
            
            metadata = SingleStreamClipMetadata(
                video_base=video_base,
                clip_index=clip_idx,
                clip_name=clip_name,
                start_frame=jittered_start,
                end_frame=end_frame,
                window_size=window_size,
                lh_label=lh_label,
                lh_label_id=lh_label_id,
                rh_label=rh_label,
                rh_label_id=rh_label_id,
                is_null=is_null,
                is_longtail=is_longtail,
                clip_path=clip_path,
                fps=fps
            )
            
            candidates.append(metadata)
            clip_idx += 1
        
        return candidates
    
    def _intelligent_sample(
        self, candidates: List[SingleStreamClipMetadata]
    ) -> List[SingleStreamClipMetadata]:
        """Apply intelligent sampling"""
        null_clips = []
        longtail_clips = []
        regular_clips = []
        
        for clip in candidates:
            if clip.is_null:
                null_clips.append(clip)
            elif clip.is_longtail:
                longtail_clips.append(clip)
            else:
                regular_clips.append(clip)
        
        total_candidates = len(candidates)
        if total_candidates == 0:
            return []
        
        target_null_count = int(total_candidates * self.null_ratio)
        
        if len(null_clips) > target_null_count:
            sampled_null = random.sample(null_clips, target_null_count)
        else:
            sampled_null = null_clips
        
        upsample_count = int(len(longtail_clips) * self.longtail_upsample_factor)
        if upsample_count > len(longtail_clips):
            sampled_longtail = longtail_clips + random.choices(
                longtail_clips, k=upsample_count - len(longtail_clips)
            )
        else:
            sampled_longtail = longtail_clips
        
        selected = sampled_null + sampled_longtail + regular_clips
        random.shuffle(selected)
        
        return selected


# ============================================================================
# Dataset Builder
# ============================================================================

class SingleStreamDatasetBuilder:
    """Build single-stream dual-hand dataset"""
    
    def __init__(
        self,
        video_dir: Path,
        lh_annotation_dir: Path,
        rh_annotation_dir: Path,
        output_dir: Path,
        lh_label_mapping: Dict[str, int],
        rh_label_mapping: Dict[str, int],
        window_sizes: List[int] = [16],
        stride_ratio: float = 0.5,
        label_strategy: str = 'majority',
        null_ratio: float = 0.15,
        longtail_threshold: float = 0.3,
        longtail_upsample_factor: float = 3.0,
        temporal_jitter: int = 0,
        save_clips: bool = True
    ):
        self.video_dir = Path(video_dir)
        self.lh_annotation_dir = Path(lh_annotation_dir)
        self.rh_annotation_dir = Path(rh_annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lh_label_mapping = lh_label_mapping
        self.rh_label_mapping = rh_label_mapping
        self.window_sizes = window_sizes
        
        self.analyzer = LongTailAnalyzer(longtail_threshold)
        self.extractor = None
        
        self.stride_ratio = stride_ratio
        self.null_ratio = null_ratio
        self.longtail_upsample_factor = longtail_upsample_factor
        self.temporal_jitter = temporal_jitter
        self.save_clips = save_clips
    
    def build_dataset(
        self,
        split_list: Optional[Path] = None,
        output_prefix: str = 'train'
    ):
        """Build dataset with two-pass approach"""
        
        print(f"Building single-stream dual-hand dataset...")
        
        # Get video list
        if split_list and split_list.exists():
            video_bases = read_bundle_list(split_list)
            video_files = [self.video_dir / f"{base}.mp4" for base in video_bases]
            video_files = [vf for vf in video_files if vf.exists()]
            video_bases = [vf.stem for vf in video_files]
        else:
            video_files = sorted(self.video_dir.glob('*.mp4'))
            video_bases = [vf.stem for vf in video_files]
        
        if not video_files:
            print(f"No video files found in {self.video_dir}")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # FIRST PASS: Analyze
        print("\n" + "=" * 80)
        print("FIRST PASS: Analyzing Action Distribution")
        print("=" * 80)
        self.analyzer.analyze_annotations(
            self.lh_annotation_dir,
            self.rh_annotation_dir,
            video_bases
        )
        
        # SECOND PASS: Extract
        print("\n" + "=" * 80)
        print("SECOND PASS: Extracting Clips")
        print("=" * 80)
        
        self.extractor = SingleStreamDualHandExtractor(
            self.video_dir,
            self.lh_annotation_dir,
            self.rh_annotation_dir,
            self.output_dir,
            {**self.lh_label_mapping, **self.rh_label_mapping},
            self.analyzer,
            window_sizes=self.window_sizes,
            stride_ratio=self.stride_ratio,
            null_ratio=self.null_ratio,
            longtail_upsample_factor=self.longtail_upsample_factor,
            temporal_jitter=self.temporal_jitter,
            save_clips=self.save_clips
        )
        
        all_clips_metadata = []
        for video_base in tqdm(video_bases, desc="Extracting clips"):
            clips = self.extractor.extract_from_video(video_base)
            all_clips_metadata.extend(clips)
        
        # Save outputs
        self._save_outputs(all_clips_metadata, output_prefix)
        
        # Print statistics
        self._print_statistics()
    
    def _save_outputs(self, clips_metadata: List[SingleStreamClipMetadata], output_prefix: str):
        """Save output files - TWO index files, ONE set of clips"""
        
        # Use parent directory for index files (e.g., "dataset/single_stream_aa/")
        # while videos are in subdirectory (e.g., "dataset/single_stream_aa/videos_train/")
        index_dir = Path(self.output_dir).parent
        
        # Save LEFT HAND index
        lh_index = index_dir / f"{output_prefix}_list_video_lh.txt"
        with open(lh_index, 'w') as f:
            for clip in clips_metadata:
                f.write(f"{clip.clip_path} {clip.lh_label_id}\n")
        print(f"\nSaved LH index: {lh_index}")
        
        # Save RIGHT HAND index
        rh_index = index_dir / f"{output_prefix}_list_video_rh.txt"
        with open(rh_index, 'w') as f:
            for clip in clips_metadata:
                f.write(f"{clip.clip_path} {clip.rh_label_id}\n")
        print(f"Saved RH index: {rh_index}")
        
        # Save dual-hand pairs index
        pairs_index = index_dir / f"{output_prefix}_pairs_index.txt"
        with open(pairs_index, 'w') as f:
            f.write("# Format: clip_path lh_label_id rh_label_id is_longtail\n")
            for clip in clips_metadata:
                f.write(f"{clip.clip_path} {clip.lh_label_id} {clip.rh_label_id} {int(clip.is_longtail)}\n")
        print(f"Saved pairs index: {pairs_index}")
        
        # Save metadata
        metadata_list = []
        for clip in clips_metadata:
            metadata_list.append({
                'video_base': clip.video_base,
                'clip_name': clip.clip_name,
                'clip_path': clip.clip_path,
                'start_frame': clip.start_frame,
                'end_frame': clip.end_frame,
                'window_size': clip.window_size,
                'lh_label': clip.lh_label,
                'lh_label_id': clip.lh_label_id,
                'rh_label': clip.rh_label,
                'rh_label_id': clip.rh_label_id,
                'is_null': clip.is_null,
                'is_longtail': clip.is_longtail,
                'fps': clip.fps
            })
        
        metadata_file = index_dir / f"{output_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"Saved metadata: {metadata_file}")
        
        # Save statistics
        stats = {
            'total_clips': self.extractor.total_clips,
            'null_clips': self.extractor.null_clips,
            'longtail_clips': self.extractor.longtail_clips,
            'regular_clips': self.extractor.regular_clips,
            'lh_label_distribution': dict(self.extractor.lh_label_counts),
            'rh_label_distribution': dict(self.extractor.rh_label_counts),
        }
        
        stats_file = index_dir / f"{output_prefix}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics: {stats_file}")
    
    def _print_statistics(self):
        """Print final statistics"""
        print("\n" + "=" * 80)
        print("FINAL DATASET STATISTICS")
        print("=" * 80)
        print(f"\nTotal clips extracted: {self.extractor.total_clips}")
        print(f"  Null clips: {self.extractor.null_clips}")
        print(f"  Long-tail clips: {self.extractor.longtail_clips}")
        print(f"  Regular clips: {self.extractor.regular_clips}")
        print(f"\nDisk space saved: 50% (single-stream vs dual-stream)")
        print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate single-stream dual-hand dataset')
    
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--lh_annotation_dir', type=str, required=True)
    parser.add_argument('--rh_annotation_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lh_mapping_file', type=str, required=True)
    parser.add_argument('--rh_mapping_file', type=str, required=True)
    parser.add_argument('--window_sizes', type=str, default='8,16,24,32,40')
    parser.add_argument('--stride_ratio', type=float, default=0.5)
    parser.add_argument('--label_strategy', type=str, default='majority')
    parser.add_argument('--null_ratio', type=float, default=0.15)
    parser.add_argument('--longtail_threshold', type=float, default=0.3)
    parser.add_argument('--longtail_upsample_factor', type=float, default=3.0)
    parser.add_argument('--temporal_jitter', type=int, default=0)
    parser.add_argument('--split_list', type=str, default=None)
    parser.add_argument('--output_prefix', type=str, default='train')
    parser.add_argument('--no_save_clips', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    window_sizes = [int(ws.strip()) for ws in args.window_sizes.split(',')]
    
    lh_label_mapping = load_label_mapping(Path(args.lh_mapping_file))
    rh_label_mapping = load_label_mapping(Path(args.rh_mapping_file))
    
    builder = SingleStreamDatasetBuilder(
        video_dir=Path(args.video_dir),
        lh_annotation_dir=Path(args.lh_annotation_dir),
        rh_annotation_dir=Path(args.rh_annotation_dir),
        output_dir=Path(args.output_dir),
        lh_label_mapping=lh_label_mapping,
        rh_label_mapping=rh_label_mapping,
        window_sizes=window_sizes,
        stride_ratio=args.stride_ratio,
        null_ratio=args.null_ratio,
        longtail_threshold=args.longtail_threshold,
        longtail_upsample_factor=args.longtail_upsample_factor,
        temporal_jitter=args.temporal_jitter,
        save_clips=not args.no_save_clips
    )
    
    split_list = Path(args.split_list) if args.split_list else None
    builder.build_dataset(split_list=split_list, output_prefix=args.output_prefix)
    
    print("\n✓ Single-stream dual-hand dataset generation complete!")
    print(f"\nKey improvements:")
    print(f"  ✅ Clips saved ONCE (50% disk space savings)")
    print(f"  ✅ Perfect temporal synchronization guaranteed")
    print(f"  ✅ Two index files: *_list_video_lh.txt and *_list_video_rh.txt")
    print(f"  ✅ Matches your single-stream, dual-prediction architecture")


if __name__ == '__main__':
    main()

