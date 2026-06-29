#!/usr/bin/env python3
"""
Utility script to validate compositional dual-hand annotation files.
Checks format, counts classes, and identifies potential issues.
"""

import argparse
import os
from collections import Counter


def validate_annotation_file(filepath, check_videos=False, video_dir=None):
    """
    Validate a compositional annotation file.
    
    Args:
        filepath: Path to annotation file
        check_videos: Whether to check if video files exist
        video_dir: Directory containing videos (if check_videos=True)
    
    Returns:
        dict with validation results
    """
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return None
    
    results = {
        'total_lines': 0,
        'valid_lines': 0,
        'errors': [],
        'verb_ids': Counter(),
        'manip_obj_ids': Counter(),
        'target_obj_ids': Counter(),
        'tool_ids': Counter(),
        'missing_videos': []
    }
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            results['total_lines'] += 1
            parts = line.split()
            
            # Check format: should have exactly 5 parts (video + 4 labels)
            if len(parts) != 5:
                error = f"Line {line_num}: Expected 5 values, got {len(parts)}: {line}"
                results['errors'].append(error)
                continue
            
            video_name = parts[0]
            
            # Parse labels
            try:
                verb_id = int(parts[1])
                manip_obj_id = int(parts[2])
                target_obj_id = int(parts[3])
                tool_id = int(parts[4])
            except ValueError as e:
                error = f"Line {line_num}: Invalid label format (must be integers): {line}"
                results['errors'].append(error)
                continue
            
            # Check for negative IDs
            if any(x < 0 for x in [verb_id, manip_obj_id, target_obj_id, tool_id]):
                error = f"Line {line_num}: Negative label IDs not allowed: {line}"
                results['errors'].append(error)
                continue
            
            # Check if video exists (optional)
            if check_videos and video_dir:
                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    results['missing_videos'].append(video_name)
            
            # Count label occurrences
            results['verb_ids'][verb_id] += 1
            results['manip_obj_ids'][manip_obj_id] += 1
            results['target_obj_ids'][target_obj_id] += 1
            results['tool_ids'][tool_id] += 1
            
            results['valid_lines'] += 1
    
    # Print summary
    print(f"📊 Summary:")
    print(f"  Total lines: {results['total_lines']}")
    print(f"  Valid lines: {results['valid_lines']}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\n❌ Errors found:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    # Print class statistics
    print(f"\n📈 Class Statistics:")
    print(f"  Verbs: {len(results['verb_ids'])} unique classes")
    print(f"    Range: {min(results['verb_ids'].keys()) if results['verb_ids'] else 'N/A'} - {max(results['verb_ids'].keys()) if results['verb_ids'] else 'N/A'}")
    print(f"    Most common: {results['verb_ids'].most_common(3)}")
    
    print(f"  Manipulated Objects: {len(results['manip_obj_ids'])} unique classes")
    print(f"    Range: {min(results['manip_obj_ids'].keys()) if results['manip_obj_ids'] else 'N/A'} - {max(results['manip_obj_ids'].keys()) if results['manip_obj_ids'] else 'N/A'}")
    print(f"    Most common: {results['manip_obj_ids'].most_common(3)}")
    
    print(f"  Target Objects: {len(results['target_obj_ids'])} unique classes")
    print(f"    Range: {min(results['target_obj_ids'].keys()) if results['target_obj_ids'] else 'N/A'} - {max(results['target_obj_ids'].keys()) if results['target_obj_ids'] else 'N/A'}")
    print(f"    Most common: {results['target_obj_ids'].most_common(3)}")
    
    print(f"  Tools: {len(results['tool_ids'])} unique classes")
    print(f"    Range: {min(results['tool_ids'].keys()) if results['tool_ids'] else 'N/A'} - {max(results['tool_ids'].keys()) if results['tool_ids'] else 'N/A'}")
    print(f"    Most common: {results['tool_ids'].most_common(3)}")
    
    # Check for class imbalance
    print(f"\n⚖️  Class Imbalance Check:")
    for name, counter in [
        ('Verb', results['verb_ids']),
        ('Manip Obj', results['manip_obj_ids']),
        ('Target Obj', results['target_obj_ids']),
        ('Tool', results['tool_ids'])
    ]:
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            least_common_count = counter.most_common()[-1][1]
            ratio = most_common_count / least_common_count if least_common_count > 0 else float('inf')
            
            if ratio > 10:
                print(f"  ⚠️  {name}: High imbalance (ratio: {ratio:.1f}x)")
            else:
                print(f"  ✓ {name}: Balanced (ratio: {ratio:.1f}x)")
    
    # Check for missing videos
    if check_videos:
        if results['missing_videos']:
            print(f"\n❌ Missing Videos: {len(results['missing_videos'])}")
            for vid in results['missing_videos'][:5]:
                print(f"  {vid}")
            if len(results['missing_videos']) > 5:
                print(f"  ... and {len(results['missing_videos']) - 5} more")
        else:
            print(f"\n✓ All videos found")
    
    # Final verdict
    if results['errors'] or (check_videos and results['missing_videos']):
        print(f"\n❌ Validation FAILED")
    else:
        print(f"\n✓ Validation PASSED")
    
    return results


def generate_training_config(lh_results, rh_results):
    """Generate recommended training configuration based on annotation statistics."""
    print(f"\n{'='*60}")
    print(f"Recommended Training Configuration")
    print(f"{'='*60}\n")
    
    # Get max class IDs (add 1 since IDs are 0-indexed)
    lh_num_verbs = max(lh_results['verb_ids'].keys()) + 1 if lh_results['verb_ids'] else 1
    lh_num_manip = max(lh_results['manip_obj_ids'].keys()) + 1 if lh_results['manip_obj_ids'] else 1
    lh_num_target = max(lh_results['target_obj_ids'].keys()) + 1 if lh_results['target_obj_ids'] else 1
    lh_num_tools = max(lh_results['tool_ids'].keys()) + 1 if lh_results['tool_ids'] else 1
    
    rh_num_verbs = max(rh_results['verb_ids'].keys()) + 1 if rh_results['verb_ids'] else 1
    rh_num_manip = max(rh_results['manip_obj_ids'].keys()) + 1 if rh_results['manip_obj_ids'] else 1
    rh_num_target = max(rh_results['target_obj_ids'].keys()) + 1 if rh_results['target_obj_ids'] else 1
    rh_num_tools = max(rh_results['tool_ids'].keys()) + 1 if rh_results['tool_ids'] else 1
    
    print("Add these parameters to your training script:\n")
    print(f"LH_NUM_VERBS={lh_num_verbs}")
    print(f"LH_NUM_MANIP_OBJS={lh_num_manip}")
    print(f"LH_NUM_TARGET_OBJS={lh_num_target}")
    print(f"LH_NUM_TOOLS={lh_num_tools}")
    print()
    print(f"RH_NUM_VERBS={rh_num_verbs}")
    print(f"RH_NUM_MANIP_OBJS={rh_num_manip}")
    print(f"RH_NUM_TARGET_OBJS={rh_num_target}")
    print(f"RH_NUM_TOOLS={rh_num_tools}")
    print()
    
    # Generate full command snippet
    print("Or use in Python script:")
    print(f"""
--lh_num_verbs {lh_num_verbs} \\
--lh_num_manip_objs {lh_num_manip} \\
--lh_num_target_objs {lh_num_target} \\
--lh_num_tools {lh_num_tools} \\
--rh_num_verbs {rh_num_verbs} \\
--rh_num_manip_objs {rh_num_manip} \\
--rh_num_target_objs {rh_num_target} \\
--rh_num_tools {rh_num_tools}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Validate compositional dual-hand annotation files"
    )
    parser.add_argument('--lh_train', type=str, required=True,
                       help='Path to left hand training annotation file')
    parser.add_argument('--lh_val', type=str, required=True,
                       help='Path to left hand validation annotation file')
    parser.add_argument('--rh_train', type=str, required=True,
                       help='Path to right hand training annotation file')
    parser.add_argument('--rh_val', type=str, required=True,
                       help='Path to right hand validation annotation file')
    parser.add_argument('--lh_video_dir', type=str, default=None,
                       help='Directory containing left hand videos (for video existence check)')
    parser.add_argument('--rh_video_dir', type=str, default=None,
                       help='Directory containing right hand videos (for video existence check)')
    parser.add_argument('--check_videos', action='store_true',
                       help='Check if video files actually exist')
    
    args = parser.parse_args()
    
    print("="*60)
    print("COMPOSITIONAL ANNOTATION VALIDATOR")
    print("="*60)
    
    # Validate all files
    lh_train_results = validate_annotation_file(
        args.lh_train, 
        check_videos=args.check_videos,
        video_dir=os.path.join(args.lh_video_dir, 'videos_train') if args.lh_video_dir else None
    )
    
    lh_val_results = validate_annotation_file(
        args.lh_val,
        check_videos=args.check_videos,
        video_dir=os.path.join(args.lh_video_dir, 'videos_val') if args.lh_video_dir else None
    )
    
    rh_train_results = validate_annotation_file(
        args.rh_train,
        check_videos=args.check_videos,
        video_dir=os.path.join(args.rh_video_dir, 'videos_train') if args.rh_video_dir else None
    )
    
    rh_val_results = validate_annotation_file(
        args.rh_val,
        check_videos=args.check_videos,
        video_dir=os.path.join(args.rh_video_dir, 'videos_val') if args.rh_video_dir else None
    )
    
    # Generate training config if all validations passed
    if all([lh_train_results, lh_val_results, rh_train_results, rh_val_results]):
        # Merge train and val results for each hand
        lh_merged = {
            'verb_ids': lh_train_results['verb_ids'] + lh_val_results['verb_ids'],
            'manip_obj_ids': lh_train_results['manip_obj_ids'] + lh_val_results['manip_obj_ids'],
            'target_obj_ids': lh_train_results['target_obj_ids'] + lh_val_results['target_obj_ids'],
            'tool_ids': lh_train_results['tool_ids'] + lh_val_results['tool_ids'],
        }
        rh_merged = {
            'verb_ids': rh_train_results['verb_ids'] + rh_val_results['verb_ids'],
            'manip_obj_ids': rh_train_results['manip_obj_ids'] + rh_val_results['manip_obj_ids'],
            'target_obj_ids': rh_train_results['target_obj_ids'] + rh_val_results['target_obj_ids'],
            'tool_ids': rh_train_results['tool_ids'] + rh_val_results['tool_ids'],
        }
        generate_training_config(lh_merged, rh_merged)
    
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)


if __name__ == '__main__':
    main()

