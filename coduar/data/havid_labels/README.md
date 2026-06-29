# HAVID Compositional Action Labels

This directory contains the compositional decomposition of HAVID action labels.

## 📊 Summary Statistics

- **Total Actions**: 74 classes
- **Verbs**: 6 unique classes
- **Manipulated Objects**: 25 unique classes
- **Target Objects**: 25 unique classes
- **Tools**: 5 unique classes

## 📁 Files

### 1. `havid_compositional_mapping.txt`
Complete mapping from HAVID action class ID to compositional element IDs.

**Format**: `class_id label verb_id manip_obj_id target_obj_id tool_id`

**Example**:
```
0 ibacb 1 1 8 0
# Class 0 (ibacb) = verb:i(1), manip:ba(1), target:cb(8), tool:null(0)
```

### 2. `label_map_verbs.txt`
Mapping of verb IDs to verb codes.

**Verbs** (6 total):
- `0`: null (no action)
- `1`: i (insert/in)
- `2`: l (lift/load)
- `3`: p (place/put)
- `4`: r (remove)
- `5`: s (screw/secure)

### 3. `label_map_manip_objs.txt`
Mapping of manipulated object IDs to object codes.

**Key Objects** (25 total, selected examples):
- `0`: null (no object)
- `1`: ba (battery)
- `4`: cb (cable)
- `5`: cc (cable clip)
- `6`: ck (cable keeper)
- `22`: sh (shaft)
- `23`: sp (spring)

### 4. `label_map_target_objs.txt`
Mapping of target object IDs to object codes.

**Key Targets** (25 total, selected examples):
- `0`: null (no target)
- `4-7`: c1, c2, c3, c4 (compartments 1-4)
- `8`: cb (cable)
- `12-14`: g1, g2, g3 (gears 1-3)
- `17-23`: n1-n6 (nuts 1-6)

### 5. `label_map_tools.txt`
Mapping of tool IDs to tool codes.

**Tools** (5 total):
- `0`: null (no tool/hand only)
- `1`: dh (drill handle)
- `2`: dp (drill power)
- `3`: wn (wrench)
- `4`: ws (wrench socket)

## 🔤 Label Structure

HAVID labels follow the pattern: `[verb][manip_obj][target_obj][tool]`

Where:
- Position 1: **Verb** (1 character)
- Positions 2-3: **Manipulated Object** (2 characters)
- Positions 4-5: **Target Object** (2 characters, optional)
- Positions 6-7: **Tool** (2 characters, optional)

### Examples:

1. **`ibacb`** (Insert battery into cable)
   - `i` = insert (verb)
   - `ba` = battery (manipulated object)
   - `cb` = cable (target object)
   - (no tool)
   - **Compositional**: verb=1, manip=1, target=8, tool=0

2. **`sshc1dh`** (Screw shaft to compartment 1 using drill handle)
   - `s` = screw (verb)
   - `sh` = shaft (manipulated object)
   - `c1` = compartment 1 (target object)
   - `dh` = drill handle (tool)
   - **Compositional**: verb=5, manip=22, target=4, tool=1

3. **`lck`** (Lift cable keeper)
   - `l` = lift (verb)
   - `ck` = cable keeper (manipulated object)
   - (no target)
   - (no tool)
   - **Compositional**: verb=2, manip=6, target=0, tool=0

4. **`null`** (No action)
   - Special case for no action
   - **Compositional**: verb=0, manip=0, target=0, tool=0

## 🔄 Converting HAVID Annotations

To convert your HAVID annotations from single-label to compositional format:

```bash
cd /home/hao/Polyphony/VideoMAEv2/compositional_dual_hand

# Convert both left and right hand annotations
python convert_havid_to_compositional.py \
    --lh_data_dir /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_dir /home/hao/Polyphony/data/havid_mmaction/rh_v0

# This converts:
#   train_list_video.txt → train_list_compositional.txt
#   val_list_video.txt → val_list_compositional.txt
```

### Before (Single-Label):
```
video001.mp4 0
video002.mp4 25
video003.mp4 47
```

### After (Compositional):
```
video001.mp4 1 1 8 0
video002.mp4 2 6 0 0
video003.mp4 5 8 12 0
```

## 🎯 Training Configuration

When training the compositional model with HAVID dataset, use these class counts:

```bash
# For both left and right hands
LH_NUM_VERBS=6
LH_NUM_MANIP_OBJS=25
LH_NUM_TARGET_OBJS=25
LH_NUM_TOOLS=5

RH_NUM_VERBS=6
RH_NUM_MANIP_OBJS=25
RH_NUM_TARGET_OBJS=25
RH_NUM_TOOLS=5
```

## 📝 Usage Example

```bash
# 1. Parse HAVID labels (already done, generated these files)
python parse_havid_labels.py

# 2. Convert annotations
python convert_havid_to_compositional.py \
    --lh_data_dir /home/hao/Polyphony/data/havid_mmaction/lh_v0 \
    --rh_data_dir /home/hao/Polyphony/data/havid_mmaction/rh_v0

# 3. Validate
./run.sh validate \
    --lh_train /home/hao/Polyphony/data/havid_mmaction/lh_v0/train_list_compositional.txt \
    --lh_val /home/hao/Polyphony/data/havid_mmaction/lh_v0/val_list_compositional.txt \
    --rh_train /home/hao/Polyphony/data/havid_mmaction/rh_v0/train_list_compositional.txt \
    --rh_val /home/hao/Polyphony/data/havid_mmaction/rh_v0/val_list_compositional.txt

# 4. Update training script and train
# Edit scripts/train_compositional_dual_hands.sh
./run.sh train
```

## 📊 Action Distribution

### Most Common Actions (by verb):
- **Insert (i)**: 24 actions - Various insertion tasks with different objects
- **Screw (s)**: 29 actions - Assembly tasks with screws, many with tools
- **Place (p)**: 11 actions - Placing objects in compartments
- **Lift (l)**: 2 actions - Lifting operations
- **Remove (r)**: 4 actions - Removal operations
- **Null**: 1 action - No action state

### Tool Usage:
- **No tool (hand only)**: 64 actions (86.5%)
- **Drill handle (dh)**: 4 actions (5.4%)
- **Drill power (dp)**: 4 actions (5.4%)
- **Wrench (wn)**: 3 actions (4.1%)
- **Wrench socket (ws)**: 2 actions (2.7%)

Note: Some actions use multiple tools in different variants (e.g., `sftg1` vs `sftg1ws`)

## 🔍 Verification

To verify the compositional decomposition:

```python
import sys
sys.path.insert(0, '..')

# Load mapping
mapping = {}
with open('havid_compositional_mapping.txt') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        label = parts[1]
        verb, manip, target, tool = map(int, parts[2:6])
        mapping[class_id] = (label, verb, manip, target, tool)

# Check a specific action
class_id = 0  # ibacb
label, verb, manip, target, tool = mapping[class_id]
print(f"Class {class_id}: {label}")
print(f"  Verb: {verb}, Manip: {manip}, Target: {target}, Tool: {tool}")
```

## 📚 References

- Original HAVID task mapping: `/home/hao/Polyphony/data/havid_mmaction/task_mapping.txt`
- Parser script: `../parse_havid_labels.py`
- Converter script: `../convert_havid_to_compositional.py`
- Training script: `../scripts/train_compositional_dual_hands.sh`

