# RGB Feature Implementation - Code Changes Documentation

This document details all code changes made to support RGB features (in addition to XYZ coordinates) throughout the PointNet++ training and inference pipeline for rock joint classification.

## Overview

The implementation adds optional RGB support to the existing XYZ-only pipeline. When enabled, the model processes 6-channel input (XYZ+RGB) instead of 3-channel (XYZ only). RGB values are normalized to [0, 1] and remain unnormalized during data augmentation and coordinate normalization.

### Key Design Decisions

1. **RGB as optional feature**: The `use_rgb` flag allows toggling between 3-channel (XYZ) and 6-channel (XYZ+RGB) modes
2. **RGB normalization**: RGB values extracted from LAS files (16-bit, 0-65535) are normalized to [0, 1]
3. **Separate normalization**: XYZ coordinates are normalized (center/center_scale), but RGB values remain in [0, 1]
4. **Augmentation strategy**: Only XYZ coordinates are augmented (rotation, scaling, jittering); RGB values are preserved
5. **Backward compatibility**: All changes are backward compatible with XYZ-only workflows

---

## File-by-File Changes

### 1. `preprocessing_utils.py`

**Purpose**: Extract RGB from LAS files and save to preprocessed data

#### Function: `prepare_data_for_training()`

**Line 200-221**: Updated function signature and added RGB extraction logic

```python
# OLD
def prepare_data_for_training(las_file, no_joints_dxf, joints_dxf,
                              train_point_a, train_point_b):

# NEW
def prepare_data_for_training(las_file, no_joints_dxf, joints_dxf,
                              train_point_a, train_point_b, use_rgb=False):
```

**Lines 230-240**: Added RGB extraction after loading point cloud

```python
# ADDED
# Extract RGB if requested
rgb_array = None
if use_rgb:
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # LAS RGB values are typically 16-bit (0-65535), normalize to 0-1
        rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
        rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
        print(f"Extracted RGB colors (normalized to [0, 1])")
    else:
        print("WARNING: RGB requested but not available in LAS file. Proceeding without RGB.")
        use_rgb = False
```

**Line 302-303**: Updated return statement to include `rgb_array`

```python
# OLD
return (xyz_array, label_array, train_mask, test_mask,
        polygons_dict_train, polygons_dict_test)

# NEW
return (xyz_array, rgb_array, label_array, train_mask, test_mask,
        polygons_dict_train, polygons_dict_test)
```

**Lines 323-333**: Updated example usage to save RGB data

```python
# OLD
np.savez('preprocessed_data.npz',
         xyz_array=xyz_array,
         label_array=label_array,
         train_mask=train_mask,
         test_mask=test_mask)

# NEW
save_dict = {
    'xyz_array': xyz_array,
    'label_array': label_array,
    'train_mask': train_mask,
    'test_mask': test_mask
}
if rgb_array is not None:
    save_dict['rgb_array'] = rgb_array

np.savez('preprocessed_data.npz', **save_dict)
```

---

### 2. `dataset.py`

**Purpose**: Handle 6-channel (XYZ+RGB) patch extraction, normalization, and augmentation

#### Class: `RockJointDataset`

**Lines 17-36**: Updated `__init__()` to accept and store RGB data

```python
# OLD
def __init__(self, xyz_array, label_array, polygons_dict,
             patch_size=2048, normalize_mode='none', augment=False):

# NEW
def __init__(self, xyz_array, label_array, polygons_dict,
             patch_size=2048, normalize_mode='none', augment=False, rgb_array=None):
    """
    Args:
        ...
        rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
    """
    self.xyz_array = xyz_array
    self.rgb_array = rgb_array  # ADDED
    self.label_array = label_array
    self.polygons_dict = polygons_dict
    self.patch_size = patch_size
    self.normalize_mode = normalize_mode
    self.augment = augment
    self.use_rgb = rgb_array is not None  # ADDED
```

**Lines 45-48**: Updated dataset creation message

```python
# OLD
print(f"Dataset created with {len(self.patches)} patches")

# NEW
feature_dim = 6 if self.use_rgb else 3
print(f"Dataset created with {len(self.patches)} patches ({feature_dim}D features)")
```

#### Method: `_extract_patches_with_coherence()`

**Lines 311-315**: Extract RGB for each patch

```python
# Get points and labels
patch_points = self.xyz_array[indices]
patch_labels = self.label_array[indices]
if self.use_rgb:  # ADDED
    patch_rgb = self.rgb_array[indices]  # ADDED
```

**Lines 370-381**: Handle RGB during sampling/padding

```python
patch_points = patch_points[choice]
patch_labels = patch_labels[choice]
if self.use_rgb:  # ADDED
    patch_rgb = patch_rgb[choice]  # ADDED
```

```python
# Pad by repeating random points
n_repeat = self.patch_size - len(patch_points)
repeat_indices = np.random.choice(len(patch_points), n_repeat, replace=True)
patch_points = np.vstack([patch_points, patch_points[repeat_indices]])
patch_labels = np.concatenate([patch_labels, patch_labels[repeat_indices]])
if self.use_rgb:  # ADDED
    patch_rgb = np.vstack([patch_rgb, patch_rgb[repeat_indices]])  # ADDED
```

**Lines 383-390**: Concatenate XYZ and RGB features

```python
# ADDED
# Concatenate XYZ and RGB if using RGB
if self.use_rgb:
    patch_features = np.hstack([patch_points, patch_rgb])  # [patch_size, 6]
else:
    patch_features = patch_points  # [patch_size, 3]

# Store patch
self.patches.append(patch_features)
```

#### Method: `_normalize()`

**Lines 492-527**: Updated to handle 6D features (only normalize XYZ)

```python
# OLD
def _normalize(self, points):
    if self.normalize_mode == 'none':
        return points
    elif self.normalize_mode == 'center':
        centroid = points.mean(axis=0)
        points = points - centroid
        return points
    # ...

# NEW
def _normalize(self, points):
    """
    Apply normalization based on mode
    Note: Only normalizes XYZ coordinates, RGB remains in [0, 1]
    """
    if self.normalize_mode == 'none':
        return points

    # Separate XYZ and RGB if using RGB
    if self.use_rgb:
        xyz = points[:, :3]
        rgb = points[:, 3:]
    else:
        xyz = points

    if self.normalize_mode == 'center':
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
    elif self.normalize_mode == 'center_scale':
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.linalg.norm(xyz, axis=1))
        if max_dist > 0:
            xyz = xyz / max_dist
    else:
        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    # Concatenate back if using RGB
    if self.use_rgb:
        return np.hstack([xyz, rgb])
    else:
        return xyz
```

#### Method: `_augment()`

**Lines 529-575**: Updated to handle 6D features (only augment XYZ)

```python
# OLD
def _augment(self, points):
    # Random rotation around Z-axis
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([...])
    points = points @ rotation_matrix.T
    # ... scale, jitter, dropout ...
    return points

# NEW
def _augment(self, points):
    """
    Data augmentation: random rotation, scaling, jittering
    Note: Only augments XYZ coordinates, RGB remains unchanged
    """
    # Separate XYZ and RGB if using RGB
    if self.use_rgb:
        xyz = points[:, :3]
        rgb = points[:, 3:]
    else:
        xyz = points

    # Random rotation around Z-axis (vertical)
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([...])
    xyz = xyz @ rotation_matrix.T

    # Random scaling (90% to 110%)
    scale = np.random.uniform(0.9, 1.1)
    xyz = xyz * scale

    # Random jittering
    jitter = np.random.normal(0, 0.01, size=xyz.shape)
    xyz = xyz + jitter

    # Random point dropout
    if np.random.random() > 0.5:
        dropout_ratio = np.random.uniform(0, 0.1)
        n_keep = int(len(xyz) * (1 - dropout_ratio))
        keep_indices = np.random.choice(len(xyz), n_keep, replace=False)
        n_duplicate = len(xyz) - n_keep
        duplicate_indices = np.random.choice(keep_indices, n_duplicate, replace=True)
        all_indices = np.concatenate([keep_indices, duplicate_indices])
        xyz = xyz[all_indices]
        if self.use_rgb:
            rgb = rgb[all_indices]

    # Concatenate back if using RGB
    if self.use_rgb:
        return np.hstack([xyz, rgb])
    else:
        return xyz
```

#### Class: `RockJointInferenceDataset`

**Lines 582-596**: Updated `__init__()` to accept RGB

```python
# OLD
def __init__(self, xyz_array, patch_size=2048, stride=1024, normalize_mode='none'):

# NEW
def __init__(self, xyz_array, patch_size=2048, stride=1024, normalize_mode='none', rgb_array=None):
    """
    Args:
        ...
        rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
    """
    self.xyz_array = xyz_array
    self.rgb_array = rgb_array  # ADDED
    self.patch_size = patch_size
    self.stride = stride
    self.normalize_mode = normalize_mode
    self.use_rgb = rgb_array is not None  # ADDED
```

**Lines 601-602**: Updated dataset creation message

```python
# OLD
print(f"Inference dataset created with {len(self.patch_centers)} patches")

# NEW
feature_dim = 6 if self.use_rgb else 3
print(f"Inference dataset created with {len(self.patch_centers)} patches ({feature_dim}D features)")
```

#### Method: `__getitem__()` (Inference)

**Lines 661-714**: Updated to handle RGB features

```python
# OLD
def __getitem__(self, idx):
    center = self.patch_centers[idx]
    indices = self.patch_indices[idx]
    patch_points = self.xyz_array[indices]
    # ... sample/pad ...
    # Normalize
    if self.normalize_mode == 'center':
        centroid = patch_points.mean(axis=0)
        patch_points = patch_points - centroid
    # ...
    return torch.FloatTensor(patch_points), torch.LongTensor(selected_indices)

# NEW
def __getitem__(self, idx):
    center = self.patch_centers[idx]
    indices = self.patch_indices[idx]

    # Get points
    patch_points = self.xyz_array[indices]
    if self.use_rgb:
        patch_rgb = self.rgb_array[indices]

    # Sample or pad to patch_size
    if len(patch_points) >= self.patch_size:
        choice = np.random.choice(len(patch_points), self.patch_size, replace=False)
        selected_indices = np.array(indices)[choice]
        patch_points = patch_points[choice]
        if self.use_rgb:
            patch_rgb = patch_rgb[choice]
    else:
        n_repeat = self.patch_size - len(patch_points)
        repeat_choice = np.random.choice(len(patch_points), n_repeat, replace=True)
        selected_indices = np.array(list(indices) + [indices[i] for i in repeat_choice])
        patch_points = np.vstack([patch_points, patch_points[repeat_choice]])
        if self.use_rgb:
            patch_rgb = np.vstack([patch_rgb, patch_rgb[repeat_choice]])

    # Separate XYZ and RGB if using RGB
    if self.use_rgb:
        xyz = patch_points
        rgb = patch_rgb
    else:
        xyz = patch_points

    # Normalize (same as training) - only XYZ
    if self.normalize_mode == 'center':
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
    elif self.normalize_mode == 'center_scale':
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.linalg.norm(xyz, axis=1))
        if max_dist > 0:
            xyz = xyz / max_dist

    # Concatenate XYZ and RGB if using RGB
    if self.use_rgb:
        patch_features = np.hstack([xyz, rgb])
    else:
        patch_features = xyz

    return torch.FloatTensor(patch_features), torch.LongTensor(selected_indices)
```

---

### 3. `pointnet2_model.py`

**Purpose**: Model architecture already supports configurable `input_channels`

**No changes required** - The existing model architecture already accepts `input_channels` parameter:

```python
class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=2, input_channels=3):  # Already configurable!
        super().__init__()
        self.input_channels = input_channels
        # ...
```

The first Set Abstraction layer uses `input_channels` directly:

```python
self.sa1 = PointNetSetAbstraction(
    npoint=512, radius=0.2, nsample=32,
    in_channel=input_channels,  # Uses the parameter
    mlp=[64, 64, 128], group_all=False
)
```

---

### 4. `train_end2end.py`

**Purpose**: Integrate RGB through the end-to-end training pipeline

**Lines 43-45**: Added `use_rgb` configuration parameter

```python
# OLD
'input_channels': 3,  # XYZ only (change to 6 for XYZ+RGB)

# NEW
'use_rgb': True,      # Whether to use RGB features
'input_channels': 6,  # Will be set to 3 or 6 based on use_rgb
```

**Lines 114-127**: Load RGB from preprocessed data

```python
# OLD
data = np.load(preprocessed_path)
xyz_array = data['xyz_array']
label_array = data['label_array']
train_mask = data['train_mask']
test_mask = data['test_mask']

# NEW
data = np.load(preprocessed_path)
xyz_array = data['xyz_array']
label_array = data['label_array']
train_mask = data['train_mask']
test_mask = data['test_mask']
rgb_array = data['rgb_array'] if 'rgb_array' in data else None  # ADDED

print(f"Loaded point cloud: {len(xyz_array):,} points")
if rgb_array is not None:  # ADDED
    print(f"RGB features available: Yes")
else:
    print(f"RGB features available: No")
```

**Lines 144-152**: Call preprocessing with `use_rgb` parameter

```python
# OLD
(xyz_array, label_array, train_mask, test_mask,
 polygons_dict_train, polygons_dict_test) = prepare_data_for_training(
    las_file=config['las_file'],
    no_joints_dxf=config['no_joints_dxf'],
    joints_dxf=config['joints_dxf'],
    train_point_a=point_a,
    train_point_b=point_b
)

# NEW
(xyz_array, rgb_array, label_array, train_mask, test_mask,
 polygons_dict_train, polygons_dict_test) = prepare_data_for_training(
    las_file=config['las_file'],
    no_joints_dxf=config['no_joints_dxf'],
    joints_dxf=config['joints_dxf'],
    train_point_a=point_a,
    train_point_b=point_b,
    use_rgb=config['use_rgb']  # ADDED
)
```

**Lines 155-163**: Save RGB to preprocessed data

```python
# OLD
np.savez(preprocessed_path,
         xyz_array=xyz_array,
         label_array=label_array,
         train_mask=train_mask,
         test_mask=test_mask)

# NEW
save_dict = {
    'xyz_array': xyz_array,
    'label_array': label_array,
    'train_mask': train_mask,
    'test_mask': test_mask
}
if rgb_array is not None:
    save_dict['rgb_array'] = rgb_array
np.savez(preprocessed_path, **save_dict)
```

**Lines 191-225**: Set `input_channels` dynamically and pass RGB to datasets

```python
# ADDED
# Update input_channels based on RGB availability
if rgb_array is not None and config['use_rgb']:
    config['input_channels'] = 6
    train_rgb = rgb_array[train_mask]
    test_rgb = rgb_array[test_mask]
else:
    config['input_channels'] = 3
    train_rgb = None
    test_rgb = None
    if config['use_rgb']:
        print("\nWARNING: RGB requested but not available. Using XYZ only.")

print(f"\nUsing {config['input_channels']}-channel input (XYZ{'+RGB' if config['input_channels'] == 6 else ''})")

print("\nCreating training dataset...")
train_dataset = RockJointDataset(
    xyz_array=xyz_array[train_mask],
    label_array=label_array[train_mask],
    polygons_dict=polygons_dict_train,
    patch_size=config['patch_size'],
    normalize_mode=config['normalize_mode'],
    augment=config['augment_train'],
    rgb_array=train_rgb  # ADDED
)

print("\nCreating test dataset...")
test_dataset = RockJointDataset(
    xyz_array=xyz_array[test_mask],
    label_array=label_array[test_mask],
    polygons_dict=polygons_dict_test,
    patch_size=config['patch_size'],
    normalize_mode=config['normalize_mode'],
    augment=False,
    rgb_array=test_rgb  # ADDED
)
```

---

### 5. `inference.py`

**Purpose**: Use RGB features during inference on full point clouds

#### Class: `PointCloudInference`

**Lines 49-71**: Updated `predict_point_cloud()` to accept RGB

```python
# OLD
def predict_point_cloud(self, xyz_array, batch_size=16):

# NEW
def predict_point_cloud(self, xyz_array, rgb_array=None, batch_size=16):
    """
    Args:
        xyz_array: Full point cloud [N, 3]
        rgb_array: Optional RGB colors [N, 3]
        batch_size: Batch size for inference
    """
    # ...
    inference_dataset = RockJointInferenceDataset(
        xyz_array=xyz_array,
        rgb_array=rgb_array,  # ADDED
        patch_size=self.config['patch_size'],
        stride=self.config['stride'],
        normalize_mode=self.config['normalize_mode']
    )
```

#### Function: `main()`

**Lines 240-242**: Added `use_rgb` to config

```python
# OLD
'num_classes': 2,
'input_channels': 3,

# NEW
'num_classes': 2,
'use_rgb': True,       # Whether to use RGB features (must match training)
'input_channels': 6,   # Will be set to 3 or 6 based on use_rgb
```

**Lines 257-277**: Extract RGB from LAS file

```python
# OLD
print("Loading point cloud...")
las = laspy.read(config['input_las'])
xyz_array = np.vstack([las.x, las.y, las.z]).transpose()
print(f"Loaded {len(xyz_array)} points")

# NEW
print("Loading point cloud...")
las = laspy.read(config['input_las'])
xyz_array = np.vstack([las.x, las.y, las.z]).transpose()
print(f"Loaded {len(xyz_array)} points")

# Extract RGB if requested
rgb_array = None
if config['use_rgb']:
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
        rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
        print(f"Extracted RGB colors")
        config['input_channels'] = 6
    else:
        print("WARNING: RGB requested but not available in LAS file. Using XYZ only.")
        config['use_rgb'] = False
        config['input_channels'] = 3
else:
    config['input_channels'] = 3

print(f"Using {config['input_channels']}-channel input (XYZ{'+RGB' if config['input_channels'] == 6 else ''})")
```

**Lines 290-294**: Pass RGB to prediction

```python
# OLD
predictions, probabilities = inference_engine.predict_point_cloud(
    xyz_array=xyz_array,
    batch_size=config['batch_size']
)

# NEW
predictions, probabilities = inference_engine.predict_point_cloud(
    xyz_array=xyz_array,
    rgb_array=rgb_array,  # ADDED
    batch_size=config['batch_size']
)
```

---

### 6. `train.py`

**Purpose**: Training loop (Trainer class)

**No changes required** - The `Trainer` class operates on DataLoaders that provide batches of patches. Since the dataset classes now return 6-channel tensors when RGB is enabled, the training loop automatically handles this without modification.

---

## Summary of Changes

### Files Modified
1. ✅ `preprocessing_utils.py` - Extract and save RGB from LAS
2. ✅ `dataset.py` - Handle 6-channel patches, normalization, augmentation
3. ✅ `train_end2end.py` - Pass RGB through training pipeline
4. ✅ `inference.py` - Use RGB during inference
5. ✅ `pointnet2_model.py` - No changes (already supports `input_channels`)
6. ✅ `train.py` - No changes (operates on DataLoader batches)

### Key Configuration Changes

#### To enable RGB features:

In `train_end2end.py`:
```python
config = {
    'use_rgb': True,          # Enable RGB
    'input_channels': 6,      # Will be set automatically
    # ... other settings
}
```

In `inference.py`:
```python
config = {
    'use_rgb': True,          # Enable RGB (must match training)
    'input_channels': 6,      # Will be set automatically
    # ... other settings
}
```

#### To use XYZ only (original behavior):

Set `'use_rgb': False` in both config dictionaries.

---

## Testing Checklist

- [ ] Test preprocessing with RGB extraction
- [ ] Verify RGB normalization to [0, 1] range
- [ ] Test training with 6-channel input
- [ ] Test training with 3-channel input (XYZ only)
- [ ] Verify augmentation only affects XYZ, not RGB
- [ ] Verify normalization only affects XYZ, not RGB
- [ ] Test inference with RGB features
- [ ] Test inference with XYZ only
- [ ] Verify backward compatibility with existing preprocessed data
- [ ] Check that model `input_channels` matches data dimensionality

---

## Implementation Notes

### RGB Normalization

LAS files typically store RGB as 16-bit unsigned integers (0-65535). The implementation normalizes these to [0, 1]:

```python
rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
```

### Feature Concatenation

Features are always concatenated as `[XYZ, RGB]`:
- XYZ: 3 channels (columns 0-2)
- RGB: 3 channels (columns 3-5)

```python
patch_features = np.hstack([xyz, rgb])  # Shape: [N, 6]
```

### Backward Compatibility

All changes maintain backward compatibility:
- `rgb_array=None` defaults ensure XYZ-only mode
- `use_rgb=False` flag preserves original behavior
- Loading preprocessed data without RGB falls back gracefully

---

## Additional Feature: Spatial Subsampling for Validation

### Problem
The validation region may have denser polygon annotations than the training region, leading to many overlapping patches and an imbalanced number of training vs. validation steps.

### Solution
Added `min_patch_distance` parameter to `RockJointDataset` that enforces a minimum spatial distance between patch centers, preventing redundant overlapping patches.

### Implementation

**dataset.py - Lines 17-40**: Added `min_patch_distance` parameter

```python
def __init__(self, xyz_array, label_array, polygons_dict,
             patch_size=2048, normalize_mode='none', augment=False, rgb_array=None,
             min_patch_distance=None):
    """
    Args:
        ...
        min_patch_distance: Minimum distance between patch centers (for spatial subsampling)
                           If None, no spatial subsampling is applied
    """
    self.min_patch_distance = min_patch_distance
```

**dataset.py - Lines 289-328**: Spatial subsampling logic during patch extraction

```python
# Track patch centers for spatial subsampling
accepted_patch_centers = []
accepted_tree = None

for idx, center_idx in enumerate(center_indices):
    center = self.xyz_array[center_idx]

    # Check spatial subsampling constraint
    if self.min_patch_distance is not None and len(accepted_patch_centers) > 0:
        # Rebuild tree if we have new patches
        if accepted_tree is None or len(accepted_patch_centers) % 100 == 0:
            accepted_tree = KDTree(np.array(accepted_patch_centers))

        # Check distance to nearest accepted patch
        dist, _ = accepted_tree.query(center)
        if dist < self.min_patch_distance:
            skipped_too_close += 1
            continue

    # ... extract patch ...

    # Track this patch center for spatial subsampling
    if self.min_patch_distance is not None:
        accepted_patch_centers.append(center)
```

**train_end2end.py - Lines 51-52**: Configuration parameter

```python
'test_min_patch_distance': 1.5,  # Minimum distance between test patch centers (in meters)
                                  # Set to None to disable spatial subsampling
```

**train_end2end.py - Lines 219-228**: Apply to test dataset

```python
test_dataset = RockJointDataset(
    xyz_array=xyz_array[test_mask],
    label_array=label_array[test_mask],
    polygons_dict=polygons_dict_test,
    patch_size=config['patch_size'],
    normalize_mode=config['normalize_mode'],
    augment=False,
    rgb_array=test_rgb,
    min_patch_distance=config.get('test_min_patch_distance', None)  # ADDED
)
```

### Usage

To reduce validation patches, adjust `test_min_patch_distance`:

```python
config = {
    'test_min_patch_distance': 1.5,  # Larger value = fewer patches, more spacing
    # Set to None to disable and use all patches
}
```

**Recommended values:**
- `None`: No subsampling (use all patches)
- `0.5 - 1.0`: Light subsampling (slight reduction)
- `1.0 - 2.0`: Moderate subsampling (balanced coverage and speed)
- `2.0+`: Aggressive subsampling (minimal patches, fast validation)

### Benefits

1. **Faster validation**: Fewer validation steps per epoch
2. **Non-random subsampling**: Maintains spatial coverage across the validation region
3. **Prevents redundancy**: Avoids heavily overlapping patches
4. **Configurable**: Easy to tune the trade-off between coverage and speed

---

## Future Enhancements

Potential improvements for future iterations:

1. **RGB augmentation**: Add color jittering, brightness/contrast adjustments
2. **Alternative color spaces**: Support HSV, LAB color representations
3. **Feature scaling**: Experiment with different RGB normalization ranges
4. **Feature ablation**: Analyze RGB contribution to classification performance
5. **Multi-modal fusion**: Experiment with separate feature extractors for XYZ and RGB
6. **Adaptive subsampling**: Automatically determine optimal `min_patch_distance` based on point density

---

**Document Version**: 1.1
**Last Updated**: 2025-10-31
**Author**: Claude (Anthropic)
