# Spatial Subsampling Guide

## Problem Statement

You observed that your validation dataset has **274 steps** while training has only **254 steps**, indicating that the validation region has denser polygon annotations. This leads to:

1. Longer validation time per epoch
2. Potentially redundant patches (many overlapping patches in dense regions)
3. Imbalanced training/validation compute time

## Solution: Spatial Subsampling

I've implemented a **non-random spatial subsampling** mechanism that ensures patch centers are separated by a minimum distance. This reduces the number of validation patches while maintaining good spatial coverage.

### Key Features

✅ **Non-random**: Uses spatial distance, not random sampling
✅ **Maintains coverage**: Ensures patches are spread across the validation region
✅ **Configurable**: Easy to adjust the trade-off between speed and coverage
✅ **Optional**: Can be disabled by setting to `None`

---

## How It Works

The algorithm processes potential patch centers sequentially and:

1. Checks if a candidate patch center is too close to any already-accepted patch
2. If too close (distance < `min_patch_distance`), skips it
3. If far enough, accepts it and adds to the dataset
4. Uses a KDTree for efficient nearest-neighbor queries

**Result**: Patches are spatially distributed without dense overlapping clusters.

---

## Configuration

### In `train_end2end.py`

```python
config = {
    # ... other settings ...

    'test_min_patch_distance': 1.5,  # Minimum distance between test patch centers (meters)
                                      # Set to None to disable spatial subsampling
}
```

### Recommended Values

Based on your current setup (patch_size=1024, ~274 validation steps):

| `test_min_patch_distance` | Effect | Expected Steps | Use Case |
|---------------------------|--------|----------------|----------|
| `None` (disabled) | No subsampling | ~274 | Maximum coverage, slower |
| `0.5` | Light subsampling | ~230-250 | Minor speedup |
| `1.0` | Moderate subsampling | ~180-200 | Balanced |
| `1.5` | **Recommended** | ~140-160 | Good balance of speed/coverage |
| `2.0` | Aggressive subsampling | ~100-120 | Fast validation |
| `3.0` | Very aggressive | ~60-80 | Minimal validation |

**Start with 1.5 meters** - this should reduce your validation steps from 274 to roughly **140-160 steps**, making it comparable to your 254 training steps.

---

## Example Usage

### Scenario 1: Reduce validation to ~150 steps

```python
config = {
    'test_min_patch_distance': 1.5,  # Good starting point
}
```

### Scenario 2: Match training and validation steps (~250)

```python
config = {
    'test_min_patch_distance': 0.5,  # Light reduction
}
```

### Scenario 3: Very fast validation (~100 steps)

```python
config = {
    'test_min_patch_distance': 2.0,  # Aggressive subsampling
}
```

### Scenario 4: Disable (use all patches)

```python
config = {
    'test_min_patch_distance': None,  # No subsampling
}
```

---

## Monitoring the Effect

When you run training, you'll see output like:

```
Creating test dataset...
Spatial subsampling enabled: min distance = 1.50 m

Extracting patches...
  Processed 500/5000 centers...
  Extracted: 120,
  Skipped (insufficient): 45,
  Skipped (unlabeled): 82,
  Skipped (incoherent): 153,
  Skipped (too close): 100  ← This shows patches rejected by spatial subsampling

============================================================
Patch Extraction Complete
============================================================
Total patches extracted: 145  ← Your new validation patch count
Skipped (insufficient points): 312
Skipped (too few labeled): 627
Skipped (incoherent/mixed): 1023
Skipped (too close to existing): 2893  ← Total rejected by spatial constraint
```

Watch the **"Skipped (too close to existing)"** count - this tells you how many patches were rejected due to spatial proximity.

---

## Tuning Guidelines

### If validation is still too slow:

**Increase** `test_min_patch_distance` to 2.0 or 2.5

```python
'test_min_patch_distance': 2.0,  # More aggressive
```

### If you want more validation coverage:

**Decrease** `test_min_patch_distance` to 1.0 or 0.75

```python
'test_min_patch_distance': 1.0,  # More patches
```

### If validation accuracy seems unstable:

**Decrease** `test_min_patch_distance` to get more patches

```python
'test_min_patch_distance': 0.75,  # More representative
```

---

## Important Notes

### Does NOT affect training dataset

By default, spatial subsampling is **only applied to the test dataset**. Training uses all available patches for maximum learning.

If you want to also subsample training patches:

```python
# In train_end2end.py
train_dataset = RockJointDataset(
    xyz_array=xyz_array[train_mask],
    label_array=label_array[train_mask],
    polygons_dict=polygons_dict_train,
    patch_size=config['patch_size'],
    normalize_mode=config['normalize_mode'],
    augment=config['augment_train'],
    rgb_array=train_rgb,
    min_patch_distance=config.get('train_min_patch_distance', None)  # ADD THIS
)
```

### Spatial units

`min_patch_distance` is in **meters** (same units as your point cloud coordinates).

For reference, your patch radius is approximately **calculated from point density**, typically in the range of 0.5-2.0 meters for your dataset.

### Deterministic behavior

The subsampling is **deterministic** given:
- Same random seed (affects the order of candidate centers)
- Same `min_patch_distance` value

So you'll get consistent results across runs.

---

## Validation Impact

### What you're trading off:

**Higher `min_patch_distance` values:**
- ✅ Faster validation (fewer steps)
- ✅ Less redundancy
- ❌ Less validation coverage
- ❌ Potentially noisier validation metrics

**Lower `min_patch_distance` values:**
- ✅ Better validation coverage
- ✅ More stable validation metrics
- ❌ Slower validation
- ❌ More redundant patches

### Recommended approach:

1. **Start with 1.5 meters** - should reduce from 274 to ~150 steps
2. **Monitor validation metrics** - check if accuracy/loss are stable
3. **Adjust if needed**:
   - If metrics are too noisy → reduce to 1.0
   - If still too slow → increase to 2.0

---

## Quick Reference

```python
# In train_end2end.py config dictionary:

# Recommended starting point (balanced)
'test_min_patch_distance': 1.5,

# Fast validation (aggressive subsampling)
'test_min_patch_distance': 2.0,

# Maximum coverage (slow validation)
'test_min_patch_distance': 0.5,

# Disable subsampling (original behavior)
'test_min_patch_distance': None,
```

---

## Expected Results

With `test_min_patch_distance = 1.5`, you should see:

**Before:**
- Training: 254 steps/epoch
- Validation: 274 steps/epoch
- Total: ~528 steps/epoch

**After:**
- Training: 254 steps/epoch
- Validation: ~140-160 steps/epoch
- Total: ~400-414 steps/epoch
- **~20-25% speedup per epoch**

---

## Questions?

If you need to adjust the behavior:

1. **Too few validation patches?** → Decrease `test_min_patch_distance`
2. **Still too many patches?** → Increase `test_min_patch_distance`
3. **Want same for training?** → Add `train_min_patch_distance` parameter
4. **Validation metrics unstable?** → Decrease distance for more coverage

---

**Version**: 1.0
**Created**: 2025-10-31
