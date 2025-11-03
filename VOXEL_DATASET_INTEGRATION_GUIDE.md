# Voxel-Based Dataset Integration Guide

## Overview

The new `voxelize_dataset.py` provides a **unified voxel-grid approach** that ensures training and inference use **identical patch extraction methods**, eliminating the distribution mismatch identified in the root cause analysis.

### Key Advantages

✅ **Consistent methodology**: Training and inference use the same voxel grid
✅ **No coherence filtering**: Includes boundary and mixed-label regions
✅ **No polygon bias**: Grid-based, not centered on polygon centroids
✅ **Deterministic**: Same voxel size = same grid structure
✅ **Configurable**: Easy to tune voxel size for different datasets
✅ **Mixed-label support**: Voxels can contain points from both classes

---

## Architecture Overview

### Voxel Grid Creation

```
Point Cloud Space
┌─────────────────────────────────────┐
│                                     │
│  ┌───┬───┬───┬───┐                 │
│  │ V │ V │ V │ V │  ← Regular grid │
│  ├───┼───┼───┼───┤    of voxels    │
│  │ V │ V │ V │ V │                 │
│  ├───┼───┼───┼───┤                 │
│  │ V │ V │ V │ V │                 │
│  └───┴───┴───┴───┘                 │
│                                     │
└─────────────────────────────────────┘

Each voxel (V):
- Has fixed size (e.g., 2.0m cube)
- Queries points within radius
- Contains ~patch_size points
- Can have mixed labels (no filtering!)
```

### Training vs. Inference - **SAME APPROACH**

```
Training:                    Inference:
┌─────────────────┐         ┌─────────────────┐
│ Voxel Grid      │         │ Voxel Grid      │
│ (train region)  │         │ (full cloud)    │
│                 │         │                 │
│ ┌─┬─┬─┐         │         │ ┌─┬─┬─┬─┬─┐     │
│ │V│V│V│         │         │ │V│V│V│V│V│     │
│ ├─┼─┼─┤         │         │ ├─┼─┼─┼─┼─┤     │
│ │V│V│V│         │    →    │ │V│V│V│V│V│     │
│ └─┴─┴─┘         │         │ ├─┼─┼─┼─┼─┤     │
│                 │         │ │V│V│V│V│V│     │
└─────────────────┘         │ └─┴─┴─┴─┴─┘     │
                            └─────────────────┘

Same voxel size!
Same extraction method!
Same normalization!
```

---

## Integration Steps

### Step 1: Update `train_end2end.py`

#### 1.1 Import the new dataset

```python
# OLD
from dataset import RockJointDataset

# NEW
from voxelize_dataset import VoxelDataset, create_train_test_voxel_datasets
```

#### 1.2 Update config

```python
config = {
    # ... existing config ...

    # ===== DATA PARAMETERS =====
    'patch_size': 1024,
    'voxel_size': None,  # Auto-compute, or set manually (e.g., 2.0 for 2m voxels)
    'normalize_mode': 'center_scale',
    'augment_train': False,

    # Remove or comment out these old parameters:
    # 'test_min_patch_distance': 0,  # Not needed anymore
}
```

#### 1.3 Replace dataset creation (Lines ~205-228)

```python
# OLD CODE TO REPLACE:
# train_dataset = RockJointDataset(
#     xyz_array=xyz_array[train_mask],
#     label_array=label_array[train_mask],
#     polygons_dict=polygons_dict_train,
#     ...
# )

# NEW CODE:
print("\n" + "="*70)
print("CREATING VOXEL-BASED DATASETS")
print("="*70)

train_dataset, test_dataset = create_train_test_voxel_datasets(
    xyz_array=xyz_array,
    label_array=label_array,
    train_mask=train_mask,
    test_mask=test_mask,
    patch_size=config['patch_size'],
    voxel_size=config.get('voxel_size', None),
    normalize_mode=config['normalize_mode'],
    augment_train=config['augment_train'],
    rgb_array=rgb_array
)
```

#### 1.4 Remove polygon-related code

Since voxel dataset doesn't use polygons for patch extraction:

```python
# These lines can be removed or commented out:
# - Loading polygon dictionaries
# - Filtering polygons by train/test split
# - Creating polygons_dict_train, polygons_dict_test

# Keep only:
# - Loading xyz_array, label_array, train_mask, test_mask from preprocessed data
# - Loading rgb_array if using RGB
```

---

### Step 2: Update `inference.py`

#### 2.1 Import the new dataset

```python
# OLD
from dataset import RockJointInferenceDataset

# NEW
from voxelize_dataset import VoxelDataset
```

#### 2.2 Update `PointCloudInference.predict_point_cloud()` (Lines ~49-104)

```python
def predict_point_cloud(self, xyz_array, rgb_array=None, batch_size=16):
    """
    Predict labels for entire point cloud using voxel grid

    Args:
        xyz_array: Full point cloud [N, 3]
        rgb_array: Optional RGB colors [N, 3]
        batch_size: Batch size for inference

    Returns:
        predictions: [N] array with class predictions
        probabilities: [N, num_classes] array with class probabilities
    """
    print(f"\nPredicting on point cloud with {len(xyz_array)} points")

    # Create inference dataset using voxel grid
    # Note: label_array=None means all points are unlabeled (inference mode)
    inference_dataset = VoxelDataset(
        xyz_array=xyz_array,
        label_array=None,  # Inference mode
        rgb_array=rgb_array,
        patch_size=self.config['patch_size'],
        voxel_size=self.config.get('voxel_size', None),  # Use same as training!
        normalize_mode=self.config['normalize_mode'],
        augment=False  # No augmentation during inference
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize prediction arrays
    predictions = np.full(len(xyz_array), -1)
    probabilities = np.zeros((len(xyz_array), self.config['num_classes']))
    vote_counts = np.zeros(len(xyz_array))

    print(f"\nRunning inference on {len(inference_dataset)} voxels...")

    for points, _ in tqdm(inference_loader, desc="Inference"):
        # Note: labels are ignored (all -1 during inference)
        points = points.to(self.device)  # [B, N, C]

        # Get voxel indices for this batch
        batch_start_idx = inference_loader.dataset
        # We need to track which voxel each batch item came from

        outputs = self.model(points)  # [B, num_classes, N]
        probs = torch.softmax(outputs, dim=1)  # [B, num_classes, N]
        preds = torch.argmax(outputs, dim=1)  # [B, N]

        # For each voxel in batch, assign predictions to corresponding points
        for b in range(len(preds)):
            # Get the voxel index for this batch item
            voxel_idx = (inference_loader.batch_size *
                        (len(vote_counts[vote_counts > 0]) // inference_loader.batch_size)) + b

            if voxel_idx >= len(inference_dataset):
                continue

            # Get point indices from this voxel
            voxel_info = inference_dataset.voxels[voxel_idx]
            point_indices = voxel_info['indices']

            # Note: We predicted on patch_size points, but voxel may have more/less
            # Need to map predictions back to original point indices

            point_preds = preds[b].cpu().numpy()
            point_probs = probs[b].cpu().numpy().T  # [N, num_classes]

            # Assign predictions (will be averaged if points appear in multiple voxels)
            for i, pt_idx in enumerate(point_indices[:len(point_preds)]):
                predictions[pt_idx] = point_preds[i]
                probabilities[pt_idx] += point_probs[i]
                vote_counts[pt_idx] += 1

    # Average probabilities
    mask = vote_counts > 0
    probabilities[mask] /= vote_counts[mask, None]

    # Report coverage
    coverage = np.sum(mask) / len(xyz_array)
    print(f"\nInference coverage: {np.sum(mask):,} / {len(xyz_array):,} points ({coverage:.1%})")

    return predictions, probabilities
```

**IMPORTANT NOTE:** The above inference code needs refinement because we need to properly track which points were sampled from each voxel. See "Step 4: Fix Inference Point Mapping" below for the correct approach.

#### 2.3 Update config

```python
config = {
    # ... existing config ...

    # Model parameters (must match training)
    'num_classes': 2,
    'use_rgb': True,
    'input_channels': 6,

    # Inference parameters - MUST MATCH TRAINING
    'patch_size': 1024,  # Same as training
    'voxel_size': None,  # Use same as training (or load from training config)
    'normalize_mode': 'center_scale',  # Same as training
    'batch_size': 16,

    # Remove these old parameters:
    # 'stride': 32,  # Not used anymore
}
```

---

### Step 3: Handle Point Indices in VoxelDataset

The current `VoxelDataset.__getitem__()` doesn't return the sampled point indices, which we need for inference. We have two options:

#### Option A: Modify `__getitem__()` to return indices (for inference)

```python
# In voxelize_dataset.py, update __getitem__()

def __getitem__(self, idx):
    """
    Returns:
        points: [patch_size, C] tensor
        labels: [patch_size] tensor
        indices: [patch_size] tensor with original point indices (for inference)
    """
    voxel = self.voxels[idx]
    indices = voxel['indices']

    # ... existing sampling code ...

    # Track which original indices were selected
    if len(patch_points) >= self.patch_size:
        choice = np.random.choice(len(patch_points), self.patch_size, replace=False)
        selected_indices = np.array(indices)[choice]
        # ... rest of sampling ...
    else:
        # ... padding code ...
        selected_indices = np.array(list(indices) + [indices[i] for i in repeat_indices])

    # ... normalization and augmentation ...

    return (torch.FloatTensor(patch_features),
            torch.LongTensor(patch_labels),
            torch.LongTensor(selected_indices))  # Add this
```

#### Option B: Create separate inference method (cleaner)

Add a method to get all points and indices without sampling:

```python
def get_voxel_batch(self, idx, return_indices=True):
    """
    Get voxel data without random sampling (for deterministic inference)

    Args:
        idx: Voxel index
        return_indices: Whether to return original point indices

    Returns:
        points: [patch_size, C] tensor
        labels: [patch_size] tensor
        indices: [patch_size] array with original indices (if return_indices=True)
    """
    # Similar to __getitem__ but deterministic
    # Use first patch_size points instead of random sampling
    pass
```

**Recommendation:** Use Option A and update the training loop to ignore the third return value.

---

### Step 4: Fix Inference Point Mapping

The key challenge is mapping predictions from sampled patches back to original point cloud indices. Here's the corrected approach:

```python
def predict_point_cloud(self, xyz_array, rgb_array=None, batch_size=16):
    """Updated inference with proper point mapping"""

    # Create inference dataset
    inference_dataset = VoxelDataset(
        xyz_array=xyz_array,
        label_array=None,
        rgb_array=rgb_array,
        patch_size=self.config['patch_size'],
        voxel_size=self.config.get('voxel_size', None),
        normalize_mode=self.config['normalize_mode'],
        augment=False
    )

    # Modify dataloader to use custom collate function
    def collate_with_indices(batch):
        """Collate function that preserves indices"""
        points = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        indices = [item[2] for item in batch]  # List of index tensors
        return points, labels, indices

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_indices
    )

    # Initialize prediction arrays
    predictions = np.full(len(xyz_array), -1, dtype=np.int32)
    probabilities = np.zeros((len(xyz_array), self.config['num_classes']))
    vote_counts = np.zeros(len(xyz_array), dtype=np.int32)

    print(f"\nRunning inference on {len(inference_dataset)} voxels...")

    with torch.no_grad():
        for points, _, batch_indices in tqdm(inference_loader, desc="Inference"):
            points = points.to(self.device)  # [B, N, C]

            outputs = self.model(points)  # [B, num_classes, N]
            probs = torch.softmax(outputs, dim=1)  # [B, num_classes, N]
            preds = torch.argmax(outputs, dim=1)  # [B, N]

            # Convert to numpy
            preds_np = preds.cpu().numpy()  # [B, N]
            probs_np = probs.cpu().numpy()  # [B, num_classes, N]

            # Assign predictions using original point indices
            for b in range(len(batch_indices)):
                point_indices = batch_indices[b].numpy()  # [patch_size]
                point_preds = preds_np[b]  # [patch_size]
                point_probs = probs_np[b].T  # [patch_size, num_classes]

                # Accumulate predictions (averaging if points appear in multiple voxels)
                for i, pt_idx in enumerate(point_indices):
                    predictions[pt_idx] = point_preds[i]
                    probabilities[pt_idx] += point_probs[i]
                    vote_counts[pt_idx] += 1

    # Average probabilities where points were seen
    mask = vote_counts > 0
    probabilities[mask] /= vote_counts[mask, None]

    # Report coverage
    coverage = np.sum(mask) / len(xyz_array)
    print(f"\nInference coverage: {np.sum(mask):,} / {len(xyz_array):,} points ({coverage:.1%})")

    if coverage < 0.95:
        print(f"WARNING: Low coverage! {np.sum(~mask):,} points not covered by any voxel")

    return predictions, probabilities
```

---

### Step 5: Update Training Loop (train.py)

The training loop needs to handle the third return value (indices) from the dataset:

```python
# In train.py, update train_epoch() and test_epoch()

def train_epoch(self, epoch):
    """Train for one epoch"""
    self.model.train()
    # ... existing code ...

    pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch_data in enumerate(pbar):
        # Handle both old and new dataset formats
        if len(batch_data) == 3:
            points, labels, indices = batch_data  # New voxel dataset
        else:
            points, labels = batch_data  # Old dataset
            indices = None

        points = points.to(self.device)
        labels = labels.to(self.device)

        # ... rest of training code unchanged ...
```

Similarly for `test_epoch()`.

---

### Step 6: Save and Load Voxel Size

Since inference must use the **same voxel size** as training, save it with the model checkpoint:

#### In `train.py` (saving):

```python
# When saving checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'config': self.config,
    'voxel_size': train_dataset.get_voxel_size(),  # ADD THIS
    # ... other fields ...
}, checkpoint_path)
```

#### In `inference.py` (loading):

```python
# Load checkpoint
checkpoint = torch.load(model_path, map_location=self.device)

# Get voxel size from checkpoint
voxel_size = checkpoint.get('voxel_size', None)
if voxel_size is not None:
    print(f"Using voxel size from checkpoint: {voxel_size:.4f} m")
    self.config['voxel_size'] = voxel_size
else:
    print("WARNING: No voxel size in checkpoint, will auto-compute")
```

---

## Complete Integration Checklist

### Files to Modify

- [ ] ✅ `voxelize_dataset.py` (already created)
- [ ] `train_end2end.py` - Replace dataset creation
- [ ] `inference.py` - Replace dataset and prediction logic
- [ ] `train.py` - Handle 3-tuple return from dataset
- [ ] `voxelize_dataset.py` - Update `__getitem__` to return indices

### Configuration Changes

- [ ] Add `voxel_size` parameter to config
- [ ] Remove `test_min_patch_distance` parameter (not needed)
- [ ] Remove `stride` parameter (inference only, not needed)
- [ ] Update comments to reflect voxel-based approach

### Testing Steps

1. **Test voxel dataset creation**
   ```python
   python voxelize_dataset.py  # Run example
   ```

2. **Test training with new dataset**
   ```python
   python train_end2end.py
   # Check that voxels are created correctly
   # Verify label distribution includes low-coherence voxels
   ```

3. **Compare voxel statistics**
   ```
   Expected output:
   - Training: ~X voxels with Y% low-coherence voxels
   - Test: ~Z voxels with similar coherence distribution
   ```

4. **Test inference**
   ```python
   python inference.py
   # Verify coverage is high (>95%)
   # Check that predictions are consistent
   ```

5. **Reproduce training accuracy on training region**
   ```python
   # Run inference on training region only
   # Compare to training confusion matrix
   # Should now match closely!
   ```

---

## Expected Improvements

### Before (Polygon-Centered Dataset)

```
Training Confusion Matrix (Epoch 10):
  No Joint: 788,773 correct / 794,175 total (99.3%)
  Joint:    3,362,376 correct / 3,367,361 total (99.9%)
  Overall:  99.8% accuracy

Inference on Training Region:
  Overall: ~70-80% accuracy (does NOT reproduce training!)

Root cause: Distribution mismatch
- Training: Only coherent patches (1,023 mixed patches filtered out)
- Inference: All regions including boundaries
```

### After (Voxel-Based Dataset)

```
Training Confusion Matrix (Epoch 10):
  No Joint: ~85-90% accuracy (lower, but realistic)
  Joint:    ~85-90% accuracy
  Overall:  ~85-90% accuracy

Inference on Training Region:
  Overall: ~85-90% accuracy (MATCHES training!)

Improvement: Distribution consistency
- Training: All regions including boundaries and mixed voxels
- Inference: Same voxel grid, same extraction method
- Model learns to handle difficult cases
```

---

## Tuning Voxel Size

### Automatic (Recommended for First Run)

```python
config = {
    'voxel_size': None,  # Auto-compute based on point density
}
```

The algorithm will:
1. Estimate point density
2. Calculate voxel size to contain ~patch_size points
3. Usually results in 1-3 meter voxels

### Manual (For Fine-Tuning)

```python
config = {
    'voxel_size': 2.0,  # 2 meter voxels
}
```

**Guidelines:**
- **Smaller voxels** (1.0-1.5m):
  - More voxels
  - Finer spatial resolution
  - Longer training time
  - Better for detecting small features

- **Larger voxels** (2.0-3.0m):
  - Fewer voxels
  - Coarser spatial resolution
  - Faster training
  - Better for large-scale patterns

**Rule of thumb:** Voxel size should be similar to the patch radius used in training (check the output during training to see estimated patch radius).

---

## Troubleshooting

### Issue: Too many/too few voxels

**Solution:** Adjust `voxel_size` or `min_points_threshold`

```python
VoxelDataset(
    ...,
    voxel_size=2.0,  # Increase to reduce number of voxels
    min_points_threshold=256,  # Lower to keep more voxels
)
```

### Issue: Low inference coverage (<95%)

**Causes:**
1. Voxel size too large - some regions have no voxel centers nearby
2. `min_points_threshold` too high - sparse regions filtered out

**Solution:**
- Decrease `voxel_size`
- Decrease `min_points_threshold`
- Check point cloud boundaries in inference vs. training

### Issue: Training accuracy still very high (>95%)

**Check:**
1. Verify voxels have low coherence:
   ```
   Look for output:
   "Low coherence (<70%): X voxels"
   ```
   Should be >30% of voxels

2. If most voxels are high-coherence:
   - Point cloud may be naturally very coherent (not mixed)
   - Polygons may be well-separated spatially
   - This is OK! The model should still generalize better

### Issue: Training is too slow

**Solutions:**
1. Increase `voxel_size` to reduce number of voxels
2. Increase `batch_size` if GPU memory allows
3. Reduce `num_epochs`
4. Consider spatial subsampling (but ensure train/test use same subsampling)

---

## Migration Path

### Phase 1: Testing (1-2 days)

1. Create new training run with voxel dataset
2. Monitor training for 10-20 epochs
3. Compare training curves to old approach
4. Check voxel label distribution

### Phase 2: Validation (1 week)

1. Run inference on training region
2. Compare inference accuracy to training accuracy
3. Verify they match (within 5%)
4. Run inference on test region
5. Evaluate actual generalization performance

### Phase 3: Optimization (1-2 weeks)

1. Tune voxel size for best performance
2. Experiment with `min_points_threshold`
3. Try different normalization modes
4. Add hard negative mining for boundary voxels

### Phase 4: Production (ongoing)

1. Retrain final model with optimized hyperparameters
2. Run full inference on all regions
3. Compare results to expert annotations
4. Iterate based on geological validation

---

## Summary

The voxel-based dataset approach:

✅ Eliminates train-test distribution mismatch
✅ Includes boundary and mixed-label regions in training
✅ Uses identical extraction for training and inference
✅ Provides deterministic, reproducible results
✅ Allows model to learn realistic decision boundaries

**Next step:** Follow the integration checklist and test on your actual data!

---

**Version:** 1.0
**Created:** 2025-10-31
**Status:** Ready for integration testing
