# Training vs. Inference Performance Mismatch - Root Cause Analysis

## Executive Summary

**Problem:** The model achieves ~99% accuracy during training evaluation (epoch 10 confusion matrix shows 788,773 correct "No Joint" and 3,362,376 correct "Joint" predictions), but this performance cannot be reproduced during inference on the same training region.

**Root Cause:** There are **fundamental differences** between how patches are extracted and evaluated during training vs. inference, leading to a severe **train-test distribution mismatch**.

---

## Critical Issue #1: Different Patch Extraction Methods

### Training: Polygon-Centered Coherent Patches

**Location:** `dataset.py` - `RockJointDataset._extract_patches_with_coherence()` (Lines 233-502)

**Method:**
1. **Centers on labeled polygon centroids**
2. **Filters for coherence**: Only keeps patches where ≥70% of labeled points match the center point's label
3. **Filters for label density**: Only keeps patches with ≥50% labeled points
4. **Uses polygon-derived patch radius**: Calculates radius based on point density to contain ~patch_size points
5. **Result**: Creates **highly curated, coherent patches** where most points have the same label

**Key Code:**
```python
# Lines 283-284
min_coherence = 0.70  # At least 70% of labeled points must match center label
min_labeled_ratio = 0.50  # At least 50% of points in patch must be labeled

# Lines 325-331
coherence = np.sum(labeled_point_labels == center_label) / len(labeled_point_labels)
if coherence < min_coherence:
    skipped_incoherent += 1
    continue
```

**What this means:**
- Training patches are **spatially pure** - mostly uniform class distribution
- The model learns to classify **easy, coherent regions**
- Mixed boundary regions are largely **excluded** from training

---

### Inference: Grid-Based Sliding Window

**Location:** `dataset.py` - `RockJointInferenceDataset._create_spatial_grid()` (Lines 637-689)

**Method:**
1. **Regular 3D spatial grid** with uniform spacing
2. **No coherence filtering** - includes all regions
3. **No label density requirement** - includes unlabeled regions
4. **Different patch radius calculation**: Uses `stride` parameter to determine spacing
5. **Result**: Creates patches that **include boundary regions, mixed labels, and unlabeled areas**

**Key Code:**
```python
# Lines 665-667
x_range = np.arange(min_bounds[0], max_bounds[0], spatial_stride)
y_range = np.arange(min_bounds[1], max_bounds[1], spatial_stride)
z_range = np.arange(min_bounds[2], max_bounds[2], spatial_stride)

# Lines 680-681
indices = tree.query_ball_point(center, spatial_stride)
```

**What this means:**
- Inference patches include **all spatial regions** regardless of label coherence
- Includes **boundary regions** where classes mix
- Includes **unlabeled regions** that were never seen during training
- The distribution is **fundamentally different** from training

---

## Critical Issue #2: Evaluation Granularity Mismatch

### Training Evaluation: Per-Point Within Coherent Patches

**Location:** `train.py` - `train_epoch()` (Lines 88-138)

**What happens:**
1. Model receives coherent patches (filtered by `_extract_patches_with_coherence`)
2. Model predicts per-point labels for all 1024 points in each patch
3. **Only labeled points are used for metrics** (Lines 128-131):
   ```python
   if -1 in all_labels:
       valid_mask = all_labels != -1
       all_preds = all_preds[valid_mask]
       all_labels = all_labels[valid_mask]
   ```
4. Confusion matrix is computed on these **labeled points within coherent patches**

**Result:** The 99% training accuracy represents performance on:
- **Coherent, spatially pure patches**
- **Only labeled points** (which are mostly uniform within each patch due to coherence filtering)
- **Excludes difficult boundary regions**

---

### Inference Evaluation: Per-Point Across Entire Point Cloud

**Location:** `inference.py` - `predict_point_cloud()` (Lines 49-104)

**What happens:**
1. Model receives **all spatial regions** via sliding window grid
2. Model predicts per-point labels for entire point cloud
3. **All points receive predictions** via majority voting:
   ```python
   # Lines 98-101
   predictions[point_indices] = point_preds
   probabilities[point_indices] += point_probs
   vote_counts[point_indices] += 1
   ```
4. Evaluation includes:
   - **Boundary regions** (never seen in training)
   - **Mixed-label regions** (filtered out during training)
   - **Unlabeled regions** (no ground truth, but model still predicts)

**Result:** Inference accuracy represents performance on:
- **All spatial regions** including difficult boundaries
- **Mixed coherence patches** that were filtered out during training
- **A fundamentally different distribution** than training

---

## Critical Issue #3: The "Coherence Paradox"

### The Training Dataset is Systematically Biased

Looking at the training dataset extraction output:

```
Patch Extraction Complete
Total patches extracted: 254
Skipped (insufficient points): 312
Skipped (too few labeled): 627
Skipped (incoherent/mixed): 1023  ← THIS IS THE PROBLEM
```

**What this means:**
- **1,023 patches** were skipped because they had **mixed labels** (low coherence)
- Only **254 patches** were kept (the "easy" ones with uniform labels)
- The model is trained **only on spatially pure regions**

**The paradox:**
1. Training accuracy is high because the model only sees **easy, coherent patches**
2. These patches have mostly uniform labels within them
3. The model learns to predict uniform regions very well
4. But **real-world inference** requires predicting on:
   - Boundary regions (joint vs. no-joint transitions)
   - Mixed regions (complex geological features)
   - Spatially heterogeneous regions

**Analogy:**
This is like training a student only on multiple-choice questions where all 4 options are the same letter (A, A, A, A), then testing them on questions with different options (A, B, C, D). The training accuracy will be 100%, but test accuracy will be random.

---

## Visual Evidence from Confusion Matrix

### Training Confusion Matrix (Epoch 10)

```
                    Predicted
              No Joint    Joint
True
No Joint      788,773     5,402    ← 99.3% accuracy on "No Joint"
Joint          4,985    3,362,376  ← 99.9% accuracy on "Joint"
```

**Total points evaluated:** 4,161,536 points
**These are only the labeled points within the 254 coherent patches!**

**What the model actually learned:**
- "If I'm in a coherent 'No Joint' patch → predict 'No Joint' everywhere"
- "If I'm in a coherent 'Joint' patch → predict 'Joint' everywhere"

**What the model CANNOT do:**
- Distinguish boundaries between classes
- Handle mixed-label regions
- Generalize to spatial heterogeneity

---

## Issue #4: Patch Radius vs. Stride Mismatch

### Training Patch Radius

**Calculation** (Lines 263-270):
```python
point_density = 1.0 / (avg_point_spacing ** 3)
target_volume = self.patch_size / point_density
patch_radius = (3 * target_volume / (4 * np.pi)) ** (1/3)
```

**Typical value:** ~0.5-2.0 meters (depends on point density)

### Inference Spatial Stride

**From config:**
```python
'stride': 32,  # Number of points
```

**Converted to spatial distance** (Line 662):
```python
spatial_stride = avg_spacing * np.sqrt(self.stride)
```

With `avg_spacing = 0.06m` and `stride = 32`:
- `spatial_stride = 0.06 * sqrt(32) = 0.06 * 5.66 = 0.34 meters`

**Problem:**
- Training patches have radius ~1.0-2.0m → diameter ~2.0-4.0m
- Inference patches have radius ~0.34m → diameter ~0.68m
- **The patch sizes are completely different!**

---

## Issue #5: Random Sampling in Inference

**Location:** `dataset.py` - `RockJointInferenceDataset.__getitem__()` (Line 710)

```python
if len(patch_points) >= self.patch_size:
    choice = np.random.choice(len(patch_points), self.patch_size, replace=False)
```

**Problem:**
- During **training**, patches are centered on polygon centroids and carefully curated
- During **inference**, patches are **randomly sampled** from grid regions
- Even if the same region is processed, different random samples will be selected
- This adds **additional stochasticity** that wasn't present during training

---

## Issue #6: Point Cloud Coverage Mismatch

### Training Coverage

From the coherence-filtered extraction:
- Only **254 patches** cover the training region
- Each patch centers on a **polygon centroid**
- Patches have **50-70% labeled points** minimum
- **Large areas** of the point cloud are **never seen** during training

**Estimated training coverage:**
- 254 patches × 1024 points/patch = 260,096 points
- But with unlabeled points, actual unique labeled points is lower
- From confusion matrix: 4,161,536 labeled point evaluations
- This suggests ~16 evaluations per point (due to overlapping patches in DataLoader)

### Inference Coverage

- **Sliding window covers the entire point cloud**
- Grid-based approach ensures **uniform spatial coverage**
- Includes regions that were:
  - Filtered out during training (low coherence)
  - Never labeled
  - Boundary regions

**The model is being asked to predict on regions it has never seen!**

---

## Summary of Root Causes

### 1. **Coherence Filtering Bias** (MOST CRITICAL)
   - Training uses only coherent patches (70%+ uniform labels)
   - Inference includes all regions (including mixed/boundary areas)
   - Model never learns to handle difficult boundary cases

### 2. **Patch Extraction Method Mismatch**
   - Training: Polygon-centered, coherent patches
   - Inference: Grid-based, uniform sampling
   - Completely different spatial distributions

### 3. **Patch Size Mismatch**
   - Training patches: Large radius (~1-2m)
   - Inference patches: Small radius (~0.34m from stride=32)
   - Different receptive field sizes

### 4. **Evaluation Granularity Difference**
   - Training: Per-point on labeled points in coherent patches
   - Inference: Per-point on all points in grid patches
   - Training metrics are misleadingly high

### 5. **Spatial Coverage Difference**
   - Training: Sparse coverage of "easy" regions only
   - Inference: Dense coverage of all regions
   - Model asked to generalize to unseen regions

### 6. **Random Sampling Stochasticity**
   - Inference uses random point sampling from grid regions
   - Training uses curated polygon-centered sampling
   - Additional source of variance

---

## Why Training Accuracy is Misleadingly High

The training confusion matrix shows:
```
True No Joint: 788,773 + 5,402 = 794,175 points
True Joint:    4,985 + 3,362,376 = 3,367,361 points
Total: 4,161,536 labeled points evaluated
```

**These points come from:**
- 254 coherent patches
- Each patch has ≥70% coherence (mostly uniform labels)
- Each patch is evaluated multiple times (due to batch shuffling)

**What the model is actually doing:**
1. Receives a coherent "No Joint" patch
2. Predicts "No Joint" for most/all points (easy!)
3. Receives a coherent "Joint" patch
4. Predicts "Joint" for most/all points (easy!)

**The model is NOT learning:**
- How to distinguish joints from no-joints in mixed regions
- How to identify boundaries
- How to handle spatial heterogeneity
- How to generalize beyond polygon interiors

---

## Implications

### Why Inference Fails to Reproduce Training Accuracy

1. **Distribution Shift:** Inference sees a completely different distribution (mixed patches, boundaries, unlabeled regions)

2. **Unseen Regions:** Model is tested on spatial regions that were filtered out during training

3. **Different Receptive Field:** Inference patches are smaller (0.34m radius) vs. training patches (1-2m radius)

4. **No Coherence Guarantee:** Inference patches can have any label distribution, not just coherent ones

5. **Full Coverage:** Inference must predict on ALL points, not just the "easy" ones in polygon centers

### Why the Model Appears to "Overfit"

The model isn't technically overfitting in the traditional sense (training on training set, failing on validation set). Instead:

1. **The training evaluation is misleading** - it only measures performance on easy, coherent patches
2. **The inference task is fundamentally different** - it requires generalization to all spatial regions
3. **The model learned the wrong thing** - it learned to classify coherent patches, not individual points or boundaries

---

## Recommendations

### Immediate Fixes (High Priority)

1. **Remove coherence filtering** from training patch extraction
   - Allow mixed-label patches
   - Include boundary regions
   - Lower or remove `min_coherence` threshold

2. **Match patch extraction methods**
   - Use the same grid-based extraction for training as inference
   - Or use the same polygon-based extraction for inference as training

3. **Match patch radii**
   - Ensure training and inference use the same spatial patch size
   - Fix the stride calculation to match training patch radius

4. **Stratified sampling**
   - Instead of filtering out mixed patches, sample them proportionally
   - Include boundary regions in training

### Medium-Term Improvements

5. **Add boundary region mining**
   - Explicitly identify and oversample boundary regions during training
   - Create patches that span class boundaries

6. **Multi-scale training**
   - Train on multiple patch sizes
   - Help model learn features at different scales

7. **Augmentation of patch coherence**
   - Artificially create mixed-label training patches
   - Combine regions from different classes

### Long-Term Architectural Changes

8. **True segmentation architecture**
   - Current approach treats this as classification (one label per patch)
   - Should be true point-wise segmentation (one label per point)
   - Use proper segmentation loss (not just classification on majority label)

9. **Attention mechanisms**
   - Add attention to focus on discriminative regions
   - Help model identify boundaries explicitly

10. **Curriculum learning**
    - Start with coherent patches (easy)
    - Gradually introduce mixed patches (hard)
    - Progressive difficulty increase

---

## Testing the Hypothesis

To verify this analysis, you could:

1. **Create inference dataset with coherence filtering**
   ```python
   # Apply same coherence filtering to inference patches
   # Expected: inference accuracy should match training
   ```

2. **Remove coherence filtering from training**
   ```python
   # Set min_coherence = 0.0
   # Expected: training accuracy drops, but inference improves
   ```

3. **Compare patch statistics**
   ```python
   # Compute coherence distribution for training vs. inference patches
   # Expected: training patches are all high-coherence, inference are mixed
   ```

4. **Visualize patch centers**
   ```python
   # Plot spatial locations of training patches vs. inference patches
   # Expected: training patches cluster at polygon centers, inference are uniform
   ```

---

## Conclusion

The 99% training accuracy is **not representative** of model performance. It's an artifact of:

1. Training only on curated, coherent patches (filtering out 1,023 mixed patches)
2. Evaluating only on labeled points within these coherent patches
3. Using a fundamentally different data distribution than inference

The model has learned to classify **coherent spatial regions**, not **individual points** or **boundaries**. When asked to predict on the full point cloud (including boundaries, mixed regions, and unlabeled areas), it fails because it never saw such data during training.

**The fix is not to improve the model architecture, but to fix the data pipeline to ensure training and inference distributions match.**

---

**Report Version:** 1.0
**Date:** 2025-10-31
**Critical Finding:** Training-inference distribution mismatch due to coherence filtering
