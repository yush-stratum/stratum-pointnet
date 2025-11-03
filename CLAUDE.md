# PointNet++ for Rock Joint Classification in Open Pit Mines

## Project Overview

This project implements a deep learning pipeline using PointNet++ for binary classification of rock discontinuities (joints vs no joints) in 3D point cloud data from open pit mine rock faces. The goal is to automate the detection and classification of geological features that are traditionally identified manually by geotechnical experts.

## Problem Statement

Geotechnical engineers need to identify and classify rock joints and discontinuities in open pit mines for safety and operational planning. Manual annotation of these features in large point clouds (10-20 million points) is:
- Time-consuming and labor-intensive
- Subject to human error and inconsistency
- Difficult to scale across multiple mine sites
- Requires specialized geological expertise

This project automates this process using deep learning on 3D point cloud data.

## Objectives

1. Train a PointNet++ model to classify rock joints vs non-jointed rock faces
2. Achieve 85%+ accuracy on held-out test data
3. Process full point clouds (10-20M points) efficiently
4. Provide per-point predictions with confidence scores
5. Output results in standard LAS format for GIS integration

## Technology Stack

### Deep Learning Framework
- **PyTorch 2.0+**: Primary deep learning framework
- **CUDA**: GPU acceleration for training and inference

### Point Cloud Processing
- **laspy 2.4+**: Reading and writing LAS point cloud files
- **ezdxf 1.0+**: Extracting polygon annotations from DXF files
- **NumPy 1.24+**: Numerical operations on point arrays
- **SciPy 1.10+**: Spatial operations (KDTree, nearest neighbors)

### Machine Learning & Evaluation
- **scikit-learn 1.3+**: Metrics, train/test evaluation
- **pandas**: Metrics tracking and CSV output

### Visualization
- **matplotlib**: Training curves and confusion matrices
- **seaborn**: Enhanced visualization aesthetics
- **plotly**: Interactive 3D point cloud visualization

## Architecture

### Model: PointNet++

PointNet++ is a hierarchical neural network designed for point cloud processing. It learns spatial features at multiple scales through set abstraction layers.

**Architecture Details:**

```
Input: [Batch, 2048, 3] - Patches of 2048 points with XYZ coordinates

Set Abstraction Layer 1:
  - Farthest Point Sampling: 2048 â†’ 512 points
  - Ball Query: radius=0.2m, k=32 neighbors
  - PointNet: MLP [64, 64, 128]

Set Abstraction Layer 2:
  - Farthest Point Sampling: 512 â†’ 128 points
  - Ball Query: radius=0.4m, k=64 neighbors
  - PointNet: MLP [128, 128, 256]

Set Abstraction Layer 3:
  - Global Pooling: 128 â†’ 1 point
  - PointNet: MLP [256, 512, 1024]

Classification Head:
  - FC: 1024 â†’ 512 (dropout 0.4)
  - FC: 512 â†’ 256 (dropout 0.4)
  - FC: 256 â†’ 2 (class logits)

Output: [Batch, 2] - Class probabilities
```

**Key Features:**
- Hierarchical feature learning (local to global)
- Permutation invariant (order doesn't matter)
- Approximately 1.5M parameters (lightweight)
- Handles irregular point sampling naturally

### Why PointNet++?

**Advantages for Geological Applications:**
1. **Direct point cloud processing**: No voxelization or meshing required
2. **Multi-scale feature learning**: Captures both fine details and global structure
3. **Rotation invariant features**: Works regardless of point cloud orientation
4. **Efficient**: Processes patches quickly on GPU
5. **State-of-the-art**: Proven performance on 3D point cloud tasks

**Why not pre-trained models?**
- Pre-trained models (ModelNet, ShapeNet, ScanNet) are trained on indoor/outdoor scenes or objects
- Massive domain mismatch with geological features
- Rock discontinuities have unique geometric signatures
- Training from scratch with domain-specific data yields better results

## Data Pipeline

### Input Data

1. **Point Cloud (LAS format)**
   - XYZ coordinates (required)
   - RGB color (optional)
   - Typical size: 10-20 million points
   - Resolution: 6cm point spacing

2. **Polygon Annotations (DXF format)**
   - Two DXF files: one for "no joints" regions, one for "joints" regions
   - Polygons drawn as closed LINE entities
   - Created by domain experts using CAD software

### Preprocessing Pipeline

```
1. Load LAS point cloud â†’ xyz_array [N, 3]

2. Extract polygons from DXF files
   - Parse LINE entities
   - Connect lines into closed polygons using graph traversal
   - Validate polygon closure

3. Label points within polygons
   - Project polygons and points to 2D plane
   - Use ray casting algorithm for point-in-polygon test
   - Apply configurable depth tolerance
   - Output: label_array [N] with {-1: unlabeled, 0: no joints, 1: joints}

4. Create spatial train/test split
   - Define split line using two 3D points
   - Use cross product to assign points to train or test
   - Ensures spatial separation (no data leakage)

5. Filter polygons by split
   - Assign polygons to train or test based on centroid location
   - Maintain polygon-point coordinate correspondence

6. Save preprocessed data
   - Arrays: preprocessed_data.npz (xyz, labels, masks)
   - Polygons: polygons_dict.pkl (train/test polygon dictionaries)
   - Location: ./data/preprocessed/
```

### Training Data Generation

```
For each polygon:
  1. Compute polygon centroid
  2. Query points within search radius using KDTree
  3. Sample or pad to exactly 2048 points
  4. Apply normalization (center, center_scale, or none)
  5. Apply augmentation (if training):
     - Random rotation around Z-axis
     - Random scaling (90-110%)
     - Gaussian jittering (Ïƒ=0.01)
     - Random point dropout (0-10%)
  6. Create patch: [2048, 3] tensor

Result: Dataset of labeled patches for training
```

## Normalization Strategies

The pipeline supports three normalization modes (toggleable via config):

### 1. None
- Use absolute XYZ coordinates without modification
- Preserves spatial context and absolute position
- Risk: Model may overfit to specific locations
- Use case: When absolute position is informative

### 2. Center (Recommended)
- Subtract patch centroid to create zero-mean coordinates
- Learns relative spatial patterns
- Recommended for geological features where local geometry matters
- Formula: `points_normalized = points - centroid`

### 3. Center + Scale
- Subtract centroid and normalize to unit sphere
- Scale-invariant learning
- Use when feature sizes vary significantly
- Formula: `points_normalized = (points - centroid) / max_distance`

**Critical**: Normalization mode must match between training and inference.

## Training Methodology

### Configuration

```python
config = {
    # Data
    'patch_size': 2048,           # Points per patch
    'normalize_mode': 'center',   # Normalization strategy
    'augment_train': True,        # Enable augmentation

    # Training
    'batch_size': 16,             # Adjust based on GPU memory
    'num_epochs': 100,            # Training epochs
    'learning_rate': 0.001,       # Initial learning rate
    'weight_decay': 1e-4,         # L2 regularization

    # Learning rate schedule
    'lr_decay_step': 20,          # Decay every N epochs
    'lr_decay_rate': 0.7,         # Multiply LR by this factor

    # Class balancing (optional)
    'class_weights': None,        # [w0, w1] if imbalanced
}
```

### Training Process

```
For each epoch:
  1. Forward pass: patches â†’ predictions
  2. Compute loss: CrossEntropyLoss(predictions, labels)
  3. Backward pass: compute gradients
  4. Optimizer step: update weights (Adam)
  5. Evaluate on test set
  6. Update learning rate (StepLR scheduler)
  7. Save checkpoint if best accuracy

Tracking:
  - Loss (train/test)
  - Accuracy (train/test)
  - Precision (train/test)
  - Recall (train/test)
  - F1 score (test)
```

### Outputs

Saved in `./checkpoints/{run_name}/`:
```
- config.json                      # Configuration used
- best_model_acc{score}.pth        # Best model checkpoint
- checkpoint_epoch{N}.pth          # Periodic checkpoints
- training_curves.png              # Loss, accuracy, precision, recall curves
- training_metrics.csv             # Per-epoch metrics
- confusion_matrix.png             # Final confusion matrix
- classification_report.json       # Precision, recall, F1 per class
```

## Inference Methodology

### Sliding Window Approach

Full point cloud inference uses overlapping spatial patches with voting:

```
1. Create spatial grid over point cloud
   - Estimate point density
   - Determine spatial stride based on desired overlap
   - Generate grid of patch centers

2. For each patch center:
   - Extract nearest 2048 points using KDTree
   - Apply same normalization as training
   - Predict class (0 or 1)
   - Store prediction for all points in patch

3. Aggregate predictions via majority voting
   - Each point receives votes from overlapping patches
   - Final prediction = most common class
   - Confidence = average probability across votes

4. Save to LAS file
   - classification field: predictions (0=uncovered, 1=no joints, 2=joints)
   - prob_class_0, prob_class_1: confidence scores
   - Optionally: ground_truth and correctness comparison
```

### Stride Parameter

- **stride = 1024**: 50% overlap (recommended balance)
- **stride = 2048**: No overlap (faster, less smooth)
- **stride = 512**: 75% overlap (slower, smoother predictions)

Lower stride = more overlap = slower inference but smoother predictions

## Project Structure

```
.
â”œâ”€â”€ data/
â”‚   â””â”€â”€ preprocessed/              # Preprocessed data (not in version control)
â”‚       â”œâ”€â”€ {dataset}_preprocessed.npz
â”‚       â””â”€â”€ {dataset}_polygons.pkl
â”‚
â”œâ”€â”€ checkpoints/
â”‚   â””â”€â”€ {run_name}/                # Training outputs per run
â”‚       â”œâ”€â”€ config.json
â”‚       â”œâ”€â”€ best_model.pth
â”‚       â”œâ”€â”€ checkpoint_epoch{N}.pth
â”‚       â”œâ”€â”€ training_curves.png
â”‚       â”œâ”€â”€ training_metrics.csv
â”‚       â”œâ”€â”€ confusion_matrix.png
â”‚       â””â”€â”€ classification_report.json
â”‚
â”œâ”€â”€ predictions/                   # Inference outputs
â”‚   â””â”€â”€ {dataset}_predicted.las
â”‚
â”œâ”€â”€ pointnet2_model.py            # PointNet++ architecture
â”œâ”€â”€ dataset.py                     # Dataset classes (train/inference)
â”œâ”€â”€ preprocessing_utils.py         # Preprocessing functions
â”œâ”€â”€ train.py                       # Trainer class
â”œâ”€â”€ train_end_to_end.py           # Main training script
â”œâ”€â”€ inference.py                   # Inference script
â”œâ”€â”€ requirements.txt               # Python dependencies
â””â”€â”€ README.md                      # Documentation
```

## Success Criteria

### Quantitative Metrics

**Primary Metric:**
- Test accuracy greater than or equal to 85%

**Secondary Metrics:**
- Precision greater than or equal to 0.80 (both classes)
- Recall greater than or equal to 0.80 (both classes)
- F1 score greater than or equal to 0.80 (both classes)
- Balanced performance across classes

**Inference Quality:**
- Coverage greater than or equal to 95% of point cloud
- Average confidence greater than or equal to 0.70
- Spatial coherence (no random scattered predictions)

### Qualitative Validation

1. **Visual inspection**: Predictions match expert annotations
2. **Spatial continuity**: Joint boundaries are clear and coherent
3. **Generalization**: Model works on unseen regions
4. **Confidence calibration**: High confidence on clear features, low on ambiguous

## Key Implementation Details

### Polygon Extraction

- DXF LINE entities connected via graph traversal
- Closed loops detected by returning to start point
- Minimum 3 vertices required
- Self-intersecting polygons handled gracefully

### Point Labeling

- 2D projection for efficient point-in-polygon testing
- Ray casting algorithm for containment check
- Configurable depth tolerance (default: 5.0 meters)
- Projection axis automatically determined by polygon orientation

### Patch Extraction

- One patch per polygon centroid
- KDTree for efficient nearest neighbor search
- Adaptive search radius based on polygon size
- Points sampled or padded to exact patch_size

### Train/Test Split

- Spatial split using cross product of 2D line
- Ensures geographic separation (no data leakage)
- Polygons filtered by centroid location
- Maintains coordinate system consistency

### LAS Classification Encoding

- LAS format: 5-bit unsigned integer (0-31 range)
- Encoding: predictions shifted by +1
  - 0 = Uncovered (no prediction)
  - 1 = No joints (original class 0)
  - 2 = Joints (original class 1)
