"""
KNN-Based Point Dataset - Fully Deterministic Approach

Key Features:
- Each point queries its K nearest neighbors
- No random sampling - fully deterministic
- No shuffling during extraction - reproducible
- Efficient KNN computation with caching
- Identical extraction for training and inference
- Supports batched KNN queries for performance
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from tqdm import tqdm
import pickle
import os


class KNNPointDataset(Dataset):
    """
    Fully deterministic dataset where each point queries K nearest neighbors.

    Key Design Principles:
    1. Deterministic: No random sampling, same K neighbors always
    2. Point-centric: Each sample is centered on a specific point
    3. Reproducible: Same point → same neighbors → same patch
    4. Efficient: Pre-compute and cache KNN indices
    5. Identical for train/inference: Same methodology everywhere
    """

    def __init__(self, xyz_array, label_array=None,
                 k_neighbors=1024, normalize_mode='center', augment=False,
                 rgb_array=None, cache_path=None, point_indices=None,
                 stride=1, min_labeled_ratio=0.0, max_samples_per_class=None):
        """
        Args:
            xyz_array: Full point cloud coordinates [N, 3]
            label_array: Optional labels for each point [N]. -1 for unlabeled.
            k_neighbors: Number of nearest neighbors to query (patch size)
            normalize_mode: 'none', 'center', or 'center_scale'
            augment: Whether to apply data augmentation (training only)
            rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
            cache_path: Path to save/load pre-computed KNN indices
            point_indices: Optional specific point indices to use as centers
                          If None, uses every 'stride'-th labeled point
            stride: Sample every N-th point as center (for memory efficiency)
            min_labeled_ratio: Minimum ratio of labeled neighbors required (0.0-1.0)
                              0.0 = include all points, 1.0 = only fully labeled neighborhoods
            max_samples_per_class: Maximum number of center points per class (for label balancing)
                                  None = use all points (after stride)
                                  Dict or int: e.g., {0: 50000, 1: 100000, 2: 100000} or 100000
        """
        self.xyz_array = xyz_array
        self.label_array = label_array if label_array is not None else np.full(len(xyz_array), -1)
        self.rgb_array = rgb_array
        self.k_neighbors = k_neighbors
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.use_rgb = rgb_array is not None
        self.stride = stride
        self.min_labeled_ratio = min_labeled_ratio
        self.max_samples_per_class = max_samples_per_class

        print(f"\n{'='*70}")
        print("Creating KNN Point Dataset")
        print(f"{'='*70}")
        print(f"Total points in cloud: {len(xyz_array):,}")
        print(f"K neighbors per sample: {k_neighbors}")
        print(f"Normalization: {normalize_mode}")
        print(f"Augmentation: {augment}")
        print(f"Using RGB: {self.use_rgb}")
        print(f"Point stride: {stride} (use every {stride}th point)")
        print(f"Min labeled ratio: {min_labeled_ratio:.0%}")
        if max_samples_per_class is not None:
            print(f"Label sampling: {max_samples_per_class}")

        # Build KDTree for KNN queries
        print("\nBuilding KDTree...")
        self.tree = KDTree(self.xyz_array)
        print("KDTree built successfully")

        # Determine center points
        if point_indices is not None:
            # Use provided point indices
            self.center_point_indices = point_indices
            print(f"Using {len(point_indices):,} provided center points")
        else:
            # Use strided sampling from ALL points (3-class includes background as class 0)
            all_indices = np.arange(len(self.xyz_array))
            self.center_point_indices = all_indices[::stride]
            print(f"Using every {stride}th point as center (all points have labels in 3-class mode)")
            print(f"Center points selected: {len(self.center_point_indices):,}")

        # Pre-compute or load KNN indices (for ALL center points after stride)
        self.knn_indices = self._compute_or_load_knn(cache_path)

        # Store all indices before sampling (for epoch resampling)
        self.all_center_indices = self.center_point_indices.copy()
        self.all_knn_indices = self.knn_indices.copy()

        # Apply label sampling if requested (RUNTIME SAMPLING - can change without recaching)
        # This creates an active subset that can be resampled each epoch
        self.active_sample_indices = None
        if max_samples_per_class is not None:
            self._sample_active_indices(seed=42)  # Initial sampling with fixed seed

        # Filter by labeled ratio if needed
        if min_labeled_ratio > 0 and not np.all(self.label_array == -1):
            self._filter_by_labeled_ratio()

        feature_dim = 6 if self.use_rgb else 3
        print(f"\nKNNPointDataset created:")
        print(f"  Samples (center points): {len(self.center_point_indices):,}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Total neighbor queries: {len(self.center_point_indices) * k_neighbors:,}")
        print(f"{'='*70}\n")

    def _compute_or_load_knn(self, cache_path):
        """
        Compute or load pre-computed KNN indices.

        This is the most expensive operation, so we cache it.
        """
        if cache_path is not None and os.path.exists(cache_path):
            print(f"\nLoading cached KNN indices from {cache_path}...")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Verify cache matches current settings
            if (cached_data['k_neighbors'] == self.k_neighbors and
                cached_data['n_points'] == len(self.xyz_array) and
                len(cached_data['knn_indices']) == len(self.center_point_indices)):
                print("Cache valid, using cached KNN indices")
                return cached_data['knn_indices']
            else:
                print("Cache invalid (settings changed), recomputing...")

        # Compute KNN indices
        print(f"\nComputing KNN indices for {len(self.center_point_indices):,} center points...")
        print("This may take a while for large point clouds...")

        knn_indices = []
        batch_size = 10000  # Process in batches for progress tracking

        for i in tqdm(range(0, len(self.center_point_indices), batch_size),
                     desc="Computing KNN"):
            batch_centers = self.center_point_indices[i:i+batch_size]
            batch_points = self.xyz_array[batch_centers]

            # Query K+1 neighbors (includes the center point itself)
            distances, indices = self.tree.query(batch_points, k=self.k_neighbors + 1)

            # Store indices (excluding the first one which is the point itself)
            for idx_array in indices:
                knn_indices.append(idx_array[1:])  # Exclude self

        knn_indices = np.array(knn_indices)
        print(f"KNN computation complete: {knn_indices.shape}")

        # Cache results if path provided
        if cache_path is not None:
            print(f"Saving KNN indices to {cache_path}...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'knn_indices': knn_indices,
                    'k_neighbors': self.k_neighbors,
                    'n_points': len(self.xyz_array),
                    'center_indices': self.center_point_indices
                }, f)
            print("Cache saved")

        return knn_indices

    def _sample_active_indices(self, seed=None):
        """
        Sample active indices from the full cached set based on max_samples_per_class.
        This is called at initialization and can be called again for epoch resampling.

        Args:
            seed: Random seed for reproducibility. If None, uses random sampling.
        """
        # Get labels for ALL cached center points
        center_labels = self.label_array[self.all_center_indices]

        # Determine max samples per class
        if isinstance(self.max_samples_per_class, dict):
            max_per_class = self.max_samples_per_class
        else:
            # If int, apply same limit to all classes
            unique_labels = np.unique(center_labels)
            max_per_class = {label: self.max_samples_per_class for label in unique_labels}

        # Sample from each class
        selected_indices = []
        for label in sorted(np.unique(center_labels)):
            # Get indices for this class (indices into all_center_indices array)
            class_mask = center_labels == label
            class_indices = np.where(class_mask)[0]

            # Determine how many to sample
            max_for_this_class = max_per_class.get(label, len(class_indices))
            n_to_sample = min(max_for_this_class, len(class_indices))

            # Random sampling
            if seed is not None:
                np.random.seed(seed + int(label))  # Deterministic per class
            sampled = np.random.choice(class_indices, size=n_to_sample, replace=False)
            selected_indices.extend(sampled)

        # Store active sample indices (these index into all_center_indices/all_knn_indices)
        self.active_sample_indices = np.array(selected_indices, dtype=np.int64)

        # Print summary only on first call
        if seed == 42:
            print(f"\n{'='*70}")
            print("LABEL SAMPLING (Runtime - No Cache Recomputation Needed)")
            print(f"{'='*70}")
            print(f"\nTotal cached center points: {len(self.all_center_indices):,}")

            print(f"\nClass distribution in CACHE:")
            for label in sorted(np.unique(center_labels)):
                count = np.sum(center_labels == label)
                print(f"  Class {label}: {count:,} cached points")

            active_labels = self.label_array[self.all_center_indices[self.active_sample_indices]]
            print(f"\nClass distribution in ACTIVE SAMPLE:")
            for label in sorted(np.unique(active_labels)):
                count = np.sum(active_labels == label)
                pct = 100 * count / len(active_labels)
                print(f"  Class {label}: {count:,} samples ({pct:.1f}%)")

            print(f"\nActive samples: {len(self.active_sample_indices):,}")
            print(f"Can resample each epoch without recomputing KNN cache ✅")
            print(f"{'='*70}\n")

    def resample_active_indices(self, epoch=None):
        """
        Resample active indices for a new epoch (training only).
        Uses epoch number as seed for deterministic-but-different sampling each epoch.

        Args:
            epoch: Epoch number to use as seed. If None, uses random sampling.
        """
        if self.max_samples_per_class is None:
            return  # No sampling configured

        seed = (1000 + epoch) if epoch is not None else None
        self._sample_active_indices(seed=seed)

        if epoch is not None:
            print(f"Resampled training data for epoch {epoch} (seed={seed})")

    def _filter_by_labeled_ratio(self):
        """
        Filter out samples where too few neighbors are labeled.
        In 3-class mode, can be used to focus on specific classes.
        """
        print(f"\nFiltering by labeled ratio (min: {self.min_labeled_ratio:.0%})...")

        valid_samples = []
        filtered_knn = []

        for i, neighbor_indices in enumerate(self.knn_indices):
            neighbor_labels = self.label_array[neighbor_indices]

            # In 3-class mode, all points are labeled (0, 1, 2)
            # This filter can be used to ensure minimum ratio of non-background points
            # For now, we consider all points as "labeled"
            labeled_mask = neighbor_labels >= 0  # All valid labels
            labeled_ratio = np.sum(labeled_mask) / len(neighbor_labels)

            if labeled_ratio >= self.min_labeled_ratio:
                valid_samples.append(i)
                filtered_knn.append(neighbor_indices)

        print(f"Samples before filtering: {len(self.center_point_indices):,}")
        print(f"Samples after filtering: {len(valid_samples):,}")
        print(f"Filtered out: {len(self.center_point_indices) - len(valid_samples):,} "
              f"({100 * (1 - len(valid_samples)/len(self.center_point_indices)):.1f}%)")

        # Update arrays
        self.center_point_indices = self.center_point_indices[valid_samples]
        self.knn_indices = np.array(filtered_knn)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Uses active_sample_indices if runtime sampling is enabled.
        """
        if self.active_sample_indices is not None:
            return len(self.active_sample_indices)
        else:
            return len(self.all_center_indices)

    def __getitem__(self, idx):
        """
        Returns:
            points: [k_neighbors, C] tensor where C=3 (XYZ) or C=6 (XYZ+RGB)
            labels: [k_neighbors] tensor with per-point labels (0, 1, or 2 in 3-class mode)
            center_idx: Scalar tensor with the center point index (for tracking)
        """
        # Map through active sampling if enabled (runtime sampling from cached KNN)
        if self.active_sample_indices is not None:
            actual_idx = self.active_sample_indices[idx]
            center_idx = self.all_center_indices[actual_idx]
            neighbor_indices = self.all_knn_indices[actual_idx]
        else:
            # No sampling - use cached indices directly
            center_idx = self.all_center_indices[idx]
            neighbor_indices = self.all_knn_indices[idx]

        # Get neighbor coordinates and labels
        neighbor_points = self.xyz_array[neighbor_indices].copy()
        neighbor_labels = self.label_array[neighbor_indices].copy()

        if self.use_rgb:
            neighbor_rgb = self.rgb_array[neighbor_indices].copy()

        # Get center point coordinates (for normalization reference)
        center_point = self.xyz_array[center_idx]

        # Concatenate XYZ and RGB if using RGB
        if self.use_rgb:
            neighbor_features = np.hstack([neighbor_points, neighbor_rgb])  # [K, 6]
        else:
            neighbor_features = neighbor_points  # [K, 3]

        # Normalize relative to center point
        neighbor_features = self._normalize(neighbor_features, center_point)

        # Augmentation (training only)
        # NOTE: Augmentation breaks determinism, but is necessary for training
        # During inference, augment=False ensures determinism
        if self.augment:
            neighbor_features = self._augment(neighbor_features)

        return (torch.FloatTensor(neighbor_features),
                torch.LongTensor(neighbor_labels),
                torch.LongTensor([center_idx]))

    def _normalize(self, points, center_point):
        """
        Apply normalization relative to center point
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
            # Center on the query point (relative coordinates)
            xyz = xyz - center_point

        elif self.normalize_mode == 'center_scale':
            # Center and scale to unit sphere
            xyz = xyz - center_point
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

    def _augment(self, points):
        """
        Data augmentation: random rotation, scaling, jittering
        Note: Only augments XYZ coordinates, RGB remains unchanged

        WARNING: This breaks determinism! Only use during training.
        """
        # Separate XYZ and RGB if using RGB
        if self.use_rgb:
            xyz = points[:, :3]
            rgb = points[:, 3:]
        else:
            xyz = points

        # Random rotation around Z-axis (vertical)
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        xyz = xyz @ rotation_matrix.T

        # Random scaling (90% to 110%)
        scale = np.random.uniform(0.9, 1.1)
        xyz = xyz * scale

        # Random jittering
        jitter = np.random.normal(0, 0.01, size=xyz.shape)
        xyz = xyz + jitter

        # Concatenate back if using RGB
        if self.use_rgb:
            return np.hstack([xyz, rgb])
        else:
            return xyz

    def get_sample_info(self, idx):
        """
        Get detailed information about a sample

        Returns:
            dict with keys: center_idx, center_point, neighbor_indices,
                           neighbor_labels, label_stats
        """
        center_idx = self.center_point_indices[idx]
        neighbor_indices = self.knn_indices[idx]
        neighbor_labels = self.label_array[neighbor_indices]

        # Compute label statistics
        labeled_mask = neighbor_labels != -1
        n_labeled = np.sum(labeled_mask)

        label_stats = {
            'center_label': int(self.label_array[center_idx]),
            'n_neighbors': len(neighbor_indices),
            'n_labeled': n_labeled,
            'n_unlabeled': len(neighbor_indices) - n_labeled,
            'labeled_ratio': n_labeled / len(neighbor_indices)
        }

        if n_labeled > 0:
            labeled_vals = neighbor_labels[labeled_mask]
            n_class_0 = np.sum(labeled_vals == 0)
            n_class_1 = np.sum(labeled_vals == 1)

            label_stats.update({
                'n_class_0': n_class_0,
                'n_class_1': n_class_1,
                'class_0_ratio': n_class_0 / n_labeled,
                'class_1_ratio': n_class_1 / n_labeled,
                'coherence': max(n_class_0, n_class_1) / n_labeled
            })

        return {
            'center_idx': center_idx,
            'center_point': self.xyz_array[center_idx],
            'neighbor_indices': neighbor_indices,
            'neighbor_labels': neighbor_labels,
            'label_stats': label_stats
        }

    def analyze_dataset_statistics(self):
        """
        Analyze and print comprehensive dataset statistics
        """
        print(f"\n{'='*70}")
        print("Dataset Statistics Analysis")
        print(f"{'='*70}")

        # Basic stats
        print(f"Total samples: {len(self):,}")
        print(f"K neighbors per sample: {self.k_neighbors}")

        if np.all(self.label_array == -1):
            print("Mode: Inference (no labels)")
            print(f"{'='*70}\n")
            return

        # Label statistics across all samples
        all_coherences = []
        all_labeled_ratios = []
        center_label_dist = {-1: 0, 0: 0, 1: 0}

        for idx in range(len(self)):
            info = self.get_sample_info(idx)
            stats = info['label_stats']

            center_label_dist[stats['center_label']] += 1

            if stats['n_labeled'] > 0:
                all_coherences.append(stats['coherence'])
                all_labeled_ratios.append(stats['labeled_ratio'])

        # Print statistics
        print(f"\nCenter point label distribution:")
        print(f"  Unlabeled: {center_label_dist[-1]:,}")
        print(f"  Class 0 (No Joint): {center_label_dist[0]:,}")
        print(f"  Class 1 (Joint): {center_label_dist[1]:,}")

        if len(all_coherences) > 0:
            print(f"\nNeighborhood statistics:")
            print(f"  Average coherence: {np.mean(all_coherences):.2%} "
                  f"(std: {np.std(all_coherences):.2%})")
            print(f"  Coherence range: {np.min(all_coherences):.2%} to {np.max(all_coherences):.2%}")
            print(f"  Average labeled ratio: {np.mean(all_labeled_ratios):.2%} "
                  f"(std: {np.std(all_labeled_ratios):.2%})")

            # Coherence distribution
            low_coherence = np.sum(np.array(all_coherences) < 0.7)
            med_coherence = np.sum((np.array(all_coherences) >= 0.7) &
                                  (np.array(all_coherences) < 0.9))
            high_coherence = np.sum(np.array(all_coherences) >= 0.9)

            print(f"\nCoherence distribution:")
            print(f"  Low coherence (<70%): {low_coherence} ({100*low_coherence/len(all_coherences):.1f}%)")
            print(f"  Medium coherence (70-90%): {med_coherence} ({100*med_coherence/len(all_coherences):.1f}%)")
            print(f"  High coherence (>90%): {high_coherence} ({100*high_coherence/len(all_coherences):.1f}%)")

        print(f"{'='*70}\n")


# Convenience function for creating train/test datasets
def create_train_test_knn_datasets(xyz_array, label_array, train_mask, test_mask,
                                   k_neighbors=1024, normalize_mode='center',
                                   augment_train=False, rgb_array=None,
                                   train_stride=1, test_stride=1,
                                   min_labeled_ratio=0.0,
                                   cache_dir='./data/knn_cache',
                                   max_samples_per_class=None):
    """
    Create training and test KNN datasets

    Args:
        xyz_array: Full point cloud [N, 3]
        label_array: Labels [N]
        train_mask: Boolean mask for training points [N]
        test_mask: Boolean mask for test points [N]
        k_neighbors: Number of nearest neighbors
        normalize_mode: Normalization method
        augment_train: Whether to augment training data
        rgb_array: Optional RGB colors [N, 3]
        train_stride: Sample every N-th training point
        test_stride: Sample every N-th test point
        min_labeled_ratio: Minimum labeled ratio for training samples
        cache_dir: Directory to save/load KNN caches
        max_samples_per_class: Maximum training samples per class (for label balancing)
                              None, int, or dict: {0: 50000, 1: 100000, 2: 100000}

    Returns:
        train_dataset, test_dataset
    """
    os.makedirs(cache_dir, exist_ok=True)

    print("="*70)
    print("CREATING TRAINING KNN DATASET")
    print("="*70)

    train_cache = os.path.join(cache_dir, 'train_knn_cache.pkl')
    train_dataset = KNNPointDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        k_neighbors=k_neighbors,
        normalize_mode=normalize_mode,
        augment=augment_train,
        rgb_array=rgb_array[train_mask] if rgb_array is not None else None,
        cache_path=train_cache,
        stride=train_stride,
        min_labeled_ratio=min_labeled_ratio,
        max_samples_per_class=max_samples_per_class  # Label sampling for training only
    )

    print("\n" + "="*70)
    print("CREATING TEST KNN DATASET")
    print("="*70)

    test_cache = os.path.join(cache_dir, 'test_knn_cache.pkl')
    test_dataset = KNNPointDataset(
        xyz_array=xyz_array[test_mask],
        label_array=label_array[test_mask],
        k_neighbors=k_neighbors,
        normalize_mode=normalize_mode,
        augment=False,
        rgb_array=rgb_array[test_mask] if rgb_array is not None else None,
        cache_path=test_cache,
        stride=test_stride,
        min_labeled_ratio=0.0  # Don't filter test set
    )

    print("\n" + "="*70)
    print("DATASET CREATION SUMMARY")
    print("="*70)
    print(f"K neighbors: {k_neighbors}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Training stride: {train_stride}")
    print(f"Test stride: {test_stride}")
    print(f"Normalize mode: {normalize_mode}")
    print(f"Augmentation: Train={augment_train}, Test=False")
    print(f"Min labeled ratio (train): {min_labeled_ratio:.0%}")
    print("="*70 + "\n")

    return train_dataset, test_dataset


# Example usage
if __name__ == '__main__':
    print("KNNPointDataset - Example Usage\n")

    # Simulate data
    np.random.seed(42)
    n_points = 10000
    xyz = np.random.randn(n_points, 3) * 10
    labels = np.random.randint(0, 2, n_points)

    # Create dataset
    dataset = KNNPointDataset(
        xyz_array=xyz,
        label_array=labels,
        k_neighbors=128,
        normalize_mode='center',
        augment=False,
        stride=10  # Use every 10th point
    )

    print(f"Dataset created with {len(dataset)} samples")

    if len(dataset) > 0:
        # Get sample
        points, labels, center_idx = dataset[0]
        print(f"\nSample 0:")
        print(f"  Points shape: {points.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Center index: {center_idx.item()}")

        # Get info
        info = dataset.get_sample_info(0)
        print(f"\nSample 0 detailed info:")
        print(f"  Center point: {info['center_point']}")
        print(f"  Label stats: {info['label_stats']}")

        # Analyze statistics
        dataset.analyze_dataset_statistics()
