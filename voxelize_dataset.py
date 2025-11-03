"""
Voxel-Based Dataset for Rock Joint Point Cloud Classification

Key Innovation: Training and inference use IDENTICAL voxel grid extraction
- No coherence filtering
- No polygon-centered bias
- Allows mixed-label voxels
- Ensures distribution consistency
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from tqdm import tqdm


class VoxelDataset(Dataset):
    """
    Unified voxel-based dataset for both training and inference.

    Key principles:
    1. Create regular 3D voxel grid over point cloud
    2. Each voxel contains ~patch_size points
    3. Extract patches by querying points within each voxel
    4. NO filtering based on label coherence
    5. Same extraction method for training and inference
    """

    def __init__(self, xyz_array, label_array=None,
                 patch_size=1024, voxel_size=None,
                 normalize_mode='center', augment=False,
                 rgb_array=None, spatial_bounds=None,
                 min_points_threshold=None):
        """
        Args:
            xyz_array: Full point cloud coordinates [N, 3]
            label_array: Optional labels for each point [N]. -1 for unlabeled.
                        If None, assumes inference mode (all points unlabeled)
            patch_size: Target number of points per voxel
            voxel_size: Size of each voxel cube (meters). If None, auto-computed
            normalize_mode: 'none', 'center', or 'center_scale'
            augment: Whether to apply data augmentation (training only)
            rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
            spatial_bounds: Optional [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
                           If None, computed from xyz_array
            min_points_threshold: Minimum points in voxel to keep. If None, use patch_size // 4
        """
        self.xyz_array = xyz_array
        self.label_array = label_array if label_array is not None else np.full(len(xyz_array), -1)
        self.rgb_array = rgb_array
        self.patch_size = patch_size
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.use_rgb = rgb_array is not None

        # Set minimum points threshold
        self.min_points_threshold = min_points_threshold if min_points_threshold is not None else patch_size // 4

        # Determine spatial bounds
        if spatial_bounds is not None:
            self.bounds = spatial_bounds
        else:
            self.bounds = [
                (xyz_array[:, 0].min(), xyz_array[:, 0].max()),
                (xyz_array[:, 1].min(), xyz_array[:, 1].max()),
                (xyz_array[:, 2].min(), xyz_array[:, 2].max())
            ]

        # Compute or use provided voxel size
        if voxel_size is not None:
            self.voxel_size = voxel_size
        else:
            self.voxel_size = self._estimate_voxel_size()

        print(f"Voxel size: {self.voxel_size:.4f} m")

        # Build KDTree for efficient spatial queries
        self.tree = KDTree(self.xyz_array)

        # Create voxel grid
        self.voxels = []  # List of (voxel_center, point_indices)
        self._create_voxel_grid()

        feature_dim = 6 if self.use_rgb else 3
        print(f"VoxelDataset created: {len(self.voxels)} voxels, {feature_dim}D features")

    def _estimate_voxel_size(self):
        """
        Estimate voxel size to contain approximately patch_size points

        Algorithm:
        1. Sample points and estimate point density
        2. Calculate voxel volume needed for patch_size points
        3. Compute voxel edge length (cube root of volume)
        """
        # Sample points to estimate density
        sample_size = min(10000, len(self.xyz_array))
        sample_indices = np.random.choice(len(self.xyz_array), sample_size, replace=False)
        sample_points = self.xyz_array[sample_indices]

        # Build temporary tree
        sample_tree = KDTree(sample_points)

        # Estimate average spacing between points
        k = min(50, sample_size - 1)
        distances, _ = sample_tree.query(sample_points, k=k+1)
        avg_point_spacing = np.mean(distances[:, 1:])  # Exclude self

        # Estimate point density (points per cubic meter)
        point_density = 1.0 / (avg_point_spacing ** 3)

        # Calculate volume needed to contain patch_size points
        target_volume = self.patch_size / point_density

        # Voxel size is cube root of volume (assuming cubic voxels)
        voxel_size = target_volume ** (1/3)

        print(f"Estimated point density: {point_density:.2f} points/m³")
        print(f"Average point spacing: {avg_point_spacing:.4f} m")

        return voxel_size

    def _create_voxel_grid(self):
        """
        Create regular 3D voxel grid and extract point indices for each voxel

        Process:
        1. Divide space into regular cubic voxels
        2. For each voxel center, query points within voxel radius
        3. Keep voxels with sufficient points
        4. Store voxel centers and point indices
        """
        # Calculate voxel radius (half diagonal of cube)
        voxel_radius = self.voxel_size * np.sqrt(3) / 2

        # Create voxel grid coordinates
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_min, z_max = self.bounds[2]

        # Generate grid coordinates
        x_coords = np.arange(x_min, x_max + self.voxel_size, self.voxel_size)
        y_coords = np.arange(y_min, y_max + self.voxel_size, self.voxel_size)
        z_coords = np.arange(z_min, z_max + self.voxel_size, self.voxel_size)

        print(f"\nCreating voxel grid:")
        print(f"  X: {len(x_coords)} voxels ({x_min:.2f} to {x_max:.2f})")
        print(f"  Y: {len(y_coords)} voxels ({y_min:.2f} to {y_max:.2f})")
        print(f"  Z: {len(z_coords)} voxels ({z_min:.2f} to {z_max:.2f})")
        print(f"  Total potential voxels: {len(x_coords) * len(y_coords) * len(z_coords):,}")
        print(f"  Voxel radius (query): {voxel_radius:.4f} m")
        print(f"  Min points threshold: {self.min_points_threshold}")

        # Track statistics
        total_voxels_tried = 0
        voxels_kept = 0
        voxels_skipped_empty = 0
        voxels_skipped_insufficient = 0

        # Extract voxels
        print("\nExtracting voxels...")
        for x in tqdm(x_coords, desc="Voxel grid X"):
            for y in y_coords:
                for z in z_coords:
                    total_voxels_tried += 1

                    # Voxel center
                    center = np.array([x, y, z])

                    # Query points within voxel radius
                    indices = self.tree.query_ball_point(center, voxel_radius)

                    # Skip empty voxels
                    if len(indices) == 0:
                        voxels_skipped_empty += 1
                        continue

                    # Skip voxels with too few points
                    if len(indices) < self.min_points_threshold:
                        voxels_skipped_insufficient += 1
                        continue

                    # Keep this voxel
                    self.voxels.append({
                        'center': center,
                        'indices': indices,
                        'num_points': len(indices)
                    })
                    voxels_kept += 1

        print(f"\n{'='*60}")
        print("Voxel Grid Creation Complete")
        print(f"{'='*60}")
        print(f"Total voxels tried: {total_voxels_tried:,}")
        print(f"Voxels kept: {voxels_kept:,}")
        print(f"Voxels skipped (empty): {voxels_skipped_empty:,}")
        print(f"Voxels skipped (insufficient points): {voxels_skipped_insufficient:,}")

        if voxels_kept > 0:
            # Analyze voxel statistics
            points_per_voxel = [v['num_points'] for v in self.voxels]
            print(f"\nVoxel statistics:")
            print(f"  Mean points per voxel: {np.mean(points_per_voxel):.1f}")
            print(f"  Median points per voxel: {np.median(points_per_voxel):.1f}")
            print(f"  Min points per voxel: {np.min(points_per_voxel)}")
            print(f"  Max points per voxel: {np.max(points_per_voxel)}")

            # Analyze label statistics (if labels provided)
            if not np.all(self.label_array == -1):
                self._analyze_label_distribution()
        else:
            print("\nWARNING: No voxels were created!")
            print("Try adjusting voxel_size or min_points_threshold")

        print(f"{'='*60}\n")

    def _analyze_label_distribution(self):
        """Analyze distribution of labels across voxels"""
        voxel_label_stats = []

        for voxel in self.voxels:
            indices = voxel['indices']
            voxel_labels = self.label_array[indices]

            # Count label distribution
            labeled_mask = voxel_labels != -1
            n_labeled = np.sum(labeled_mask)
            n_unlabeled = len(voxel_labels) - n_labeled

            if n_labeled > 0:
                labeled_vals = voxel_labels[labeled_mask]
                n_class_0 = np.sum(labeled_vals == 0)
                n_class_1 = np.sum(labeled_vals == 1)

                # Calculate coherence (purity)
                majority_class = 0 if n_class_0 > n_class_1 else 1
                majority_count = max(n_class_0, n_class_1)
                coherence = majority_count / n_labeled if n_labeled > 0 else 0

                voxel_label_stats.append({
                    'n_labeled': n_labeled,
                    'n_unlabeled': n_unlabeled,
                    'n_class_0': n_class_0,
                    'n_class_1': n_class_1,
                    'coherence': coherence,
                    'labeled_ratio': n_labeled / len(voxel_labels)
                })

        if len(voxel_label_stats) > 0:
            # Compute statistics
            coherences = [s['coherence'] for s in voxel_label_stats]
            labeled_ratios = [s['labeled_ratio'] for s in voxel_label_stats]

            print(f"\nLabel distribution across voxels:")
            print(f"  Voxels with labels: {len(voxel_label_stats)} / {len(self.voxels)}")
            print(f"  Average coherence: {np.mean(coherences):.2%} "
                  f"(min: {np.min(coherences):.2%}, max: {np.max(coherences):.2%})")
            print(f"  Average labeled ratio: {np.mean(labeled_ratios):.2%} "
                  f"(min: {np.min(labeled_ratios):.2%}, max: {np.max(labeled_ratios):.2%})")

            # Count voxels by coherence bins
            low_coherence = np.sum(np.array(coherences) < 0.7)
            med_coherence = np.sum((np.array(coherences) >= 0.7) & (np.array(coherences) < 0.9))
            high_coherence = np.sum(np.array(coherences) >= 0.9)

            print(f"\nVoxel coherence distribution:")
            print(f"  Low coherence (<70%): {low_coherence} voxels ({100*low_coherence/len(coherences):.1f}%)")
            print(f"  Medium coherence (70-90%): {med_coherence} voxels ({100*med_coherence/len(coherences):.1f}%)")
            print(f"  High coherence (>90%): {high_coherence} voxels ({100*high_coherence/len(coherences):.1f}%)")

            # Total label counts
            total_class_0 = sum(s['n_class_0'] for s in voxel_label_stats)
            total_class_1 = sum(s['n_class_1'] for s in voxel_label_stats)
            total_labeled = total_class_0 + total_class_1

            print(f"\nTotal labeled points in voxels:")
            print(f"  Class 0 (No Joint): {total_class_0:,} ({100*total_class_0/total_labeled:.1f}%)")
            print(f"  Class 1 (Joint): {total_class_1:,} ({100*total_class_1/total_labeled:.1f}%)")

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        """
        Returns:
            points: [patch_size, C] tensor where C=3 (XYZ) or C=6 (XYZ+RGB)
            labels: [patch_size] tensor with per-point labels (-1 for unlabeled)
        """
        voxel = self.voxels[idx]
        indices = voxel['indices']

        # Get points and labels
        patch_points = self.xyz_array[indices]
        patch_labels = self.label_array[indices]

        if self.use_rgb:
            patch_rgb = self.rgb_array[indices]

        # Sample or pad to exactly patch_size points
        if len(patch_points) >= self.patch_size:
            # Random sample (maintains diversity)
            choice = np.random.choice(len(patch_points), self.patch_size, replace=False)
            selected_indices = np.array(indices)[choice]
            patch_points = patch_points[choice]
            patch_labels = patch_labels[choice]
            if self.use_rgb:
                patch_rgb = patch_rgb[choice]
        else:
            # Pad by repeating points
            n_repeat = self.patch_size - len(patch_points)
            repeat_indices = np.random.choice(len(patch_points), n_repeat, replace=True)
            patch_points = np.vstack([patch_points, patch_points[repeat_indices]])
            patch_labels = np.concatenate([patch_labels, patch_labels[repeat_indices]])
            selected_indices = np.array(list(indices) + [indices[i] for i in repeat_indices])
            if self.use_rgb:
                patch_rgb = np.vstack([patch_rgb, patch_rgb[repeat_indices]])

        # Concatenate XYZ and RGB if using RGB
        if self.use_rgb:
            patch_features = np.hstack([patch_points, patch_rgb])  # [patch_size, 6]
        else:
            patch_features = patch_points  # [patch_size, 3]

        # Normalize
        patch_features = self._normalize(patch_features)

        # Augmentation (training only)
        if self.augment:
            patch_features = self._augment(patch_features)

        return (torch.FloatTensor(patch_features),
                torch.LongTensor(patch_labels),
                torch.LongTensor(selected_indices))

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
            # Zero-mean (relative geometry)
            centroid = xyz.mean(axis=0)
            xyz = xyz - centroid

        elif self.normalize_mode == 'center_scale':
            # Zero-mean + unit sphere
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

        # Random point dropout
        if np.random.random() > 0.5:
            dropout_ratio = np.random.uniform(0, 0.1)
            n_keep = int(len(xyz) * (1 - dropout_ratio))
            keep_indices = np.random.choice(len(xyz), n_keep, replace=False)
            # Duplicate random points to maintain patch_size
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

    def get_voxel_info(self, idx):
        """
        Get information about a specific voxel

        Returns:
            dict with keys: center, indices, num_points, label_stats
        """
        voxel = self.voxels[idx]
        indices = voxel['indices']
        labels = self.label_array[indices]

        # Compute label statistics
        labeled_mask = labels != -1
        n_labeled = np.sum(labeled_mask)

        label_stats = {
            'n_total': len(indices),
            'n_labeled': n_labeled,
            'n_unlabeled': len(indices) - n_labeled
        }

        if n_labeled > 0:
            labeled_vals = labels[labeled_mask]
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
            'center': voxel['center'],
            'indices': indices,
            'num_points': voxel['num_points'],
            'label_stats': label_stats
        }

    def get_spatial_bounds(self):
        """Return the spatial bounds used for voxel grid"""
        return self.bounds

    def get_voxel_size(self):
        """Return the voxel size used"""
        return self.voxel_size


# Convenience function for creating train/test datasets with same voxel grid
def create_train_test_voxel_datasets(xyz_array, label_array, train_mask, test_mask,
                                     patch_size=1024, voxel_size=None,
                                     normalize_mode='center', augment_train=True,
                                     rgb_array=None):
    """
    Create training and test datasets with consistent voxel grids

    Args:
        xyz_array: Full point cloud [N, 3]
        label_array: Labels [N]
        train_mask: Boolean mask for training points [N]
        test_mask: Boolean mask for test points [N]
        patch_size: Target points per voxel
        voxel_size: Voxel size in meters (if None, auto-computed)
        normalize_mode: Normalization method
        augment_train: Whether to augment training data
        rgb_array: Optional RGB colors [N, 3]

    Returns:
        train_dataset, test_dataset
    """
    print("="*70)
    print("CREATING TRAINING VOXEL DATASET")
    print("="*70)

    train_dataset = VoxelDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        patch_size=patch_size,
        voxel_size=voxel_size,
        normalize_mode=normalize_mode,
        augment=augment_train,
        rgb_array=rgb_array[train_mask] if rgb_array is not None else None
    )

    print("\n" + "="*70)
    print("CREATING TEST VOXEL DATASET")
    print("="*70)

    # Use same voxel size as training for consistency
    test_dataset = VoxelDataset(
        xyz_array=xyz_array[test_mask],
        label_array=label_array[test_mask],
        patch_size=patch_size,
        voxel_size=train_dataset.get_voxel_size(),  # Same voxel size!
        normalize_mode=normalize_mode,
        augment=False,
        rgb_array=rgb_array[test_mask] if rgb_array is not None else None
    )

    print("\n" + "="*70)
    print("DATASET CREATION SUMMARY")
    print("="*70)
    print(f"Voxel size: {train_dataset.get_voxel_size():.4f} m (shared)")
    print(f"Training voxels: {len(train_dataset)}")
    print(f"Test voxels: {len(test_dataset)}")
    print(f"Patch size: {patch_size} points")
    print(f"Normalize mode: {normalize_mode}")
    print(f"Augmentation: Train={augment_train}, Test=False")
    print("="*70 + "\n")

    return train_dataset, test_dataset


# Example usage
if __name__ == '__main__':
    # Example: Create voxel datasets
    print("VoxelDataset - Example Usage\n")

    # Simulate some data
    np.random.seed(42)
    n_points = 100000
    xyz = np.random.randn(n_points, 3) * 10  # Random point cloud
    labels = np.random.randint(0, 2, n_points)  # Binary labels

    # Create dataset
    dataset = VoxelDataset(
        xyz_array=xyz,
        label_array=labels,
        patch_size=256,
        voxel_size=4.0,  # 2 meter voxels
        normalize_mode='center',
        augment=False
    )

    print(f"\nDataset created with {len(dataset)} voxels")

    # Get a sample
    if len(dataset) > 0:
        # print(dataset[0])
        points, labels, idx = dataset[0]
        print(f"\nSample voxel:")
        print(f"  Points shape: {points.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels unique: {torch.unique(labels).numpy()}")

        # Get voxel info
        info = dataset.get_voxel_info(0)
        print(f"\nVoxel 0 info:")
        print(f"  Center: {info['center']}")
        print(f"  Total points: {info['num_points']}")
        print(f"  Label stats: {info['label_stats']}")
