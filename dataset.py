"""
Dataset class for Rock Joint Point Cloud Classification
Handles patch extraction, normalization, and data loading
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import ConvexHull
import laspy


class RockJointDataset(Dataset):
    """
    Dataset for binary classification of rock joints from point cloud patches
    """
    def __init__(self, xyz_array, label_array, polygons_dict,
                 patch_size=2048, normalize_mode='none', augment=False, rgb_array=None,
                 min_patch_distance=None):
        """
        Args:
            xyz_array: Full point cloud coordinates [N, 3]
            label_array: Labels for each point [N]
            polygons_dict: Dict with 'label_0': [...polygons...], 'label_1': [...polygons...]
            patch_size: Number of points per patch
            normalize_mode: 'none', 'center', or 'center_scale'
            augment: Whether to apply data augmentation
            rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
            min_patch_distance: Minimum distance between patch centers (for spatial subsampling)
                               If None, no spatial subsampling is applied
        """
        self.xyz_array = xyz_array
        self.rgb_array = rgb_array
        self.label_array = label_array
        self.polygons_dict = polygons_dict
        self.patch_size = patch_size
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.use_rgb = rgb_array is not None
        self.min_patch_distance = min_patch_distance

        # Extract patches centered on polygon centroids
        self.patches = []  # Will store XYZ or XYZ+RGB
        self.labels = []
        # self._extract_patches()
        self._extract_patches_with_coherence()
        self.labels = np.array(self.labels)

        feature_dim = 6 if self.use_rgb else 3
        print(f"Dataset created with {len(self.patches)} patches ({feature_dim}D features)")
        print(f"  Class 0 (No Joints): {sum(self.labels == 0)} patches")
        print(f"  Class 1 (Joints): {sum(self.labels == 1)} patches")

    # def _extract_patches(self):
    #     """
    #     Extract patches with PER-POINT labels for segmentation.
    #     Creates a spatial grid of patch centers and assigns labels to each point.

    #     Algorithm details:
    #     - Randomly sample atleast 10k points
    #     - Estimate the mean distance between at least K of their NN
    #     - This gives us an idea about point density we would like per patch
    #     - Since we know the density we have, we can estimate the patch radius, also helps with finding spatial stride.
    #     - For each {x/y/z} in sliding window:
    #       - Query radius, if not dense enough or not labelled enough, reject.
    #       - else: we either randomly sample or pad the patch depending on number of points in patch
    #     """
    #     from scipy.spatial import KDTree

    #     # Build KDTree for efficient nearest neighbor search
    #     tree = KDTree(self.xyz_array)

    #     # Get bounds of labeled points only
    #     labeled_mask = self.label_array != -1
    #     labeled_points = self.xyz_array[labeled_mask]

    #     if len(labeled_points) == 0:
    #         print("WARNING: No labeled points found!")
    #         return

    #     # Compute bounds for patch center sampling
    #     min_bounds = labeled_points.min(axis=0)
    #     max_bounds = labeled_points.max(axis=0)

    #     # Estimate point density to determine spatial stride
    #     sample_size = min(10000, len(labeled_points))
    #     sample_indices = np.random.choice(len(labeled_points), sample_size, replace=False)
    #     sample_points = labeled_points[sample_indices]
    #     #randomly sample 10k from all labelled points. Why?

    #     # Build KDTree for sample to estimate spacing
    #     sample_tree = KDTree(sample_points)
    #     k = min(50, sample_size - 1)
    #     distances, _ = sample_tree.query(sample_points, k=k+1) #Find KNN for 10k sample points
    #     avg_point_spacing = np.mean(distances[:, 1:])  # Exclude self

    #     print(f"Average point spacing: {avg_point_spacing:.4f} m")

    #     # Determine patch radius (should contain ~patch_size points)
    #     # Heuristic: radius such that sphere contains roughly patch_size points
    #     # Volume = (4/3) * pi * r^3
    #     # point_density ≈ patch_size / volume
    #     point_density = 1.0 / (avg_point_spacing ** 3)
    #     target_volume = self.patch_size / point_density
    #     patch_radius = (3 * target_volume / (4 * np.pi)) ** (1/3)

    #     print(f"Estimated patch radius: {patch_radius:.4f} m")

    #     # Create spatial grid for patch centers
    #     # Stride = patch_radius to ensure coverage with some overlap
    #     spatial_stride = patch_radius * 0.75  # 25% overlap

    #     x_range = np.arange(min_bounds[0], max_bounds[0] + spatial_stride, spatial_stride)
    #     y_range = np.arange(min_bounds[1], max_bounds[1] + spatial_stride, spatial_stride)
    #     z_range = np.arange(min_bounds[2], max_bounds[2] + spatial_stride, spatial_stride)

    #     print(f"\nCreating spatial grid:")
    #     print(f"  X: {len(x_range)} samples")
    #     print(f"  Y: {len(y_range)} samples")
    #     print(f"  Z: {len(z_range)} samples")
    #     print(f"  Total potential patches: {len(x_range) * len(y_range) * len(z_range)}")

    #     patch_count = 0
    #     skipped_insufficient = 0
    #     skipped_unlabeled = 0

    #     # Generate patches from grid
    #     for x in x_range:
    #         for y in y_range:
    #             for z in z_range:
    #                 center = np.array([x, y, z])

    #                 # Query points within radius
    #                 indices = tree.query_ball_point(center, patch_radius)

    #                 # Patch not dense enough
    #                 if len(indices) < self.patch_size // 4:
    #                     skipped_insufficient += 1
    #                     continue

    #                 # Get points and labels per point
    #                 patch_points = self.xyz_array[indices]
    #                 patch_labels = self.label_array[indices]

    #                 # Check if patch has any labeled points
    #                 labeled_in_patch = np.sum(patch_labels != -1)

    #                 # Skip patches with too few labeled points
    #                 if labeled_in_patch < self.patch_size // 8:
    #                     skipped_unlabeled += 1
    #                     continue

    #                 # Sample or pad to exactly patch_size
    #                 if len(patch_points) >= self.patch_size:
    #                     # Prefer sampling labeled points if available
    #                     labeled_mask_patch = patch_labels != -1
    #                     n_labeled = np.sum(labeled_mask_patch)

    #                     if n_labeled >= self.patch_size:
    #                         # Sample only from labeled points
    #                         labeled_indices = np.where(labeled_mask_patch)[0]
    #                         choice = np.random.choice(labeled_indices, self.patch_size, replace=False)
    #                     elif n_labeled > 0:
    #                         # Include all labeled points, fill rest with random
    #                         labeled_indices = np.where(labeled_mask_patch)[0]
    #                         unlabeled_indices = np.where(~labeled_mask_patch)[0]
    #                         n_unlabeled_needed = self.patch_size - n_labeled

    #                         if len(unlabeled_indices) >= n_unlabeled_needed:
    #                             unlabeled_choice = np.random.choice(
    #                                 unlabeled_indices, n_unlabeled_needed, replace=False
    #                             )
    #                         else:
    #                             unlabeled_choice = unlabeled_indices

    #                         choice = np.concatenate([labeled_indices, unlabeled_choice])

    #                         # If still not enough, sample with replacement
    #                         if len(choice) < self.patch_size:
    #                             additional = np.random.choice(
    #                                 len(patch_points),
    #                                 self.patch_size - len(choice),
    #                                 replace=True
    #                             )
    #                             choice = np.concatenate([choice, additional])
    #                     else:
    #                         # Random sampling when n_labelled == 0
    #                         choice = np.random.choice(len(patch_points), self.patch_size, replace=False)

    #                     patch_points = patch_points[choice]
    #                     patch_labels = patch_labels[choice]
    #                 else:
    #                     # Pad by repeating random points
    #                     n_repeat = self.patch_size - len(patch_points)
    #                     repeat_indices = np.random.choice(len(patch_points), n_repeat, replace=True)
    #                     patch_points = np.vstack([patch_points, patch_points[repeat_indices]])
    #                     patch_labels = np.concatenate([patch_labels, patch_labels[repeat_indices]])

    #                 # Store patch with per-point labels
    #                 self.patches.append(patch_points)
    #                 self.labels.append(patch_labels)
    #                 patch_count += 1

    #                 if patch_count % 100 == 0:
    #                     print(f"  Extracted {patch_count} patches...")

    #     # Convert labels to array for easier manipulation
    #     self.labels = np.array(self.labels)  # Shape: [num_patches, patch_size]

    #     print(f"\n=== Patch Extraction Complete ===")
    #     print(f"Total patches extracted: {len(self.patches)}")
    #     print(f"Skipped (insufficient points): {skipped_insufficient}")
    #     print(f"Skipped (too few labeled points): {skipped_unlabeled}")

    #     # Analyze label distribution across patches
    #     if len(self.labels) > 0:
    #         total_points = len(self.labels) * self.patch_size
    #         class_0_points = np.sum(self.labels == 0)
    #         class_1_points = np.sum(self.labels == 1)
    #         unlabeled_points = np.sum(self.labels == -1)

    #         print(f"\nLabel distribution across all patches:")
    #         print(f"  Class 0 (No Joints): {class_0_points:,} points ({100*class_0_points/total_points:.2f}%)")
    #         print(f"  Class 1 (Joints): {class_1_points:,} points ({100*class_1_points/total_points:.2f}%)")
    #         print(f"  Unlabeled: {unlabeled_points:,} points ({100*unlabeled_points/total_points:.2f}%)")

    #         # Per-patch statistics
    #         avg_labeled_per_patch = np.mean(np.sum(self.labels != -1, axis=1))
    #         print(f"\nAverage labeled points per patch: {avg_labeled_per_patch:.1f} / {self.patch_size}")
    #     else:
    #         print("WARNING: No patches were extracted!")

    def _extract_patches_with_coherence(self):
        """
        Extract patches with spatial coherence - nearby points should have similar labels.
        Centers patches on labeled points and filters out heterogeneous patches.
        """
        from scipy.spatial import KDTree

        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(self.xyz_array)

        # Get only labeled points as potential patch centers
        labeled_mask = self.label_array != -1
        labeled_indices = np.where(labeled_mask)[0]

        if len(labeled_indices) == 0:
            print("WARNING: No labeled points found!")
            return

        print(f"Total labeled points: {len(labeled_indices):,}")
        print(f"  Class 0: {np.sum(self.label_array == 0):,}")
        print(f"  Class 1: {np.sum(self.label_array == 1):,}")

        # Estimate point density for patch radius
        sample_size = min(10000, len(labeled_indices))
        sample_indices = np.random.choice(labeled_indices, sample_size, replace=False)
        sample_points = self.xyz_array[sample_indices]

        sample_tree = KDTree(sample_points)
        k = min(50, sample_size - 1)
        distances, _ = sample_tree.query(sample_points, k=k+1)
        avg_point_spacing = np.mean(distances[:, 1:])

        print(f"Average point spacing: {avg_point_spacing:.4f} m")

        # Calculate patch radius to contain ~patch_size points
        point_density = 1.0 / (avg_point_spacing ** 3)
        target_volume = self.patch_size / point_density
        patch_radius = (3 * target_volume / (4 * np.pi)) ** (1/3)

        print(f"Target patch radius: {patch_radius:.4f} m")

        # Determine number of patches to sample
        # Aim for good coverage: sample more centers than we expect to keep
        n_centers_to_try = min(len(labeled_indices), 5000)  # Try up to 5000 patches

        # Sample potential patch centers from labeled points
        print(f"\nSampling {n_centers_to_try} potential patch centers...")
        center_indices = np.random.choice(labeled_indices, size=n_centers_to_try, replace=False)

        # Extraction parameters
        min_coherence = 0.40  # At least 70% of labeled points must match center label
        min_labeled_ratio = 0.20  # At least 50% of points in patch must be labeled

        print(f"Coherence threshold: {min_coherence:.0%}")
        print(f"Min labeled ratio: {min_labeled_ratio:.0%}")

        # Track statistics
        patches_extracted = 0
        skipped_insufficient = 0
        skipped_unlabeled = 0
        skipped_incoherent = 0
        skipped_too_close = 0

        # Track patch centers for spatial subsampling
        accepted_patch_centers = []

        # Build KDTree for accepted patches (updated dynamically)
        accepted_tree = None

        print("\nExtracting patches...")
        if self.min_patch_distance is not None:
            print(f"Spatial subsampling enabled: min distance = {self.min_patch_distance:.2f} m")

        for idx, center_idx in enumerate(center_indices):
            if idx % 500 == 0 and idx > 0:
                print(f"  Processed {idx}/{n_centers_to_try} centers... "
                      f"Extracted: {patches_extracted}, "
                      f"Skipped (insufficient): {skipped_insufficient}, "
                      f"Skipped (unlabeled): {skipped_unlabeled}, "
                      f"Skipped (incoherent): {skipped_incoherent}, "
                      f"Skipped (too close): {skipped_too_close}")

            center = self.xyz_array[center_idx]
            center_label = self.label_array[center_idx]

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

            # Query points within radius
            indices = tree.query_ball_point(center, patch_radius)

            # Need enough points
            if len(indices) < self.patch_size // 4:
                skipped_insufficient += 1
                continue

            # Get points and labels
            patch_points = self.xyz_array[indices]
            patch_labels = self.label_array[indices]
            if self.use_rgb:
                patch_rgb = self.rgb_array[indices]

            # Check labeled ratio
            n_labeled = np.sum(patch_labels != -1)
            labeled_ratio = n_labeled / len(patch_labels)

            if labeled_ratio < min_labeled_ratio:
                skipped_unlabeled += 1
                continue

            # Check coherence: what percentage of labeled points match the center label?
            labeled_points_mask = patch_labels != -1
            labeled_point_labels = patch_labels[labeled_points_mask]

            if len(labeled_point_labels) == 0:
                skipped_unlabeled += 1
                continue

            coherence = np.sum(labeled_point_labels == center_label) / len(labeled_point_labels)

            if coherence < min_coherence:
                skipped_incoherent += 1
                continue

            # This patch passes all filters - now sample/pad to exact patch_size
            if len(patch_points) >= self.patch_size:
                # Prefer keeping labeled points
                labeled_mask_patch = patch_labels != -1
                labeled_indices_patch = np.where(labeled_mask_patch)[0]
                unlabeled_indices_patch = np.where(~labeled_mask_patch)[0]

                n_labeled_in_patch = len(labeled_indices_patch)

                if n_labeled_in_patch >= self.patch_size:
                    # Sample only from labeled points
                    choice = np.random.choice(labeled_indices_patch, self.patch_size, replace=False)
                elif n_labeled_in_patch > 0:
                    # Take all labeled, fill rest with unlabeled
                    n_unlabeled_needed = self.patch_size - n_labeled_in_patch

                    if len(unlabeled_indices_patch) >= n_unlabeled_needed:
                        unlabeled_choice = np.random.choice(
                            unlabeled_indices_patch, n_unlabeled_needed, replace=False
                        )
                    else:
                        # Not enough unlabeled - sample with replacement
                        unlabeled_choice = np.random.choice(
                            unlabeled_indices_patch, n_unlabeled_needed, replace=True
                        )

                    choice = np.concatenate([labeled_indices_patch, unlabeled_choice])
                else:
                    # All unlabeled (shouldn't happen due to earlier filter)
                    choice = np.random.choice(len(patch_points), self.patch_size, replace=False)

                patch_points = patch_points[choice]
                patch_labels = patch_labels[choice]
                if self.use_rgb:
                    patch_rgb = patch_rgb[choice]
            else:
                # Pad by repeating random points
                n_repeat = self.patch_size - len(patch_points)
                repeat_indices = np.random.choice(len(patch_points), n_repeat, replace=True)
                patch_points = np.vstack([patch_points, patch_points[repeat_indices]])
                patch_labels = np.concatenate([patch_labels, patch_labels[repeat_indices]])
                if self.use_rgb:
                    patch_rgb = np.vstack([patch_rgb, patch_rgb[repeat_indices]])

            # Concatenate XYZ and RGB if using RGB
            if self.use_rgb:
                patch_features = np.hstack([patch_points, patch_rgb])  # [patch_size, 6]
            else:
                patch_features = patch_points  # [patch_size, 3]

            # Store patch
            self.patches.append(patch_features)
            self.labels.append(patch_labels)
            patches_extracted += 1

            # Track this patch center for spatial subsampling
            if self.min_patch_distance is not None:
                accepted_patch_centers.append(center)

        # Convert labels to array
        self.labels = np.array(self.labels)  # Shape: [num_patches, patch_size]

        print(f"\n{'='*60}")
        print("Patch Extraction Complete")
        print(f"{'='*60}")
        print(f"Total patches extracted: {patches_extracted}")
        print(f"Skipped (insufficient points): {skipped_insufficient}")
        print(f"Skipped (too few labeled): {skipped_unlabeled}")
        print(f"Skipped (incoherent/mixed): {skipped_incoherent}")
        if self.min_patch_distance is not None:
            print(f"Skipped (too close to existing): {skipped_too_close}")

        if patches_extracted == 0:
            print("\nWARNING: No patches were extracted!")
            print("Try adjusting:")
            print("  - min_coherence (currently 0.70)")
            print("  - min_labeled_ratio (currently 0.50)")
            print("  - patch_radius")
            return

        # Analyze extracted patches
        print(f"\n{'='*60}")
        print("Patch Analysis")
        print(f"{'='*60}")

        total_points = len(self.labels) * self.patch_size
        class_0_points = np.sum(self.labels == 0)
        class_1_points = np.sum(self.labels == 1)
        unlabeled_points = np.sum(self.labels == -1)

        print(f"\nLabel distribution across all patches:")
        print(f"  Class 0 (No Joints): {class_0_points:,} points ({100*class_0_points/total_points:.2f}%)")
        print(f"  Class 1 (Joints):    {class_1_points:,} points ({100*class_1_points/total_points:.2f}%)")
        print(f"  Unlabeled:           {unlabeled_points:,} points ({100*unlabeled_points/total_points:.2f}%)")

        # Per-patch coherence statistics
        patch_coherences = []
        patch_labeled_ratios = []

        for patch_labels in self.labels:
            labeled_mask = patch_labels != -1
            n_labeled = np.sum(labeled_mask)

            if n_labeled > 0:
                labeled_ratio = n_labeled / len(patch_labels)
                patch_labeled_ratios.append(labeled_ratio)

                # Calculate coherence (how pure is this patch?)
                labeled_vals = patch_labels[labeled_mask]
                majority_class = np.bincount(labeled_vals).argmax()
                coherence = np.sum(labeled_vals == majority_class) / len(labeled_vals)
                patch_coherences.append(coherence)

        if len(patch_coherences) > 0:
            print(f"\nPatch quality metrics:")
            print(f"  Average coherence: {np.mean(patch_coherences):.2%} "
                  f"(min: {np.min(patch_coherences):.2%}, max: {np.max(patch_coherences):.2%})")
            print(f"  Average labeled ratio: {np.mean(patch_labeled_ratios):.2%} "
                  f"(min: {np.min(patch_labeled_ratios):.2%}, max: {np.max(patch_labeled_ratios):.2%})")

        # Distribution of patches by majority class
        patch_majority_classes = []
        for patch_labels in self.labels:
            labeled_vals = patch_labels[patch_labels != -1]
            if len(labeled_vals) > 0:
                majority = np.bincount(labeled_vals).argmax()
                patch_majority_classes.append(majority)

        if len(patch_majority_classes) > 0:
            class_0_patches = np.sum(np.array(patch_majority_classes) == 0)
            class_1_patches = np.sum(np.array(patch_majority_classes) == 1)

            print(f"\nPatch distribution by majority class:")
            print(f"  Majority Class 0: {class_0_patches} patches ({100*class_0_patches/len(patch_majority_classes):.1f}%)")
            print(f"  Majority Class 1: {class_1_patches} patches ({100*class_1_patches/len(patch_majority_classes):.1f}%)")

        print(f"\n{'='*60}\n")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Returns:
            points: [patch_size, 3] tensor
            label: scalar tensor
        """
        patch = self.patches[idx].copy()
        labels = self.labels[idx].copy()

        # Normalize
        patch = self._normalize(patch)

        # Augmentation
        if self.augment:
            patch = self._augment(patch)

        return torch.FloatTensor(patch), torch.LongTensor(labels)

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


class RockJointInferenceDataset(Dataset):
    """
    Dataset for inference on full point cloud using sliding window
    """
    def __init__(self, xyz_array, patch_size=2048, stride=1024, normalize_mode='none', rgb_array=None):
        """
        Args:
            xyz_array: Full point cloud coordinates [N, 3]
            patch_size: Number of points per patch
            stride: Overlap between patches (in points)
            normalize_mode: Same as training
            rgb_array: Optional RGB colors [N, 3] normalized to [0, 1]
        """
        self.xyz_array = xyz_array
        self.rgb_array = rgb_array
        self.patch_size = patch_size
        self.stride = stride
        self.normalize_mode = normalize_mode
        self.use_rgb = rgb_array is not None

        # Create spatial grid for patch extraction
        self.patch_centers, self.patch_indices = self._create_spatial_grid()

        feature_dim = 6 if self.use_rgb else 3
        print(f"Inference dataset created with {len(self.patch_centers)} patches ({feature_dim}D features)")

    def _create_spatial_grid(self):
        """
        Create overlapping spatial grid for patch extraction
        """
        from scipy.spatial import KDTree

        tree = KDTree(self.xyz_array)

        # Determine grid spacing based on point cloud bounds
        min_bounds = self.xyz_array.min(axis=0)
        max_bounds = self.xyz_array.max(axis=0)

        # Create grid with spacing = stride
        # Convert stride (number of points) to spatial distance
        # Estimate point density
        sample_size = min(10000, len(self.xyz_array))
        sample_indices = np.random.choice(len(self.xyz_array), sample_size, replace=False)
        sample_points = self.xyz_array[sample_indices]

        # Estimate average distance to k-th nearest neighbor
        k = min(50, sample_size - 1)
        distances, _ = tree.query(sample_points, k=k+1)
        avg_spacing = np.mean(distances[:, 1:])  # Exclude self

        # Spatial stride
        spatial_stride = avg_spacing * np.sqrt(self.stride)

        # Create grid
        x_range = np.arange(min_bounds[0], max_bounds[0], spatial_stride)
        y_range = np.arange(min_bounds[1], max_bounds[1], spatial_stride)
        z_range = np.arange(min_bounds[2], max_bounds[2], spatial_stride)

        patch_centers = []
        patch_indices = []

        print(f"Creating spatial grid: {len(x_range)} x {len(y_range)} x {len(z_range)}")

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    center = np.array([x, y, z])

                    # Query points within radius
                    # radius = spatial_stride * 1.5 ## Note: WHAT?
                    indices = tree.query_ball_point(center, spatial_stride)

                    if len(indices) < self.patch_size // 4:
                        continue

                    patch_centers.append(center)
                    patch_indices.append(indices)

        return patch_centers, patch_indices

    def __len__(self):
        return len(self.patch_centers)

    def __getitem__(self, idx):
        """
        Returns:
            points: [patch_size, 3 or 6] tensor
            indices: point indices in original cloud
        """
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
