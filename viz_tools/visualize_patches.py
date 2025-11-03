"""
Patch Visualization Tool for PointNet++ Training
Generates LAS files to visualize patches in CloudCompare

Two visualization modes:
1. Batch Overview: Shows full point cloud with colored patches from a batch
2. Label Inspection: Shows individual patches with per-point labels highlighted
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import laspy
import os
from tqdm import tqdm
import json

from dataset import RockJointDataset


class PatchVisualizer:
    """
    Visualizes training patches for inspection in CloudCompare
    """

    def __init__(self, dataset, xyz_array, output_dir='./visualizations'):
        """
        Args:
            dataset: RockJointDataset instance
            xyz_array: Full point cloud coordinates [N, 3]
            output_dir: Directory to save visualization files
        """
        self.dataset = dataset
        self.xyz_array = xyz_array
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Store original centroids before normalization for matching
        self._compute_patch_centroids()

    def _compute_patch_centroids(self):
        """Compute centroids of patches before normalization"""
        print("Computing patch centroids for matching...")
        self.patch_centroids = []

        for patch in tqdm(self.dataset.patches, desc="Computing centroids"):
            centroid = patch.mean(axis=0)
            self.patch_centroids.append(centroid)

        self.patch_centroids = np.array(self.patch_centroids)
        print(f"Computed {len(self.patch_centroids)} centroids")

    def visualize_batch_overview(self, batch_idx=0, batch_size=32):
        """
        Mode 1: Visualize a batch of patches on the full point cloud
        Each patch gets a unique color

        Args:
            batch_idx: Which batch to visualize
            batch_size: Number of patches to show
        """
        print(f"\n{'='*60}")
        print(f"Mode 1: Batch Overview Visualization")
        print(f"{'='*60}")

        # Sample patches for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.dataset))
        n_patches = end_idx - start_idx

        print(f"Visualizing patches {start_idx} to {end_idx-1}")
        print(f"Total patches in batch: {n_patches}")

        # Create a color array for the full point cloud
        # Default: gray (128, 128, 128)
        colors = np.ones((len(self.xyz_array), 3), dtype=np.uint16) * 32768
        patch_membership = np.full(len(self.xyz_array), -1, dtype=np.int32)

        # Generate distinct colors for each patch
        patch_colors = self._generate_distinct_colors(n_patches)

        # Match each patch back to the original point cloud
        from scipy.spatial import KDTree
        tree = KDTree(self.xyz_array)

        print("\nMatching patches to original point cloud...")
        for i, patch_idx in enumerate(tqdm(range(start_idx, end_idx))):
            # Get original patch (before normalization)
            original_patch = self.dataset.patches[patch_idx]
            centroid = self.patch_centroids[patch_idx]

            # Find these points in the original cloud
            # Query a radius around the centroid
            search_radius = self._estimate_patch_radius(original_patch)
            candidate_indices = tree.query_ball_point(centroid, search_radius * 1.5)

            if len(candidate_indices) == 0:
                continue

            # Match points by finding nearest neighbors
            candidate_points = self.xyz_array[candidate_indices]
            patch_tree = KDTree(original_patch)

            # For each point in the patch, find its closest match in candidates
            distances, matched_indices = patch_tree.query(candidate_points)

            # Keep only close matches (within tolerance)
            tolerance = 0.01  # 1cm
            close_matches = distances < tolerance

            if np.sum(close_matches) > 0:
                matched_cloud_indices = np.array(candidate_indices)[close_matches]

                # Color these points
                colors[matched_cloud_indices] = patch_colors[i]
                patch_membership[matched_cloud_indices] = patch_idx

        # Create LAS file
        output_path = os.path.join(
            self.output_dir,
            f'batch_overview_batch{batch_idx}_size{n_patches}.las'
        )

        print(f"\nSaving to {output_path}")
        self._save_colored_las(
            self.xyz_array, colors, output_path,
            extra_fields={'patch_id': patch_membership}
        )

        # Save metadata
        metadata = {
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'n_patches': n_patches,
            'patch_range': [start_idx, end_idx-1],
            'patch_colors': patch_colors.tolist()
        }

        metadata_path = os.path.join(
            self.output_dir,
            f'batch_overview_batch{batch_idx}_metadata.json'
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Batch overview saved!")
        print(f"  - Point cloud: {output_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"\nOpen in CloudCompare to inspect colored patches")

    def visualize_label_inspection(self, patch_indices=None, n_patches=10):
        """
        Mode 2: Inspect labels within patches
        Shows patches with different colors for different labels

        Args:
            patch_indices: List of specific patch indices to visualize, or None for random
            n_patches: Number of patches to visualize if patch_indices is None
        """
        print(f"\n{'='*60}")
        print(f"Mode 2: Label Inspection Visualization")
        print(f"{'='*60}")

        if patch_indices is None:
            # Sample random patches
            patch_indices = np.random.choice(
                len(self.dataset),
                min(n_patches, len(self.dataset)),
                replace=False
            )

        print(f"Visualizing {len(patch_indices)} patches")

        # Define label colors
        label_colors = {
            -1: np.array([128, 128, 128]),  # Gray - unlabeled
            0: np.array([0, 255, 0]),        # Green - no joints
            1: np.array([255, 0, 0])         # Red - joints
        }

        for patch_idx in tqdm(patch_indices, desc="Creating patch visualizations"):
            # Get original patch and labels
            original_patch = self.dataset.patches[patch_idx].copy()
            patch_labels = self.dataset.labels[patch_idx].copy()
            centroid = self.patch_centroids[patch_idx]

            # Compute label statistics
            n_labeled = np.sum(patch_labels != -1)
            n_class_0 = np.sum(patch_labels == 0)
            n_class_1 = np.sum(patch_labels == 1)
            n_unlabeled = np.sum(patch_labels == -1)

            # Skip patches with no labeled points
            if n_labeled == 0:
                continue

            # Compute coherence
            labeled_vals = patch_labels[patch_labels != -1]
            majority_class = np.bincount(labeled_vals).argmax()
            coherence = np.sum(labeled_vals == majority_class) / len(labeled_vals)

            # Color points by label
            colors = np.zeros((len(original_patch), 3), dtype=np.uint16)
            for label, color in label_colors.items():
                mask = patch_labels == label
                colors[mask] = (color * 257).astype(np.uint16)  # Convert to 16-bit

            # Create output path
            output_path = os.path.join(
                self.output_dir,
                f'patch_{patch_idx:04d}_coherence{coherence:.2f}.las'
            )

            # Save patch as LAS
            self._save_colored_las(
                original_patch, colors, output_path,
                extra_fields={'label': (patch_labels + 1).astype(np.uint8)}
            )

            # Save metadata
            metadata = {
                'patch_idx': patch_idx,
                'centroid': centroid.tolist(),
                'n_points': len(original_patch),
                'n_labeled': int(n_labeled),
                'n_class_0': int(n_class_0),
                'n_class_1': int(n_class_1),
                'n_unlabeled': int(n_unlabeled),
                'coherence': float(coherence),
                'majority_class': int(majority_class)
            }

            metadata_path = os.path.join(
                self.output_dir,
                f'patch_{patch_idx:04d}_metadata.json'
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"\n✓ Label inspection files saved to {self.output_dir}")
        print(f"\nColor coding:")
        print(f"  🟢 Green  = Class 0 (No Joints)")
        print(f"  🔴 Red    = Class 1 (Joints)")
        print(f"  ⚪ Gray   = Unlabeled")
        print(f"\nOpen individual patch files in CloudCompare to inspect labels")

    def visualize_patch_neighborhoods(self, patch_indices=None, n_patches=5,
                                     neighborhood_radius=2.0):
        """
        Mode 3: Show patches in context with their neighborhoods

        Args:
            patch_indices: Specific patches to visualize
            n_patches: Number of patches if patch_indices is None
            neighborhood_radius: Radius to include surrounding context
        """
        print(f"\n{'='*60}")
        print(f"Mode 3: Patch Neighborhood Visualization")
        print(f"{'='*60}")

        if patch_indices is None:
            patch_indices = np.random.choice(
                len(self.dataset),
                min(n_patches, len(self.dataset)),
                replace=False
            )

        from scipy.spatial import KDTree
        tree = KDTree(self.xyz_array)

        for patch_idx in tqdm(patch_indices, desc="Creating neighborhood views"):
            original_patch = self.dataset.patches[patch_idx].copy()
            patch_labels = self.dataset.labels[patch_idx].copy()
            centroid = self.patch_centroids[patch_idx]

            # Get neighborhood points
            neighbor_indices = tree.query_ball_point(centroid, neighborhood_radius)
            neighborhood_points = self.xyz_array[neighbor_indices]

            # Create combined point cloud: patch + context
            # Patch points in color, context in gray
            n_patch = len(original_patch)
            n_neighbor = len(neighborhood_points)

            combined_points = np.vstack([original_patch, neighborhood_points])

            # Colors
            colors = np.ones((n_patch + n_neighbor, 3), dtype=np.uint16) * 32768

            # Color patch points by label
            label_colors = {
                -1: np.array([128, 128, 128]),
                0: np.array([0, 255, 0]),
                1: np.array([255, 0, 0])
            }

            for label, color in label_colors.items():
                mask = patch_labels == label
                colors[:n_patch][mask] = (color * 257).astype(np.uint16)

            # Mark patch vs context
            patch_marker = np.zeros(n_patch + n_neighbor, dtype=np.uint8)
            patch_marker[:n_patch] = 1  # 1 = patch, 0 = context

            # Save
            output_path = os.path.join(
                self.output_dir,
                f'neighborhood_patch_{patch_idx:04d}.las'
            )

            self._save_colored_las(
                combined_points, colors, output_path,
                extra_fields={
                    'is_patch': patch_marker,
                    'label': np.concatenate([
                        (patch_labels + 1).astype(np.uint8),
                        np.zeros(n_neighbor, dtype=np.uint8)
                    ])
                }
            )

        print(f"\n✓ Neighborhood visualizations saved to {self.output_dir}")
        print(f"\nVisualization shows:")
        print(f"  - Colored points: patch points with labels")
        print(f"  - Gray points: surrounding context")

    def _estimate_patch_radius(self, patch):
        """Estimate the spatial radius of a patch"""
        centroid = patch.mean(axis=0)
        distances = np.linalg.norm(patch - centroid, axis=1)
        return np.percentile(distances, 95)  # 95th percentile

    def _generate_distinct_colors(self, n):
        """Generate N visually distinct colors"""
        colors = []

        if n <= 12:
            # Use predefined distinct colors for small N
            base_colors = [
                [255, 0, 0],      # Red
                [0, 255, 0],      # Green
                [0, 0, 255],      # Blue
                [255, 255, 0],    # Yellow
                [255, 0, 255],    # Magenta
                [0, 255, 255],    # Cyan
                [255, 128, 0],    # Orange
                [128, 0, 255],    # Purple
                [0, 255, 128],    # Spring green
                [255, 0, 128],    # Rose
                [128, 255, 0],    # Chartreuse
                [0, 128, 255],    # Sky blue
            ]
            colors = base_colors[:n]
        else:
            # Generate colors using HSV space
            import colorsys
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
                colors.append([int(c * 255) for c in rgb])

        # Convert to 16-bit
        colors = np.array(colors, dtype=np.uint16) * 257
        return colors

    def _save_colored_las(self, xyz, colors, output_path, extra_fields=None):
        """
        Save point cloud with colors to LAS file

        Args:
            xyz: [N, 3] coordinates
            colors: [N, 3] RGB colors (uint16)
            output_path: Output file path
            extra_fields: Dict of additional fields to add
        """
        # Create LAS header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = xyz.min(axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])

        # Create LAS data
        las = laspy.LasData(header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]

        # Add colors
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        # Add extra fields if provided
        if extra_fields:
            for name, data in extra_fields.items():
                try:
                    # Determine dtype
                    if data.dtype == np.float32 or data.dtype == np.float64:
                        dtype = np.float32
                    elif data.dtype == np.int8:
                        dtype = np.int8
                    elif data.dtype == np.uint8:
                        dtype = np.uint8
                    elif data.dtype == np.int16:
                        dtype = np.int16
                    elif data.dtype == np.uint16:
                        dtype = np.uint16
                    elif data.dtype == np.int32:
                        dtype = np.int32
                    else:
                        dtype = np.int32

                    las.add_extra_dim(laspy.ExtraBytesParams(
                        name=name,
                        type=dtype
                    ))
                    setattr(las, name, data.astype(dtype))
                except Exception as e:
                    print(f"Warning: Could not add field '{name}': {e}")

        # Write file
        las.write(output_path)

    def create_summary_visualization(self):
        """Create a summary showing all patches colored by their majority class"""
        print(f"\n{'='*60}")
        print(f"Creating Summary Visualization")
        print(f"{'='*60}")

        # Color the full point cloud by patch membership and class
        colors = np.ones((len(self.xyz_array), 3), dtype=np.uint16) * 32768
        patch_ids = np.full(len(self.xyz_array), -1, dtype=np.int32)
        patch_classes = np.full(len(self.xyz_array), -1, dtype=np.int8)

        from scipy.spatial import KDTree
        tree = KDTree(self.xyz_array)

        # Define class colors
        class_colors = {
            0: np.array([0, 255, 0]),    # Green - no joints
            1: np.array([255, 0, 0])     # Red - joints
        }

        print("Matching all patches...")
        for patch_idx in tqdm(range(len(self.dataset))):
            original_patch = self.dataset.patches[patch_idx]
            patch_labels = self.dataset.labels[patch_idx]
            centroid = self.patch_centroids[patch_idx]

            # Find majority class
            labeled_vals = patch_labels[patch_labels != -1]
            if len(labeled_vals) == 0:
                continue

            majority_class = np.bincount(labeled_vals).argmax()

            # Match to original cloud
            search_radius = self._estimate_patch_radius(original_patch)
            candidate_indices = tree.query_ball_point(centroid, search_radius * 1.5)

            if len(candidate_indices) == 0:
                continue

            candidate_points = self.xyz_array[candidate_indices]
            patch_tree = KDTree(original_patch)
            distances, _ = patch_tree.query(candidate_points)

            close_matches = distances < 0.01
            if np.sum(close_matches) > 0:
                matched_indices = np.array(candidate_indices)[close_matches]

                # Color by majority class
                color = class_colors[majority_class]
                colors[matched_indices] = (color * 257).astype(np.uint16)
                patch_ids[matched_indices] = patch_idx
                patch_classes[matched_indices] = majority_class

        # Save
        output_path = os.path.join(self.output_dir, 'summary_all_patches.las')
        print(f"\nSaving summary to {output_path}")

        self._save_colored_las(
            self.xyz_array, colors, output_path,
            extra_fields={
                'patch_id': patch_ids,
                'patch_class': (patch_classes + 1).astype(np.uint8)
            }
        )

        print(f"\n✓ Summary visualization saved!")
        print(f"\nColor coding:")
        print(f"  🟢 Green  = Patches with majority Class 0 (No Joints)")
        print(f"  🔴 Red    = Patches with majority Class 1 (Joints)")
        print(f"  ⚪ Gray   = Not covered by any patch")


def main():
    """
    Main function to run patch visualizations
    """
    print("\n" + "="*70)
    print("PATCH VISUALIZATION TOOL")
    print("="*70)

    # =========================
    # Configuration
    # =========================
    config = {
        'data_ckpt': './data/preprocessed/preprocessed_data.npz',
        'polygon_ckpt': './data/preprocessed/polygons_dict.pkl',

        'patch_size': 1024,
        'normalize_mode': 'center_scale',
        'augment_train': False,

        'output_dir': './visualizations',

        # Visualization parameters
        'batch_idx': 0,
        'batch_size': 32,
        'n_label_inspect': 20,
        'n_neighborhood': 5,
    }

    # =========================
    # Load Data
    # =========================
    print("\nLoading preprocessed data...")
    data = np.load(config['data_ckpt'])
    xyz_array = data['xyz_array']
    label_array = data['label_array']
    train_mask = data['train_mask']
    test_mask = data['test_mask']

    import pickle
    with open(config['polygon_ckpt'], 'rb') as f:
        polygons_data = pickle.load(f)
        polygons_dict_train = polygons_data['train']
        polygons_dict_test = polygons_data['test']

    print(f"Loaded {len(xyz_array):,} points")
    print(f"Training points: {np.sum(train_mask):,}")
    print(f"Test points: {np.sum(test_mask):,}")

    # =========================
    # Create Dataset
    # =========================
    print("\nCreating training dataset...")
    train_dataset = RockJointDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        polygons_dict=polygons_dict_train,
        patch_size=config['patch_size'],
        normalize_mode=config['normalize_mode'],
        augment=config['augment_train']
    )

    # =========================
    # Create Visualizer
    # =========================
    print("\nInitializing visualizer...")
    visualizer = PatchVisualizer(
        dataset=train_dataset,
        xyz_array=xyz_array[train_mask],
        output_dir=config['output_dir']
    )

    # =========================
    # Generate Visualizations
    # =========================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Mode 1: Batch overview
    visualizer.visualize_batch_overview(
        batch_idx=config['batch_idx'],
        batch_size=config['batch_size']
    )

    # Mode 2: Label inspection
    visualizer.visualize_label_inspection(
        n_patches=config['n_label_inspect']
    )

    # Mode 3: Neighborhood context
    visualizer.visualize_patch_neighborhoods(
        n_patches=config['n_neighborhood']
    )

    # Summary visualization
    visualizer.create_summary_visualization()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll files saved to: {config['output_dir']}/")
    print("\nRecommended CloudCompare workflow:")
    print("  1. Open 'batch_overview_*.las' to see patch distribution")
    print("  2. Open individual 'patch_*.las' files to inspect labels")
    print("  3. Open 'neighborhood_*.las' to see patches in context")
    print("  4. Open 'summary_all_patches.las' for overall view")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
