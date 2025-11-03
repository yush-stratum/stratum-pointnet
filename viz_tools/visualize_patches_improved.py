"""
Improved Patch Visualization with Centroid Mapping
Stores original centroids during dataset creation to enable proper matching after normalization
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import laspy
import os
import pickle
from tqdm import tqdm
import json
import colorsys

os.sys.path.insert(0,'..')

def create_patch_centroid_mapping(dataset):
    """
    Create mapping between patches and their original centroids
    Run this once after dataset creation

    Args:
        dataset: RockJointDataset instance

    Returns:
        centroids: [N_patches, 3] array of original centroids
    """
    print("Creating patch centroid mapping...")
    centroids = []

    for patch in tqdm(dataset.patches):
        # Compute centroid of original (pre-normalization) patch
        centroid = patch.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    return centroids


class ImprovedPatchVisualizer:
    """
    Visualizer that properly handles normalized patches
    """

    def __init__(self, dataset, xyz_array_full, train_mask, output_dir='./visualizations'):
        """
        Args:
            dataset: RockJointDataset
            xyz_array_full: Full point cloud [N_total, 3]
            train_mask: Boolean mask for training points
            output_dir: Output directory
        """
        self.dataset = dataset
        self.xyz_full = xyz_array_full
        self.xyz_train = xyz_array_full[train_mask]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build mapping
        print("Building centroid mapping...")
        self.centroids = create_patch_centroid_mapping(dataset)

        # Build KDTree for fast lookup
        from scipy.spatial import KDTree
        self.tree_train = KDTree(self.xyz_train)

        print(f"Visualizer ready with {len(self.centroids)} patches")

    def visualize_mode1_batch_overview(self, batch_idx=0, batch_size=32,
                                      use_full_cloud=False):
        """
        Mode 1: Show full/train point cloud with colored patches from a batch

        Args:
            batch_idx: Which batch to visualize
            batch_size: Size of batch
            use_full_cloud: If True, show full cloud; if False, only training region
        """
        print(f"\n{'='*70}")
        print(f"Mode 1: Batch Overview Visualization")
        print(f"{'='*70}")

        # Select patches for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.dataset))
        n_patches = end_idx - start_idx

        if n_patches == 0:
            print("No patches in this batch!")
            return

        print(f"Batch {batch_idx}: patches {start_idx} to {end_idx-1} ({n_patches} total)")

        # Choose base cloud
        if use_full_cloud:
            base_cloud = self.xyz_full
            cloud_name = "full"
        else:
            base_cloud = self.xyz_train
            cloud_name = "train"

        # Initialize colors (gray background)
        colors = np.ones((len(base_cloud), 3), dtype=np.uint16) * 32768
        patch_id_field = np.full(len(base_cloud), -1, dtype=np.int32)

        # Generate distinct colors for each patch
        patch_colors = self._generate_distinct_colors(n_patches)

        # Build KDTree for base cloud
        from scipy.spatial import KDTree
        base_tree = KDTree(base_cloud)

        print(f"Coloring {n_patches} patches on {cloud_name} cloud...")

        for i, patch_idx in enumerate(tqdm(range(start_idx, end_idx))):
            # Get original patch and its centroid
            original_patch = self.dataset.patches[patch_idx]
            centroid = self.centroids[patch_idx]

            # Find points in base cloud near this centroid
            search_radius = self._estimate_patch_radius(original_patch) * 1.2
            candidate_indices = base_tree.query_ball_point(centroid, search_radius)

            if len(candidate_indices) == 0:
                continue

            # Get candidate points
            candidate_points = base_cloud[candidate_indices]

            # Match patch points to candidates using nearest neighbor
            from scipy.spatial import KDTree as KDTree2
            candidate_tree = KDTree2(candidate_points)

            # For each point in original patch, find closest in candidates
            distances, indices = candidate_tree.query(original_patch)

            # Keep only close matches (within tolerance)
            tolerance = 0.02  # 2cm
            close_matches = distances < tolerance

            if np.sum(close_matches) < len(original_patch) * 0.3:
                # Too few matches, skip
                continue

            # Map back to base cloud indices
            matched_local = indices[close_matches]
            matched_global = np.array(candidate_indices)[matched_local]

            # Color these points
            colors[matched_global] = patch_colors[i]
            patch_id_field[matched_global] = patch_idx

        # Save
        output_path = os.path.join(
            self.output_dir,
            f'mode1_batch{batch_idx}_{cloud_name}_cloud.las'
        )

        print(f"\nSaving to {output_path}...")
        self._save_las(base_cloud, colors, output_path, {
            'patch_id': patch_id_field
        })

        # Save metadata (convert all numpy types to Python native types)
        metadata = {
            'visualization_mode': 'batch_overview',
            'batch_idx': int(batch_idx),
            'batch_size': int(batch_size),
            'n_patches': int(n_patches),
            'patch_range': [int(start_idx), int(end_idx-1)],
            'cloud_type': cloud_name,
            'n_points': int(len(base_cloud))
        }

        metadata_path = output_path.replace('.las', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Mode 1 visualization saved!")
        print(f"  Files: {output_path}")
        print(f"         {metadata_path}")
        print(f"\nOpen in CloudCompare to see {n_patches} colored patches")

    def visualize_mode2_label_inspection(self, patch_indices=None, n_patches=10):
        """
        Mode 2: Inspect individual patches with per-point label coloring

        Args:
            patch_indices: Specific patches to visualize, or None for random
            n_patches: Number of patches if patch_indices is None
        """
        print(f"\n{'='*70}")
        print(f"Mode 2: Label Inspection Visualization")
        print(f"{'='*70}")

        # Select patches
        if patch_indices is None:
            patch_indices = np.random.choice(
                len(self.dataset),
                min(n_patches, len(self.dataset)),
                replace=False
            )

        print(f"Creating label inspection for {len(patch_indices)} patches...")

        # Label color scheme
        label_colors = {
            -1: np.array([128, 128, 128]),  # Gray - unlabeled
            0: np.array([0, 255, 0]),        # Green - no joints
            1: np.array([255, 0, 0])         # Red - joints
        }

        for patch_idx in tqdm(patch_indices):
            # Get original patch and labels
            original_patch = self.dataset.patches[patch_idx]
            patch_labels = self.dataset.labels[patch_idx]
            centroid = self.centroids[patch_idx]

            # Compute statistics
            n_labeled = np.sum(patch_labels != -1)
            n_class_0 = np.sum(patch_labels == 0)
            n_class_1 = np.sum(patch_labels == 1)
            n_unlabeled = np.sum(patch_labels == -1)

            if n_labeled == 0:
                continue  # Skip patches with no labels

            # Compute coherence
            labeled_vals = patch_labels[patch_labels != -1]
            majority_class = np.bincount(labeled_vals).argmax()
            coherence = np.sum(labeled_vals == majority_class) / len(labeled_vals)

            # Color points by their labels
            colors = np.zeros((len(original_patch), 3), dtype=np.uint16)
            for label, color in label_colors.items():
                mask = patch_labels == label
                colors[mask] = (color * 257).astype(np.uint16)

            # Save individual patch
            output_path = os.path.join(
                self.output_dir,
                f'mode2_patch{patch_idx:04d}_coherence{coherence:.2f}.las'
            )

            self._save_las(original_patch, colors, output_path, {
                'label': (patch_labels + 1).astype(np.uint8)  # Shift by 1 for LAS
            })

            # Save metadata (convert all numpy types to Python native types)
            metadata = {
                'visualization_mode': 'label_inspection',
                'patch_idx': int(patch_idx),
                'centroid': [float(x) for x in centroid],
                'n_points': int(len(original_patch)),
                'n_labeled': int(n_labeled),
                'labeled_ratio': float(n_labeled / len(original_patch)),
                'n_class_0': int(n_class_0),
                'n_class_1': int(n_class_1),
                'n_unlabeled': int(n_unlabeled),
                'coherence': float(coherence),
                'majority_class': int(majority_class),
                'color_scheme': {
                    'green': 'Class 0 (No Joints)',
                    'red': 'Class 1 (Joints)',
                    'gray': 'Unlabeled'
                }
            }

            metadata_path = output_path.replace('.las', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"\n✓ Mode 2 visualizations saved to {self.output_dir}/")
        print(f"\nColor coding:")
        print(f"  🟢 Green = Class 0 (No Joints)")
        print(f"  🔴 Red   = Class 1 (Joints)")
        print(f"  ⚪ Gray  = Unlabeled")
        print(f"\nEach patch saved as separate LAS file for detailed inspection")

    def visualize_mode2_with_context(self, patch_indices=None, n_patches=5,
                                    context_radius=3.0):
        """
        Mode 2 Enhanced: Show patches with surrounding context
        Patch colored by labels, context in faint gray

        Args:
            patch_indices: Patches to visualize
            n_patches: Number if patch_indices is None
            context_radius: Radius for surrounding points
        """
        print(f"\n{'='*70}")
        print(f"Mode 2 Enhanced: Label Inspection with Context")
        print(f"{'='*70}")

        if patch_indices is None:
            patch_indices = np.random.choice(
                len(self.dataset),
                min(n_patches, len(self.dataset)),
                replace=False
            )

        label_colors = {
            -1: np.array([128, 128, 128]),
            0: np.array([0, 255, 0]),
            1: np.array([255, 0, 0])
        }

        print(f"Creating {len(patch_indices)} patches with context...")

        for patch_idx in tqdm(patch_indices):
            original_patch = self.dataset.patches[patch_idx]
            patch_labels = self.dataset.labels[patch_idx]
            centroid = self.centroids[patch_idx]

            # Get context points
            context_indices = self.tree_train.query_ball_point(centroid, context_radius)
            context_points = self.xyz_train[context_indices]

            # Combine: patch + context
            combined_xyz = np.vstack([original_patch, context_points])
            n_patch = len(original_patch)
            n_context = len(context_points)

            # Colors: patch by labels, context gray
            colors = np.ones((n_patch + n_context, 3), dtype=np.uint16) * 16384  # Faint gray

            # Color patch points
            for label, color in label_colors.items():
                mask = patch_labels == label
                colors[:n_patch][mask] = (color * 257).astype(np.uint16)

            # Create marker field
            is_patch = np.zeros(n_patch + n_context, dtype=np.uint8)
            is_patch[:n_patch] = 1

            # Save
            output_path = os.path.join(
                self.output_dir,
                f'mode2_context_patch{patch_idx:04d}.las'
            )

            self._save_las(combined_xyz, colors, output_path, {
                'is_patch': is_patch,
                'label': np.concatenate([
                    (patch_labels + 1).astype(np.uint8),
                    np.zeros(n_context, dtype=np.uint8)
                ])
            })

        print(f"\n✓ Context visualizations saved!")
        print(f"View in CloudCompare:")
        print(f"  - Colored points = patch with labels")
        print(f"  - Faint gray = surrounding context")

    def _estimate_patch_radius(self, patch):
        """Estimate spatial extent of patch"""
        centroid = patch.mean(axis=0)
        distances = np.linalg.norm(patch - centroid, axis=1)
        return np.percentile(distances, 95)

    def _generate_distinct_colors(self, n):
        """Generate N visually distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
            color = np.array([int(c * 255) for c in rgb], dtype=np.uint16) * 257
            colors.append(color)
        return np.array(colors)

    def _save_las(self, xyz, colors, path, extra_fields=None):
        """Save point cloud with colors to LAS"""
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = xyz.min(axis=0)
        header.scales = [0.001, 0.001, 0.001]

        las = laspy.LasData(header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        if extra_fields:
            for name, data in extra_fields.items():
                try:
                    if data.dtype in [np.float32, np.float64]:
                        dtype = np.float32
                    elif data.dtype == np.int32:
                        dtype = np.int32
                    else:
                        dtype = np.uint8

                    las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
                    setattr(las, name, data.astype(dtype))
                except:
                    pass

        las.write(path)

    def save_centroid_mapping(self, path='../data/preprocessed/patch_centroids.npy'):
        """Save centroid mapping for later use"""
        np.save(path, self.centroids)
        print(f"Saved centroid mapping to {path}")

    @classmethod
    def load_with_centroids(cls, dataset, xyz_array_full, train_mask,
                           centroid_path, output_dir='./visualizations'):
        """Load visualizer with pre-computed centroids"""
        instance = cls.__new__(cls)
        instance.dataset = dataset
        instance.xyz_full = xyz_array_full
        instance.xyz_train = xyz_array_full[train_mask]
        instance.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load centroids
        instance.centroids = np.load(centroid_path)

        from scipy.spatial import KDTree
        instance.tree_train = KDTree(instance.xyz_train)

        print(f"Loaded visualizer with {len(instance.centroids)} centroids")
        return instance


def main():
    """Run visualization pipeline"""
    print("\n" + "="*70)
    print("IMPROVED PATCH VISUALIZATION TOOL")
    print("="*70)

    # Load data
    print("\nLoading data...")
    data = np.load('../data/preprocessed/preprocessed_data.npz')
    xyz_array = data['xyz_array']
    label_array = data['label_array']
    train_mask = data['train_mask']

    import pickle
    with open('../data/preprocessed/polygons_dict.pkl', 'rb') as f:
        polygons_data = pickle.load(f)
        polygons_dict_train = polygons_data['train']

    print(f"Loaded {len(xyz_array):,} points")

    # Create dataset
    print("\nCreating dataset...")
    from dataset import RockJointDataset

    train_dataset = RockJointDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        polygons_dict=polygons_dict_train,
        patch_size=1024,
        normalize_mode='center_scale',
        augment=False
    )

    # Create visualizer
    print("\nInitializing visualizer...")
    visualizer = ImprovedPatchVisualizer(
        dataset=train_dataset,
        xyz_array_full=xyz_array,
        train_mask=train_mask,
        output_dir='./visualizations'
    )

    # Save centroids for future use
    visualizer.save_centroid_mapping()

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Mode 1: Batch overview
    print("\n" + "-"*70)
    visualizer.visualize_mode1_batch_overview(
        batch_idx=0,
        batch_size=32,
        use_full_cloud=False  # Set True to see full point cloud
    )

    # Mode 2: Label inspection
    print("\n" + "-"*70)
    visualizer.visualize_mode2_label_inspection(n_patches=15)

    # Mode 2 Enhanced: With context
    print("\n" + "-"*70)
    visualizer.visualize_mode2_with_context(n_patches=5)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: ./visualizations/")
    print("\nRecommended CloudCompare workflow:")
    print("  1. Mode 1: Open 'mode1_batch*' to see patch distribution")
    print("  2. Mode 2: Open 'mode2_patch*' files to inspect labels")
    print("  3. Mode 2 Enhanced: Open 'mode2_context*' for patches with surroundings")


if __name__ == '__main__':
    main()
