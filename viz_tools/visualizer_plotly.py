"""
Interactive Plotly Patch Visualizer
Generates interactive HTML files with Plotly Scatter3D for unlimited zoom
"""

import numpy as np
import os
import pickle
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Install with: pip install plotly")
    exit(1)


class PlotlyPatchVisualizer:
    """
    Interactive 3D visualizer using Plotly
    Much better for zooming and inspecting details
    """

    def __init__(self, preprocessed_npz, polygons_pkl, output_dir='./plotly_viz'):
        """
        Args:
            preprocessed_npz: Path to preprocessed_data.npz
            polygons_pkl: Path to polygons_dict.pkl
            output_dir: Output directory for HTML files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("Loading preprocessed data...")
        data = np.load(preprocessed_npz)
        self.xyz_full = data['xyz_array']
        self.label_array = data['label_array']
        self.train_mask = data['train_mask']
        self.test_mask = data['test_mask']

        self.xyz_train = self.xyz_full[self.train_mask]
        self.labels_train = self.label_array[self.train_mask]

        print(f"Loaded {len(self.xyz_full):,} points")
        print(f"  Training: {np.sum(self.train_mask):,}")

        # Load polygons
        print("Loading polygons...")
        with open(polygons_pkl, 'rb') as f:
            polygons_data = pickle.load(f)
            self.polygons_train = polygons_data['train']

        print(f"Train polygons: Class 0={len(self.polygons_train['label_0'])}, "
              f"Class 1={len(self.polygons_train['label_1'])}")

        # Extract patches
        print("Extracting patches from polygons...")
        self.patches, self.patch_labels, self.centroids, self.patch_classes = \
            self._extract_patches_from_polygons(
                self.xyz_train,
                self.labels_train,
                self.polygons_train,
                patch_size=1024
            )

        print(f"Extracted {len(self.patches)} patches")

    def _extract_patches_from_polygons(self, xyz_array, label_array, polygons_dict, patch_size=1024):
        """Extract patches from polygon centroids"""
        from scipy.spatial import KDTree

        patches = []
        labels = []
        centroids = []
        patch_classes = []

        tree = KDTree(xyz_array)

        for class_id, class_name in [(0, 'label_0'), (1, 'label_1')]:
            polygons = polygons_dict[class_name]

            for polygon in tqdm(polygons, desc=f"Class {class_id}"):
                poly_array = np.array(polygon)
                centroid = poly_array.mean(axis=0)
                centroids.append(centroid)
                patch_classes.append(class_id)

                max_dist = np.max(np.linalg.norm(poly_array - centroid, axis=1))
                search_radius = max_dist * 1.5

                indices = tree.query_ball_point(centroid, search_radius)

                if len(indices) < patch_size // 4:
                    continue

                patch_points = xyz_array[indices]
                patch_labels_arr = label_array[indices]

                # Sample or pad
                if len(patch_points) >= patch_size:
                    choice = np.random.choice(len(patch_points), patch_size, replace=False)
                    patch_points = patch_points[choice]
                    patch_labels_arr = patch_labels_arr[choice]
                else:
                    n_repeat = patch_size - len(patch_points)
                    repeat_indices = np.random.choice(len(patch_points), n_repeat, replace=True)
                    patch_points = np.vstack([patch_points, patch_points[repeat_indices]])
                    patch_labels_arr = np.concatenate([patch_labels_arr, patch_labels_arr[repeat_indices]])

                patches.append(patch_points)
                labels.append(patch_labels_arr)

        return patches, np.array(labels), np.array(centroids), np.array(patch_classes)

    def visualize_mode1_interactive_batch(self, batch_idx=0, batch_size=32,
                                         use_full_cloud=True, downsample_bg=10):
        """
        Mode 1: Interactive batch overview with Plotly

        Args:
            batch_idx: Batch index
            batch_size: Size of batch
            use_full_cloud: Show full cloud or just training region
            downsample_bg: Downsample background cloud by factor (for performance)
        """
        print(f"\n{'='*70}")
        print(f"Mode 1 Interactive: Batch Overview")
        print(f"{'='*70}")

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.patches))
        n_patches = end_idx - start_idx

        if n_patches == 0:
            print("No patches in this batch!")
            return

        print(f"Batch {batch_idx}: patches {start_idx} to {end_idx-1} ({n_patches} total)")

        # Prepare data
        base_cloud = self.xyz_full if use_full_cloud else self.xyz_train
        cloud_name = "full" if use_full_cloud else "train"

        # Downsample background for performance
        print(f"Downsampling background cloud by {downsample_bg}x for performance...")
        bg_indices = np.arange(0, len(base_cloud), downsample_bg)
        bg_cloud = base_cloud[bg_indices]

        print(f"Background cloud: {len(bg_cloud):,} points (downsampled from {len(base_cloud):,})")

        # Create figure
        fig = go.Figure()

        # Add background cloud (gray)
        fig.add_trace(go.Scatter3d(
            x=bg_cloud[:, 0],
            y=bg_cloud[:, 1],
            z=bg_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='lightgray',
                opacity=0.3
            ),
            name='Background',
            hoverinfo='skip'
        ))

        # Add each patch with unique color
        print(f"Adding {n_patches} patches...")
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

        for i, patch_idx in enumerate(tqdm(range(start_idx, end_idx))):
            patch = self.patches[patch_idx]
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter3d(
                x=patch[:, 0],
                y=patch[:, 1],
                z=patch[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.8
                ),
                name=f'Patch {patch_idx}',
                hovertemplate=f'<b>Patch {patch_idx}</b><br>' +
                             'X: %{x:.2f}<br>' +
                             'Y: %{y:.2f}<br>' +
                             'Z: %{z:.2f}<br>' +
                             '<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title=f'Mode 1 - Batch {batch_idx} Overview ({n_patches} patches on {cloud_name} cloud)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1400,
            height=900,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Save
        output_path = os.path.join(
            self.output_dir,
            f'mode1_batch{batch_idx}_interactive.html'
        )

        print(f"\nSaving interactive HTML to {output_path}...")
        fig.write_html(output_path)

        print(f"\n✓ Mode 1 interactive visualization saved!")
        print(f"  File: {output_path}")
        print(f"\nOpen in browser for:")
        print(f"  - Unlimited zoom")
        print(f"  - Rotate and pan")
        print(f"  - Click legend to show/hide patches")
        print(f"  - Hover for coordinates")

    def visualize_mode2_interactive_patches(self, patch_indices=None, n_patches=10):
        """
        Mode 2: Interactive individual patch inspection

        Args:
            patch_indices: Specific patches to visualize
            n_patches: Number of patches if patch_indices is None
        """
        print(f"\n{'='*70}")
        print(f"Mode 2 Interactive: Label Inspection")
        print(f"{'='*70}")

        if patch_indices is None:
            patch_indices = np.random.choice(
                len(self.patches),
                min(n_patches, len(self.patches)),
                replace=False
            )

        print(f"Creating {len(patch_indices)} interactive patch visualizations...")

        for patch_idx in tqdm(patch_indices):
            patch = self.patches[patch_idx]
            labels = self.patch_labels[patch_idx]
            centroid = self.centroids[patch_idx]
            patch_class = self.patch_classes[patch_idx]

            # Compute statistics
            n_labeled = np.sum(labels != -1)
            if n_labeled == 0:
                continue

            n_class_0 = np.sum(labels == 0)
            n_class_1 = np.sum(labels == 1)
            n_unlabeled = np.sum(labels == -1)

            labeled_vals = labels[labels != -1]
            majority_class = np.bincount(labeled_vals).argmax()
            coherence = np.sum(labeled_vals == majority_class) / len(labeled_vals)

            # Create figure
            fig = go.Figure()

            # Add points colored by label
            for label, color, name in [
                (-1, 'gray', 'Unlabeled'),
                (0, 'green', 'Class 0 (No Joints)'),
                (1, 'red', 'Class 1 (Joints)')
            ]:
                mask = labels == label
                n_points = np.sum(mask)

                if n_points > 0:
                    fig.add_trace(go.Scatter3d(
                        x=patch[mask, 0],
                        y=patch[mask, 1],
                        z=patch[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=color,
                            opacity=0.8
                        ),
                        name=f'{name} ({n_points} pts)',
                        hovertemplate=f'<b>{name}</b><br>' +
                                     'X: %{x:.2f}<br>' +
                                     'Y: %{y:.2f}<br>' +
                                     'Z: %{z:.2f}<br>' +
                                     '<extra></extra>'
                    ))

            # Update layout with statistics
            fig.update_layout(
                title=f'Mode 2 - Patch {patch_idx} (Coherence: {coherence:.1%}, Class: {majority_class})<br>' +
                      f'<sup>Class 0: {n_class_0} | Class 1: {n_class_1} | Unlabeled: {n_unlabeled}</sup>',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                width=1200,
                height=900,
                hovermode='closest'
            )

            # Save
            output_path = os.path.join(
                self.output_dir,
                f'mode2_patch{patch_idx:04d}_coherence{coherence:.2f}.html'
            )

            fig.write_html(output_path)

        print(f"\n✓ Mode 2 interactive visualizations saved!")
        print(f"  Directory: {self.output_dir}/")
        print(f"\nOpen HTML files in browser for:")
        print(f"  - Unlimited zoom into individual points")
        print(f"  - Click legend to show/hide label classes")
        print(f"  - Rotate to inspect from any angle")

    def visualize_mode2_with_context(self, patch_indices=None, n_patches=5,
                                    context_radius=3.0, downsample_context=5):
        """
        Mode 2 Enhanced: Patches with surrounding context

        Args:
            patch_indices: Patches to visualize
            n_patches: Number if patch_indices is None
            context_radius: Radius for context points
            downsample_context: Downsample context by factor
        """
        print(f"\n{'='*70}")
        print(f"Mode 2 Enhanced: Patches with Context")
        print(f"{'='*70}")

        if patch_indices is None:
            patch_indices = np.random.choice(
                len(self.patches),
                min(n_patches, len(self.patches)),
                replace=False
            )

        from scipy.spatial import KDTree
        tree = KDTree(self.xyz_train)

        print(f"Creating {len(patch_indices)} patches with context...")

        for patch_idx in tqdm(patch_indices):
            patch = self.patches[patch_idx]
            labels = self.patch_labels[patch_idx]
            centroid = self.centroids[patch_idx]

            # Get context
            context_indices = tree.query_ball_point(centroid, context_radius)
            context_indices = context_indices[::downsample_context]  # Downsample
            context_points = self.xyz_train[context_indices]

            # Create figure
            fig = go.Figure()

            # Add context (faint gray)
            fig.add_trace(go.Scatter3d(
                x=context_points[:, 0],
                y=context_points[:, 1],
                z=context_points[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color='lightgray',
                    opacity=0.3
                ),
                name='Context',
                hoverinfo='skip'
            ))

            # Add patch colored by labels
            for label, color, name in [
                (-1, 'gray', 'Unlabeled'),
                (0, 'green', 'Class 0 (No Joints)'),
                (1, 'red', 'Class 1 (Joints)')
            ]:
                mask = labels == label
                n_points = np.sum(mask)

                if n_points > 0:
                    fig.add_trace(go.Scatter3d(
                        x=patch[mask, 0],
                        y=patch[mask, 1],
                        z=patch[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=color,
                            opacity=0.9
                        ),
                        name=f'{name} ({n_points})',
                        # hovertemplate=f'<b>{name}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                    ))

            # Layout
            fig.update_layout(
                title=f'Mode 2 Context - Patch {patch_idx} with {context_radius}m surroundings',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                width=1200,
                height=900,
                hovermode='closest'
            )

            # Save
            output_path = os.path.join(
                self.output_dir,
                f'mode2_context_patch{patch_idx:04d}.html'
            )

            fig.write_html(output_path)

        print(f"\n✓ Context visualizations saved!")
        print(f"  Directory: {self.output_dir}/")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("PLOTLY INTERACTIVE PATCH VISUALIZER")
    print("="*70)

    # Configuration
    config = {
        'preprocessed_npz': '../data/preprocessed/preprocessed_data.npz',
        'polygons_pkl': '../data/preprocessed/polygons_dict.pkl',
        'output_dir': './plotly_viz',

        # Mode 1 settings
        'batch_idx': 0,
        'batch_size': 32,
        'use_full_cloud': False,  # True shows entire rock face
        'downsample_bg': 1,      # Downsample background by 10x for speed

        # Mode 2 settings
        'n_label_inspect': 10,
        'n_context': 5,
        'context_radius': 3.0,
        'downsample_context': 1
    }

    # Check files
    if not os.path.exists(config['preprocessed_npz']):
        print(f"\nERROR: Cannot find {config['preprocessed_npz']}")
        print("Run train_end2end.py first")
        return

    if not os.path.exists(config['polygons_pkl']):
        print(f"\nERROR: Cannot find {config['polygons_pkl']}")
        print("Run train_end2end.py first")
        return

    # Create visualizer
    visualizer = PlotlyPatchVisualizer(
        preprocessed_npz=config['preprocessed_npz'],
        polygons_pkl=config['polygons_pkl'],
        output_dir=config['output_dir']
    )

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING INTERACTIVE VISUALIZATIONS")
    print("="*70)

    # Mode 1: Batch overview
    print("\n" + "-"*70)
    visualizer.visualize_mode1_interactive_batch(
        batch_idx=config['batch_idx'],
        batch_size=config['batch_size'],
        use_full_cloud=config['use_full_cloud'],
        downsample_bg=config['downsample_bg']
    )

    # Mode 2: Label inspection
    print("\n" + "-"*70)
    visualizer.visualize_mode2_interactive_patches(
        n_patches=config['n_label_inspect']
    )

    # Mode 2 Enhanced: With context
    print("\n" + "-"*70)
    visualizer.visualize_mode2_with_context(
        n_patches=config['n_context'],
        context_radius=config['context_radius'],
        downsample_context=config['downsample_context']
    )

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nInteractive HTML files saved to: {config['output_dir']}/")
    print("\nOpen in web browser:")
    print("  - mode1_*.html for batch overview")
    print("  - mode2_patch*.html for individual patches")
    print("  - mode2_context*.html for patches with surroundings")
    print("\nFeatures:")
    print("  ✓ Unlimited zoom - inspect individual points")
    print("  ✓ Rotate, pan, and orbit")
    print("  ✓ Click legend to show/hide elements")
    print("  ✓ Hover for point coordinates")
    print("  ✓ No rendering limits!")


if __name__ == '__main__':
    main()
