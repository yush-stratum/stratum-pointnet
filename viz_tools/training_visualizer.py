"""
Real-time Batch Visualization During Training
Hooks into the training loop to save batch visualizations at specified intervals
"""

import numpy as np
import torch
import laspy
import os
from pathlib import Path
import json


class TrainingBatchVisualizer:
    """
    Visualizer that hooks into training to save batch snapshots
    """

    def __init__(self, xyz_array, output_dir='./training_viz',
                 visualize_every_n_batches=50):
        """
        Args:
            xyz_array: Full point cloud for matching
            output_dir: Where to save visualizations
            visualize_every_n_batches: Frequency of visualization
        """
        self.xyz_array = xyz_array
        self.output_dir = output_dir
        self.visualize_every_n_batches = visualize_every_n_batches
        self.batch_counter = 0

        os.makedirs(output_dir, exist_ok=True)

        from scipy.spatial import KDTree
        self.tree = KDTree(xyz_array)

        print(f"Training visualizer initialized")
        print(f"Will visualize every {visualize_every_n_batches} batches")

    def visualize_batch(self, batch_points, batch_labels, epoch, batch_idx,
                       predictions=None, losses=None):
        """
        Visualize a batch during training

        Args:
            batch_points: [B, N, 3] tensor
            batch_labels: [B, N] tensor
            epoch: Current epoch
            batch_idx: Current batch index
            predictions: [B, N] tensor (optional)
            losses: [B] tensor (optional)
        """
        self.batch_counter += 1

        if self.batch_counter % self.visualize_every_n_batches != 0:
            return

        print(f"\n  📸 Saving batch visualization (epoch {epoch}, batch {batch_idx})...")

        # Convert to numpy
        batch_points = batch_points.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

        if predictions is not None:
            predictions = predictions.detach().cpu().numpy()
        if losses is not None:
            losses = losses.detach().cpu().numpy()

        batch_size = len(batch_points)

        # Create epoch directory
        epoch_dir = os.path.join(self.output_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)

        # === Mode 1: Full cloud with colored batch patches ===
        self._create_batch_overview(
            batch_points, batch_labels, predictions,
            epoch, batch_idx, epoch_dir, batch_size
        )

        # === Mode 2: Individual patches with labels ===
        self._create_patch_details(
            batch_points, batch_labels, predictions, losses,
            epoch, batch_idx, epoch_dir, batch_size
        )

        print(f"  ✓ Saved to {epoch_dir}/")

    def _create_batch_overview(self, batch_points, batch_labels, predictions,
                               epoch, batch_idx, epoch_dir, batch_size):
        """Create overview showing all patches in the batch on full cloud"""

        # Color full cloud
        colors = np.ones((len(self.xyz_array), 3), dtype=np.uint16) * 32768  # Gray
        patch_membership = np.full(len(self.xyz_array), -1, dtype=np.int32)
        is_correct = np.full(len(self.xyz_array), -1, dtype=np.int8)

        # Generate colors for patches
        patch_colors = self._generate_colors(batch_size)

        # Match each patch to original cloud
        for b in range(batch_size):
            patch = batch_points[b]

            # Denormalize if needed - estimate centroid from neighborhood
            centroid_estimate = self._estimate_original_centroid(patch)

            if centroid_estimate is None:
                continue

            # Find matching points
            matched_indices = self._match_patch_to_cloud(patch, centroid_estimate)

            if len(matched_indices) > 0:
                colors[matched_indices] = patch_colors[b]
                patch_membership[matched_indices] = b

                # Mark correctness if predictions available
                if predictions is not None:
                    patch_preds = predictions[b]
                    patch_labels_b = batch_labels[b]

                    # For each matched point, check if correctly classified
                    for idx in matched_indices:
                        # Find which point in the patch this corresponds to
                        # Use closest point heuristic
                        pass  # Simplified for now

        # Save
        output_path = os.path.join(
            epoch_dir,
            f'batch_overview_e{epoch}_b{batch_idx}.las'
        )

        self._save_las(self.xyz_array, colors, output_path, {
            'patch_id': patch_membership
        })

    def _create_patch_details(self, batch_points, batch_labels, predictions, losses,
                             epoch, batch_idx, epoch_dir, batch_size):
        """Create detailed view of interesting patches"""

        # Select interesting patches to visualize
        interesting_indices = []

        if losses is not None:
            # Find highest loss patches
            n_visualize = min(5, batch_size)
            top_loss_indices = np.argsort(losses)[-n_visualize:]
            interesting_indices.extend(top_loss_indices)
        else:
            # Just take first few
            interesting_indices = list(range(min(5, batch_size)))

        for b in interesting_indices:
            patch = batch_points[b]
            labels = batch_labels[b]

            # Color by labels
            colors = self._color_by_labels(labels, predictions[b] if predictions is not None else None)

            # Save patch
            output_path = os.path.join(
                epoch_dir,
                f'patch_detail_e{epoch}_b{batch_idx}_p{b}.las'
            )

            extra_fields = {'gt_label': (labels + 1).astype(np.uint8)}
            if predictions is not None:
                extra_fields['pred_label'] = (predictions[b] + 1).astype(np.uint8)
                extra_fields['correct'] = (predictions[b] == labels).astype(np.uint8)

            self._save_las(patch, colors, output_path, extra_fields)

            # Save metadata
            metadata = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'patch_in_batch': b,
                'n_points': len(patch),
                'n_labeled': int(np.sum(labels != -1)),
                'n_class_0': int(np.sum(labels == 0)),
                'n_class_1': int(np.sum(labels == 1)),
            }

            if predictions is not None:
                labeled_mask = labels != -1
                if np.sum(labeled_mask) > 0:
                    accuracy = np.sum(predictions[b][labeled_mask] == labels[labeled_mask]) / np.sum(labeled_mask)
                    metadata['accuracy'] = float(accuracy)

            if losses is not None:
                metadata['loss'] = float(losses[b])

            metadata_path = output_path.replace('.las', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def _estimate_original_centroid(self, normalized_patch):
        """
        Attempt to estimate where this normalized patch came from
        This is challenging - we use the fact that local geometry is preserved
        """
        # For 'center' normalization, we've only translated
        # For 'center_scale', we've also scaled
        # Without storing the original centroid, we need to search

        # Strategy: Search for similar local geometry in the full cloud
        # This is expensive, so we'll use a sampling approach

        # Sample a subset of points from full cloud
        n_samples = min(10000, len(self.xyz_array))
        sample_indices = np.random.choice(len(self.xyz_array), n_samples, replace=False)
        sample_points = self.xyz_array[sample_indices]

        # For each sample point as potential centroid, check if geometry matches
        # This is a simplified heuristic - just return the mean of sample for now
        # In practice, you'd want to store original centroids during dataset creation

        return np.mean(sample_points, axis=0)

    def _match_patch_to_cloud(self, patch, centroid_estimate, tolerance=0.05):
        """Match normalized patch back to original cloud points"""

        # Query points near estimated centroid
        search_radius = 2.0  # meters
        candidate_indices = self.tree.query_ball_point(centroid_estimate, search_radius)

        if len(candidate_indices) < len(patch) // 2:
            return []

        candidate_points = self.xyz_array[candidate_indices]

        # Build KDTree for candidates
        from scipy.spatial import KDTree
        candidate_tree = KDTree(candidate_points)

        # Try to match patch points to candidates
        # Account for normalization by trying different translations/scales
        best_matches = []
        best_score = 0

        for scale in [1.0, 0.5, 2.0]:  # Try different scales
            for offset in [centroid_estimate, np.zeros(3)]:
                transformed_patch = patch * scale + offset

                distances, indices = candidate_tree.query(transformed_patch)

                n_close = np.sum(distances < tolerance)
                if n_close > best_score:
                    best_score = n_close
                    close_mask = distances < tolerance
                    best_matches = np.array(candidate_indices)[indices[close_mask]]

        return best_matches

    def _color_by_labels(self, labels, predictions=None):
        """Color points by their labels (and optionally predictions)"""
        colors = np.zeros((len(labels), 3), dtype=np.uint16)

        if predictions is None:
            # Just color by ground truth
            label_colors = {
                -1: np.array([128, 128, 128]),  # Gray
                0: np.array([0, 255, 0]),       # Green
                1: np.array([255, 0, 0])        # Red
            }

            for label, color in label_colors.items():
                mask = labels == label
                colors[mask] = (color * 257).astype(np.uint16)
        else:
            # Color by correctness
            labeled_mask = labels != -1
            correct_mask = (predictions == labels) & labeled_mask
            incorrect_mask = (predictions != labels) & labeled_mask
            unlabeled_mask = labels == -1

            colors[correct_mask] = np.array([0, 255, 0]) * 257  # Green - correct
            colors[incorrect_mask] = np.array([255, 0, 0]) * 257  # Red - wrong
            colors[unlabeled_mask] = np.array([128, 128, 128]) * 257  # Gray

        return colors

    def _generate_colors(self, n):
        """Generate N distinct colors"""
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(np.array([int(c * 255) for c in rgb], dtype=np.uint16) * 257)
        return np.array(colors)

    def _save_las(self, xyz, colors, output_path, extra_fields=None):
        """Save point cloud to LAS with colors"""
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = xyz.min(axis=0) if len(xyz) > 0 else [0, 0, 0]
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
                    dtype = data.dtype if hasattr(data, 'dtype') else type(data[0])
                    if dtype == np.float32 or dtype == np.float64:
                        dtype = np.float32
                    elif dtype == np.int32:
                        dtype = np.int32
                    else:
                        dtype = np.uint8

                    las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
                    setattr(las, name, data.astype(dtype))
                except Exception as e:
                    pass

        las.write(output_path)


# Example integration with training loop
def example_usage():
    """
    Example of how to integrate with train.py
    """

    # In your train.py, modify the Trainer class:

    class TrainerWithVisualization:
        def __init__(self, model, train_loader, test_loader, config, xyz_array=None):
            # ... existing init code ...

            # Add visualizer
            if xyz_array is not None and config.get('visualize_training', False):
                self.visualizer = TrainingBatchVisualizer(
                    xyz_array=xyz_array,
                    output_dir=config.get('viz_dir', './training_viz'),
                    visualize_every_n_batches=config.get('viz_every_n', 50)
                )
            else:
                self.visualizer = None

        def train_epoch(self, epoch):
            # ... existing training code ...

            for batch_idx, (points, labels) in enumerate(pbar):
                points = points.to(self.device)
                labels = labels.to(self.device)

                # ... forward pass ...
                outputs = self.model(points)

                # ... compute loss ...
                loss = self.criterion(outputs, labels)

                # Get predictions for visualization
                preds = torch.argmax(outputs, dim=1)

                # ... backward pass ...

                # VISUALIZE
                if self.visualizer is not None:
                    self.visualizer.visualize_batch(
                        batch_points=points,
                        batch_labels=labels,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        predictions=preds,
                        losses=None  # or per-sample losses if available
                    )


if __name__ == '__main__':
    print("This module provides TrainingBatchVisualizer for real-time visualization")
    print("Import it in your training script to visualize batches during training")
