"""
Inference script for PointNet++ on full point clouds
Outputs per-point predictions to new LAS file

IMPORTANT: Automatically loads training configuration from checkpoint directory
to ensure inference uses the same settings (features, normalization, etc.) as training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import laspy
import os
import json
from collections import defaultdict

from pointnet2_model import PointNet2Segmentation
from dataset import RockJointInferenceDataset
from dataset_factory import DatasetFactory

class PointCloudInference:
    """
    Inference engine for full point cloud prediction

    Automatically loads training configuration to ensure consistency.
    """
    def __init__(self, model_path, inference_overrides=None):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            inference_overrides: Optional dict with inference-specific overrides
                                (e.g., batch_size, stride). Training config is loaded
                                automatically and overrides are applied on top.

        Example:
            inference = PointCloudInference(
                model_path='./checkpoints/run_1/best_model.pth',
                inference_overrides={
                    'batch_size': 32,  # Override for inference
                    'dataset': {
                        'params': {
                            'test_stride': 1  # Override stride for full coverage
                        }
                    }
                }
            )
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"\nLoading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load training config from checkpoint directory
        checkpoint_dir = os.path.dirname(model_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')

        if os.path.exists(config_path):
            print(f"Loading training config from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("✅ Training configuration loaded successfully")
        else:
            # Fallback to checkpoint config if config.json not found
            print(f"⚠️  WARNING: config.json not found in {checkpoint_dir}")
            if 'config' in checkpoint:
                print("Using config from checkpoint (may be incomplete)")
                self.config = checkpoint['config']
            else:
                raise FileNotFoundError(
                    f"No config.json found in {checkpoint_dir} and no config in checkpoint. "
                    f"Cannot proceed with inference without training configuration."
                )

        # Apply inference overrides if provided
        if inference_overrides is not None:
            print("\nApplying inference overrides:")
            self._apply_overrides(self.config, inference_overrides)

        # Print key configuration
        print(f"\n{'='*70}")
        print("INFERENCE CONFIGURATION")
        print(f"{'='*70}")
        print(f"Model architecture:")
        print(f"  - Input channels: {self.config['input_channels']}")
        print(f"  - Number of classes: {self.config['num_classes']}")
        print(f"\nDataset settings:")
        print(f"  - Type: {self.config['dataset']['type']}")
        print(f"  - Use RGB: {self.config.get('use_rgb', False)}")

        feature_names = self.config['dataset']['params'].get('feature_names', None)
        if feature_names:
            print(f"  - Geometric features: {feature_names}")
        else:
            print(f"  - Geometric features: None")

        print(f"  - Normalization: {self.config['dataset']['params'].get('normalize_mode', 'center')}")
        print(f"  - K neighbors: {self.config['dataset']['params'].get('k_neighbors', 'N/A')}")
        print(f"{'='*70}\n")

        # Handle voxel size if present
        voxel_size = checkpoint.get('voxel_size', None)
        if voxel_size is not None:
            print(f"Using voxel size from checkpoint: {voxel_size:.4f} m")
            self.config['voxel_size'] = voxel_size

        # Initialize model with training configuration
        self.model = PointNet2Segmentation(
            num_classes=self.config['num_classes'],
            input_channels=self.config['input_channels']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully")
        print(f"   - Trained to epoch: {checkpoint['epoch']}")
        print(f"   - Training test accuracy: {checkpoint.get('test_acc', 'N/A'):.4f}" if checkpoint.get('test_acc') else "")

    def _apply_overrides(self, base_config, overrides):
        """
        Recursively apply override values to base config
        """
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                # Recursive merge for nested dicts
                self._apply_overrides(base_config[key], value)
                print(f"  - Overriding {key}: <nested dict>")
            else:
                # Direct override
                old_value = base_config.get(key, '<not set>')
                base_config[key] = value
                print(f"  - Overriding {key}: {old_value} → {value}")

    # def predict_point_cloud(self, xyz_array, rgb_array=None, batch_size=16):
    #     """
    #     Predict labels for entire point cloud using sliding window

    #     Args:
    #         xyz_array: Full point cloud [N, 3]
    #         rgb_array: Optional RGB colors [N, 3]
    #         batch_size: Batch size for inference

    #     Returns:
    #         predictions: [N] array with class predictions
    #         probabilities: [N, num_classes] array with class probabilities
    #     """
    #     print(f"\nPredicting on point cloud with {len(xyz_array)} points")

    #     # Create inference dataset
    #     inference_dataset = RockJointInferenceDataset(
    #         xyz_array=xyz_array,
    #         rgb_array=rgb_array,
    #         patch_size=self.config['patch_size'],
    #         stride=self.config['stride'],
    #         normalize_mode=self.config['normalize_mode']
    #     )

    #     inference_loader = DataLoader(
    #         inference_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=True
    #     )

    #     predictions = np.full(len(xyz_array), -1)
    #     probabilities = np.zeros((len(xyz_array), self.config['num_classes']))
    #     vote_counts = np.zeros(len(xyz_array))

    #     for points, indices in inference_loader:
    #         points = points.to(self.device)  # [B, N, 3]

    #         outputs = self.model(points)  # [B, num_classes, N]
    #         probs = torch.softmax(outputs, dim=1)  # [B, num_classes, N]
    #         preds = torch.argmax(outputs, dim=1)  # [B, N]

    #         # Directly assign to corresponding point indices
    #         for b in range(len(preds)):
    #             point_indices = indices[b].detach().cpu().numpy()
    #             point_preds = preds[b].detach().cpu().numpy()
    #             point_probs = probs[b].detach().cpu().numpy().T  # [N, num_classes]

    #             # Average if points appear in multiple patches
    #             predictions[point_indices] = point_preds
    #             probabilities[point_indices] += point_probs
    #             vote_counts[point_indices] += 1

    #     probabilities /= vote_counts[:, None]
    #     return predictions, probabilities


    def predict_point_cloud(self, xyz_array, rgb_array=None, label_array=None, batch_size=16):
        """Updated inference with proper point mapping and dataset factory support"""

        # Create inference dataset using factory
        inference_dataset = DatasetFactory.create_inference_dataset(
            config=self.config,
            xyz_array=xyz_array,
            rgb_array=rgb_array,
            label_array=label_array
        )

        # Check dataset type to determine return format
        dataset_type = self.config.get('dataset', {}).get('type', 'polygon')

        # Modify dataloader to use custom collate function
        def collate_with_indices(batch):
            """Collate function that preserves indices"""
            points = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])

            # Handle both 2-tuple and 3-tuple returns
            if len(batch[0]) == 3:
                indices = [item[2] for item in batch]  # List of index tensors
                return points, labels, indices
            else:
                return points, labels, None

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

        print(f"\nRunning inference on {len(inference_dataset)} patches...")

        with torch.no_grad():
            for batch_data in tqdm(inference_loader, desc="Inference"):
                points, _, batch_indices = batch_data
                points = points.to(self.device)  # [B, N, C]

                outputs = self.model(points)  # [B, num_classes, N]
                probs = torch.softmax(outputs, dim=1)  # [B, num_classes, N]
                preds = torch.argmax(outputs, dim=1)  # [B, N]

                # Convert to numpy
                preds_np = preds.cpu().numpy()  # [B, N]
                probs_np = probs.cpu().numpy()  # [B, num_classes, N]

                # Handle different dataset types
                if batch_indices is not None and dataset_type in ['knn', 'voxel']:
                    # Direct point-to-prediction mapping (KNN, voxel)
                    for b in range(len(batch_indices)):
                        point_indices = batch_indices[b].numpy() if hasattr(batch_indices[b], 'numpy') else batch_indices[b]
                        point_preds = preds_np[b]  # [patch_size]
                        point_probs = probs_np[b].T  # [patch_size, num_classes]

                        # Accumulate predictions (averaging if points appear in multiple patches)
                        for i, pt_idx in enumerate(point_indices):
                            predictions[pt_idx] = point_preds[i]
                            probabilities[pt_idx] += point_probs[i]
                            vote_counts[pt_idx] += 1
                else:
                    # For polygon dataset, would need spatial aggregation (not implemented here)
                    raise NotImplementedError(f"Inference aggregation for {dataset_type} dataset not yet implemented")

        # Average probabilities where points were seen
        mask = vote_counts > 0
        probabilities[mask] /= vote_counts[mask, None]

        # Report coverage
        coverage = np.sum(mask) / len(xyz_array)
        print(f"\nInference coverage: {np.sum(mask):,} / {len(xyz_array):,} points ({coverage:.1%})")

        if coverage < 0.95:
            print(f"WARNING: Low coverage! {np.sum(~mask):,} points not covered by any voxel")

        return predictions, probabilities

    def get_train_test_mask(self, data_ckpt):
        """
        Load train/test masks and ground truth labels from preprocessed data.

        Handles both binary and multiclass modes:
        - Binary: labels are 0 (no-joint), 1 (joint), -1 (unlabeled) → saved as 1, 2, 0
        - Multiclass: labels are 0 (background), 1 (no-joint), 2 (joint) → saved as 1, 2, 3
        """
        data = np.load(data_ckpt)
        xyz_array = data['xyz_array']
        label_array = data['label_array']
        train_mask = data['train_mask']
        test_mask = data['test_mask']

        # Get classification mode from config
        classification_mode = self.config.get('classification_mode', 'multiclass')

        train_classification_array = np.zeros_like(label_array)
        train_classification_array -= 1  # Start with -1 for all
        train_classification_array[np.where(train_mask)[0]] = label_array[np.where(train_mask)[0]]

        test_classification_array = np.zeros_like(label_array)
        test_classification_array -= 1
        test_classification_array[np.where(test_mask)[0]] = label_array[np.where(test_mask)[0]]

        # Shift by +1 for LAS storage (0 reserved for "unclassified")
        train_classification_array += 1
        test_classification_array += 1
        ground_truth = (label_array + 1).astype(np.uint8)

        return train_classification_array.astype(np.uint8), test_classification_array.astype(np.uint8), ground_truth

    def save_predictions_to_las(self, input_las_path, output_las_path,
                                predictions, probabilities, data_ckpt):
        """
        Save predictions to new LAS file

        Args:
            input_las_path: Original LAS file path
            output_las_path: Output LAS file path
            predictions: [N] array with class predictions
            probabilities: [N, num_classes] array with probabilities
            data_ckpt: checkpoint for preprocessed data
        """
        print(f"\nSaving predictions to {output_las_path}")

        # Read original LAS
        las = laspy.read(input_las_path)

        # Create new LAS with same header
        header = laspy.LasHeader(point_format=las.header.point_format,
                                  version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        new_las = laspy.LasData(header)

        # Copy coordinates
        new_las.x = las.x
        new_las.y = las.y
        new_las.z = las.z

        # Copy original attributes if they exist
        if hasattr(las, 'red'):
            new_las.red = las.red
            new_las.green = las.green
            new_las.blue = las.blue

        if hasattr(las, 'intensity'):
            new_las.intensity = las.intensity

        # Add predictions as classification, shift by 1
        new_las.classification = (predictions+1).astype(np.uint8)

        # Train/Test masks
        train_mask, test_mask, label_arr = self.get_train_test_mask(data_ckpt)

        # Add probabilities as extra dimensions if supported
        try:
            # Dynamically add probability fields for all classes
            num_classes = self.config['num_classes']
            classification_mode = self.config.get('classification_mode', 'multiclass')

            print(f"\nAdding {num_classes} probability fields to LAS file...")

            # Get class names for better field naming
            if classification_mode == 'binary':
                class_names = ['no_joint', 'joint']
            elif num_classes == 3:
                class_names = ['background', 'no_joint', 'joint']
            else:
                class_names = [f'class_{i}' for i in range(num_classes)]

            # Add probability field for each class
            for class_id in range(num_classes):
                field_name = f"prob_{class_names[class_id]}"
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                    name=field_name,
                    type=np.float32
                ))
                setattr(new_las, field_name, probabilities[:, class_id])
                print(f"  Added field: {field_name}")

            # Add train/test masks
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="train_mask",
                type=np.uint8
            ))
            new_las.train_mask = train_mask

            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="test_mask",
                type=np.uint8
            ))
            new_las.test_mask = test_mask

            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="label_arr",
                type=np.uint8
            ))
            new_las.label_arr = label_arr

            print("✅ Successfully added all probability and mask fields to LAS file")
        except Exception as e:
            print(f"⚠️  Warning: Could not add probability fields: {e}")

        # Write to file
        new_las.write(output_las_path)
        print(f"Successfully saved predictions to {output_las_path}")

        # Print statistics
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70)

        num_classes = self.config['num_classes']
        classification_mode = self.config.get('classification_mode', 'multiclass')

        # Get class names
        if classification_mode == 'binary':
            class_display_names = ['No-Joint', 'Joint']
        elif num_classes == 3:
            class_display_names = ['Background', 'No-Joint', 'Joint']
        else:
            class_display_names = [f'Class {i}' for i in range(num_classes)]

        for class_id in range(num_classes):
            count = np.sum(predictions == class_id)
            percentage = 100 * count / len(predictions)
            avg_prob = np.mean(probabilities[predictions == class_id, class_id]) if count > 0 else 0
            class_name = class_display_names[class_id] if class_id < len(class_display_names) else f'Class {class_id}'
            print(f"  {class_name} (class {class_id}): {count:,} points ({percentage:.2f}%) "
                  f"| Avg confidence: {avg_prob:.3f}")

        print("="*70)


def main():
    """
    Main inference function

    NOTE: Training configuration is automatically loaded from checkpoint directory.
    You only need to specify:
    1. Model path
    2. Input/output paths
    3. Optional inference-specific overrides (batch_size, stride, etc.)
    """

    # =========================
    # Configuration
    # =========================
    # Model checkpoint path (must point to .pth file in checkpoint directory with config.json)
    model_path = '/home/yush/local_backup_geotech/geotech_pointnet/checkpoints/test_run_knn_normals_rgb/checkpoint_epoch25.pth'

    # Input/Output paths
    input_las = '/home/yush/local_backup_geotech/geotech_pointnet/data/w_E_p1_6cm_all.las'
    output_las = './predictions/w_E_p1_6cm_all_predicted_knn_1024.las'
    data_ckpt = "./data/preprocessed/preprocessed_data.npz"

    # Optional: Inference-specific overrides
    # Training config is loaded automatically, but you can override specific values
    inference_overrides = {
        'batch_size': 1024,  # Override batch size for inference
        'input_channels':10,
        'dataset': {
            'params': {
                'inference_stride': 10,  # Override stride for faster/slower inference
                # 'cache_dir' is inherited from training config
            }
        }
    }

    # Create output directory
    os.makedirs(os.path.dirname(output_las), exist_ok=True)

    # =========================
    # Initialize Inference Engine
    # =========================
    # Training config (including features, normalization, RGB, etc.) loaded automatically
    inference_engine = PointCloudInference(
        model_path=model_path,
        inference_overrides=inference_overrides
    )

    # =========================
    # Load Point Cloud
    # =========================
    print("\n" + "="*70)
    print("LOADING POINT CLOUD")
    print("="*70)
    print(f"Loading point cloud from: {input_las}")
    las = laspy.read(input_las)
    xyz_array = np.vstack([las.x, las.y, las.z]).transpose()
    print(f"✅ Loaded {len(xyz_array):,} points")

    # Extract RGB if training used RGB
    rgb_array = None
    if inference_engine.config.get('use_rgb', False):
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
            rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
            print(f"✅ Extracted RGB colors (training used RGB)")
        else:
            print("⚠️  WARNING: Training used RGB but not available in LAS file.")
            print("    Inference may fail or produce incorrect results.")
            raise ValueError("RGB required by training config but not found in LAS file")
    else:
        print("RGB not used (training config: use_rgb=False)")

    # =========================
    # Run Prediction
    # =========================
    predictions, probabilities = inference_engine.predict_point_cloud(
        xyz_array=xyz_array,
        rgb_array=rgb_array,
        batch_size=inference_engine.config.get('batch_size', 1024)
    )

    # =========================
    # Save Results
    # =========================
    inference_engine.save_predictions_to_las(
        input_las_path=input_las,
        output_las_path=output_las,
        predictions=predictions,
        probabilities=probabilities,
        data_ckpt=data_ckpt
    )

    print("\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70)
    print(f"✅ Predictions saved to: {output_las}")
    print("="*70)


if __name__ == '__main__':
    main()

