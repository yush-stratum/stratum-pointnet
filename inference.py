"""
Inference script for PointNet++ on full point clouds
Outputs per-point predictions to new LAS file
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import laspy
import os
from collections import defaultdict

from pointnet2_model import PointNet2Segmentation
from dataset import RockJointInferenceDataset

from voxelize_dataset import VoxelDataset

class PointCloudInference:
    """
    Inference engine for full point cloud prediction
    """
    def __init__(self, model_path, config):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Dict with inference configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        voxel_size = checkpoint.get('voxel_size', None)
        if voxel_size is not None:
            print(f"Using voxel size from checkpoint: {voxel_size:.4f} m")
            self.config['voxel_size'] = voxel_size
        else:
            print("WARNING: No voxel size in checkpoint, will auto-compute")

        # Initialize model (use classification model for patch-based inference)
        from pointnet2_model import PointNet2Classification, PointNet2Segmentation
        self.model = PointNet2Segmentation(
            num_classes=config['num_classes'],
            input_channels=config['input_channels']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (trained to epoch {checkpoint['epoch']})")
        print(f"Training accuracy: {checkpoint.get('test_acc', 'N/A')}")

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


    def predict_point_cloud(self, xyz_array, rgb_array=None, batch_size=16):
        """Updated inference with proper point mapping"""

        # Create inference dataset
        inference_dataset = VoxelDataset(
            xyz_array=xyz_array,
            label_array=None,
            rgb_array=rgb_array,
            patch_size=self.config['patch_size'],
            voxel_size=self.config.get('voxel_size', None),
            normalize_mode=self.config['normalize_mode'],
            augment=False
        )

        # Modify dataloader to use custom collate function
        def collate_with_indices(batch):
            """Collate function that preserves indices"""
            points = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            indices = [item[2] for item in batch]  # List of index tensors
            return points, labels, indices

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

        print(f"\nRunning inference on {len(inference_dataset)} voxels...")

        with torch.no_grad():
            for points, _, batch_indices in tqdm(inference_loader, desc="Inference"):
                points = points.to(self.device)  # [B, N, C]

                outputs = self.model(points)  # [B, num_classes, N]
                probs = torch.softmax(outputs, dim=1)  # [B, num_classes, N]
                preds = torch.argmax(outputs, dim=1)  # [B, N]

                # Convert to numpy
                preds_np = preds.cpu().numpy()  # [B, N]
                probs_np = probs.cpu().numpy()  # [B, num_classes, N]

                # Assign predictions using original point indices
                for b in range(len(batch_indices)):
                    point_indices = batch_indices[b].numpy()  # [patch_size]
                    point_preds = preds_np[b]  # [patch_size]
                    point_probs = probs_np[b].T  # [patch_size, num_classes]

                    # Accumulate predictions (averaging if points appear in multiple voxels)
                    for i, pt_idx in enumerate(point_indices):
                        predictions[pt_idx] = point_preds[i]
                        probabilities[pt_idx] += point_probs[i]
                        vote_counts[pt_idx] += 1

        # Average probabilities where points were seen
        mask = vote_counts > 0
        probabilities[mask] /= vote_counts[mask, None]

        # Report coverage
        coverage = np.sum(mask) / len(xyz_array)
        print(f"\nInference coverage: {np.sum(mask):,} / {len(xyz_array):,} points ({coverage:.1%})")

        if coverage < 0.95:
            print(f"WARNING: Low coverage! {np.sum(~mask):,} points not covered by any voxel")

        return predictions, probabilities

    def get_train_test_mask(self):
        data = np.load(self.config['data_ckpt'])
        xyz_array = data['xyz_array']
        label_array = data['label_array']
        train_mask = data['train_mask']
        test_mask = data['test_mask']

        train_classification_array = np.zeros_like(label_array)
        train_classification_array -= 1 #label array is -1 for unassigned
        train_classification_array[np.where(train_mask)[0]] = label_array[np.where(train_mask)[0]]

        test_classification_array = np.zeros_like(label_array)
        test_classification_array -= 1
        test_classification_array[np.where(test_mask)[0]] = label_array[np.where(test_mask)[0]]

        train_classification_array += 1
        test_classification_array += 1

        return train_classification_array.astype(np.uint8), test_classification_array.astype(np.uint8), (label_array+1).astype(np.uint8)

    def save_predictions_to_las(self, input_las_path, output_las_path,
                                predictions, probabilities):
        """
        Save predictions to new LAS file

        Args:
            input_las_path: Original LAS file path
            output_las_path: Output LAS file path
            predictions: [N] array with class predictions
            probabilities: [N, num_classes] array with probabilities
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
        train_mask, test_mask, label_arr = self.get_train_test_mask()

        # Add probabilities as extra dimensions if supported
        try:
            # Add probability for class 0
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="prob_class_0",
                type=np.float32
            ))
            new_las.prob_class_0 = probabilities[:, 0]

            # Add probability for class 1
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="prob_class_1",
                type=np.float32
            ))
            new_las.prob_class_1 = probabilities[:, 1]


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


            print("Added probability fields to LAS file")
        except Exception as e:
            print(f"Warning: Could not add probability fields: {e}")

        # Write to file
        new_las.write(output_las_path)
        print(f"Successfully saved predictions to {output_las_path}")

        # Print statistics
        print("\nPrediction Statistics:")
        for class_id in range(self.config['num_classes']):
            count = np.sum(predictions == class_id)
            percentage = 100 * count / len(predictions)
            avg_prob = np.mean(probabilities[predictions == class_id, class_id]) if count > 0 else 0
            print(f"  Class {class_id}: {count} points ({percentage:.2f}%) "
                  f"| Avg confidence: {avg_prob:.3f}")


def main():
    """Main inference function"""

    # =========================
    # Configuration
    # =========================
    config = {
        # Model path
        'model_path': '/home/yush/local_backup_geotech/geotech_pointnet/checkpoints/test_run_rgb_voxel_2m256p/best_model_acc0.8158.pth',  # Update with your best model

        # Input/Output
        'input_las': '/home/yush/local_backup_geotech/geotech_pointnet/data/w_E_p1_6cm_all.las',
        'output_las': './predictions/w_E_p1_6cm_all_predicted_fix_xyz_voxel.las',
        "data_ckpt": "./data/preprocessed/preprocessed_data.npz",

        # Model parameters (must match training)
        'num_classes': 2,
        'use_rgb': False,       # Whether to use RGB features (must match training)
        'input_channels': 3,   # Will be set to 3 or 6 based on use_rgb

        # Inference parameters
        'patch_size': 256,
        'stride': 32,
        'normalize_mode': 'center_scale',  # Must match training
        'batch_size': 16,
    }

    # Create output directory
    os.makedirs(os.path.dirname(config['output_las']), exist_ok=True)

    # =========================
    # Load Point Cloud
    # =========================
    print("Loading point cloud...")
    las = laspy.read(config['input_las'])
    xyz_array = np.vstack([las.x, las.y, las.z]).transpose()
    print(f"Loaded {len(xyz_array)} points")

    # Extract RGB if requested
    rgb_array = None
    if config['use_rgb']:
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
            rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
            print(f"Extracted RGB colors")
            config['input_channels'] = 6
        else:
            print("WARNING: RGB requested but not available in LAS file. Using XYZ only.")
            config['use_rgb'] = False
            config['input_channels'] = 3
    else:
        config['input_channels'] = 3

    print(f"Using {config['input_channels']}-channel input (XYZ{'+RGB' if config['input_channels'] == 6 else ''})")

    # =========================
    # Initialize Inference Engine
    # =========================
    inference_engine = PointCloudInference(
        model_path=config['model_path'],
        config=config
    )

    # =========================
    # Run Prediction
    # =========================
    predictions, probabilities = inference_engine.predict_point_cloud(
        xyz_array=xyz_array,
        rgb_array=rgb_array,
        batch_size=config['batch_size']
    )

    # =========================
    # Save Results
    # =========================
    inference_engine.save_predictions_to_las(
        input_las_path=config['input_las'],
        output_las_path=config['output_las'],
        predictions=predictions,
        probabilities=probabilities
    )

    print("\n" + "="*50)
    print("Inference Complete!")
    print(f"Predictions saved to: {config['output_las']}")
    print("="*50)


if __name__ == '__main__':
    main()
