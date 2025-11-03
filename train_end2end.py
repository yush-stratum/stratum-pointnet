"""
End-to-End Training Script for Rock Joint Classification
Integrates preprocessing, training, and evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

from pointnet2_model import PointNet2Classification, PointNet2Segmentation
from dataset import RockJointDataset
from preprocessing_utils import prepare_data_for_training
from train import Trainer


# Voxel-Based Dataset Integration Guide
from voxelize_dataset import VoxelDataset, create_train_test_voxel_datasets

def main():
    """Complete training pipeline from raw data to trained model"""

    # =========================
    # Configuration
    # =========================
    config = {
        'run_name': 'test_run_rgb_voxel_2m256p',

        # ===== DATA PATHS =====
        'las_file': '/home/yush/local_backup_geotech/geotech_pointnet/data/w_E_p1_6cm_all.las',
        'no_joints_dxf': '/home/yush/local_backup_geotech/geotech_cv/notebooks/dxfs/29 sept/no_joints_3_oct.dxf',
        'joints_dxf': '/home/yush/local_backup_geotech/geotech_cv/notebooks/dxfs/29 sept/polygon joints 29 sept.dxf',
        'data_dir': './data/preprocessed',
        # 'data_ckpt':'./checkpoints/preprocessed_data.npz',
        # 'polygon_ckpt':'./checkpoints/polygons_dict.pkl',

        # ===== TRAIN/TEST SPLIT =====
        # Define the line that separates train/test regions
        'split_point_a': [464275, 9175697, 2546],
        'split_point_b': [464294, 9175699, 2552],

        # ===== MODEL PARAMETERS =====
        'use_rgb': True,      # Whether to use RGB features
        'input_channels': 6,  # Will be set to 3 or 6 based on use_rgb
        'num_classes': 2,     # Binary classification

        # ===== DATA PARAMETERS =====
        'patch_size': 256,
        'voxel_size': 2 ,
        'normalize_mode': 'center_scale',  # Options: 'none', 'center', 'center_scale'
        'augment_train': False,
        # 'test_min_patch_distance': 1,  # Minimum distance between test patch centers (in meters)
        #                                   # Set to None to disable spatial subsampling

        # ===== TRAINING PARAMETERS =====
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'lr_decay_step': 20,
        'lr_decay_rate': 0.7,

        # ===== CLASS BALANCING (OPTIONAL) =====
        # Set to [w0, w1] if classes are imbalanced (e.g., [1.0, 2.0])
        'class_weights': None,

        # ===== SAVING =====
        'save_dir': './checkpoints',
        'save_interval': 5,

        # ===== REPRODUCIBILITY =====
        'seed': 42
    }

    data_dir = config['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    # Create run-specific directory
    run_dir = os.path.join(config['save_dir'], config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    config['run_dir'] = run_dir  # Add to config for easy access


    # Save config
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    print("\n" + "="*70)
    print("POINTNET++ ROCK JOINT CLASSIFICATION - END-TO-END PIPELINE")
    print("="*70)

    # =========================
    # Step 1: Data Preparation
    # =========================


      # Define paths for preprocessed data
    preprocessed_path = os.path.join(data_dir,'preprocessed_data.npz')
    polygons_path = os.path.join(data_dir, 'polygons_dict.pkl')

    # Check if preprocessed data exists
    if os.path.exists(preprocessed_path) and os.path.exists(polygons_path):
        print("\n" + "="*70)
        print("STEP 1: LOADING PREPROCESSED DATA")
        print("="*70)
        print(f"Loading from: {preprocessed_path}")

        # Load arrays
        data = np.load(preprocessed_path)
        xyz_array = data['xyz_array']
        label_array = data['label_array']
        train_mask = data['train_mask']
        test_mask = data['test_mask']
        rgb_array = data['rgb_array'] if 'rgb_array' in data else None

        print(f"Loaded point cloud: {len(xyz_array):,} points")
        if rgb_array is not None:
            print(f"RGB features available: Yes")
        else:
            print(f"RGB features available: No")
        print(f"Training points: {np.sum(train_mask):,}")
        print(f"Test points: {np.sum(test_mask):,}")

    #     # Load polygon dictionaries
    #     import pickle
    #     with open(polygons_path, 'rb') as f:
    #         polygons_data = pickle.load(f)
    #         polygons_dict_train = polygons_data['train']
    #         polygons_dict_test = polygons_data['test']

    #     print(f"Loaded polygon dictionaries from: {polygons_path}")
    #     print(f"Train polygons - Class 0: {len(polygons_dict_train['label_0'])}, Class 1: {len(polygons_dict_train['label_1'])}")
    #     print(f"Test polygons - Class 0: {len(polygons_dict_test['label_0'])}, Class 1: {len(polygons_dict_test['label_1'])}")

    else:
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION")
        print("="*70)
        print("No preprocessed data found. Running preprocessing pipeline...")

        point_a = np.array(config['split_point_a'])
        point_b = np.array(config['split_point_b'])

        (xyz_array, rgb_array, label_array, train_mask, test_mask,
         polygons_dict_train, polygons_dict_test) = prepare_data_for_training(
            las_file=config['las_file'],
            no_joints_dxf=config['no_joints_dxf'],
            joints_dxf=config['joints_dxf'],
            train_point_a=point_a,
            train_point_b=point_b,
            use_rgb=config['use_rgb']
        )

        # Save preprocessed data
        save_dict = {
            'xyz_array': xyz_array,
            'label_array': label_array,
            'train_mask': train_mask,
            'test_mask': test_mask
        }
        if rgb_array is not None:
            save_dict['rgb_array'] = rgb_array
        np.savez(preprocessed_path, **save_dict)
        print(f"\nPreprocessed data saved to: {preprocessed_path}")

        # import pandas as pd
        # df_exp = pd.DataFrame(np.column_stack([xyz_array,(label_array + 1)]),columns=['X','Y','Z','class'])
        # df_exp.to_csv('xyz_label.csv',index=False)
        # print(f"DEBUG: Saved xyz_label.csv")
        # Save polygon dictionaries
        import pickle
        with open(polygons_path, 'wb') as f:
            pickle.dump({
                'train': polygons_dict_train,
                'test': polygons_dict_test
            }, f)
        print(f"Polygon dictionaries saved to: {polygons_path}")

    # =========================
    # Step 2: Create Datasets
    # =========================
    print("\n" + "="*70)
    print("STEP 2: CREATING DATASETS")
    print("="*70)

    # Update input_channels based on RGB availability
    if rgb_array is not None and config['use_rgb']:
        config['input_channels'] = 6
        train_rgb = rgb_array[train_mask]
        test_rgb = rgb_array[test_mask]
    else:
        config['input_channels'] = 3
        train_rgb = None
        test_rgb = None
        if config['use_rgb']:
            print("\nWARNING: RGB requested but not available. Using XYZ only.")

    print(f"\nUsing {config['input_channels']}-channel input (XYZ{'+RGB' if config['input_channels'] == 6 else ''})")

    # print("\nCreating training dataset...")
    # train_dataset = RockJointDataset(
    #     xyz_array=xyz_array[train_mask],
    #     label_array=label_array[train_mask],
    #     polygons_dict=polygons_dict_train,
    #     patch_size=config['patch_size'],
    #     normalize_mode=config['normalize_mode'],
    #     augment=config['augment_train'],
    #     rgb_array=train_rgb
    # )

    # print("\nCreating test dataset...")
    # test_dataset = RockJointDataset(
    #     xyz_array=xyz_array[test_mask],
    #     label_array=label_array[test_mask],
    #     polygons_dict=polygons_dict_test,
    #     patch_size=config['patch_size'],
    #     normalize_mode=config['normalize_mode'],
    #     augment=False,
    #     rgb_array=test_rgb,
    #     min_patch_distance=config.get('test_min_patch_distance', None)
    # )

    # NEW CODE:
    print("\n" + "="*70)
    print("CREATING VOXEL-BASED DATASETS")
    print("="*70)

    train_dataset, test_dataset = create_train_test_voxel_datasets(
        xyz_array=xyz_array,
        label_array=label_array,
        train_mask=train_mask,
        test_mask=test_mask,
        patch_size=config['patch_size'],
        voxel_size=config.get('voxel_size', None),
        normalize_mode=config['normalize_mode'],
        augment_train=config['augment_train'],
        rgb_array=rgb_array
    )

    # Create DataLoaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # # train_dataset = RockJointDataset(...)
    # for i in range(10):
    #     patch, assigned_label = train_dataset.patches[i], train_dataset.labels[i]
    #     # Get actual labels of points in patch
    #     # You'll need to track indices to do this properly
    #     print(f"Patch {i}: Assigned label={assigned_label}")

    # # Check class balance
    # # print(train_dataset.labels)
    # train_labels_flat = train_dataset.labels.flatten()
    # test_labels_flat = test_dataset.labels.flatten()

    # # Remove unlabeled points (-1) from counting
    # train_labels_flat = train_labels_flat[train_labels_flat != -1]
    # test_labels_flat = test_labels_flat[test_labels_flat != -1]

    # train_class_counts = np.bincount(train_labels_flat)
    # test_class_counts = np.bincount(test_labels_flat)

    # print("\nClass distribution:")
    # print(f"  Training:   Class 0: {train_class_counts[0]}, Class 1: {train_class_counts[1]}")
    # print(f"  Test:       Class 0: {test_class_counts[0]}, Class 1: {test_class_counts[1]}")

    # # Suggest class weights if imbalanced
    # if train_class_counts[0] / train_class_counts[1] > 2.0 or train_class_counts[1] / train_class_counts[0] > 2.0:
    #     total = train_class_counts.sum()
    #     suggested_weights = [total / (2 * train_class_counts[0]),
    #                         total / (2 * train_class_counts[1])]
    #     print(f"\n  ⚠ Classes are imbalanced! Consider using class_weights: {suggested_weights}")

    # =========================
    # Step 3: Create Model
    # =========================
    print("\n" + "="*70)
    print("STEP 3: INITIALIZING MODEL")
    print("="*70)

    model = PointNet2Segmentation(
        num_classes=config['num_classes'],
        input_channels=config['input_channels']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: PointNet2Segmentation")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input: [{config['batch_size']}, {config['patch_size']}, {config['input_channels']}]")
    print(f"Output: [{config['batch_size']}, {config['num_classes']}]")

    # =========================
    # Step 4: Train
    # =========================
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)

    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()

    # =========================
    # Step 5: Summary
    # =========================
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\nBest Test Accuracy: {trainer.best_test_acc:.4f}")
    print(f"Best Model Saved: {trainer.best_model_path}")
    print(f"\nGenerated Files:")
    print(f"  - {trainer.best_model_path}")
    print(f"  - {config['save_dir']}/confusion_matrix.png")
    print(f"  - {config['save_dir']}/training_curves.png")
    print(f"  - {config['save_dir']}/classification_report.json")
    print(f"  - {config['save_dir']}/config.json")
    print(f"  - {preprocessed_path}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review training curves and confusion matrix")
    print("2. If results are good, run inference on full point cloud:")
    print(f"   python inference.py --model {trainer.best_model_path}")
    print("\n3. To improve results, try:")
    print("   - Different normalize_mode ('center_scale' or 'none')")
    print("   - Adjust learning_rate or num_epochs")
    print("   - Use class_weights if classes are imbalanced")
    print("   - Increase patch_size (if GPU memory allows)")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
