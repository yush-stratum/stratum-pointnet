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
from dataset_factory import DatasetFactory


def calculate_input_channels(config):
    """
    Calculate the total number of input channels based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        int: Total number of input channels

    Breakdown:
        - XYZ: 3 channels (always)
        - RGB: 3 channels (if use_rgb=True)
        - Geometric features: variable (based on feature_names)
          - normals: 3 channels
          - curvature, roughness, linearity, planarity, sphericity: 1 channel each
    """
    # Start with XYZ (always 3)
    total_channels = 3

    # Add RGB if enabled
    if config.get('use_rgb', False):
        total_channels += 3

    # Add geometric features if configured
    dataset_params = config.get('dataset', {}).get('params', {})
    feature_names = dataset_params.get('feature_names', None)

    if feature_names is not None and len(feature_names) > 0:
        for feature_name in feature_names:
            if feature_name == 'normals':
                total_channels += 3
            else:
                # Scalar features
                total_channels += 1

    return total_channels

def apply_smote_augmentation(xyz_array, label_array, rgb_array, feature_dict,
                             sampling_strategy, k_neighbors=5, random_state=42):
    """
    Apply SMOTE to generate synthetic training samples.

    Args:
        xyz_array: XYZ coordinates [N, 3]
        label_array: Class labels [N]
        rgb_array: RGB values [N, 3] or None
        feature_dict: Dictionary of features {name: array}
        total_target_samples: Target total number of samples after SMOTE
        k_neighbors: Number of neighbors for SMOTE interpolation
        random_state: Random seed

    Returns:
        Dictionary with augmented arrays
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn")

    print("\n" + "="*70)
    print("SMOTE AUGMENTATION")
    print("="*70)

    # Show original distribution
    print("\nOriginal class distribution:")
    unique_labels, counts = np.unique(label_array, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count:,} ({100*count/len(label_array):.2f}%)")
    print(f"  Total: {len(label_array):,} samples")

    # Build feature matrix
    feature_list = [xyz_array]
    if rgb_array is not None:
        feature_list.append(rgb_array)
    if feature_dict is not None:
        for fname in sorted(feature_dict.keys()):
            feat = feature_dict[fname]
            if feat.ndim == 1:
                feat = feat[:, np.newaxis]
            feature_list.append(feat)

    X = np.hstack(feature_list)  # [N, total_features]
    y = label_array

    # # Calculate sampling strategy to reach total_target_samples
    # current_total = len(y)
    # if total_target_samples <= current_total:
    #     print(f"\n⚠️  Target samples ({total_target_samples:,}) <= current ({current_total:,})")
    #     print("Skipping SMOTE augmentation")
    #     return None

    # # Distribute additional samples proportionally across classes
    # n_to_add = total_target_samples - current_total
    # class_proportions = counts / current_total

    # sampling_strategy = {}
    # for label, count, prop in zip(unique_labels, counts, class_proportions):
    #     target_count = int(count + n_to_add * prop)
    #     sampling_strategy[int(label)] = target_count

    # print(f"\nSMOTE strategy (target: {total_target_samples:,} total samples):")
    # for label in sorted(sampling_strategy.keys()):
    #     orig_count = counts[unique_labels == label][0]
    #     target_count = sampling_strategy[label]
    #     print(f"  Class {label}: {orig_count:,} → {target_count:,} (+{target_count - orig_count:,} synthetic)")

    # Apply SMOTE
    print(f"\nApplying SMOTE with k_neighbors={k_neighbors}...")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Show final distribution
    print("\nFinal class distribution after SMOTE:")
    unique_labels_final, counts_final = np.unique(y_resampled, return_counts=True)
    for label, count in zip(unique_labels_final, counts_final):
        print(f"  Class {label}: {count:,} ({100*count/len(y_resampled):.2f}%)")
    print(f"  Total: {len(y_resampled):,} samples")
    print(f"  Synthetic samples generated: {len(y_resampled) - len(y):,}")
    print("="*70 + "\n")

    # Split back into components
    idx = 0
    xyz_resampled = X_resampled[:, :3]
    idx += 3

    rgb_resampled = None
    if rgb_array is not None:
        rgb_resampled = X_resampled[:, idx:idx+3]
        idx += 3

    features_resampled = {}
    if feature_dict is not None:
        for fname in sorted(feature_dict.keys()):
            feat_dim = feature_dict[fname].shape[1] if feature_dict[fname].ndim == 2 else 1
            features_resampled[fname] = X_resampled[:, idx:idx+feat_dim]
            if feat_dim == 1:
                features_resampled[fname] = features_resampled[fname].flatten()
            idx += feat_dim

    return {
        'xyz_array': xyz_resampled,
        'label_array': y_resampled,
        'rgb_array': rgb_resampled,
        'features': features_resampled,
        'n_original': len(y),
        'n_synthetic': len(y_resampled) - len(y)
    }


def main():
    """Complete training pipeline from raw data to trained model"""

    # =========================
    # Configuration
    # =========================
    config = {
        'run_name': 'test_run_knn_binary_normals_smote',

        # ===== CLASSIFICATION MODE =====
        # Options: 'binary' or 'multiclass'
        # - 'binary': Only Joints (class 1) vs No-Joints (class 0). Unlabeled points ignored.
        # - 'multiclass': Background (class 0), No-Joints (class 1), Joints (class 2). Unlabeled points become Background.
        'classification_mode': 'binary',  # Change to 'binary' for 2-class classification

        # ===== DATA PATHS =====
        'las_file': '/home/yush/local_backup_geotech/geotech_pointnet/data/w_E_p1_6cm_all.las',
        'no_joints_dxf': '/home/yush/local_backup_geotech/geotech_cv/notebooks/dxfs/29 sept/no_joints_3_oct.dxf',
        'joints_dxf': '/home/yush/local_backup_geotech/geotech_cv/notebooks/dxfs/29 sept/polygon joints 29 sept.dxf',
        # 'rgb_array': '/home/yush/local_backup_geotech/geotech_pointnet/data/rgb_array.npy',
        'rgb_array':None,
        'data_dir': './data/multi_bench_test/',

        # ===== TRAIN/TEST SPLIT =====
        # Define the line that separates train/test regions
        # 'split_point_a': [464275, 9175697, 2546],
        # 'split_point_b': [464294, 9175699, 2552],

        'split_point_a':  [464293.65 , 9175861.23 , 2565.35 ],
        'split_point_b':  [464232.35 , 9175853.74 , 2516.821 ],

        # ===== DATASET CONFIGURATION =====
        'dataset': {
            'type': 'knn',  # Options: 'polygon', 'voxel', 'knn'
            'params': {
                'k_neighbors': 256,
                'normalize_mode': 'center',
                'augment_train': False,
                'train_stride': 10,  # Use every 10th point for initial sampling
                'test_stride': 10,    # Use every 3rd test point
                'min_labeled_ratio': 0.0,
                'cache_dir': './data/knn_cache/big_test',
                'feature_names': ['normals', 'curvature'],
                'feature_k_neighbors': 30,
                'feature_cache_dir': './data/feature_cache/big_test',

                # Label sampling for class balancing (limits samples per class)
                # None = no limit, int = same limit for all classes
                # dict = per-class limits: {0: 50000, 1: 100000, 2: 100000}
                # 'max_samples_per_class': {
                #     0: 10000,   # Background: max 50K samples
                #     1: 5000000,  # No-Joint: max 100K samples
                #     2: 5000000   # Joint: max 100K samples
                # }
            }
        },

        # ===== MODEL PARAMETERS =====
        'use_rgb': False,      # Whether to use RGB features
        'input_channels': 6,  # Will be calculated automatically based on features
        'num_classes': 2,  # Will be set automatically based on classification_mode (2 for binary, 3 for multiclass)

        # ===== DATA PARAMETERS (Backward compatibility) =====
        'patch_size': 1024,
        'normalize_mode': 'center',
        'augment_train': False,

        # ===== DATA AUGMENTATION (SMOTE) =====
        # Set total_train_samples to enable SMOTE augmentation
        # None = no SMOTE, use original data
        # int = target number of total training samples after SMOTE
        'sampling_strategy': {0: 700000, 1:500000},  # e.g., 500000 to upsample to 500k samples
        'smote_k_neighbors': 10,  # Number of neighbors for SMOTE interpolation

        # ===== TRAINING PARAMETERS =====
        'batch_size': 512,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'lr_decay_step': 20,
        'lr_decay_rate': 0.7,

        # ===== CLASS BALANCING (OPTIONAL) =====
        # Binary mode: [w0, w1] for [No-Joint, Joint]
        # Multiclass mode: [w0, w1, w2] for [Background, No-Joint, Joint]
        # Set to None for no weighting
        'class_weights': None,  # Example for multiclass: [Background, No-Joint, Joint]

        # ===== SAVING =====
        'save_dir': './checkpoints',
        'save_interval': 5,

        # ===== REPRODUCIBILITY =====
        'seed': 42
    }

    # =========================
    # Validate and Set Classification Mode
    # =========================
    classification_mode = config.get('classification_mode', 'multiclass')
    if classification_mode not in ['binary', 'multiclass']:
        raise ValueError(f"classification_mode must be 'binary' or 'multiclass', got '{classification_mode}'")

    # Set num_classes based on mode (override any manual setting)
    if classification_mode == 'binary':
        config['num_classes'] = 2  # No-Joint (0), Joint (1)
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION MODE: BINARY (2 classes)")
        print(f"  Class 0: No-Joint")
        print(f"  Class 1: Joint")
        print(f"  Note: Unlabeled points (label=-1) will be excluded from training")
        print(f"{'='*70}\n")
    else:  # multiclass
        config['num_classes'] = 3  # Background (0), No-Joint (1), Joint (2)
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION MODE: MULTICLASS (3 classes)")
        print(f"  Class 0: Background (unlabeled points)")
        print(f"  Class 1: No-Joint")
        print(f"  Class 2: Joint")
        print(f"{'='*70}\n")

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
        if config['use_rgb']:
          rgb_array = np.load(config['rgb_array'])
        else:
          rgb_array = None

        print(f"Loaded point cloud: {len(xyz_array):,} points")
        if rgb_array is not None:
            print(f"RGB features available: Yes")
        else:
            print(f"RGB features available: No")
        print(f"Training points: {np.sum(train_mask):,}")
        print(f"Test points: {np.sum(test_mask):,}")

        # CRITICAL: Check if labels need to be remapped based on classification mode
        unique_labels = np.unique(label_array)
        print(f"\nLoaded label range: {unique_labels}")

        if classification_mode == 'binary':
            # Binary mode expects labels: -1 (unlabeled), 0 (no-joint), 1 (joint)
            # If labels are [0, 1, 2], they were saved in multiclass mode - need to remap back
            if np.min(unique_labels) >= 0 and np.max(unique_labels) >= 2:
                print("⚠️  WARNING: Preprocessed data has multiclass labels [0, 1, 2]")
                print("⚠️  Binary mode requires [-1, 0, 1]. Remapping back...")
                # Remap: 0 → -1, 1 → 0, 2 → 1
                label_array = label_array - 1
                print(f"✅ Labels remapped: {np.unique(label_array)}")
        elif classification_mode == 'multiclass':
            # Multiclass mode expects labels already as [0, 1, 2] after loading
            # If labels are [-1, 0, 1], they need to be shifted
            if np.min(unique_labels) < 0:
                print("Labels are in binary format [-1, 0, 1], will be shifted to [0, 1, 2]")
            else:
                print("Labels already in multiclass format [0, 1, 2]")

    #     # Load polygon dictionaries
    #     import pickle
    #     with open(polygons_path, 'rb') as f:
    #         polygons_data = pickle.load(f)
    #         polygons_dict_train = polygons_data['train']
    #         polygons_dict_test = polygons_data['test']

        # Load polygon dictionaries (only required for 'polygon' dataset type)
        dataset_type = config.get('dataset', {}).get('type', 'polygon')
        if dataset_type == 'polygon':
            import pickle
            with open(polygons_path, 'rb') as f:
                polygons_data = pickle.load(f)
                polygons_dict_train = polygons_data['train']
                polygons_dict_test = polygons_data['test']

            print(f"Loaded polygon dictionaries from: {polygons_path}")
            print(f"Train polygons - Class 0: {len(polygons_dict_train['label_0'])}, Class 1: {len(polygons_dict_train['label_1'])}")
            print(f"Test polygons - Class 0: {len(polygons_dict_test['label_0'])}, Class 1: {len(polygons_dict_test['label_1'])}")
        else:
            polygons_dict_train = None
            polygons_dict_test = None
            print(f"Dataset type '{dataset_type}' does not require polygon dictionaries")

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

    # Load polygon dictionaries if they exist and dataset type requires them
    dataset_type = config.get('dataset', {}).get('type', 'polygon')
    if dataset_type == 'polygon':
        if not os.path.exists(polygons_path):
            raise FileNotFoundError(f"Polygon dataset requires {polygons_path} but it doesn't exist. Run preprocessing first.")
        import pickle
        with open(polygons_path, 'rb') as f:
            polygons_data = pickle.load(f)
            polygons_dict_train = polygons_data['train']
            polygons_dict_test = polygons_data['test']
        print(f"\nLoaded polygon dictionaries from: {polygons_path}")
    else:
        polygons_dict_train = None
        polygons_dict_test = None
        print(f"\nDataset type '{dataset_type}' does not require polygon dictionaries")

    # =========================
    # Handle Label Remapping for Multi-Class Classification
    # =========================
    # NOTE: Binary mode filtering happens in DatasetFactory._create_knn_datasets()
    # to ensure KDTree is built only on labeled points

    if classification_mode == 'multiclass':
        print("\n" + "="*70)
        print("MULTICLASS MODE: Converting unlabeled points to Background")
        print("="*70)
        print("Original labels: -1 (unlabeled), 0 (no-joint), 1 (joint)")
        print("New labels: 0 (background), 1 (no-joint), 2 (joint)")

        # Remap: -1 → 0 (background), 0 → 1 (no-joint), 1 → 2 (joint)
        original_label_counts = np.bincount(label_array[label_array >= 0])
        unlabeled_count = np.sum(label_array == -1)

        # Shift all labels by +1 (this makes -1→0, 0→1, 1→2)
        label_array = label_array + 1

        print(f"\nLabel counts after remapping:")
        print(f"  Background (class 0): {unlabeled_count:,} points (was unlabeled)")
        if len(original_label_counts) > 0:
            print(f"  No-Joint (class 1): {original_label_counts[0]:,} points (was class 0)")
        if len(original_label_counts) > 1:
            print(f"  Joint (class 2): {original_label_counts[1]:,} points (was class 1)")
        print("="*70 + "\n")

    # =========================
    # Step 2: Create Datasets
    # =========================
    print("\n" + "="*70)
    print("STEP 2: CREATING DATASETS")
    print("="*70)

    # DEBUG: Verify label distribution before dataset creation
    print(f"\nDEBUG - Label distribution check:")
    print(f"  Full point cloud: {len(xyz_array):,} points")
    print(f"  Train mask: {np.sum(train_mask):,} points selected")
    print(f"  Test mask: {np.sum(test_mask):,} points selected")
    train_label_unique = np.unique(label_array[train_mask])
    test_label_unique = np.unique(label_array[test_mask])
    print(f"  Train labels present: {train_label_unique}")
    print(f"  Test labels present: {test_label_unique}")
    for label in train_label_unique:
        count = np.sum(label_array[train_mask] == label)
        print(f"    Train class {label}: {count:,} points")
    for label in test_label_unique:
        count = np.sum(label_array[test_mask] == label)
        print(f"    Test class {label}: {count:,} points")
    print(f"  Expected num_classes: {config['num_classes']}")
    print()

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

    # Set feature cache key based on LAS filename for consistent caching
    las_basename = os.path.splitext(os.path.basename(config['las_file']))[0]
    config['dataset']['params']['feature_cache_key'] = las_basename
    print(f"\nFeature cache key set to: {las_basename}")

    # =========================
    # SMOTE Augmentation (Optional)
    # =========================
    smote_applied = False
    if config.get('total_train_samples') is not None:
        # Load features if needed for SMOTE
        feature_names = config.get('dataset', {}).get('params', {}).get('feature_names', None)
        feature_dict_train = None

        if feature_names is not None and len(feature_names) > 0:
            from feature_factory import FeatureFactory
            feature_cache_dir = config.get('dataset', {}).get('params', {}).get('feature_cache_dir', './data/feature_cache')
            feature_k = config.get('dataset', {}).get('params', {}).get('feature_k_neighbors', 30)

            print(f"\nLoading features for SMOTE: {feature_names}")
            full_features = FeatureFactory.get_features(
                xyz_array=xyz_array,
                feature_names=feature_names,
                k_neighbors=feature_k,
                cache_dir=feature_cache_dir,
                cache_key=las_basename
            )

            # Extract train features
            feature_dict_train = {}
            for fname in feature_names:
                feature_dict_train[fname] = full_features[fname][train_mask]

        # Apply SMOTE to training data only
        smote_result = apply_smote_augmentation(
            xyz_array=xyz_array[train_mask],
            label_array=label_array[train_mask],
            rgb_array=rgb_array[train_mask] if rgb_array is not None else None,
            feature_dict=feature_dict_train,
            sampling_strategy=config.get('sampling_strategy',None),
            k_neighbors=config.get('smote_k_neighbors', 5),
            random_state=config.get('seed', 42)
        )

        if smote_result is not None:
            # Replace training data with SMOTE-augmented data
            # We need to update xyz_array, label_array, rgb_array to include synthetic points
            # and update train_mask to include the new synthetic points

            n_original = len(xyz_array)
            n_synthetic = smote_result['n_synthetic']

            # Append synthetic points to arrays
            xyz_array = np.vstack([xyz_array, smote_result['xyz_array'][smote_result['n_original']:]])
            label_array = np.concatenate([label_array, smote_result['label_array'][smote_result['n_original']:]])

            if rgb_array is not None:
                rgb_array = np.vstack([rgb_array, smote_result['rgb_array'][smote_result['n_original']:]])

            # Extend train_mask to include synthetic points (all marked as train)
            synthetic_train_mask = np.ones(n_synthetic, dtype=bool)
            train_mask = np.concatenate([train_mask, synthetic_train_mask])

            # Extend test_mask (synthetic points not in test)
            synthetic_test_mask = np.zeros(n_synthetic, dtype=bool)
            test_mask = np.concatenate([test_mask, synthetic_test_mask])

            smote_applied = True
            print(f"✅ SMOTE applied: {n_original:,} → {len(xyz_array):,} total points")
            print(f"   Training points: {np.sum(train_mask):,} (including {n_synthetic:,} synthetic)")

    # Create datasets using factory
    print("\n" + "="*70)
    print("DATASET CREATION" + (" (with SMOTE augmentation)" if smote_applied else ""))
    print("="*70)

    train_dataset, test_dataset = DatasetFactory.create_train_test_datasets(
        config=config,
        xyz_array=xyz_array,
        label_array=label_array,
        train_mask=train_mask,
        test_mask=test_mask,
        rgb_array=rgb_array,
        polygons_dict_train=polygons_dict_train,
        polygons_dict_test=polygons_dict_test
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

    # =========================
    # Class Distribution Analysis
    # =========================
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)

    # Get labels from datasets
    if hasattr(train_dataset, 'label_array') and hasattr(train_dataset, 'center_point_indices'):
        # KNN dataset - get labels of center points
        train_labels = train_dataset.label_array[train_dataset.center_point_indices]
        test_labels = test_dataset.label_array[test_dataset.center_point_indices]
    else:
        # Other dataset types - would need different handling
        print("Warning: Class distribution analysis not available for this dataset type")
        train_labels = None
        test_labels = None

    if train_labels is not None:
        # Count samples per class
        unique_train = np.unique(train_labels)
        unique_test = np.unique(test_labels)

        # Get class names based on classification mode
        if classification_mode == 'binary':
            class_names = ['No-Joint', 'Joint']
        else:
            class_names = ['Background', 'No-Joint', 'Joint']

        print("\nTraining Dataset:")
        train_total = len(train_labels)
        for cls in sorted(unique_train):
            count = np.sum(train_labels == cls)
            pct = 100 * count / train_total
            class_name = class_names[int(cls)] if cls < len(class_names) else f'Class {cls}'
            print(f"  {class_name} (class {cls}): {count:,} samples ({pct:.2f}%)")
        print(f"  Total: {train_total:,} samples")

        print("\nTest Dataset:")
        test_total = len(test_labels)
        for cls in sorted(unique_test):
            count = np.sum(test_labels == cls)
            pct = 100 * count / test_total
            class_name = class_names[int(cls)] if cls < len(class_names) else f'Class {cls}'
            print(f"  {class_name} (class {cls}): {count:,} samples ({pct:.2f}%)")
        print(f"  Total: {test_total:,} samples")

        # Calculate class imbalance ratio
        train_counts = [np.sum(train_labels == cls) for cls in unique_train]
        max_count = max(train_counts)
        min_count = min(train_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 10:
            print("  ⚠️  SEVERE IMBALANCE - Consider using class weights or label sampling")
            # Suggest class weights (inverse frequency)
            weights = [train_total / (len(unique_train) * count) for count in train_counts]
            print(f"  Suggested class_weights: {weights}")
        elif imbalance_ratio > 3:
            print("  ⚠️  MODERATE IMBALANCE - May benefit from class weights")

    print("="*70 + "\n")

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
    # Step 3: Calculate Input Channels & Create Model
    # =========================
    print("\n" + "="*70)
    print("STEP 3: CALCULATING INPUT CHANNELS & INITIALIZING MODEL")
    print("="*70)

    # Calculate actual input channels based on dataset configuration
    actual_input_channels = calculate_input_channels(config)

    # Print breakdown
    print(f"\nCalculating input channels:")
    print(f"  - XYZ: 3 channels")
    if config.get('use_rgb', False):
        print(f"  - RGB: 3 channels")

    feature_names = config['dataset']['params'].get('feature_names', None)
    if feature_names is not None and len(feature_names) > 0:
        print(f"  - Geometric features:")
        for feature_name in feature_names:
            if feature_name == 'normals':
                print(f"    • normals: 3 channels")
            else:
                print(f"    • {feature_name}: 1 channel")

    print(f"\n  Total input channels: {actual_input_channels}")

    # Check if manual value matches calculated value
    if config.get('input_channels') is not None and config['input_channels'] != actual_input_channels:
        print(f"  ⚠️  WARNING: Config has input_channels={config['input_channels']}, but calculated {actual_input_channels}")
        print(f"  ⚠️  Using calculated value: {actual_input_channels}")

    # Override config with calculated value
    config['input_channels'] = actual_input_channels

    model = PointNet2Segmentation(
        num_classes=config['num_classes'],
        input_channels=actual_input_channels
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: PointNet2Segmentation")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: [B={config['batch_size']}, N={config['dataset']['params']['k_neighbors']}, C={actual_input_channels}]")
    print(f"Output shape: [B={config['batch_size']}, num_classes={config['num_classes']}, N={config['dataset']['params']['k_neighbors']}]")

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
