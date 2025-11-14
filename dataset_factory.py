"""
Dataset Factory - Modular Dataset Management

Provides a clean interface for creating different dataset types:
- PolygonDataset (original): Polygon-centered patches with coherence filtering
- VoxelDataset: Regular voxel grid, no filtering
- KNNPointDataset: K-nearest neighbors, fully deterministic

Usage:
    from dataset_factory import DatasetFactory

    # Create train/test datasets
    train_ds, test_ds = DatasetFactory.create_train_test_datasets(
        config=config,
        xyz_array=xyz,
        label_array=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        rgb_array=rgb
    )
"""

import numpy as np
from typing import Dict, Tuple, Optional
import os


class DatasetFactory:
    """
    Factory class for creating different types of datasets

    Supported dataset types:
    - 'polygon': Original polygon-centered approach (dataset.RockJointDataset)
    - 'voxel': Voxel grid approach (voxelize_dataset.VoxelDataset)
    - 'knn': K-nearest neighbors approach (knn_point_dataset.KNNPointDataset)
    """

    SUPPORTED_TYPES = ['polygon', 'voxel', 'knn']

    @staticmethod
    def create_train_test_datasets(
        config: Dict,
        xyz_array: np.ndarray,
        label_array: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        rgb_array: Optional[np.ndarray] = None,
        polygons_dict_train: Optional[Dict] = None,
        polygons_dict_test: Optional[Dict] = None
    ) -> Tuple:
        """
        Create training and test datasets based on config

        Args:
            config: Configuration dictionary with 'dataset' section
            xyz_array: Full point cloud [N, 3]
            label_array: Labels [N]
            train_mask: Boolean mask for training points
            test_mask: Boolean mask for test points
            rgb_array: Optional RGB colors [N, 3]
            polygons_dict_train: Required for 'polygon' type
            polygons_dict_test: Required for 'polygon' type

        Returns:
            (train_dataset, test_dataset)

        Config Structure:
            config = {
                'dataset': {
                    'type': 'knn',  # or 'polygon', 'voxel'
                    'params': {
                        # Type-specific parameters
                    }
                },
                # Other config fields...
            }
        """
        dataset_config = config.get('dataset', {})
        dataset_type = dataset_config.get('type', 'polygon')  # Default to original
        dataset_params = dataset_config.get('params', {})

        # Validate dataset type
        if dataset_type not in DatasetFactory.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported dataset type: '{dataset_type}'. "
                f"Must be one of {DatasetFactory.SUPPORTED_TYPES}"
            )

        print(f"\n{'='*70}")
        print(f"DATASET FACTORY: Creating '{dataset_type}' datasets")
        print(f"{'='*70}\n")

        # Route to appropriate creation method
        if dataset_type == 'polygon':
            return DatasetFactory._create_polygon_datasets(
                xyz_array, label_array, train_mask, test_mask,
                rgb_array, polygons_dict_train, polygons_dict_test,
                config, dataset_params
            )
        elif dataset_type == 'voxel':
            return DatasetFactory._create_voxel_datasets(
                xyz_array, label_array, train_mask, test_mask,
                rgb_array, config, dataset_params
            )
        elif dataset_type == 'knn':
            return DatasetFactory._create_knn_datasets(
                xyz_array, label_array, train_mask, test_mask,
                rgb_array, config, dataset_params
            )

    @staticmethod
    def _create_polygon_datasets(
        xyz_array, label_array, train_mask, test_mask,
        rgb_array, polygons_dict_train, polygons_dict_test,
        config, dataset_params
    ):
        """Create polygon-centered datasets (original approach)"""
        from dataset import RockJointDataset

        if polygons_dict_train is None or polygons_dict_test is None:
            raise ValueError("Polygon datasets require polygons_dict_train and polygons_dict_test")

        # Extract parameters with defaults
        patch_size = dataset_params.get('patch_size', config.get('patch_size', 1024))
        normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))
        augment_train = dataset_params.get('augment_train', config.get('augment_train', False))
        min_patch_distance = dataset_params.get('test_min_patch_distance',
                                               config.get('test_min_patch_distance', None))

        print("Creating training dataset (polygon-centered)...")
        train_dataset = RockJointDataset(
            xyz_array=xyz_array[train_mask],
            label_array=label_array[train_mask],
            polygons_dict=polygons_dict_train,
            patch_size=patch_size,
            normalize_mode=normalize_mode,
            augment=augment_train,
            rgb_array=rgb_array[train_mask] if rgb_array is not None else None,
            min_patch_distance=None  # No subsampling for training
        )

        print("\nCreating test dataset (polygon-centered)...")
        test_dataset = RockJointDataset(
            xyz_array=xyz_array[test_mask],
            label_array=label_array[test_mask],
            polygons_dict=polygons_dict_test,
            patch_size=patch_size,
            normalize_mode=normalize_mode,
            augment=False,
            rgb_array=rgb_array[test_mask] if rgb_array is not None else None,
            min_patch_distance=min_patch_distance
        )

        return train_dataset, test_dataset

    @staticmethod
    def _create_voxel_datasets(
        xyz_array, label_array, train_mask, test_mask,
        rgb_array, config, dataset_params
    ):
        """Create voxel grid datasets"""
        from voxelize_dataset import create_train_test_voxel_datasets

        # Extract parameters with defaults
        patch_size = dataset_params.get('patch_size', config.get('patch_size', 1024))
        voxel_size = dataset_params.get('voxel_size', None)
        normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))
        augment_train = dataset_params.get('augment_train', config.get('augment_train', False))

        return create_train_test_voxel_datasets(
            xyz_array=xyz_array,
            label_array=label_array,
            train_mask=train_mask,
            test_mask=test_mask,
            patch_size=patch_size,
            voxel_size=voxel_size,
            normalize_mode=normalize_mode,
            augment_train=augment_train,
            rgb_array=rgb_array
        )

    @staticmethod
    def _create_knn_datasets(
        xyz_array, label_array, train_mask, test_mask,
        rgb_array, config, dataset_params
    ):
        """Create KNN point datasets (fully deterministic)"""
        from knn_point_dataset import create_train_test_knn_datasets

        # Extract parameters with defaults
        k_neighbors = dataset_params.get('k_neighbors', config.get('patch_size', 1024))
        normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))
        augment_train = dataset_params.get('augment_train', config.get('augment_train', False))
        train_stride = dataset_params.get('train_stride', 1)
        test_stride = dataset_params.get('test_stride', 1)
        min_labeled_ratio = dataset_params.get('min_labeled_ratio', 0.0)
        cache_dir = dataset_params.get('cache_dir', './data/knn_cache')
        max_samples_per_class = dataset_params.get('max_samples_per_class', None)

        # Feature parameters
        feature_names = dataset_params.get('feature_names', None)
        feature_k_neighbors = dataset_params.get('feature_k_neighbors', 30)
        feature_cache_dir = dataset_params.get('feature_cache_dir', './data/feature_cache')
        feature_cache_key = dataset_params.get('feature_cache_key', None)

        # NOTE: Label filtering now happens in train_end2end.py before calling this function
        # Both binary and multiclass modes filter based on config['class_labels']

        return create_train_test_knn_datasets(
            xyz_array=xyz_array,
            label_array=label_array,
            train_mask=train_mask,
            test_mask=test_mask,
            k_neighbors=k_neighbors,
            normalize_mode=normalize_mode,
            augment_train=augment_train,
            rgb_array=rgb_array,
            train_stride=train_stride,
            test_stride=test_stride,
            min_labeled_ratio=min_labeled_ratio,
            cache_dir=cache_dir,
            max_samples_per_class=max_samples_per_class,
            feature_names=feature_names,
            feature_k_neighbors=feature_k_neighbors,
            feature_cache_dir=feature_cache_dir,
            feature_cache_key=feature_cache_key
        )

    @staticmethod
    def create_inference_dataset(
        config: Dict,
        xyz_array: np.ndarray,
        rgb_array: Optional[np.ndarray] = None,
        label_array: Optional[np.ndarray] = None
    ):
        """
        Create inference dataset based on config

        Args:
            config: Configuration dictionary with 'dataset' section
            xyz_array: Full point cloud [N, 3]
            rgb_array: Optional RGB colors [N, 3]
            label_array: Optional ground truth labels (for evaluation)

        Returns:
            inference_dataset
        """
        dataset_config = config.get('dataset', {})
        dataset_type = dataset_config.get('type', 'polygon')
        dataset_params = dataset_config.get('params', {})

        print(f"\n{'='*70}")
        print(f"DATASET FACTORY: Creating '{dataset_type}' inference dataset")
        print(f"{'='*70}\n")

        if dataset_type == 'polygon':
            # Old inference dataset (sliding window)
            from dataset import RockJointInferenceDataset

            patch_size = dataset_params.get('patch_size', config.get('patch_size', 1024))
            stride = dataset_params.get('stride', config.get('stride', 512))
            normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))

            return RockJointInferenceDataset(
                xyz_array=xyz_array,
                rgb_array=rgb_array,
                patch_size=patch_size,
                stride=stride,
                normalize_mode=normalize_mode
            )

        elif dataset_type == 'voxel':
            from voxelize_dataset import VoxelDataset

            patch_size = dataset_params.get('patch_size', config.get('patch_size', 1024))
            voxel_size = dataset_params.get('voxel_size', config.get('voxel_size', None))
            normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))

            return VoxelDataset(
                xyz_array=xyz_array,
                label_array=label_array,  # Can be None for pure inference
                rgb_array=rgb_array,
                patch_size=patch_size,
                voxel_size=voxel_size,
                normalize_mode=normalize_mode,
                augment=False
            )

        elif dataset_type == 'knn':
            from knn_point_dataset import KNNPointDataset

            k_neighbors = dataset_params.get('k_neighbors', config.get('patch_size', 1024))
            normalize_mode = dataset_params.get('normalize_mode', config.get('normalize_mode', 'center'))
            stride = dataset_params.get('inference_stride', dataset_params.get('test_stride', 1))
            cache_dir = dataset_params.get('cache_dir', './data/knn_cache')
            cache_path = os.path.join(cache_dir, 'inference_knn_cache.pkl')

            # Feature parameters (should match training config)
            feature_names = dataset_params.get('feature_names', None)
            feature_k_neighbors = dataset_params.get('feature_k_neighbors', 30)
            feature_cache_dir = dataset_params.get('feature_cache_dir', './data/feature_cache')
            feature_cache_key = dataset_params.get('feature_cache_key', None)

            return KNNPointDataset(
                xyz_array=xyz_array,
                label_array=label_array,
                rgb_array=rgb_array,
                k_neighbors=k_neighbors,
                normalize_mode=normalize_mode,
                augment=False,  # Never augment during inference
                cache_path=cache_path,
                stride=stride,
                feature_names=feature_names,
                feature_k_neighbors=feature_k_neighbors,
                feature_cache_dir=feature_cache_dir,
                feature_cache_key=feature_cache_key
            )

    @staticmethod
    def get_dataset_info(dataset_type: str) -> Dict:
        """
        Get information about a dataset type

        Args:
            dataset_type: One of 'polygon', 'voxel', 'knn'

        Returns:
            Dictionary with dataset information
        """
        info = {
            'polygon': {
                'name': 'Polygon-Centered Dataset',
                'description': 'Original approach: patches centered on polygon centroids with coherence filtering',
                'deterministic': False,
                'pros': ['Good for well-labeled regions', 'Focuses on polygon interiors'],
                'cons': ['Filters out mixed regions', 'Random sampling', 'Train-test distribution mismatch'],
                'use_case': 'Legacy compatibility, highly curated training data'
            },
            'voxel': {
                'name': 'Voxel Grid Dataset',
                'description': 'Regular 3D voxel grid with no filtering',
                'deterministic': 'Partially (grid is deterministic, sampling is random)',
                'pros': ['Uniform spatial coverage', 'Includes boundary regions', 'Same grid structure for train/test'],
                'cons': ['Still has random sampling', 'May include empty regions'],
                'use_case': 'Balanced coverage, includes difficult regions'
            },
            'knn': {
                'name': 'K-Nearest Neighbors Dataset',
                'description': 'Each point queries K nearest neighbors, fully deterministic',
                'deterministic': True,
                'pros': ['Fully deterministic', 'Reproducible', 'No random sampling', 'Point-centric'],
                'cons': ['Larger dataset (one sample per point)', 'Requires KNN computation'],
                'use_case': 'Production deployments, reproducible research, maximum determinism'
            }
        }

        if dataset_type not in info:
            return {'error': f'Unknown dataset type: {dataset_type}'}

        return info[dataset_type]

    @staticmethod
    def print_dataset_comparison():
        """Print a comparison table of all dataset types"""
        print("\n" + "="*90)
        print("DATASET TYPE COMPARISON")
        print("="*90)

        for dtype in DatasetFactory.SUPPORTED_TYPES:
            info = DatasetFactory.get_dataset_info(dtype)
            print(f"\n{info['name']} ('{dtype}')")
            print("-" * 90)
            print(f"Description: {info['description']}")
            print(f"Deterministic: {info['deterministic']}")
            print(f"Pros: {', '.join(info['pros'])}")
            print(f"Cons: {', '.join(info['cons'])}")
            print(f"Use case: {info['use_case']}")

        print("\n" + "="*90)
        print("RECOMMENDATION: Use 'knn' for maximum determinism and reproducibility")
        print("="*90 + "\n")


# Example usage
if __name__ == '__main__':
    print("DatasetFactory - Example Usage\n")

    # Print comparison
    DatasetFactory.print_dataset_comparison()

    # Example configuration for KNN dataset
    config_knn = {
        'dataset': {
            'type': 'knn',
            'params': {
                'k_neighbors': 1024,
                'normalize_mode': 'center',
                'augment_train': False,
                'train_stride': 1,
                'test_stride': 1,
                'min_labeled_ratio': 0.0,
                'cache_dir': './data/knn_cache'
            }
        },
        # Backwards compatibility (will be overridden by dataset.params)
        'patch_size': 1024,
        'normalize_mode': 'center',
        'augment_train': False
    }

    # Example configuration for Voxel dataset
    config_voxel = {
        'dataset': {
            'type': 'voxel',
            'params': {
                'patch_size': 1024,
                'voxel_size': None,  # Auto-compute
                'normalize_mode': 'center',
                'augment_train': False
            }
        }
    }

    # Example configuration for Polygon dataset
    config_polygon = {
        'dataset': {
            'type': 'polygon',
            'params': {
                'patch_size': 1024,
                'normalize_mode': 'center',
                'augment_train': False,
                'test_min_patch_distance': 1.5
            }
        }
    }

    print("\nExample KNN Config:")
    print(config_knn)

    print("\nExample Voxel Config:")
    print(config_voxel)

    print("\nExample Polygon Config:")
    print(config_polygon)
