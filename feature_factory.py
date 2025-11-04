"""
Feature Factory - Modular Geometric Feature Computation with Caching

Computes geometric features for point clouds:
- Normals: Surface normal vectors (required for other features)
- Curvature: Surface curvature estimation
- Roughness: Local surface roughness
- Linearity: 1D structure measure
- Planarity: 2D structure measure
- Sphericity: 3D structure measure

Features are cached to disk. The factory intelligently:
1. Loads existing features from cache
2. Computes only missing features
3. Saves newly computed features back to cache

Usage:
    from feature_factory import FeatureFactory

    # Request specific features
    features = FeatureFactory.get_features(
        xyz_array=xyz,
        feature_names=['normals', 'curvature', 'planarity'],
        k_neighbors=30,
        cache_dir='./data/feature_cache'
    )

    # Returns: {'normals': array, 'curvature': array, 'planarity': array}
"""

import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import hashlib


class FeatureFactory:
    """
    Factory for computing and caching geometric features from point clouds.
    """

    # All available features
    AVAILABLE_FEATURES = [
        'normals',      # [N, 3] surface normals
        'curvature',    # [N] curvature values
        'roughness',    # [N] surface roughness
        'linearity',    # [N] linear structure measure
        'planarity',    # [N] planar structure measure
        'sphericity'    # [N] spherical structure measure
    ]

    # Features that depend on normals
    NORMAL_DEPENDENT = ['curvature', 'roughness']

    @staticmethod
    def get_features(
        xyz_array: np.ndarray,
        feature_names: List[str],
        k_neighbors: int = 30,
        cache_dir: str = './data/feature_cache',
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get requested features, computing only what's missing from cache.

        Args:
            xyz_array: Point cloud [N, 3]
            feature_names: List of feature names to compute
            k_neighbors: Number of neighbors for feature computation
            cache_dir: Directory to store cached features
            force_recompute: If True, ignore cache and recompute all

        Returns:
            Dictionary mapping feature names to arrays

        Example:
            features = FeatureFactory.get_features(
                xyz_array=xyz,
                feature_names=['normals', 'planarity'],
                k_neighbors=30
            )
            # Returns: {'normals': [N,3], 'planarity': [N]}
        """
        # Validate feature names
        for name in feature_names:
            if name not in FeatureFactory.AVAILABLE_FEATURES:
                raise ValueError(
                    f"Unknown feature: '{name}'. "
                    f"Available: {FeatureFactory.AVAILABLE_FEATURES}"
                )

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Generate cache key based on point cloud and k_neighbors
        cache_key = FeatureFactory._generate_cache_key(xyz_array, k_neighbors)
        cache_path = os.path.join(cache_dir, f"features_{cache_key}.pkl")

        # Load existing cache
        cached_features = {}
        if not force_recompute and os.path.exists(cache_path):
            cached_features = FeatureFactory._load_cache(cache_path)
            print(f"\n{'='*70}")
            print(f"FEATURE FACTORY: Loading from cache")
            print(f"{'='*70}")
            print(f"Cache file: {cache_path}")
            print(f"Cached features: {list(cached_features.keys())}")

        # Determine which features need computation
        missing_features = [f for f in feature_names if f not in cached_features]

        if not missing_features:
            print(f"All requested features found in cache! ✅")
            print(f"{'='*70}\n")
            return {name: cached_features[name] for name in feature_names}

        print(f"\n{'='*70}")
        print(f"FEATURE FACTORY: Computing missing features")
        print(f"{'='*70}")
        print(f"Requested: {feature_names}")
        print(f"In cache: {[f for f in feature_names if f in cached_features]}")
        print(f"To compute: {missing_features}")
        print(f"{'='*70}\n")

        # Compute missing features
        computed_features = FeatureFactory._compute_features(
            xyz_array=xyz_array,
            feature_names=missing_features,
            k_neighbors=k_neighbors,
            existing_features=cached_features
        )

        # Merge cached and computed
        all_features = {**cached_features, **computed_features}

        # Save updated cache
        FeatureFactory._save_cache(cache_path, all_features)
        print(f"\nCache updated with new features ✅")
        print(f"Cache now contains: {list(all_features.keys())}\n")

        # Return only requested features
        return {name: all_features[name] for name in feature_names}

    @staticmethod
    def _generate_cache_key(xyz_array: np.ndarray, k_neighbors: int) -> str:
        """Generate unique cache key for point cloud and parameters"""
        # Use hash of point cloud shape + sample points + k_neighbors
        n_points = len(xyz_array)
        sample_indices = np.linspace(0, n_points-1, min(1000, n_points), dtype=int)
        sample_points = xyz_array[sample_indices].tobytes()

        hash_input = f"{n_points}_{k_neighbors}_{sample_points}".encode()
        cache_key = hashlib.md5(hash_input).hexdigest()[:16]

        return cache_key

    @staticmethod
    def _load_cache(cache_path: str) -> Dict[str, np.ndarray]:
        """Load cached features from disk"""
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            return cached
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            return {}

    @staticmethod
    def _save_cache(cache_path: str, features: Dict[str, np.ndarray]):
        """Save features to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

    @staticmethod
    def _compute_features(
        xyz_array: np.ndarray,
        feature_names: List[str],
        k_neighbors: int,
        existing_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute requested features.
        Reuses existing features where possible (e.g., normals for curvature).
        """
        computed = {}
        n_points = len(xyz_array)

        # Build KDTree for neighbor queries (used by all features)
        print(f"Building KDTree for {n_points:,} points...")
        kdtree = KDTree(xyz_array)

        # Check if we need normals (either directly or for dependent features)
        need_normals = 'normals' in feature_names or \
                      any(f in feature_names for f in FeatureFactory.NORMAL_DEPENDENT)

        # Compute or reuse normals
        if need_normals:
            if 'normals' in existing_features:
                print(f"Reusing normals from existing features ✅")
                normals = existing_features['normals']
            else:
                print(f"Computing normals (k={k_neighbors})...")
                normals = FeatureFactory._compute_normals(xyz_array, kdtree, k_neighbors)
                computed['normals'] = normals
                print(f"  ✓ Normals computed: {normals.shape}")
        else:
            normals = None

        # Compute each requested feature
        for feature_name in feature_names:
            if feature_name == 'normals':
                continue  # Already handled above

            if feature_name == 'curvature':
                print(f"Computing curvature...")
                computed['curvature'] = FeatureFactory._compute_curvature(
                    xyz_array, normals, kdtree, k_neighbors
                )
                print(f"  ✓ Curvature computed: {computed['curvature'].shape}")

            elif feature_name == 'roughness':
                print(f"Computing roughness...")
                computed['roughness'] = FeatureFactory._compute_roughness(
                    xyz_array, normals, kdtree, k_neighbors
                )
                print(f"  ✓ Roughness computed: {computed['roughness'].shape}")

            elif feature_name in ['linearity', 'planarity', 'sphericity']:
                # These are computed together from eigenvalues
                if 'linearity' not in computed and 'linearity' in feature_names:
                    print(f"Computing geometric features (linearity, planarity, sphericity)...")
                    lin, plan, spher = FeatureFactory._compute_geometric_features(
                        xyz_array, kdtree, k_neighbors
                    )
                    if 'linearity' in feature_names:
                        computed['linearity'] = lin
                    if 'planarity' in feature_names:
                        computed['planarity'] = plan
                    if 'sphericity' in feature_names:
                        computed['sphericity'] = spher
                    print(f"  ✓ Geometric features computed: {lin.shape}")

        return computed

    @staticmethod
    def _compute_normals(
        xyz_array: np.ndarray,
        kdtree: KDTree,
        k_neighbors: int
    ) -> np.ndarray:
        """
        Compute surface normals using PCA on local neighborhoods.

        Returns:
            normals: [N, 3] array of unit normal vectors
        """
        n_points = len(xyz_array)
        normals = np.zeros((n_points, 3), dtype=np.float32)

        for i in range(n_points):
            # Query k nearest neighbors
            distances, indices = kdtree.query(xyz_array[i], k=k_neighbors)
            neighbors = xyz_array[indices]

            # Center the neighborhood
            centered = neighbors - neighbors.mean(axis=0)

            # PCA to find normal (eigenvector with smallest eigenvalue)
            if len(centered) >= 3:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Normal is eigenvector corresponding to smallest eigenvalue
                normal = eigenvectors[:, 0]
                normals[i] = normal / (np.linalg.norm(normal) + 1e-10)

            # Progress
            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1:,}/{n_points:,} ({100*(i+1)/n_points:.1f}%)")

        return normals

    @staticmethod
    def _compute_curvature(
        xyz_array: np.ndarray,
        normals: np.ndarray,
        kdtree: KDTree,
        k_neighbors: int
    ) -> np.ndarray:
        """
        Estimate curvature as variation in normal direction.

        Returns:
            curvature: [N] array of curvature values
        """
        n_points = len(xyz_array)
        curvature = np.zeros(n_points, dtype=np.float32)

        print(f"  Computing curvature for {n_points:,} points...")

        for i in range(n_points):
            # Query k nearest neighbors
            distances, indices = kdtree.query(xyz_array[i], k=k_neighbors)

            # Get normals of neighbors
            neighbor_normals = normals[indices]

            # Curvature = standard deviation of normal angles
            # Higher variation = higher curvature
            dot_products = np.dot(neighbor_normals, normals[i])
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angles = np.arccos(dot_products)
            curvature[i] = np.std(angles)

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1:,}/{n_points:,} ({100*(i+1)/n_points:.1f}%)")

        return curvature

    @staticmethod
    def _compute_roughness(
        xyz_array: np.ndarray,
        normals: np.ndarray,
        kdtree: KDTree,
        k_neighbors: int
    ) -> np.ndarray:
        """
        Compute surface roughness as distance from fitted plane.

        Returns:
            roughness: [N] array of roughness values
        """
        n_points = len(xyz_array)
        roughness = np.zeros(n_points, dtype=np.float32)

        print(f"  Computing roughness for {n_points:,} points...")

        for i in range(n_points):
            # Query k nearest neighbors
            distances, indices = kdtree.query(xyz_array[i], k=k_neighbors)
            neighbors = xyz_array[indices]

            # Fit plane: normal @ (p - centroid) = 0
            centroid = neighbors.mean(axis=0)
            normal = normals[i]

            # Distance of each neighbor from plane
            distances_from_plane = np.abs(np.dot(neighbors - centroid, normal))

            # Roughness = RMS distance from plane
            roughness[i] = np.sqrt(np.mean(distances_from_plane ** 2))

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1:,}/{n_points:,} ({100*(i+1)/n_points:.1f}%)")

        return roughness

    @staticmethod
    def _compute_geometric_features(
        xyz_array: np.ndarray,
        kdtree: KDTree,
        k_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute geometric features from eigenvalue analysis.

        Based on eigenvalues λ1 >= λ2 >= λ3 of covariance matrix:
        - Linearity: (λ1 - λ2) / λ1 (1D structures like edges)
        - Planarity: (λ2 - λ3) / λ1 (2D structures like planes)
        - Sphericity: λ3 / λ1 (3D structures)

        Returns:
            linearity: [N] array
            planarity: [N] array
            sphericity: [N] array
        """
        n_points = len(xyz_array)
        linearity = np.zeros(n_points, dtype=np.float32)
        planarity = np.zeros(n_points, dtype=np.float32)
        sphericity = np.zeros(n_points, dtype=np.float32)

        print(f"  Computing geometric features for {n_points:,} points...")

        for i in range(n_points):
            # Query k nearest neighbors
            distances, indices = kdtree.query(xyz_array[i], k=k_neighbors)
            neighbors = xyz_array[indices]

            # Center the neighborhood
            centered = neighbors - neighbors.mean(axis=0)

            # Compute eigenvalues of covariance matrix
            if len(centered) >= 3:
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)  # Returns sorted ascending
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending: λ1, λ2, λ3

                # Normalize
                eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)

                lambda1, lambda2, lambda3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

                # Compute features
                if lambda1 > 1e-10:
                    linearity[i] = (lambda1 - lambda2) / lambda1
                    planarity[i] = (lambda2 - lambda3) / lambda1
                    sphericity[i] = lambda3 / lambda1

            if (i + 1) % 50000 == 0:
                print(f"  Progress: {i+1:,}/{n_points:,} ({100*(i+1)/n_points:.1f}%)")

        return linearity, planarity, sphericity

    @staticmethod
    def print_feature_info():
        """Print information about available features"""
        print("\n" + "="*70)
        print("FEATURE FACTORY - Available Features")
        print("="*70)

        features_info = {
            'normals': {
                'shape': '[N, 3]',
                'description': 'Surface normal vectors (unit length)',
                'use_case': 'Required for curvature and roughness'
            },
            'curvature': {
                'shape': '[N]',
                'description': 'Surface curvature (variation in normals)',
                'use_case': 'Detect sharp edges and curved surfaces'
            },
            'roughness': {
                'shape': '[N]',
                'description': 'Surface roughness (RMS distance from plane)',
                'use_case': 'Distinguish smooth vs rough rock surfaces'
            },
            'linearity': {
                'shape': '[N]',
                'description': '1D structure measure (edges, ridges)',
                'use_case': 'High values indicate linear features'
            },
            'planarity': {
                'shape': '[N]',
                'description': '2D structure measure (planes, surfaces)',
                'use_case': 'High values indicate planar regions'
            },
            'sphericity': {
                'shape': '[N]',
                'description': '3D structure measure (volumetric)',
                'use_case': 'High values indicate isotropic neighborhoods'
            }
        }

        for name in FeatureFactory.AVAILABLE_FEATURES:
            info = features_info[name]
            print(f"\n{name}:")
            print(f"  Shape: {info['shape']}")
            print(f"  Description: {info['description']}")
            print(f"  Use case: {info['use_case']}")

        print("\n" + "="*70)
        print("Usage:")
        print("  features = FeatureFactory.get_features(")
        print("      xyz_array=xyz,")
        print("      feature_names=['normals', 'curvature', 'planarity'],")
        print("      k_neighbors=30")
        print("  )")
        print("="*70 + "\n")


# Example usage
if __name__ == '__main__':
    # Print feature information
    FeatureFactory.print_feature_info()

    # Example: Create synthetic point cloud
    np.random.seed(42)
    n_points = 1000
    xyz = np.random.randn(n_points, 3).astype(np.float32)

    print("\n" + "="*70)
    print("EXAMPLE: Computing features")
    print("="*70)

    # First request: compute normals and planarity
    print("\nRequest 1: ['normals', 'planarity']")
    features1 = FeatureFactory.get_features(
        xyz_array=xyz,
        feature_names=['normals', 'planarity'],
        k_neighbors=20,
        cache_dir='./test_cache'
    )
    print(f"Received: {list(features1.keys())}")
    print(f"  normals: {features1['normals'].shape}")
    print(f"  planarity: {features1['planarity'].shape}")

    # Second request: reuse normals from cache, compute curvature
    print("\nRequest 2: ['normals', 'curvature']")
    features2 = FeatureFactory.get_features(
        xyz_array=xyz,
        feature_names=['normals', 'curvature'],
        k_neighbors=20,
        cache_dir='./test_cache'
    )
    print(f"Received: {list(features2.keys())}")
    print(f"  normals: {features2['normals'].shape}")
    print(f"  curvature: {features2['curvature'].shape}")

    # Third request: all features (some from cache, some new)
    print("\nRequest 3: All features")
    features3 = FeatureFactory.get_features(
        xyz_array=xyz,
        feature_names=FeatureFactory.AVAILABLE_FEATURES,
        k_neighbors=20,
        cache_dir='./test_cache'
    )
    print(f"Received: {list(features3.keys())}")
    for name, arr in features3.items():
        print(f"  {name}: {arr.shape}")
