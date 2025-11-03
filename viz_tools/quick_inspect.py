"""
Quick Patch Inspector
Fast tool for debugging and inspecting patches during development
"""

import numpy as np
import laspy
import os
import json

os.sys.path.insert(0,'..')

from dataset import RockJointDataset


def quick_inspect_batch(dataset, batch_idx=0, batch_size=32, output_dir='./quick_viz'):
    """
    Quickly save a batch of patches for inspection
    No matching to full cloud - just saves the raw patches

    Args:
        dataset: RockJointDataset
        batch_idx: Batch index
        batch_size: Batch size
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    print(f"\nQuick inspection: Batch {batch_idx}, patches {start_idx}-{end_idx-1}")

    # === Save all patches in batch as one combined LAS ===
    all_patches = []
    all_labels = []
    patch_markers = []

    for i, patch_idx in enumerate(range(start_idx, end_idx)):
        patch = dataset.patches[patch_idx]
        labels = dataset.labels[patch_idx]

        # Offset patch spatially so they don't overlap
        offset = np.array([i * 5.0, 0, 0])  # 5m spacing
        patch_offset = patch + offset

        all_patches.append(patch_offset)
        all_labels.append(labels)
        patch_markers.append(np.full(len(patch), i))

    # Combine
    combined_xyz = np.vstack(all_patches)
    combined_labels = np.concatenate(all_labels)
    combined_markers = np.concatenate(patch_markers)

    # Color by labels
    colors = np.ones((len(combined_xyz), 3), dtype=np.uint16) * 32768
    colors[combined_labels == 0] = np.array([0, 255, 0]) * 257  # Green
    colors[combined_labels == 1] = np.array([255, 0, 0]) * 257  # Red

    # Save combined view
    output_path = os.path.join(output_dir, f'quick_batch{batch_idx}_combined.las')
    save_las(combined_xyz, colors, output_path, {
        'patch_id': combined_markers.astype(np.uint8),
        'label': (combined_labels + 1).astype(np.uint8)
    })

    print(f"✓ Saved: {output_path}")
    print(f"  Contains {end_idx - start_idx} patches side-by-side")
    print(f"  🟢 Green = Class 0, 🔴 Red = Class 1, ⚪ Gray = Unlabeled")

    return output_path


def quick_inspect_interesting_patches(dataset, n_patches=10, output_dir='./quick_viz'):
    """
    Inspect patches with interesting properties:
    - Mixed labels (low coherence)
    - High coherence
    - Many unlabeled points

    Args:
        dataset: RockJointDataset
        n_patches: Number of patches per category
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nAnalyzing patches to find interesting ones...")

    # Compute metrics for all patches
    coherences = []
    unlabeled_ratios = []

    for patch_idx in range(len(dataset)):
        labels = dataset.labels[patch_idx]

        # Coherence
        labeled_mask = labels != -1
        n_labeled = np.sum(labeled_mask)

        if n_labeled > 0:
            labeled_vals = labels[labeled_mask]
            majority = np.bincount(labeled_vals).argmax()
            coherence = np.sum(labeled_vals == majority) / len(labeled_vals)
            unlabeled_ratio = 1 - (n_labeled / len(labels))
        else:
            coherence = 0
            unlabeled_ratio = 1.0

        coherences.append(coherence)
        unlabeled_ratios.append(unlabeled_ratio)

    coherences = np.array(coherences)
    unlabeled_ratios = np.array(unlabeled_ratios)

    # Find interesting patches
    # 1. Low coherence (mixed labels)
    low_coherence_indices = np.argsort(coherences)[:n_patches]

    # 2. High coherence (pure patches)
    high_coherence_indices = np.argsort(coherences)[-n_patches:]

    # 3. Many unlabeled
    high_unlabeled_indices = np.argsort(unlabeled_ratios)[-n_patches:]

    categories = {
        'low_coherence': low_coherence_indices,
        'high_coherence': high_coherence_indices,
        'high_unlabeled': high_unlabeled_indices
    }

    for category, indices in categories.items():
        print(f"\n{category.upper()}: Saving {len(indices)} patches...")

        all_patches = []
        all_labels = []
        patch_markers = []

        for i, patch_idx in enumerate(indices):
            patch = dataset.patches[patch_idx]
            labels = dataset.labels[patch_idx]

            # Offset
            offset = np.array([i * 5.0, 0, 0])
            patch_offset = patch + offset

            all_patches.append(patch_offset)
            all_labels.append(labels)
            patch_markers.append(np.full(len(patch), patch_idx))

        combined_xyz = np.vstack(all_patches)
        combined_labels = np.concatenate(all_labels)
        combined_markers = np.concatenate(patch_markers)

        # Color by labels
        colors = np.ones((len(combined_xyz), 3), dtype=np.uint16) * 32768
        colors[combined_labels == 0] = np.array([0, 255, 0]) * 257
        colors[combined_labels == 1] = np.array([255, 0, 0]) * 257

        output_path = os.path.join(output_dir, f'interesting_{category}.las')
        save_las(combined_xyz, colors, output_path, {
            'patch_idx': combined_markers.astype(np.int32),
            'label': (combined_labels + 1).astype(np.uint8)
        })

        print(f"  ✓ Saved: {output_path}")


def quick_stats(dataset):
    """Print quick statistics about the dataset"""
    print("\n" + "="*70)
    print("DATASET QUICK STATS")
    print("="*70)

    n_patches = len(dataset)
    patch_size = dataset.patch_size

    print(f"\nBasic Info:")
    print(f"  Total patches: {n_patches}")
    print(f"  Points per patch: {patch_size}")
    print(f"  Total points: {n_patches * patch_size:,}")

    # Label distribution
    all_labels = dataset.labels.flatten()
    n_total = len(all_labels)
    n_class_0 = np.sum(all_labels == 0)
    n_class_1 = np.sum(all_labels == 1)
    n_unlabeled = np.sum(all_labels == -1)

    print(f"\nLabel Distribution (all patches):")
    print(f"  Class 0: {n_class_0:,} ({100*n_class_0/n_total:.1f}%)")
    print(f"  Class 1: {n_class_1:,} ({100*n_class_1/n_total:.1f}%)")
    print(f"  Unlabeled: {n_unlabeled:,} ({100*n_unlabeled/n_total:.1f}%)")

    # Per-patch statistics
    coherences = []
    labeled_ratios = []
    patch_classes = []

    for patch_idx in range(n_patches):
        labels = dataset.labels[patch_idx]

        labeled_mask = labels != -1
        n_labeled = np.sum(labeled_mask)

        if n_labeled > 0:
            labeled_vals = labels[labeled_mask]
            majority = np.bincount(labeled_vals).argmax()
            coherence = np.sum(labeled_vals == majority) / len(labeled_vals)
            labeled_ratio = n_labeled / len(labels)

            coherences.append(coherence)
            labeled_ratios.append(labeled_ratio)
            patch_classes.append(majority)

    coherences = np.array(coherences)
    labeled_ratios = np.array(labeled_ratios)
    patch_classes = np.array(patch_classes)

    print(f"\nPatch Quality:")
    print(f"  Average coherence: {np.mean(coherences):.2%}")
    print(f"  Average labeled ratio: {np.mean(labeled_ratios):.2%}")
    print(f"  Patches with >90% coherence: {np.sum(coherences > 0.9)} ({100*np.sum(coherences > 0.9)/len(coherences):.1f}%)")
    print(f"  Patches with >80% labeled: {np.sum(labeled_ratios > 0.8)} ({100*np.sum(labeled_ratios > 0.8)/len(labeled_ratios):.1f}%)")

    print(f"\nPatch Distribution by Majority Class:")
    print(f"  Majority Class 0: {np.sum(patch_classes == 0)} patches")
    print(f"  Majority Class 1: {np.sum(patch_classes == 1)} patches")

    # Coherence histogram
    print(f"\nCoherence Histogram:")
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    hist, _ = np.histogram(coherences, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} patches")

    print("\n" + "="*70)


def save_las(xyz, colors, path, extra_fields=None):
    """Helper to save LAS file"""
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


def main():
    """Quick inspection workflow"""
    print("\n" + "="*70)
    print("QUICK PATCH INSPECTOR")
    print("="*70)

    # Load dataset
    print("\nLoading data...")
    data = np.load('../data/preprocessed/preprocessed_data.npz')
    xyz_array = data['xyz_array']
    label_array = data['label_array']
    train_mask = data['train_mask']

    import pickle
    with open('../data/preprocessed/polygons_dict.pkl', 'rb') as f:
        polygons_data = pickle.load(f)
        polygons_dict_train = polygons_data['train']

    print("Creating dataset...")
    train_dataset = RockJointDataset(
        xyz_array=xyz_array[train_mask],
        label_array=label_array[train_mask],
        polygons_dict=polygons_dict_train,
        patch_size=1024,
        normalize_mode='center',
        augment=False
    )

    # Print stats
    quick_stats(train_dataset)

    # Quick visualizations
    print("\n" + "="*70)
    print("GENERATING QUICK VISUALIZATIONS")
    print("="*70)

    # Batch inspection
    quick_inspect_batch(train_dataset, batch_idx=0, batch_size=16)

    # Interesting patches
    quick_inspect_interesting_patches(train_dataset, n_patches=10)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nFiles saved to: ./quick_viz/")
    print("\nQuick inspection files show patches side-by-side:")
    print("  - quick_batch*: Regular batch")
    print("  - interesting_*: Patches by category")
    print("\nOpen in CloudCompare for fast inspection!")


if __name__ == '__main__':
    main()
