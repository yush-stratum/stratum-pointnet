"""
Utility functions for preprocessing point clouds and polygons
Integrates with your existing workflow
"""

import numpy as np
import ezdxf
from collections import defaultdict
from tqdm import tqdm


def lines_to_polygons(dxf_doc):
    """
    Convert connected LINE entities into closed polygons
    (From your existing code)
    """
    # Extract all lines
    msp = dxf_doc.modelspace()
    lines = [entity for entity in msp if entity.dxftype() == 'LINE']

    # Build adjacency graph: point -> list of (point, line) tuples
    graph = defaultdict(list)

    for line in lines:
        start = (line.dxf.start.x, line.dxf.start.y, line.dxf.start.z)
        end = (line.dxf.end.x, line.dxf.end.y, line.dxf.end.z)

        graph[start].append((end, line))
        graph[end].append((start, line))

    # Find all closed polygons
    polygons = []
    used_lines = set()

    for start_point in graph.keys():
        for next_point, first_line in graph[start_point]:
            if id(first_line) in used_lines:
                continue

            # Try to trace a closed path
            path = [start_point, next_point]
            visited_lines = {id(first_line)}
            current = next_point

            while current != start_point:
                found_next = False

                for neighbor, line in graph[current]:
                    if id(line) not in visited_lines:
                        if neighbor == path[-2]:  # Don't backtrack
                            continue

                        path.append(neighbor)
                        visited_lines.add(id(line))
                        current = neighbor
                        found_next = True
                        break

                if not found_next:
                    break

            # Check if we closed the loop
            if current == start_point and len(path) > 3:
                polygons.append(path[:-1])  # Remove duplicate end point
                used_lines.update(visited_lines)

    return polygons


def ray_casting_2d(point, polygon):
    """
    Standard ray casting algorithm for 2D point in polygon
    (From your existing code)
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def label_points_in_polygon_2d_projection(xyz_array, label_array, polygon_array, label_id,
                                          projection_axis='x', tolerance=1.0):
    """
    Project to 2D (XY, XZ, or YZ plane) and use simple 2D polygon test.
    (From your existing code)
    """
    for polygon in tqdm(polygon_array, desc=f"Labeling class {label_id}"):
        poly_array = np.array(polygon)

        # Choose projection: drop one axis
        if projection_axis == 'z':
            poly_2d = poly_array[:, :2]  # Keep X, Y
            points_2d = xyz_array[:, :2]
            axis_idx = 2
        elif projection_axis == 'y':
            poly_2d = poly_array[:, [0, 2]]  # Keep X, Z
            points_2d = xyz_array[:, [0, 2]]
            axis_idx = 1
        else:  # 'x'
            poly_2d = poly_array[:, 1:]  # Keep Y, Z
            points_2d = xyz_array[:, 1:]
            axis_idx = 0

        # Bounding box in 2D
        min_2d = poly_2d.min(axis=0) - tolerance
        max_2d = poly_2d.max(axis=0) + tolerance

        # Also check depth tolerance
        min_depth = poly_array[:, axis_idx].min() - tolerance
        max_depth = poly_array[:, axis_idx].max() + tolerance

        # Filter candidates
        mask_2d = np.all((points_2d >= min_2d) & (points_2d <= max_2d), axis=1)
        mask_depth = (xyz_array[:, axis_idx] >= min_depth) & (xyz_array[:, axis_idx] <= max_depth)
        mask = mask_2d & mask_depth

        candidate_indices = np.where(mask)[0]

        for idx in candidate_indices:
            if ray_casting_2d(points_2d[idx], poly_2d):
                label_array[idx] = label_id

    return label_array


def create_train_test_split(xyz_array, point_a, point_b, visualize=False, max_points=100000):
    """
    Create train/test split using a line defined by two points.
    Optionally visualize train (blue) vs test (red) with 100k sampled points each.

    Args:
        xyz_array: [N, 3] point coordinates
        label_array: [N] labels (not used for split, kept for compatibility)
        point_a: [3] first point defining the split line
        point_b: [3] second point defining the split line
        visualize: bool, whether to generate and save HTML plot
        max_points: int, max points to sample per split for visualization

    Returns:
        train_mask: boolean mask for training points
        test_mask: boolean mask for test points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)

    # Vectorized 2D cross product to determine side of line (in XY plane)
    full_ab = point_b[:2] - point_a[:2]
    full_ap = xyz_array[:, :2] - point_a[:2]
    full_cross_product = full_ap[:, 0] * full_ab[1] - full_ap[:, 1] * full_ab[0]

    print(f"Train is right of the line from {point_a} to {point_b}")
    full_train_mask = full_cross_product <= 0
    full_test_mask = full_cross_product > 0

    print(f"Train mask sum: {full_train_mask.sum()}")
    print(f"Test mask sum: {full_test_mask.sum()}")

    train_mask = full_train_mask
    test_mask = full_test_mask

    # === VISUALIZATION ===
    if visualize:
        import plotly.graph_objects as go
        fig = go.Figure()

        # Sample up to max_points from each set
        train_points = xyz_array[train_mask]
        test_points = xyz_array[test_mask]

        def sample_and_plot(points, name, color):
            if len(points) == 0:
                print(f"No points in {name}")
                return None
            idx = np.random.choice(len(points), size=min(max_points, len(points)), replace=False)
            sampled = points[idx]
            # Shuffle for unbiased rendering order
            np.random.shuffle(sampled)
            return go.Scatter3d(
                x=sampled[:, 0],
                y=sampled[:, 1],
                z=sampled[:, 2],
                mode='markers',
                name=name,
                marker=dict(size=2, opacity=0.7, color=color)
            )

        t1 = sample_and_plot(train_points, 'Train', 'blue')
        t2 = sample_and_plot(test_points, 'Test', 'red')

        traces = [t for t in [t1, t2] if t is not None]
        if traces:
            fig.add_traces(traces)

            # Optional: Add split line (projected in 3D)
            line_z = np.mean(xyz_array[:, 2])  # or use min/max, or parameterize
            fig.add_trace(go.Scatter3d(
                x=[point_a[0], point_b[0]],
                y=[point_a[1], point_b[1]],
                z=[line_z, line_z],
                mode='lines',
                name='Split Line',
                line=dict(color='yellow', width=8)
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title='Train (Blue) vs Test (Red) Split',
                showlegend=True
            )
            fig.write_html('train_v_test_split.html')
            print("Visualization saved to 'train_v_test_split.html'")
        else:
            print("No points to visualize.")

    return train_mask, test_mask

def filter_polygons_by_mask(polygons, xyz_array, mask):
    """
    Filter polygons to only include those with centroids in masked region

    Args:
        polygons: List of polygons
        xyz_array: [N, 3] point coordinates
        mask: boolean mask

    Returns:
        filtered_polygons: List of polygons in masked region
    """
    from scipy.spatial import KDTree

    # Build KDTree for masked points
    masked_points = xyz_array[mask]
    tree = KDTree(masked_points)

    filtered_polygons = []

    for polygon in polygons:
        poly_array = np.array(polygon)
        centroid = poly_array.mean(axis=0)

        # Check if centroid is near any masked point
        dist, _ = tree.query(centroid)

        # If centroid is within 1 meter of any masked point, include it
        if dist < 1.0:
            filtered_polygons.append(polygon)

    return filtered_polygons


def prepare_data_for_training(las_file, no_joints_dxf, joints_dxf,
                              train_point_a, train_point_b, use_rgb=False):
    """
    Complete data preparation pipeline

    Args:
        las_file: Path to LAS file
        no_joints_dxf: Path to no-joints DXF file
        joints_dxf: Path to joints DXF file
        train_point_a: [3] first point for train/test split
        train_point_b: [3] second point for train/test split
        use_rgb: Whether to extract RGB features from LAS file

    Returns:
        xyz_array: Full point cloud
        rgb_array: RGB colors (if use_rgb=True) or None
        label_array: Labels for all points
        train_mask: Training mask
        test_mask: Test mask
        polygons_nojoint: No-joint polygons
        polygons_joints: Joint polygons
    """
    import laspy

    # Load point cloud
    print(f"Loading point cloud from {las_file}")
    las = laspy.read(las_file)
    xyz_array = np.vstack([las.x, las.y, las.z]).transpose()
    print(f"Loaded {len(xyz_array):,} points")


    # Create train/test split
    print(f"\nCreating train/test split...")
    train_mask, test_mask = create_train_test_split(
        xyz_array, train_point_a, train_point_b
    )

    print(f"Training points: {np.sum(train_mask):,}")
    print(f"Test points: {np.sum(test_mask):,}")


    # Extract RGB if requested
    rgb_array = None
    if use_rgb:
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            # LAS RGB values are typically 16-bit (0-65535), normalize to 0-1
            rgb_array = np.vstack([las.red, las.green, las.blue]).transpose().astype(np.float32)
            rgb_array = rgb_array / 65535.0  # Normalize to [0, 1]
            print(f"Extracted RGB colors (normalized to [0, 1])")
        else:
            print("WARNING: RGB requested but not available in LAS file. Proceeding without RGB.")
            use_rgb = False

    # Initialize labels
    # -1: Unlabeled (default)
    # 0: No Joints
    # 1: Joints
    label_array = np.full(len(xyz_array), -1)  # Default to unlabeled

    # Load and extract polygons
    print(f"\nLoading polygon annotations...")
    no_joints_doc = ezdxf.readfile(no_joints_dxf)
    joints_doc = ezdxf.readfile(joints_dxf)

    polygons_nojoint = lines_to_polygons(no_joints_doc)
    polygons_joints = lines_to_polygons(joints_doc)

    print(f"Found {len(polygons_nojoint)} no-joint polygons")
    print(f"Found {len(polygons_joints)} joint polygons")

    # Label points
    print(f"\nLabeling points within polygons...")
    print(f"  -1: Unlabeled (default)")
    print(f"   0: No Joints")
    print(f"   1: Joints")

    label_array = label_points_in_polygon_2d_projection(
        xyz_array, label_array, polygons_nojoint, label_id=0  # No joints = class 0
    )
    label_array = label_points_in_polygon_2d_projection(
        xyz_array, label_array, polygons_joints, label_id=1  # Joints = class 1
    )

    # Print class distribution
    print(f"\nClass distribution:")
    print(f"  -1 (Unlabeled):  {np.sum(label_array == -1):,} ({100*np.sum(label_array == -1)/len(xyz_array):.2f}%)")
    print(f"   0 (No Joints):  {np.sum(label_array == 0):,} ({100*np.sum(label_array == 0)/len(xyz_array):.2f}%)")
    print(f"   1 (Joints):     {np.sum(label_array == 1):,} ({100*np.sum(label_array == 1)/len(xyz_array):.2f}%)")


    # Filter polygons by split
    print(f"\nFiltering polygons by train/test split...")
    train_polygons_nojoint = filter_polygons_by_mask(polygons_nojoint, xyz_array, train_mask)
    train_polygons_joints = filter_polygons_by_mask(polygons_joints, xyz_array, train_mask)
    test_polygons_nojoint = filter_polygons_by_mask(polygons_nojoint, xyz_array, test_mask)
    test_polygons_joints = filter_polygons_by_mask(polygons_joints, xyz_array, test_mask)

    print(f"Train: {len(train_polygons_nojoint)} no-joint, {len(train_polygons_joints)} joint polygons")
    print(f"Test: {len(test_polygons_nojoint)} no-joint, {len(test_polygons_joints)} joint polygons")

    # Create polygon dictionaries - NEW 3-CLASS SCHEME
    # Class 0: Background (no polygons, it's the unlabeled region)
    # Class 1: No Joints
    # Class 2: Joints
    polygons_dict_train = {
        'label_0': [],  # Background has no polygons
        'label_1': train_polygons_nojoint,  # No joints
        'label_2': train_polygons_joints    # Joints
    }

    print("Number Test Polygons Joint",len(test_polygons_joints))
    print("Number Test Polygons NoJoint",len(test_polygons_nojoint))
    polygons_dict_test = {
        'label_0': [],  # Background has no polygons
        'label_1': test_polygons_nojoint,  # No joints
        'label_2': test_polygons_joints    # Joints
    }

    return (xyz_array, rgb_array, label_array, train_mask, test_mask,
            polygons_dict_train, polygons_dict_test)


# Example usage
if __name__ == '__main__':
    # Example: Prepare data
    las_file = '/path/to/pointcloud.las'
    no_joints_dxf = '/path/to/no_joints.dxf'
    joints_dxf = '/path/to/joints.dxf'

    # Define train/test split line
    point_a = np.array([464275, 9175697, 2546])
    point_b = np.array([464294, 9175699, 2552])

    # Prepare data
    (xyz_array, rgb_array, label_array, train_mask, test_mask,
     polygons_dict_train, polygons_dict_test) = prepare_data_for_training(
        las_file, no_joints_dxf, joints_dxf, point_a, point_b, use_rgb=True
    )

    # Save for later use
    save_dict = {
        'xyz_array': xyz_array,
        'label_array': label_array,
        'train_mask': train_mask,
        'test_mask': test_mask
    }
    if rgb_array is not None:
        save_dict['rgb_array'] = rgb_array

    np.savez('preprocessed_data.npz', **save_dict)

    print("\nData preparation complete!")
    print("Saved to: preprocessed_data.npz")
