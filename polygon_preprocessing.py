import ezdxf
import laspy
from tqdm import tqdm_notebook
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import KDTree

import pandas as pd


# %%

def lines_to_polygons(dxf_doc):
    
    """Convert connected LINE entities into closed polygons"""
    
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

# %%
import numpy as np
from scipy.spatial import ConvexHull

def point_in_polygon_3d(point, polygon, plane_tolerance=0.1):
    """
    Check if a 3D point is inside a 3D polygon using projection method.
    Projects both point and polygon onto the polygon's plane.
    
    plane_tolerance: distance from plane in meters (default 0.06 = 6cm)
    """
    poly_array = np.array(polygon)
    
    # Compute polygon normal via cross product
    v1 = poly_array[1] - poly_array[0]
    v2 = poly_array[2] - poly_array[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Check if point is on the polygon's plane (within tolerance)
    point_to_poly = point - poly_array[0]
    dist_to_plane = abs(np.dot(point_to_poly, normal))
    
    if dist_to_plane > plane_tolerance:
        return False
    
    # Project polygon and point onto 2D plane
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    
    # Project polygon vertices
    poly_2d = np.array([[np.dot(p - poly_array[0], u), 
                         np.dot(p - poly_array[0], v)] for p in poly_array])
    
    # Project point
    point_2d = np.array([np.dot(point - poly_array[0], u), 
                         np.dot(point - poly_array[0], v)])
    
    # Ray casting algorithm for point in polygon
    x, y = point_2d
    n = len(poly_2d)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = poly_2d[i]
        xj, yj = poly_2d[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def label_points_in_polygons(xyz_array, label_array, polygons, label_id, plane_tolerance=0.1):
    """Label points within 3D polygons."""
    labeled_count = 0
    
    for poly_idx, polygon in enumerate(polygons):
        if poly_idx % 10 == 0:
            print(f"Processing polygon {poly_idx}/{len(polygons)}")
        
        poly_array = np.array(polygon)
        
        # Bounding box with tolerance
        min_bounds = poly_array.min(axis=0) - plane_tolerance
        max_bounds = poly_array.max(axis=0) + plane_tolerance
        
        mask = np.all((xyz_array >= min_bounds) & (xyz_array <= max_bounds), axis=1)
        candidate_indices = np.where(mask)[0]
        
        for idx in candidate_indices:
            if point_in_polygon_3d(xyz_array[idx], polygon, plane_tolerance):
                label_array[idx] = label_id
                labeled_count += 1
        
        if poly_idx % 10 == 0:
            print(f"  Total labeled so far: {labeled_count}")
    
    print(f"Total points labeled: {labeled_count}")
    return label_array

# %%
def label_points_in_polygon_2d_projection(xyz_array, label_array, polygon_array, label_id,
                                          projection_axis='x', tolerance=1.0):
    """
    Project to 2D (XY, XZ, or YZ plane) and use simple 2D polygon test.
    Works when polygon was drawn from a consistent viewpoint.
    """

    for polygon in tqdm_notebook(polygon_array):
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
        
        labeled_count = 0
        for idx in candidate_indices:
            if ray_casting_2d(points_2d[idx], poly_2d):
                label_array[idx] = label_id
                labeled_count += 1
    
    return label_array

def mask_points_polygon_2d(xyz_array, polygon_array, label_id,
                                          projection_axis='x', tolerance=1.0):
    """
    Project to 2D (XY, XZ, or YZ plane) and use simple 2D polygon test.
    Works when polygon was drawn from a consistent viewpoint.
    """
    polygon_mask = np.zeros_like(xyz_array[:,0])

    for polygon in tqdm_notebook(polygon_array):
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
                polygon_mask[idx] = label_id
    
    return polygon_mask



def ray_casting_2d(point, polygon):
    """Standard ray casting algorithm for 2D point in polygon"""
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