"""
PointNet++ Model for Binary Classification of Rock Joints
Implements Set Abstraction and Feature Propagation layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    Input:
        xyz: pointcloud data, [B, N, 3] (MUST be exactly 3 channels - XYZ only)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    assert C == 3, f"FPS expects exactly 3 channels (XYZ), got {C}"

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Layer: samples points, groups neighbors, and extracts features
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(xyz, points)

        # new_xyz: [B, npoint, C]
        # new_points: [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Take the maximum along the 2nd dim
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, C, npoint]
        return new_xyz, new_points

    def sample_and_group(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, C]
            new_points: sampled points data, [B, npoint, nsample, C+D]
        """
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, xyz.shape[2])

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm

        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
        """
        device = xyz.device
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C).to(device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    Feature Propagation Layer: interpolates features from coarse to fine resolution
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # [B, D, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2Classification(nn.Module):
    """
    PointNet++ for binary classification of rock joints
    """
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()
        self.input_channels = input_channels

        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=input_channels, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, N, C]
        Return:
            x: class scores, [B, num_classes]
        """
        B, N, C = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # [B, C, N]

        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class PointNet2Segmentation(nn.Module):
    """
    PointNet++ for per-point segmentation (for inference on full point clouds)
    """
    def __init__(self, num_classes=2, input_channels=3):
      super().__init__()
      self.input_channels = input_channels

      # Set Abstraction layers - ADJUSTED FOR 1024 POINTS
      # First layer: in_channel = 3 (normalized XYZ) + (input_channels - 3) additional features
      # If input_channels=3, then in_channel=3 (just XYZ)
      # If input_channels=6, then in_channel=3+3=6 (XYZ + RGB)
      # If input_channels=7, then in_channel=3+4=7 (XYZ + 4 features)
      self.sa1 = PointNetSetAbstraction(
          npoint=512, radius=0.2, nsample=32,
          in_channel=input_channels, mlp=[32, 32, 64], group_all=False
      )
      self.sa2 = PointNetSetAbstraction(
          npoint=128, radius=0.4, nsample=32,  # 512 -> 128, nsample=32
          in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
      )
      self.sa3 = PointNetSetAbstraction(
          npoint=32, radius=0.8, nsample=32,  # 128 -> 32, nsample=32
          in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
      )
      self.sa4 = PointNetSetAbstraction(
          npoint=8, radius=1.6, nsample=16,  # 32 -> 8, nsample=16 (can't query 256 from 32!)
          in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
      )

      # Feature Propagation layers (no changes needed)
      self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
      self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
      self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
      print(f"DEBUG INPUT CHANNELS: {input_channels}")
      self.fp1 = PointNetFeaturePropagation(in_channel=128,
      mlp=[128, 128, 128])
      print("DEBUG INPUT CHANNELS:",self.fp1)

      # Segmentation head (no changes needed)
      self.conv1 = nn.Conv1d(128, 128, 1)
      self.bn1 = nn.BatchNorm1d(128)
      self.drop1 = nn.Dropout(0.5)
      self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        Input:
            xyz: input points data, [B, N, C] where C >= 3
                 First 3 channels are XYZ coordinates
                 Remaining channels (if any) are additional features (RGB, geometric features, etc.)
        Return:
            x: per-point class scores, [B, num_classes, N]
        """
        B, N, C = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # [B, C, N]

        # Separate XYZ coordinates from additional features
        l0_xyz = xyz[:, :3, :]  # [B, 3, N] - always first 3 channels
        if C > 3:
            # Additional features (RGB, normals, curvature, etc.)
            l0_points = xyz[:, 3:, :]  # [B, C-3, N]
        else:
            l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)

        return x
