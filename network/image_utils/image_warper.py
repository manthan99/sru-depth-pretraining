# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthImageWarper(nn.Module):
    def __init__(self, K_s, K_r, H_s, W_s, H_r, W_r):
        """
        Initialize the DepthImageWarper module.

        Parameters:
        - K_s: Torch tensor of shape (3, 3), intrinsic matrix of the simulated camera.
        - K_r: Torch tensor of shape (3, 3), intrinsic matrix of the real camera.
        - H_s: int, height of the simulated depth image.
        - W_s: int, width of the simulated depth image.
        - H_r: int, height of the real camera image.
        - W_r: int, width of the real camera image.
        """
        super(DepthImageWarper, self).__init__()

        # Store image dimensions
        self.H_s = H_s
        self.W_s = W_s
        self.H_r = H_r
        self.W_r = W_r

        # Register intrinsic parameters as buffers
        self.register_buffer('K_s', K_s)
        self.register_buffer('K_r', K_r)

        # Precompute the grid
        self.register_buffer('grid', self._precompute_grid())

    def _precompute_grid(self):
        """
        Precompute the sampling grid based on intrinsic parameters.
        Returns:
        - grid: Torch tensor of shape (1, H_r, W_r, 2)
        """
        # Extract intrinsic parameters
        fx_s, fy_s = self.K_s[0, 0], self.K_s[1, 1]
        cx_s, cy_s = self.K_s[0, 2], self.K_s[1, 2]

        fx_r, fy_r = self.K_r[0, 0], self.K_r[1, 1]
        cx_r, cy_r = self.K_r[0, 2], self.K_r[1, 2]

        # Create a mesh grid for the real camera image
        u_r = torch.arange(self.W_r, device=self.K_s.device)
        v_r = torch.arange(self.H_r, device=self.K_s.device)
        u_r_grid, v_r_grid = torch.meshgrid(u_r, v_r, indexing='xy')  # Shape: (W_r, H_r)

        # Compute normalized image coordinates in the real camera
        x_r = (u_r_grid - cx_r) / fx_r  # Shape: (W_r, H_r)
        y_r = (v_r_grid - cy_r) / fy_r

        # Map to simulated camera pixel coordinates
        u_s = fx_s * x_r + cx_s
        v_s = fy_s * y_r + cy_s

        # Normalize coordinates to [-1, 1] for grid_sample
        u_s_norm = (u_s / (self.W_s - 1)) * 2 - 1  # Shape: (W_r, H_r)
        v_s_norm = (v_s / (self.H_s - 1)) * 2 - 1

        # Stack and reshape to form a grid for grid_sample
        grid = torch.stack((u_s_norm, v_s_norm), dim=-1)  # Shape: (W_r, H_r, 2)
        grid = grid.permute(1, 0, 2)  # Shape: (H_r, W_r, 2)

        # Add batch dimension
        grid = grid.unsqueeze(0)  # Shape: (1, H_r, W_r, 2)

        return grid

    def forward(self, depth_simulated):
        """
        Warp the simulated depth image.

        Parameters:
        - depth_simulated: Torch tensor of shape (B, 1, H_s, W_s), batch of simulated depth images.

        Returns:
        - depth_real: Torch tensor of shape (B, 1, H_r, W_r), batch of warped depth images.
        """
        # Ensure depth_simulated is float32
        depth_simulated = depth_simulated.float()

        # Check if depth_simulated has the correct shape
        if depth_simulated.dim() == 3:
            # Add batch dimension if missing
            depth_simulated = depth_simulated.unsqueeze(0)
        elif depth_simulated.dim() != 4:
            raise ValueError("depth_simulated must be a 3D or 4D tensor.")

        B = depth_simulated.size(0)

        # Expand grid to match batch size
        grid = self.grid.expand(B, -1, -1, -1)  # Shape: (B, H_r, W_r, 2)

        # Move grid to the same device as depth_simulated
        grid = grid.to(depth_simulated.device)

        # Perform grid sampling
        depth_real = F.grid_sample(depth_simulated, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

        return depth_real
