import numpy as np
import plotly.graph_objects as go
import torch
import open3d as o3d


class RayReflection:
    """Ray reflection

    Args:
        origins: camera ray origins
        directions: original directions of camera rays
        positions: original positions of sample points
        r: n1/n2
    """

    def __init__(self, origins, directions, positions, r=None):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r  # n1/n2

    def get_reflected_directions(self, n):
        """Get new ray directions based on Fresnel Equations

        Args:
            normals: surface normals (unit vector)
        Returns:
            directions: reflected directions (unit vector)
        """
        l = self.directions  # [4096, 256, 3]
        l = l / torch.norm(l, p=2, dim=-1, keepdim=True)  # Normalize ray directions [4096, 256, 3]
        c = -torch.einsum('ijk, ijk -> ij', n, l)  # Cosine between normals and directions [4096, 256]

        # Calculate the reflected direction
        reflected_directions = l + 2 * c.unsqueeze(-1) * n
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=-1)

        return reflected_directions

    def fresnel_fn(self, normals):
        """Get new ray directions based on Fresnel Equations

        Args:
            normals: surface normals
        Returns:
            R: fraction of reflection
        """
        r = self.r  # n1/n2
        l = self.directions  # [4096, 48, 3]
        # Calculate the cosine of the angle of incidence
        cos_theta_i = -torch.einsum('ij,ij->i', l[:, 0, :], normals[:, 0, :])
        cos_theta_i = torch.clamp(cos_theta_i, -1.0, 1.0)  # Ensure values are within valid range

        sin_theta_i_squared = 1 - cos_theta_i ** 2

        # Calculate the sine of the refraction angle using Snell's law
        sin_theta_t_squared = r ** 2 * sin_theta_i_squared
        sin_theta_t_squared = torch.clamp(sin_theta_t_squared, 0.0, 1.0)  # Ensure values are within valid range
        cos_theta_t = torch.sqrt(1 - sin_theta_t_squared)

        # Add small epsilon to denominators to avoid division by zero
        epsilon = 1e-6

        # Calculate Rs and Rp using the Fresnel equations
        Rs = ((r * cos_theta_i - cos_theta_t) / (r * cos_theta_i + cos_theta_t + epsilon)) ** 2
        Rp = ((r * cos_theta_t - cos_theta_i) / (cos_theta_i + r * cos_theta_t + epsilon)) ** 2

        # Calculate the average reflectance
        R = (Rs + Rp) / 2

        # Handle any remaining NaNs
        R = torch.nan_to_num(R, nan=0.0)  # Replace NaNs with zeroes, so we don't consider reflections where there are no intersections

        return R.unsqueeze(-1)

    def update_sample_points(self, intersections, origins_new, directions_new, mask):
        """Add sample points along reflected rays
        Args:
            intersections
            directions_new
            idx
            mask
        Returns:
            mask
        """

        # Move the original sample points onto the reflected ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=-1)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]
        self.origins[mask] = origins_new[mask].clone()
        self.directions[mask] = directions_new[mask].clone()

        self.directions = torch.nn.functional.normalize(self.directions, p=2, dim=-1)

        return self.origins, self.directions, self.positions
