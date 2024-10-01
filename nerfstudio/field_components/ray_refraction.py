import os

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
import open3d as o3d


def visualization(ray_samples, idx_start, idx_end) -> None:
    positions = ray_samples.frustums.get_positions()
    # radius = 0.9
    radius = 1.0 * 0.1
    for i in range(idx_start, idx_end):
        p = positions[i, ...]
        x = p[..., 0].flatten().cpu().numpy()
        y = p[..., 1].flatten().cpu().numpy()
        z = p[..., 2].flatten().cpu().numpy()
        # Create a meshgrid of points on the sphere's surface
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        xs = radius * np.sin(phi) * np.cos(theta)
        ys = radius * np.sin(phi) * np.sin(theta)
        zs = radius * np.cos(phi)
        scatter_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))
        line_trace = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue'))
        o = ray_samples.frustums.origins[i].cpu()
        origins = go.Scatter3d(x=o[..., 0], y=o[..., 1], z=o[..., 2], mode='markers', marker=dict(size=5))
        surface_trace = go.Surface(x=xs, y=ys, z=zs, opacity=0.3)
        fig = go.Figure(data=[scatter_trace, line_trace, origins, surface_trace])
        fig.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
        # fig.show()

        file_name = f"visualization_{i}.html"
        full_path = os.path.join('logs', file_name)
        fig.write_html(full_path)

    import sys
    sys.exit(0)


class RayRefraction:
    """Ray refracting

    Args:
        origins: camera ray origins
        directions: original directions of camera rays
        positions: original positions of sample points
        r: n1/n2
    """

    def __init__(self, origins, directions, positions, r=None, radius=None):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r
        self.radius = radius

    def get_intersections_and_normals(self, condition):
        raise NotImplementedError

    # @functools.lru_cache(maxsize=128)
    def snell_fn(self, n):
        """Get new ray directions based on Snell's Law, including handling total internal reflection.

        Args:
            n: surface normals
        Returns:
            refracted directions or reflected directions in case of total internal reflection
        """

        r = self.r
        l = self.directions / torch.norm(self.directions, p=2, dim=-1,
                                         keepdim=True)  # Normalize ray directions [4096, 256, 3]
        c = -torch.einsum('ijk, ijk -> ij', n, l)  # Cosine of the angle between the surface normal and ray direction
        sqrt_term = 1 - (r ** 2) * (1 - c ** 2)
        total_internal_reflection_mask = sqrt_term < 1e-6  # Check for total internal reflection (sqrt_term <= 0)
        flag = total_internal_reflection_mask.any()  # the flag is a boolean value

        # Refracted directions for non-total-reflection cases
        refracted_directions = r * l + (r * c - torch.sqrt(torch.clamp(sqrt_term, min=0))).unsqueeze(-1) * n
        refracted_directions = torch.nn.functional.normalize(refracted_directions, dim=-1)

        # Total internal reflection case
        reflected_directions = l + 2 * c.unsqueeze(-1) * n
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=-1)

        # Return refracted directions where there's no total internal reflection, otherwise return reflected directions
        result_directions = torch.where(total_internal_reflection_mask.unsqueeze(-1), reflected_directions,
                                        refracted_directions)

        return result_directions, flag

    def update_sample_points(self, intersections, directions_new, condition, mask):
        raise NotImplementedError


class WaterBallRefraction(RayRefraction):

    def __init__(self, origins, directions, positions, r, radius):
        # super().__init__(origins, directions, positions, r, radius)
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=2, keepdim=True)
        self.positions = positions
        self.r = r
        self.radius = radius

    # @functools.lru_cache(maxsize=128)
    def get_intersections_and_normals(self, condition, ray_origins, ray_directions):
        """Get intersections and surface normals

        Args:
            condition: -
        """

        # Define the sphere
        sphere_origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.origins.device)
        sphere_radius = self.radius

        # Reshape the origins and directions tensors to (num_rays * num_samples, 3)
        ray_origins = ray_origins.reshape(-1, 3)
        ray_directions = ray_directions.reshape(-1, 3)

        # Compute the discriminant for each ray
        oc = ray_origins - sphere_origin
        a = torch.sum(ray_directions * ray_directions, dim=1)
        b = 2.0 * torch.sum(oc * ray_directions, dim=1)
        c = torch.sum(oc * oc, dim=1) - sphere_radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if condition == 'in':
            # Compute the intersections and normals for each ray
            t = (-b - torch.sqrt(discriminant)) / (2 * a)
            intersections = ray_origins + t[:, None] * ray_directions
            normals = (intersections - sphere_origin) / sphere_radius

            # Convert the intersections and normals back to their original shape
            intersections = intersections.view(*self.origins.shape)
            normals = normals.view(*self.origins.shape)
        else:
            # Compute the intersections and normals for each ray
            t = (-b + torch.sqrt(discriminant)) / (2 * a)
            intersections = ray_origins + t[:, None] * ray_directions
            normals = (intersections - sphere_origin) / sphere_radius

            # Convert the intersections and normals back to their original shape
            intersections = intersections.view(*self.origins.shape)
            normals = normals.view(*self.origins.shape)

        return intersections, normals

    # @functools.lru_cache(maxsize=128)
    def update_sample_points(self, intersections, origins_new, directions_new, condition, mask):
        """Update sample points

        Args:
            intersections: intersections of the camera ray with the surface of the object
            directions_new: refracted directions
            condition: -
            mask: -
        """
        # 1. Get the index of the first point to be updated
        if condition == 'in':
            mask = torch.logical_and((torch.norm(self.positions, dim=-1)) < self.radius, mask)
        else:
            mask = torch.logical_and((torch.norm(self.positions, dim=-1)) > self.radius, mask)

        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        masked_positions = torch.where(mask, self.positions, intersections)
        distances = torch.norm(intersections - masked_positions, dim=-1)  # Calculate Euclidean distances [4096, 48]
        first_idx = torch.argmin(torch.where(distances != 0, distances, torch.finfo(distances.dtype).max), dim=1)

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)
        # mask = torch.arange(48).to(device='cuda:0') >= first_idx
        mask = ~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=2)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]
        self.origins[mask] = origins_new[mask]
        self.directions[mask] = directions_new[mask]

        return self.origins, self.directions, self.positions, mask


class GlassCubeRefraction(RayRefraction):
    def __init__(self, origins, directions, positions, r, size=2.0):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=2, keepdim=True)
        self.positions = positions
        self.r = r
        self.size = size  # The length of the cube's edge is 2 (cube is 2x2x2)
        self.half_size = size / 2.0  # Half the size for easy boundary calculation

        # Convert the rotation angle to radians
        angle_rad = math.radians(-295.127)

        # Create the 3x3 rotation matrix for rotation around the z-axis
        self.rotation_matrix = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Compute the inverse rotation matrix (which is just the transpose for rotation matrices)
        self.inv_rotation_matrix = self.rotation_matrix.t()

    def get_intersections_and_normals(self, condition):
        """Get intersections and surface normals for a cube rotated around the z-axis"""

        # Reshape the origins and directions tensors to (num_rays * num_samples, 3)
        ray_origins = self.origins.reshape(-1, 3)
        ray_directions = self.directions.reshape(-1, 3)

        # Apply the inverse rotation matrix to transform the rays into the cube's local coordinate system
        ray_origins_rot = torch.matmul(ray_origins, self.inv_rotation_matrix)
        ray_directions_rot = torch.matmul(ray_directions, self.inv_rotation_matrix)

        # Calculate the inverse of ray directions once to save memory
        inv_directions = 1.0 / ray_directions_rot

        # Precompute constants to avoid recalculating in the loop
        t_min = torch.full_like(ray_directions_rot[:, 0], -torch.finfo(ray_directions_rot.dtype).max)
        t_max = torch.full_like(ray_directions_rot[:, 0], torch.finfo(ray_directions_rot.dtype).max)

        # Loop through each axis to calculate intersections
        for i in range(3):  # Loop through each axis
            t0 = (-self.half_size - ray_origins_rot[:, i]) * inv_directions[:, i]  # Near plane
            t1 = (self.half_size - ray_origins_rot[:, i]) * inv_directions[:, i]  # Far plane

            t_min = torch.max(t_min, torch.min(t0, t1))  # the maximum of the near plane intersections
            t_max = torch.min(t_max, torch.max(t0, t1))  # the minimum of the far plane intersections

        # Determine valid intersections based on t_min and t_max
        valid_mask = t_max >= t_min

        # Convert the condition to a boolean tensor
        use_t_min = torch.tensor(condition == 'in', dtype=torch.bool, device=ray_origins.device).expand(
            ray_origins.size(0))

        # Efficiently compute intersections
        t_intersections = torch.where(use_t_min, t_min, t_max).unsqueeze(-1)

        # Compute intersection points in the rotated space
        intersections_rot = ray_origins_rot + t_intersections * ray_directions_rot

        # Initialize normals in the rotated space
        normals_rot = torch.zeros_like(intersections_rot)

        # Determine normals by checking which face was hit in the rotated space
        hit_mask = (torch.abs(intersections_rot - self.half_size) < 1e-5) & valid_mask.unsqueeze(-1)
        normals_rot = torch.where(hit_mask, torch.sign(intersections_rot), normals_rot)

        # Rotate the intersections and normals back to the original space
        intersections = torch.matmul(intersections_rot, self.rotation_matrix)
        normals = torch.matmul(normals_rot, self.rotation_matrix)

        # Reshape the intersections and normals back to the original shape
        intersections = intersections.view(*self.origins.shape)
        normals = normals.view(*self.origins.shape)

        return intersections, normals

    def update_sample_points(self, intersections, directions_new, condition, mask):
        """Update sample points for the cube, taking into account the rotation."""

        # 1. Get the index of the first point to be updated
        if condition == 'in':
            mask = torch.logical_and((torch.abs(self.positions) < self.half_size).all(dim=-1), mask)
        else:
            mask = torch.logical_and((torch.abs(self.positions) > self.half_size).any(dim=-1), mask)

        mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        masked_positions = torch.where(mask, self.positions, intersections)
        distances = torch.norm(intersections - masked_positions, dim=-1)  # Calculate Euclidean distances [4096, 48]
        first_idx = torch.argmin(torch.where(distances != 0, distances, torch.finfo(distances.dtype).max), dim=1)

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)
        mask = ~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=2)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]

        return self.positions, mask


class MeshRefraction1(RayRefraction):

    def __init__(self, origins, directions, positions, r,):
        # super().__init__(origins, directions, positions, r)
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=-1, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r

    # @functools.lru_cache(maxsize=128)
    def get_intersections_and_normals(self, scene, origins, directions):
        """
        Get intersections and surface normals

        Args:
            scene: the scene of the 3D object
        """
        device = self.origins.device

        # Prepare rays
        rays = torch.cat((origins, directions), dim=-1).cpu().numpy()  # Prepare rays in the required format
        rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Cast rays
        results = scene.cast_rays(rays_o3d)  # Cast rays

        # Convert results to PyTorch tensors and move to the correct device
        t_hit = torch.tensor(results['t_hit'].cpu().numpy(), device=device).unsqueeze(-1)
        intersections = origins + t_hit * directions
        normals = torch.tensor(results["primitive_normals"].cpu().numpy(), device=device)

        # check if the intersection is not 'inf' and create a mask for valid intersections and normals
        mask = ~torch.isinf(t_hit).any(dim=-1)  # [4096, 256]
        intersections = torch.where(mask.unsqueeze(-1), intersections, torch.tensor(float('nan'), device=device))
        normals = torch.where(mask.unsqueeze(-1), normals, torch.tensor(float('nan'), device=device))

        return intersections, normals, mask

    def update_sample_points(self, intersections, origions_new, directions_new, condition, mask):
        """Update sample points

        Args:
            intersections: intersections of the camera ray with the surface of the object
            directions_new: refracted directions
            condition: -
            mask: -
        """
        # distances = torch.norm(intersections - self.positions, dim=-1)  # Calculate Euclidean distances [4096, 256]
        # first_idx = torch.argmin(distances, dim=-1)  # [4096]
        
        # 1. Calculate Euclidean distances [4096, 256]
        distances = torch.norm(intersections - self.positions, dim=-1)

        # Get the indices of the two smallest distances along axis -1
        top2_indices = torch.topk(distances, 2, largest=False, dim=-1).indices  # [4096, 2]
        top1_idx = top2_indices[:, 0]  # [4096]
        top2_idx = top2_indices[:, 1]  # [4096]
        first_idx = torch.max(top1_idx, top2_idx)  # get the latter index

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)  # [4096, 1]
        mask = (~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)) & mask  # [4096, 256]
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=-1)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]
        self.origins[mask] = origions_new[mask]
        self.directions[mask] = directions_new[mask]

        self.directions = torch.nn.functional.normalize(self.directions, p=2, dim=-1)

        return self.origins, self.directions, self.positions, mask


class MeshRefraction(RayRefraction):

    def __init__(self, origins, directions, positions, r, ):
        # super().__init__(origins, directions, positions, r)
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=-1, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r

    # @functools.lru_cache(maxsize=128)
    def get_intersections_and_normals(self, scene, origins, directions, indices_prev):
        """
        Get intersections and surface normals

        Args:
            scene: the scene of the 3D object
        """
        device = self.origins.device

        # Prepare rays
        rays = torch.cat((origins, directions), dim=-1).cpu().numpy()  # Prepare rays in the required format
        rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Cast rays
        results = scene.cast_rays(rays_o3d)  # Cast rays

        # Convert results to PyTorch tensors and move to the correct device
        t_hit = torch.tensor(results['t_hit'].cpu().numpy(), device=device).unsqueeze(-1)
        intersections = origins + t_hit * directions
        normals = torch.tensor(results["primitive_normals"].cpu().numpy(), device=device)

        # check if the intersection is not 'inf' and create a mask for valid intersections and normals
        mask = ~torch.isinf(t_hit).any(dim=-1)  # [4096, 256]
        intersections = torch.where(mask.unsqueeze(-1), intersections, torch.tensor(float('nan'), device=device))
        normals = torch.where(mask.unsqueeze(-1), normals, torch.tensor(float('nan'), device=device))

        # Create an indices tensor to store the indices of true values in mask
        indices = torch.nonzero(torch.all(mask, dim=-1)).squeeze(dim=-1)  # [num_of_rays]
        indices = indices_prev[indices]  # [num_of_rays], the indices of the previous True values

        return intersections, normals, mask, indices

    def update_sample_points(self, intersections, origins_new, directions_new, mask):
        """Update sample points

        Args:
            intersections: intersections of the camera ray with the surface of the object
            origins_new: refracted origins
            directions_new: refracted directions
            mask: -
        """
        # distances = torch.norm(intersections - self.positions, dim=-1)  # Calculate Euclidean distances [4096, 256]
        # first_idx = torch.argmin(distances, dim=-1)  # [4096]

        # 1. Calculate Euclidean distances [4096, 256]
        distances = torch.norm(intersections - self.positions, dim=-1)

        # Get the indices of the two smallest distances along axis -1
        top2_indices = torch.topk(distances, 2, largest=False, dim=-1).indices  # [4096, 2]
        top1_idx = top2_indices[:, 0]  # [4096]
        top2_idx = top2_indices[:, 1]  # [4096]
        first_idx = torch.max(top1_idx, top2_idx)  # get the latter index

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)  # [4096, 1]
        mask = (~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)) & mask  # [4096, 256]
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=-1)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]
        self.origins[mask] = origins_new[mask].clone()
        self.directions[mask] = directions_new[mask].clone()

        self.directions = torch.nn.functional.normalize(self.directions, p=2, dim=-1)

        return self.origins, self.directions, self.positions, mask

    def snell_fn(self, n, l):
        """Get new ray directions based on Snell's Law, including handling total internal reflection.

        Args:
            n: surface normals
        Returns:
            refracted directions or reflected directions in case of total internal reflection
        """

        r = self.r  # a tensor of shape [4096]
        l = l / torch.norm(l, p=2, dim=-1, keepdim=True)  # Normalize ray directions [4096, 256, 3]
        c = -torch.einsum('ijk, ijk -> ij', n, l)  # Cosine between normals and directions [4096, 256]

        # Adjust r's shape for broadcasting
        sqrt_term = 1 - (r[:, None] ** 2) * (1 - c ** 2)  # [4096, 256]
        total_internal_reflection_mask = sqrt_term <= 0  # [4096, 256]

        # create a [4096] mask to check if there is any total internal reflection along each ray
        tir_mask = total_internal_reflection_mask.any(dim=-1)  # [4096]

        # Refracted directions for non-total-internal-reflection cases
        refracted_directions = r[:, None, None] * l + (r[:, None] * c - torch.sqrt(torch.clamp(sqrt_term, min=0))
                                                      )[:, :, None] * n  # [4096, 256, 3]
        refracted_directions = torch.nn.functional.normalize(refracted_directions, dim=-1)

        # Total internal reflection case
        reflected_directions = l + 2 * c.unsqueeze(-1) * n
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=-1)

        # Return refracted directions where there's no total internal reflection, otherwise return reflected directions
        result_directions = torch.where(total_internal_reflection_mask.unsqueeze(-1), reflected_directions,
                                        refracted_directions)

        return result_directions, tir_mask
