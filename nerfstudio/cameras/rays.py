# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some ray datastructures.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload, List, Any

import torch
import trimesh
import open3d as o3d
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from nerfstudio.field_components.ray_refraction import WaterBallRefraction, MeshRefraction, GlassCubeRefraction
from nerfstudio.field_components.ray_reflection import RayReflection
from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[str, torch.device]
# mesh = trimesh.load_mesh('/home/projects/u7535192/projects/nerfstudio_translucent/nerfstudio/cameras/ball.ply')
# mesh_glass = trimesh.load_mesh('/home/projects/RefRef/mesh_files/simple_shapes/cube.ply')
mesh_glass = trimesh.load_mesh('/home/projects/RefRef/mesh_files/household_items/candle_holder_glass.ply')
# mesh_glass = trimesh.load_mesh('/home/projects/RefRef/mesh_files/complex_shapes/generic_sculpture.ply')
# mesh = trimesh.load_mesh('/home/projects/transdataset/medium/cat.ply')
# mesh_glass = trimesh.load_mesh('/home/projects/RefRef/mesh_files/lab_equipment/beaker_glass.ply')
# mesh_water = trimesh.load_mesh('/home/projects/RefRef/mesh_files/lab_equipment/beaker_water.ply')
IoR = 1.5

@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    pixel_area: Float[Tensor, "*bs 1"]
    """Projected area of pixel a distance 1 away from origin."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""
    intersections: Optional[Float[Tensor, "*bs 3"]] = None
    """Intersections between each ray and surfaces"""
    normals: Optional[Float[Tensor, "*bs 3"]] = None
    """normals at each intersection"""
    mask: Optional = None
    """mask"""

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_start_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "start" position of frustum.

        Returns:
            xyz positions.
        """
        return self.origins + self.directions * self.starts

    def set_offsets(self, offsets):
        """Sets offsets for this frustum for computing positions"""
        self.offsets = offsets

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates guassian approximation of conical frustum.

        Returns:
            Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        if self.offsets is not None:
            raise NotImplementedError()
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )

    def concat_frustums(self, frustums) -> "Frustums":
        self.origins = torch.cat([self.origins, frustums.origins], dim=1)
        self.directions = torch.cat([self.directions, frustums.directions], dim=1)
        self.starts = torch.cat([self.starts, frustums.starts], dim=1)
        self.ends = torch.cat([self.ends, frustums.ends], dim=1)
        self.pixel_area = torch.cat([self.pixel_area, frustums.pixel_area], dim=1)
        if self.offsets is not None:
            self.offsets = torch.cat([self.offsets, frustums.offsets], dim=1)
        if self.intersections is not None:
            self.intersections = torch.cat([self.intersections, frustums.intersections], dim=1)
        if self.normals is not None:
            self.normals = torch.cat([self.normals, frustums.normals], dim=1)
        if self.mask is not None:
            self.mask = torch.cat([self.mask, frustums.mask], dim=1)
        return Frustums(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            pixel_area=self.pixel_area,
            offsets=self.offsets,
            intersections=self.intersections,
            normals=self.normals,
            mask=self.mask,
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    camera_indices: Optional[Int[Tensor, "*bs 1"]] = None
    """Camera index."""
    deltas: Optional[Float[Tensor, "*bs 1"]] = None
    """"width" of each sample."""
    spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None
    """Start of normalized bin edges along ray [0,1], before warping is applied, ie. linear in disparity sampling."""
    spacing_to_euclidean_fn: Optional[Callable] = None
    """Function to convert bins to euclidean distance."""
    metadata: Optional[Dict[str, Shaped[Tensor, "*bs latent_dims"]]] = None
    """additional information relevant to generating ray samples"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""
    intersections: Optional[Float[Tensor, "*bs 3"]] = None

    def get_refracted_rays_ball(self) -> None:
        # 1. Get origins, directions, r1, r2
        origins = self.frustums.origins.clone()  # [4096, 256, 3]
        directions = self.frustums.directions.clone()  # [4096, 256, 3]
        positions = self.frustums.get_positions()  # [4096, 256, 3] ([num_rays_per_batch, num_samples_per_ray, 3])
        r1, r2 = 1.0 / 1.5, 1.5 / 1.0
        radius = 1.0 * 0.1

        # 2. Get normals from the geometry, calculate new directions, and update positions after the first refraction
        ray_refraction_1 = WaterBallRefraction(origins, directions, positions, r1, radius)
        intersections_1, normals_1 = ray_refraction_1.get_intersections_and_normals('in', origins, directions)  # [4096, 48, 3]
        # ray_test = MeshRefraction(origins, directions, positions, r1)
        # i1, n1, m1 = ray_test.get_intersections_and_normals(mesh)
        # # unit test: check if the two methods shapes are the same
        # nan_mask_1 = torch.isnan(intersections_1)
        # nan_mask_2 = torch.isnan(i1)
        # combined_non_nan_mask = ~nan_mask_1 & ~nan_mask_2
        # abs_diff = torch.abs(intersections_1[combined_non_nan_mask] - i1[combined_non_nan_mask])
        # num_different_values = torch.sum(abs_diff > 0.001).item()
        # num_non_nan_1 = torch.sum(~nan_mask_1).item()
        # num_non_nan_2 = torch.sum(~nan_mask_2).item()
        # print(f"Number of different values: {num_different_values}")
        # print(f"Number of non-NaN elements in intersections_1: {num_non_nan_1}")
        # print(f"Number of non-NaN elements in i1: {num_non_nan_2}")

        directions_1 = ray_refraction_1.snell_fn(normals_1)
        origins_1 = intersections_1 - directions_1 * torch.norm(origins - intersections_1, dim=-1).unsqueeze(2)
        updated_origins, updated_directions, updated_positions, mask_1 = ray_refraction_1.update_sample_points(intersections_1, origins_1, directions_1, 'in',
                                                                  torch.ones([positions.shape[0],
                                                                              positions.shape[1]],
                                                                             dtype=torch.bool,
                                                                             device=positions.device))
        # print('intersections_1:', intersections_1[2004, 0, :], intersections_1[2009, 0, :])
        # print('normals_1:', normals_1[2004, 0, :], normals_1[2009, 0, :])
        # print('directions_1:', directions_1[2004, 0, :], directions_1[2009, 0, :])
        # print('origins_1:', origins_1[2004, 0, :], origins_1[2009, 0, :])
        # print('updated_origins:', updated_origins[2004, 0, :], updated_origins[2004, 255, :])
        # print('updated_directions:', updated_directions[2004, 0, :], updated_directions[2004, 255, :])

        # 3. Get normals from the geometry, calculate new directions, and update positions after the second refraction
        ray_refraction_2 = WaterBallRefraction(updated_origins, updated_directions, updated_positions, r2, radius)
        intersections_2, normals_2 = ray_refraction_2.get_intersections_and_normals('out', intersections_1, directions_1)
        directions_2 = ray_refraction_2.snell_fn(-normals_2)
        origins_2 = intersections_2 - directions_2 * (torch.norm(origins - intersections_1, dim=-1)
                                                      + torch.norm(intersections_1 - intersections_2, dim=-1)).unsqueeze(2)
        updated_origins, updated_directions, updated_positions, mask_2 = ray_refraction_2.update_sample_points(intersections_2, origins_2, directions_2, 'out', mask_1)

        # print('intersections_2:', intersections_2[2004, 0, :], intersections_2[2009, 0, :])
        # print('normals_2:', normals_2[2004, 0, :], normals_2[2009, 0, :])
        # print('directions_2:', directions_2[2004, 0, :], directions_2[2009, 0, :])
        # print('origins_2:', origins_2[2004, 0, :], origins_2[2009, 0, :])
        # print('updated_origins:', updated_origins[2004, 0, :], updated_origins[2004, 255, :])
        # print('updated_directions:', updated_directions[2004, 0, :], updated_directions[2004, 255, :])

        self.frustums.intersections = [intersections_1, intersections_2]
        self.frustums.origins = updated_origins
        self.frustums.directions = updated_directions

        # print(self.frustums.origins[2004, :, :])
        # print(self.frustums.directions[2004, :, :])

        # print('self.frustums.origins.shape:', self.frustums.origins.shape)  # [4096, 256, 3]
        # print(self.frustums.origins[2004, 0:10, :])
        # print(self.frustums.origins[2004, 245:255, :])

        # # 4. Update ray_samples.frustums.directions
        # directions_new = directions.clone()
        # directions_new[mask_1] = directions_1[mask_1]
        # directions_new[mask_2] = directions_2[mask_2]
        #
        # # 5. Update ray_samples.frustums.origins
        # origins_new = origins.clone()
        #
        #
        # origins_new[mask_1] = origins_1[mask_1]
        # origins_new[mask_2] = origins_2[mask_2]
        #
        # self.frustums.directions = directions_new
        # self.frustums.origins = origins_new

    def solve_bg_intersection(self, origin, direction, shape='cube', radius=0.42):
        if shape == 'cube':
            # Calculate intersection t-values for each of the cube's planes
            t_min = (torch.tensor([-radius, -radius, -radius], device=origin.device) - origin) / direction
            t_max = (torch.tensor([radius, radius, radius], device=origin.device) - origin) / direction

            # For each axis, find the entry and exit points
            t_entry = torch.minimum(t_min, t_max)  # Closest intersection point along each axis
            t_exit = torch.maximum(t_min, t_max)  # Farthest intersection point along each axis

            # Find the maximum entry and minimum exit across all axes
            t_enter = torch.max(t_entry, dim=-1).values
            t_exit = torch.min(t_exit, dim=-1).values

            # Check for a valid intersection where t_enter < t_exit and t_exit > 0
            valid_mask = (t_enter < t_exit) & (t_exit > 0)

            # Choose the first positive intersection (t_enter) as the entry point to the cube
            t_positive = torch.where(t_enter > 0, t_enter, t_exit)
            return torch.where(valid_mask, t_positive, torch.tensor(float('inf'), device=direction.device))

        elif shape == 'ball':
            # Compute the coefficients a, b, c for the quadratic equation
            a = torch.sum(direction ** 2, dim=-1)  # a = dx^2 + dy^2 + dz^2
            b = 2 * torch.sum(origin * direction, dim=-1)  # b = 2 * (x0*dx + y0*dy + z0*dz)
            c = torch.sum(origin ** 2, dim=-1) - radius ** 2  # c = x0^2 + y0^2 + z0^2 - r^2
            discriminant = b ** 2 - 4 * a * c  # Compute the discriminant

            # Check for valid intersections (discriminant >= 0)
            valid_mask = discriminant >= 0

            # Calculate the two possible t values
            sqrt_discriminant = torch.sqrt(discriminant.clamp(min=0))  # Clamp to avoid NaN for negative values
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)

            t_positive = torch.where(t1 > 0, t1, t2)
            return torch.where((t_positive > 0) & valid_mask, t_positive, torch.tensor(float('inf'), device=direction.device))

    def update_far_plane(self, ray_bundle, ray_bundle_ref):
        """Update the far plane of the frustums.

        Args:
            ray_bundle: RayBundle object.
            ray_bundle_ref: RayBundle object.
        """

        # 1. Get origins, directions, r1, r2
        origins = self.frustums.origins.clone()  # [4096, 256, 3]
        directions = self.frustums.directions.clone()  # [4096, 256, 3]
        positions = self.frustums.get_positions()  # [4096, 256, 3] ([num_rays_per_batch, num_samples_per_ray, 3])
        n_air = 1.0
        n_glass = IoR
        n_water = 1.333
        # create a tensor r of shape [4096] with all elements equal to 1.0 / 1.5
        r = torch.ones(origins.shape[0], device=origins.device) * n_air / n_glass  # [4096]
        scale_factor = 0.1
        epsilon = 1e-4
        eps_far = 1e-3
        num_samples_per_ray = self.frustums.origins.shape[1]
        radius = torch.tensor(4.2 * math.sqrt(3) * 0.1, device=origins.device)

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene
        vertices_tensor = o3d.core.Tensor(mesh_glass.vertices * scale_factor,
                                          dtype=o3d.core.Dtype.Float32)  # Scale mesh.vertices!
        triangles_tensor = o3d.core.Tensor(mesh_glass.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32
        scene = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
        scene.add_triangles(vertices_tensor, triangles_tensor)  # add the triangles

        # Create some lists
        intersections_list = []
        mask_list = []
        updated_origins_list = []
        updated_directions_list = []
        indices_list = []
        t_acc_list = []
        indices = torch.arange(origins.shape[0], device=origins.device)  # a tensor of indices from 0 to 4095

        # 2. Get intersections and normals through the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions, r)
        intersections, normals, mask, indices = ray_refraction.get_intersections_and_normals(scene, origins, directions, indices)  # [4096, 256, 3]
        directions_new, tir_mask = ray_refraction.snell_fn(normals, directions)  # [4096, 256, 3]
        distance = torch.norm(origins - intersections, dim=-1)  # [4096, 256], distance from the origin to the first intersection

        # Update the far plane of the reflection frustums
        ray_reflection = RayReflection(origins, directions, positions)
        directions_reflection = ray_reflection.get_reflected_directions(normals)
        directions_reflection = directions_reflection[indices][:, 0]  # [4096, 3]
        far_new = self.solve_bg_intersection(intersections.clone()[indices][:, 0] + directions_reflection * epsilon,
                                             directions_reflection, 'cube', 0.42) + distance[indices][:, 0] + eps_far
        ray_bundle_ref.fars[indices] = far_new.unsqueeze(-1)

        origins_new = intersections - directions_new * distance.unsqueeze(-1)  # [4096, 256, 3]
        updated_origins, updated_directions, updated_positions, mask_update = ray_refraction.update_sample_points(
            intersections, origins_new, directions_new, mask)  # [4096, 256, 3], [4096, 256, 3], [4096, 256, 3], [4096, 256]
        t_acc = torch.where(mask[:, 0], distance[:, 0], 1.2)
        t_acc_list.append(t_acc)  # add the masked t_acc to the list

        intersections_list.append(intersections)
        mask_list.append(mask)
        updated_origins_list.append(updated_origins)
        updated_directions_list.append(updated_directions)
        indices_list.append(indices)

        # r = torch.where(tir_mask, r1, r2)
        r = r[mask[:, 0]]  # [4096]
        r = n_air / r
        in_out_mask = torch.ones_like(r, dtype=torch.bool,
                                      device=origins.device)  # all elements are True, True means 'in'

        # 3. Get intersections and normals through the following refractions
        i = 0
        while True:
            ray_refraction = MeshRefraction(updated_origins[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_directions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_positions[mask_list[i]].view(-1, num_samples_per_ray, 3), r)
            intersections_offset = intersections_list[i] + directions_new * epsilon
            intersections, normals, mask, indices = ray_refraction.get_intersections_and_normals(scene, intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3), indices_list[i])
            normals = torch.where(in_out_mask[:, None, None], -normals, normals)
            directions_prev = directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3).clone()
            directions_prev = directions_prev[:, 0]  # [num_of_rays, 3]
            directions_new, tir_mask = ray_refraction.snell_fn(normals, directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3))  # negative normals because the ray is inside the surface
            distance_prev = distance[mask_list[i]].reshape(-1, num_samples_per_ray)[:, 0]  # store the previous distance
            distance = distance[mask_list[i]] + torch.norm(
                intersections_list[i][mask_list[i]] - intersections.view(-1, 3), dim=-1)  # [num_of_rays, 256], the accumulated distance from the origin
            origins_new = intersections - directions_new * distance.view(-1, num_samples_per_ray).unsqueeze(-1)
            distance = distance.reshape(-1, num_samples_per_ray)  # [num_of_rays, 256]
            updated_origins, updated_directions, updated_positions, _ = ray_refraction.update_sample_points(
                intersections, origins_new, directions_new, mask)

            # intersections_last = x.unsqueeze(-1) * directions_prev[:, 0]  # [num_of_rays, 256, 3], the intersection point on the far plane
            intersections_prev = intersections_list[i][mask_list[i]]
            intersections_prev = intersections_prev.view(-1, num_samples_per_ray, 3)  # [num_of_rays, 256, 3]
            intersections_prev = intersections_prev[:, 0]  # [num_of_rays, 3]

            distance_last = self.solve_bg_intersection(intersections_prev, directions_prev, 'cube',0.42) + eps_far  # [num_of_rays], the distance from the intersection point to the far plane
            distance_last = torch.where(mask[:, 0], torch.tensor(0.0, device=origins.device), distance_last)
            t_acc = torch.where(mask[:, 0], distance[:, 0], distance_prev + distance_last)  # if mask is True, keep the accumulated distance, otherwise, add the distance to the far plane
            t_acc_list.append(t_acc)  # add the masked t_acc to the list

            r = torch.where(tir_mask, r, 1.0 / r)
            r = r[mask[:, 0]]
            in_out_mask = torch.where(tir_mask, in_out_mask, ~in_out_mask)
            in_out_mask = in_out_mask[mask[:, 0]]

            intersections_list.append(intersections)
            mask_list.append(mask)
            updated_origins_list.append(updated_origins)
            updated_directions_list.append(updated_directions)
            indices_list.append(indices)

            i += 1

            # Calculate the number of non-NaN elements in the intersections tensor
            rows_with_non_nan = ~torch.isnan(intersections).any(dim=2)  # [num_of_rays, 256]
            rows_with_non_nan = rows_with_non_nan.any(dim=1)  # [num_of_rays]
            num_non_nan_rows = rows_with_non_nan.sum().item()

            # Break the loop if no more intersections are found
            if num_non_nan_rows == 0 or i > 10:
                break

        for j in range(i):
            ray_bundle.fars[indices_list[j]] = t_acc_list[j+1].unsqueeze(-1)

    def update_far_plane1(self, ray_bundle, ray_bundle_ref):
        """Update the far plane of the frustums.

        Args:
            ray_bundle: RayBundle object.
            ray_bundle_ref: RayBundle object.
        """

        # 1. Get origins, directions, r1, r2
        origins = self.frustums.origins.clone()  # [4096, 256, 3]
        directions = self.frustums.directions.clone()  # [4096, 256, 3]
        positions = self.frustums.get_positions()  # [4096, 256, 3] ([num_rays_per_batch, num_samples_per_ray, 3])
        n_air, n_glass, n_water = 1.0, 1.5, 1.33
        # create a tensor r of shape [4096] with all elements equal to 1.0 / 1.5
        r = torch.ones(origins.shape[0], device=origins.device) * n_air / n_glass  # [4096]
        scale_factor = 0.1
        epsilon = 1e-4
        eps_far = 1e-3
        num_samples_per_ray = self.frustums.origins.shape[1]
        radius = torch.tensor(4.2 * math.sqrt(3) * 0.1, device=origins.device)

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene for glass
        vertices_glass = o3d.core.Tensor(mesh_glass.vertices * scale_factor,
                                          dtype=o3d.core.Dtype.Float32)  # Scale mesh.vertices!
        triangles_glass = o3d.core.Tensor(mesh_glass.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32
        scene_glass = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
        scene_glass.add_triangles(vertices_glass, triangles_glass)  # add the triangles

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene for water
        vertices_water = o3d.core.Tensor(mesh_water.vertices * scale_factor,
                                            dtype=o3d.core.Dtype.Float32)
        triangles_water = o3d.core.Tensor(mesh_water.faces, dtype=o3d.core.Dtype.UInt32)
        scene_water = o3d.t.geometry.RaycastingScene()
        scene_water.add_triangles(vertices_water, triangles_water)

        # Create some lists
        intersections_list = []
        mask_list = []
        updated_origins_list = []
        updated_directions_list = []
        indices_list = []
        t_acc_list = []
        indices = torch.arange(origins.shape[0], device=origins.device)  # a tensor of indices from 0 to 4095

        # 2. Get intersections and normals through the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions, r)
        intersections_glass, normals_glass, mask_glass, indices_glass = ray_refraction.get_intersections_and_normals(scene_glass, origins, directions, indices)  # [4096, 256, 3]
        intersections_water, normals_water, mask_water, indices_water = ray_refraction.get_intersections_and_normals(scene_water, origins, directions, indices)

        # remove the NaN values from the intersections mask
        glass = torch.where(mask_glass.unsqueeze(-1), intersections_glass, float('inf') * torch.ones_like(intersections_glass))
        water = torch.where(mask_water.unsqueeze(-1), intersections_water, float('inf') * torch.ones_like(intersections_water))
        mask_closer = (torch.norm(origins - glass, dim=-1) < torch.norm(origins - water, dim=-1))  # True if the glass intersection is closer, [4096, 256]

        intersections = torch.where(mask_closer.unsqueeze(-1), intersections_glass, intersections_water)
        normals = torch.where(mask_closer.unsqueeze(-1), normals_glass, normals_water)
        mask = torch.where(mask_closer, mask_glass, mask_water)  # [4096, 256]
        indices = torch.unique(torch.cat([indices_glass, indices_water], dim=0))
        # print the number of true rows in mask_closer

        ray_refraction.r = torch.where(mask_closer[:, 0], r, torch.ones_like(r) * n_air / n_water)  # update r

        directions_new, tir_mask = ray_refraction.snell_fn(normals, directions)  # [4096, 256, 3]
        distance = torch.norm(origins - intersections, dim=-1)  # [4096, 256], distance from the origin to the first intersection

        # Update the far plane of the reflection frustums
        ray_reflection = RayReflection(origins, directions, positions)
        directions_reflection = ray_reflection.get_reflected_directions(normals)
        directions_reflection = directions_reflection[indices][:, 0]  # [4096, 3]
        far_new = self.solve_bg_intersection(intersections.clone()[indices][:, 0] + directions_reflection * epsilon,
                                             directions_reflection, 'cube', 0.42) + distance[indices][:, 0] + eps_far
        ray_bundle_ref.fars[indices] = far_new.unsqueeze(-1)

        origins_new = intersections - directions_new * distance.unsqueeze(-1)  # [4096, 256, 3]
        updated_origins, updated_directions, updated_positions, mask_update = ray_refraction.update_sample_points(
            intersections, origins_new, directions_new, mask)  # [4096, 256, 3], [4096, 256, 3], [4096, 256, 3], [4096, 256]
        t_acc = torch.where(mask[:, 0], distance[:, 0], 1.2)
        t_acc_list.append(t_acc)  # add the masked t_acc to the list

        intersections_list.append(intersections)
        mask_list.append(mask)
        updated_origins_list.append(updated_origins)
        updated_directions_list.append(updated_directions)
        indices_list.append(indices)

        # r = torch.where(tir_mask, r1, r2)
        r = r[mask[:, 0]]  # [4096]
        # r = 1.0 / r  # update r for the next refraction
        mask_in = torch.ones_like(r, dtype=torch.bool, device=origins.device)  # all elements are True, True means 'in'

        # 3. Get intersections and normals through the following refractions
        i = 0
        while True:
            ray_refraction = MeshRefraction(updated_origins[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_directions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_positions[mask_list[i]].view(-1, num_samples_per_ray, 3), r)
            intersections_offset = intersections_list[i] + directions_new * epsilon
            intersections_glass, normals_glass, mask_glass, indices_glass = ray_refraction.get_intersections_and_normals(scene_glass, intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3), indices_list[i])
            intersections_water, normals_water, mask_water, indices_water = ray_refraction.get_intersections_and_normals(scene_water, intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3), indices_list[i])
            # remove the NaN values from the intersections mask
            glass = torch.where(mask_glass.unsqueeze(-1), intersections_glass, float('inf') * torch.ones_like(intersections_glass))
            water = torch.where(mask_water.unsqueeze(-1), intersections_water, float('inf') * torch.ones_like(intersections_water))
            mask_closer = (torch.norm(origins_new[mask_list[i]].view(-1, num_samples_per_ray, 3) - glass, dim=-1)
                                  < torch.norm(origins_new[mask_list[i]].view(-1, num_samples_per_ray, 3) - water, dim=-1))

            intersections = torch.where(mask_closer.unsqueeze(-1), intersections_glass, intersections_water)
            normals = torch.where(mask_closer.unsqueeze(-1), normals_glass, normals_water)
            mask = torch.where(mask_closer, mask_glass, mask_water)
            indices = torch.unique(torch.cat([indices_glass, indices_water], dim=0))

            # TODO: Update r for this refraction
            in_glass = torch.logical_and(mask_closer[:, 0], mask_in)
            in_water = torch.logical_and(~mask_closer[:, 0], mask_in)
            out_glass = torch.logical_and(mask_closer[:, 0], ~mask_in)
            out_water = torch.logical_and(~mask_closer[:, 0], ~mask_in)
            ray_refraction.r = torch.where(in_glass, n_glass/n_air, torch.ones_like(r) * n_glass / n_air)
            ray_refraction.r = torch.where(in_water, n_water/n_air, torch.ones_like(r) * n_water / n_air)
            ray_refraction.r = torch.where(out_glass, n_air/n_glass, torch.ones_like(r) * n_air / n_glass)
            ray_refraction.r = torch.where(out_water, n_air/n_water, torch.ones_like(r) * n_air / n_water)
            # # check if the two intersection points are close enough
            # mask_close = torch.norm(intersections_glass - intersections_water, dim=-1) < 1e-3
            # ray_refraction.r = torch.where(mask_closer, r, torch.ones_like(r) * n_air / n_water)  # update r

            normals = torch.where(mask_in[:, None, None], -normals, normals)
            directions_prev = directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3).clone()
            directions_prev = directions_prev[:, 0]  # [num_of_rays, 3]

            directions_new, tir_mask = ray_refraction.snell_fn(normals, directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3))  # negative normals because the ray is inside the surface
            distance_prev = distance[mask_list[i]].reshape(-1, num_samples_per_ray)[:, 0]  # store the previous distance
            distance = distance[mask_list[i]] + torch.norm(
                intersections_list[i][mask_list[i]] - intersections.view(-1, 3), dim=-1)  # [num_of_rays, 256], the accumulated distance from the origin
            origins_new = intersections - directions_new * distance.view(-1, num_samples_per_ray).unsqueeze(-1)
            distance = distance.reshape(-1, num_samples_per_ray)  # [num_of_rays, 256]
            updated_origins, updated_directions, updated_positions, _ = ray_refraction.update_sample_points(
                intersections, origins_new, directions_new, mask)

            # intersections_last = x.unsqueeze(-1) * directions_prev[:, 0]  # [num_of_rays, 256, 3], the intersection point on the far plane
            intersections_prev = intersections_list[i][mask_list[i]]
            intersections_prev = intersections_prev.view(-1, num_samples_per_ray, 3)  # [num_of_rays, 256, 3]
            intersections_prev = intersections_prev[:, 0]  # [num_of_rays, 3]

            distance_last = self.solve_bg_intersection(intersections_prev, directions_prev, 'cube',0.42) + eps_far  # [num_of_rays], the distance from the intersection point to the far plane
            distance_last = torch.where(mask[:, 0], torch.tensor(0.0, device=origins.device), distance_last)
            t_acc = torch.where(mask[:, 0], distance[:, 0], distance_prev + distance_last)  # if mask is True, keep the accumulated distance, otherwise, add the distance to the far plane
            t_acc_list.append(t_acc)  # add the masked t_acc to the list

            # r = torch.where(tir_mask, r, 1.0 / r)
            r = r[mask[:, 0]]
            mask_in = torch.where(tir_mask, mask_in, ~mask_in)
            mask_in = mask_in[mask[:, 0]]

            intersections_list.append(intersections)
            mask_list.append(mask)
            updated_origins_list.append(updated_origins)
            updated_directions_list.append(updated_directions)
            indices_list.append(indices)

            i += 1

            # Calculate the number of non-NaN elements in the intersections tensor
            rows_with_non_nan = ~torch.isnan(intersections).any(dim=2)  # [num_of_rays, 256]
            rows_with_non_nan = rows_with_non_nan.any(dim=1)  # [num_of_rays]
            num_non_nan_rows = rows_with_non_nan.sum().item()

            # Break the loop if no more intersections are found
            if num_non_nan_rows == 0 or i > 10:
                break

        for j in range(i):
            ray_bundle.fars[indices_list[j]] = t_acc_list[j+1].unsqueeze(-1)

    def get_refracted_rays1(self):
        # 1. Get origins, directions, r1, r2
        origins = self.frustums.origins.clone()  # [4096, 256, 3]
        directions = self.frustums.directions.clone()  # [4096, 256, 3]
        positions = self.frustums.get_positions()  # [4096, 256, 3] ([num_rays_per_batch, num_samples_per_ray, 3])
        n_air, n_glass, n_water = 1.0, 1.5, 1.33
        # create a tensor r of shape [4096] with all elements equal to 1.0 / 1.5
        r = torch.ones(origins.shape[0], device=origins.device) * n_air / n_glass  # [4096]
        scale_factor = 0.1
        epsilon = 1e-4
        eps_far = 1e-3
        num_samples_per_ray = self.frustums.origins.shape[1]

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene for glass
        vertices_glass = o3d.core.Tensor(mesh_glass.vertices * scale_factor,
                                         dtype=o3d.core.Dtype.Float32)  # Scale mesh.vertices!
        triangles_glass = o3d.core.Tensor(mesh_glass.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32
        scene_glass = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
        scene_glass.add_triangles(vertices_glass, triangles_glass)  # add the triangles

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene for water
        vertices_water = o3d.core.Tensor(mesh_water.vertices * scale_factor,
                                         dtype=o3d.core.Dtype.Float32)
        triangles_water = o3d.core.Tensor(mesh_water.faces, dtype=o3d.core.Dtype.UInt32)
        scene_water = o3d.t.geometry.RaycastingScene()
        scene_water.add_triangles(vertices_water, triangles_water)

        # Create some lists
        intersections_list = []
        mask_list = []
        updated_origins_list = []
        updated_directions_list = []
        indices_list = []
        indices = torch.arange(origins.shape[0], device=origins.device)  # a tensor of indices from 0 to 4095

        # 2. Get intersections and normals through the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions, r)
        intersections_glass, normals_glass, mask_glass, indices_glass = ray_refraction.get_intersections_and_normals(scene_glass, origins, directions, indices)  # [4096, 256, 3]
        intersections_water, normals_water, mask_water, indices_water = ray_refraction.get_intersections_and_normals(scene_water, origins, directions, indices)

        # remove the NaN values from the intersections mask
        glass = torch.where(mask_glass.unsqueeze(-1), intersections_glass, float('inf') * torch.ones_like(intersections_glass))
        water = torch.where(mask_water.unsqueeze(-1), intersections_water, float('inf') * torch.ones_like(intersections_water))
        mask_closer = (torch.norm(origins - glass, dim=-1) < torch.norm(origins - water, dim=-1))  # True if the glass intersection is closer, [4096, 256]

        intersections = torch.where(mask_closer.unsqueeze(-1), intersections_glass, intersections_water)
        normals = torch.where(mask_closer.unsqueeze(-1), normals_glass, normals_water)
        mask = torch.where(mask_closer, mask_glass, mask_water)  # [4096, 256]
        indices = torch.unique(torch.cat([indices_glass, indices_water], dim=0))
        ray_refraction.r = torch.where(mask_closer[:, 0], r, torch.ones_like(r) * n_air / n_water)  # update r

        normals_first = normals.clone()
        ray_reflection = RayReflection(origins, directions, positions)
        directions_reflection = ray_reflection.get_reflected_directions(normals)

        directions_new, tir_mask = ray_refraction.snell_fn(normals, directions)  # [4096, 256, 3]
        distance = torch.norm(origins - intersections, dim=-1)  # [4096, 256], distance from the origin to the first intersection
        origins_new = intersections - directions_new * distance.unsqueeze(-1)  # [4096, 256, 3]
        updated_origins, updated_directions, updated_positions, mask_update = ray_refraction.update_sample_points(
            intersections, origins_new, directions_new, mask)  # [4096, 256, 3], [4096, 256, 3], [4096, 256, 3], [4096, 256]

        intersections_list.append(intersections)
        mask_list.append(mask)
        updated_origins_list.append(updated_origins)
        updated_directions_list.append(updated_directions)
        indices_list.append(indices)

        # r = torch.where(tir_mask, r1, r2)
        r = r[mask[:, 0]]  # [4096]
        # r = 1.0 / r  # update r for the next refraction
        mask_in = torch.ones_like(r, dtype=torch.bool, device=origins.device)  # all elements are True, True means 'in'

        # 3. Get intersections and normals through the following refractions
        i = 0
        while True:
            ray_refraction = MeshRefraction(updated_origins[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_directions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_positions[mask_list[i]].view(-1, num_samples_per_ray, 3), r)
            intersections_offset = intersections_list[i] + directions_new * epsilon
            intersections_glass, normals_glass, mask_glass, indices_glass = ray_refraction.get_intersections_and_normals(scene_glass,
                                                                                                                         intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                                         directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3), indices_list[i])
            intersections_water, normals_water, mask_water, indices_water = ray_refraction.get_intersections_and_normals(scene_water,
                                                                                                                         intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                                         directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3), indices_list[i])
            # remove the NaN values from the intersections mask
            glass = torch.where(mask_glass.unsqueeze(-1), intersections_glass, float('inf') * torch.ones_like(intersections_glass))
            water = torch.where(mask_water.unsqueeze(-1), intersections_water, float('inf') * torch.ones_like(intersections_water))
            mask_closer = (torch.norm(origins_new[mask_list[i]].view(-1, num_samples_per_ray, 3) - glass, dim=-1)
                           < torch.norm(origins_new[mask_list[i]].view(-1, num_samples_per_ray, 3) - water, dim=-1))

            intersections = torch.where(mask_closer.unsqueeze(-1), intersections_glass, intersections_water)
            normals = torch.where(mask_closer.unsqueeze(-1), normals_glass, normals_water)
            mask = torch.where(mask_closer, mask_glass, mask_water)
            indices = torch.unique(torch.cat([indices_glass, indices_water], dim=0))

            # TODO: Update r for this refraction
            in_glass = torch.logical_and(mask_closer[:, 0], mask_in)
            in_water = torch.logical_and(~mask_closer[:, 0], mask_in)
            out_glass = torch.logical_and(mask_closer[:, 0], ~mask_in)
            out_water = torch.logical_and(~mask_closer[:, 0], ~mask_in)
            ray_refraction.r = torch.where(in_glass, n_glass / n_air, torch.ones_like(r) * n_glass / n_air)
            ray_refraction.r = torch.where(in_water, n_water / n_air, torch.ones_like(r) * n_water / n_air)
            ray_refraction.r = torch.where(out_glass, n_air / n_glass, torch.ones_like(r) * n_air / n_glass)
            ray_refraction.r = torch.where(out_water, n_air / n_water, torch.ones_like(r) * n_air / n_water)
            # # check if the two intersection points are close enough
            # mask_close = torch.norm(intersections_glass - intersections_water, dim=-1) < 1e-3
            # ray_refraction.r = torch.where(mask_closer, r, torch.ones_like(r) * n_air / n_water)  # update r

            normals = torch.where(mask_in[:, None, None], -normals, normals)
            directions_new, tir_mask = ray_refraction.snell_fn(normals, directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3))  # negative normals because the ray is inside the surface
            distance = distance[mask_list[i]] + torch.norm(intersections_list[i][mask_list[i]] - intersections.view(-1, 3), dim=-1)  # [num_of_rays, 256], the accumulated distance from the origin
            origins_new = intersections - directions_new * distance.view(-1, num_samples_per_ray).unsqueeze(-1)
            distance = distance.reshape(-1, num_samples_per_ray)  # [num_of_rays, 256]
            updated_origins, updated_directions, updated_positions, _ = ray_refraction.update_sample_points(
                intersections, origins_new, directions_new, mask)

            # r = torch.where(tir_mask, r, 1.0 / r)
            r = r[mask[:, 0]]
            mask_in = torch.where(tir_mask, mask_in, ~mask_in)
            mask_in = mask_in[mask[:, 0]]

            intersections_list.append(intersections)
            mask_list.append(mask)
            updated_origins_list.append(updated_origins)
            updated_directions_list.append(updated_directions)
            indices_list.append(indices)

            i += 1

            # Calculate the number of non-NaN elements in the intersections tensor
            rows_with_non_nan = ~torch.isnan(intersections).any(dim=2)  # [num_of_rays, 256]
            rows_with_non_nan = rows_with_non_nan.any(dim=1)  # [num_of_rays]
            num_non_nan_rows = rows_with_non_nan.sum().item()

            # Break the loop if no more intersections are found
            if num_non_nan_rows == 0 or i > 10:
                break

        origins_final = origins.clone()
        directions_final = directions.clone()
        intersections = intersections_list[0].unsqueeze(0).repeat(i, 1, 1, 1)

        for j in range(i-1):
            origins_final[indices_list[j]] = updated_origins_list[j+1]
            directions_final[indices_list[j]] = updated_directions_list[j+1]
            intersections[j+1][indices_list[j]] = intersections_list[j+1]

        self.frustums.origins = origins_final
        self.frustums.directions = directions_final
        self.frustums.intersections = intersections

        return intersections_list, normals_first, directions_reflection, mask_update, indices_list

    def get_refracted_rays(self):
        # 1. Get origins, directions, r1, r2
        origins = self.frustums.origins.clone()  # [4096, 256, 3]
        directions = self.frustums.directions.clone()  # [4096, 256, 3]
        positions = self.frustums.get_positions()  # [4096, 256, 3] ([num_rays_per_batch, num_samples_per_ray, 3])
        r1, r2 = 1.0 / IoR, 1.5 / IoR
        # create a tensor r of shape [4096] with all elements equal to 1/0 / 1.5
        r = torch.ones(origins.shape[0], device=origins.device) * r1  # [4096]
        scale_factor = 0.1
        epsilon = 1e-4
        num_samples_per_ray = self.frustums.origins.shape[1]

        # Convert trimesh vertices and faces to tensors and create a RaycastingScene
        vertices_tensor = o3d.core.Tensor(mesh_glass.vertices * scale_factor,
                                          dtype=o3d.core.Dtype.Float32)  # Scale mesh.vertices!
        triangles_tensor = o3d.core.Tensor(mesh_glass.faces, dtype=o3d.core.Dtype.UInt32)  # Convert to UInt32
        scene = o3d.t.geometry.RaycastingScene()  # Create a RaycastingScene
        scene.add_triangles(vertices_tensor, triangles_tensor)  # add the triangles

        # Create some lists
        intersections_list = []
        mask_list = []
        updated_origins_list = []
        updated_directions_list = []
        indices_list = []
        indices = torch.arange(origins.shape[0], device=origins.device)  # a tensor of indices from 0 to 4095

        # 2. Get intersections and normals through the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions, r)
        intersections, normals, mask, indices = ray_refraction.get_intersections_and_normals(scene, origins, directions, indices)  # [4096, 256, 3]

        normals_first = normals.clone()
        ray_reflection = RayReflection(origins, directions, positions)
        directions_reflection = ray_reflection.get_reflected_directions(normals)

        directions_new, tir_mask = ray_refraction.snell_fn(normals, directions)  # [4096, 256, 3]
        distance = torch.norm(origins - intersections, dim=-1)  # [4096, 256]
        origins_new = intersections - directions_new * distance.unsqueeze(-1)  # [4096, 256, 3]
        updated_origins, updated_directions, updated_positions, mask_update = ray_refraction.update_sample_points(
            intersections, origins_new, directions_new, mask)  # [4096, 256, 3], [4096, 256, 3], [4096, 256, 3], [4096, 256]

        intersections_list.append(intersections)
        mask_list.append(mask)
        updated_origins_list.append(updated_origins)
        updated_directions_list.append(updated_directions)
        indices_list.append(indices)

        # r = torch.where(tir_mask, r1, r2)
        r = r[mask[:, 0]]
        r = 1.0 / r
        in_out_mask = torch.ones_like(r, dtype=torch.bool, device=origins.device)  # all elements are True, True means 'in'

        # 3. Get intersections and normals through the following refractions
        i = 0
        while True:
            ray_refraction = MeshRefraction(updated_origins[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_directions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_positions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            r)
            intersections_offset = intersections_list[i] + directions_new * epsilon
            intersections, normals, mask, indices = ray_refraction.get_intersections_and_normals(scene,
                                                                                        intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                        directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 indices_list[i])
            normals = torch.where(in_out_mask[:, None, None], -normals, normals)
            directions_new, tir_mask = ray_refraction.snell_fn(normals, directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3))  # negative normals because the ray is inside the surface
            distance = distance[mask_list[i]] + torch.norm(intersections_list[i][mask_list[i]] - intersections.view(-1, 3), dim=-1)
            origins_new = intersections - directions_new * distance.view(-1, num_samples_per_ray).unsqueeze(-1)
            distance = distance.reshape(-1, num_samples_per_ray)
            updated_origins, updated_directions, updated_positions, _ = ray_refraction.update_sample_points(
                intersections, origins_new, directions_new, mask)

            r = torch.where(tir_mask, r, 1.0 / r)
            r = r[mask[:, 0]]
            in_out_mask = torch.where(tir_mask, in_out_mask, ~in_out_mask)
            in_out_mask = in_out_mask[mask[:, 0]]

            intersections_list.append(intersections)
            mask_list.append(mask)
            updated_origins_list.append(updated_origins)
            updated_directions_list.append(updated_directions)
            indices_list.append(indices)

            i += 1
            # print(f'Iteration {i} completed')

            # Calculate the number of non-NaN elements in the intersections tensor
            rows_with_non_nan = ~torch.isnan(intersections).any(dim=2)  # [4096, 256]
            rows_with_non_nan = rows_with_non_nan.any(dim=1)  # [4096]
            num_non_nan_rows = rows_with_non_nan.sum().item()

            # Break the loop if no more intersections are found
            if num_non_nan_rows == 0 or i > 10:
                break

        origins_final = origins.clone()
        directions_final = directions.clone()
        intersections = intersections_list[0].unsqueeze(0).repeat(i, 1, 1, 1)

        for j in range(i-1):
            origins_final[indices_list[j]] = updated_origins_list[j+1]
            directions_final[indices_list[j]] = updated_directions_list[j+1]
            intersections[j+1][indices_list[j]] = intersections_list[j+1]

        self.frustums.origins = origins_final
        self.frustums.directions = directions_final
        self.frustums.intersections = intersections

        return intersections_list, normals_first, directions_reflection, mask_update, indices_list

    def get_reflected_rays(self, intersections, normals, masks) -> None:
        origins = self.frustums.origins.clone()
        directions = self.frustums.directions.clone()
        positions = self.frustums.get_positions().clone()
        intersections, normals, mask = intersections.clone(), normals.clone(), masks.clone()

        # 1) Get reflective directions
        ray_reflection = RayReflection(origins, directions, positions)
        directions_new = ray_reflection.get_reflected_directions(normals)
        distance = torch.norm(origins - intersections, dim=-1)  # [4096, 256]
        origins_new = intersections - directions_new * distance.unsqueeze(-1)
        updated_origins, updated_directions = ray_reflection.update_sample_points(intersections, origins_new, directions_new, mask)
        # 2) Update ray_samples.frustums.directions
        directions_final = directions.clone()
        directions_final[mask] = updated_directions[mask]
        self.frustums.directions = directions_final

        # 3) Update ray_samples.frustums.origins
        origins_final = origins.clone()
        origins_final[mask] = updated_origins[mask]
        self.frustums.origins = origins_final

        self.frustums.intersections = intersections
        self.frustums.normals = normals
        self.frustums.mask = mask

    def get_straight_rays(self) -> None:
        # Modify the origins and directions of frustums here
        positions = self.frustums.get_positions()  # [4096, 48, 3] ([num_rays_per_batch, num_samples_per_ray, 3])

        # Modify positions based on known geometry and Snell's law
        # 1. Get origins, directions, r1, r2 directly
        origins = self.frustums.origins  # [4096, 48, 3]
        directions = self.frustums.directions  # [4096, 48, 3]
        r1, r2 = 1.0 / 1.33, 1.33 / 1.0
        # radius = 0.9
        radius = 0.9 * 0.1

        # 2. Get normals from the geometry, calculate new directions, and update positions after the first refraction
        ray_1 = WaterBallRefraction(origins, directions, positions, r1, radius)
        intersections_1, _ = ray_1.get_intersections_and_normals('in')  # [4096, 48, 3]

        # 3. Get normals from the geometry, calculate new directions, and update positions after the second refraction
        ray_2 = WaterBallRefraction(intersections_1, directions, positions, r2, radius)
        intersections_2, _ = ray_2.get_intersections_and_normals('out')

        self.frustums.intersections = [intersections_1, intersections_2]

    def get_weights(self, densities: Float[Tensor, "*batch num_samples 1"]) -> Float[Tensor, "*batch num_samples 1"]:
        """Return weights based on predicted densities

        Args:
            densities: Predicted densities for samples along ray

        Returns:
            Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance], dim=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]
        weights = torch.nan_to_num(weights)

        return weights

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[True]
    ) -> Float[Tensor, "*batch num_samples 1"]:
        ...

    @overload
    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: Literal[False] = False
    ) -> Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]]:
        ...

    @staticmethod
    def get_weights_and_transmittance_from_alphas(
            alphas: Float[Tensor, "*batch num_samples 1"], weights_only: bool = False
    ) -> Union[
        Float[Tensor, "*batch num_samples 1"],
        Tuple[Float[Tensor, "*batch num_samples 1"], Float[Tensor, "*batch num_samples 1"]],
    ]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """

        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )

        weights = alphas * transmittance[:, :-1, :]
        if weights_only:
            return weights
        return weights, transmittance

    def concat_samples(self, ray_samples) -> "RaySamples":
        """Concatenates ray samples to the current ray bundle.

        Args:
            ray_samples: RaySamples object.
        """
        self.frustums = self.frustums.concat_frustums(ray_samples.frustums)
        if self.camera_indices is not None:
            self.camera_indices = torch.cat([self.camera_indices, ray_samples.camera_indices], dim=1)
        if self.deltas is not None:
            self.deltas = torch.cat([self.deltas, ray_samples.deltas], dim=1)
        if self.spacing_starts is not None:
            self.spacing_starts = torch.cat([self.spacing_starts, ray_samples.spacing_starts], dim=1)
        if self.spacing_ends is not None:
            self.spacing_ends = torch.cat([self.spacing_ends, ray_samples.spacing_ends], dim=1)
        if self.metadata is not None:
            for key in self.metadata:
                self.metadata[key] = torch.cat([self.metadata[key], ray_samples.metadata[key]], dim=1)
        if self.times is not None:
            self.times = torch.cat([self.times, ray_samples.times], dim=1)
        return RaySamples(
            frustums=self.frustums,
            camera_indices=self.camera_indices,
            deltas=self.deltas,
            spacing_starts=self.spacing_starts,
            spacing_ends=self.spacing_ends,
            metadata=self.metadata,
            times=self.times,
        )



@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    # TODO(ethan): make sure the sizes with ... are correct
    origins: Float[Tensor, "*batch 3"]
    """Ray origins (XYZ)"""
    directions: Float[Tensor, "*batch 3"]
    """Unit ray direction vector"""
    pixel_area: Float[Tensor, "*batch 1"]
    """Projected area of pixel a distance 1 away from origin"""
    camera_indices: Optional[Int[Tensor, "*batch 1"]] = None
    """Camera indices"""
    nears: Optional[Float[Tensor, "*batch 1"]] = None
    """Distance along ray to start sampling"""
    fars: Optional[Float[Tensor, "*batch 1"]] = None
    """Rays Distance along ray to stop sampling"""
    metadata: Dict[str, Shaped[Tensor, "num_rays latent_dims"]] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation, will mimic shape of rays"""
    times: Optional[Float[Tensor, "*batch 1"]] = None
    """Times at which rays are sampled"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays: Number of rays in output RayBundle

        Returns:
            RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
            self,
            bin_starts: Float[Tensor, "*bs num_samples 1"],
            bin_ends: Float[Tensor, "*bs num_samples 1"],
            spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts  # [4096, num_samples, 1]

        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]  # add a dimension to each of the attributes of the RayBundle
        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [4096, 1, 3]
            directions=shaped_raybundle_fields.directions,  # [4096, 1, 3]
            starts=bin_starts,  # [4096, num_samples, 1], euclidean
            ends=bin_ends,  # [4096, num_samples, 1], euclidean
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,  # the viewing volume of the camera
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1], step size between adjacent sample points along the rays
            spacing_starts=spacing_starts,  # [..., num_samples, 1], normalized starting positions
            spacing_ends=spacing_ends,  # [..., num_samples, 1], normalized ending positions
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,  # mapping to euclidean space
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        # ray_samples.get_refracted_rays()

        return ray_samples
    
    def get_ray_samples_ref(
            self,
            bin_starts: Float[Tensor, "*bs num_samples 1"],
            bin_ends: Float[Tensor, "*bs num_samples 1"],
            spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
            spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        if self.camera_indices is not None:
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        shaped_raybundle_fields = self[..., None]  # add a dimension to each of the attributes of the RayBundle
        
        # TODO: modify the
        
        frustums = Frustums(
            origins=shaped_raybundle_fields.origins,  # [4096, 1, 3]
            directions=shaped_raybundle_fields.directions,  # [4096, 1, 3]
            starts=bin_starts,  # [4096, num_samples, 1], euclidean
            ends=bin_ends,  # [4096, num_samples, 1], euclidean
            pixel_area=shaped_raybundle_fields.pixel_area,  # [..., 1, 1]
        )

        ray_samples = RaySamples(
            frustums=frustums,  # the viewing volume of the camera
            camera_indices=camera_indices,  # [..., 1, 1]
            deltas=deltas,  # [..., num_samples, 1], step size between adjacent sample points along the rays
            spacing_starts=spacing_starts,  # [..., num_samples, 1], normalized starting positions
            spacing_ends=spacing_ends,  # [..., num_samples, 1], normalized ending positions
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,  # mapping to euclidean space
            metadata=shaped_raybundle_fields.metadata,
            times=None if self.times is None else self.times[..., None],  # [..., 1, 1]
        )

        # ray_samples.get_refracted_rays()

        return ray_samples
    
    def clone(self) -> "RayBundle":
        """Clones the RayBundle"""
        return RayBundle(
            origins=self.origins.clone(),
            directions=self.directions.clone(),
            pixel_area=self.pixel_area.clone(),
            camera_indices=None if self.camera_indices is None else self.camera_indices.clone(),
            nears=None if self.nears is None else self.nears.clone(),
            fars=None if self.fars is None else self.fars.clone(),
            metadata={k: v.clone() for k, v in self.metadata.items()},
            times=None if self.times is None else self.times.clone(),
        )
