import argparse
import os

import numpy as np
import mouette as M
import implicitlab as IL
from src import load_model, ImplicitRepresentation, renderers
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

"""
Code adapted from:
https://github.com/skmhrk1209/Torch-Sphere-Tracer
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("folder_path", type=str, help="path to the model '.pt' file")
    parser.add_argument("-p", "--path", type=str, default="", help="path_to_output")
    parser.add_argument("-res", "--resolution", type=int, default=256, help="grid resolution to consider")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000, help="batch size")
    args = parser.parse_args()

    device = IL.utils.get_device(args.cpu)
    print("DEVICE:", device)
    SDF : ImplicitRepresentation = load_model(args.folder_path, device)

    # def sphere(r):
    #     def sdf(p):
    #         d = torch.norm(p, dim=-1, keepdim=True) - r
    #         return d
    #     return sdf
    # SDF = sphere(0.5)

    
    num_iterations = 1000
    convergence_threshold = 1e-3

    # ---------------- camera matrix ---------------- #
    fx = fy = args.resolution
    cx = cy = args.resolution//2
    camera_matrix = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float)

    # ---------------- camera position ---------------- #

    distance = 2.5
    azimuth = 0 #np.pi / 4.0
    elevation = 0 #np.pi / 4.0

    camera_position = torch.tensor([
        +np.cos(elevation) * np.sin(azimuth), 
        -np.sin(elevation), 
        -np.cos(elevation) * np.cos(azimuth)
    ], device=device, dtype=torch.float) * distance

    # ---------------- camera rotation ---------------- #

    target_position = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
    up_direction = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float)

    camera_z_axis = target_position - camera_position
    camera_x_axis = torch.cross(up_direction, camera_z_axis, dim=-1)
    camera_y_axis = torch.cross(camera_z_axis, camera_x_axis, dim=-1)
    camera_rotation = torch.stack((camera_x_axis, camera_y_axis, camera_z_axis), dim=-1)
    camera_rotation = F.normalize(camera_rotation, dim=-2)

    # ---------------- directional light ---------------- #

    light_directions = torch.tensor([1.0, 0.5, 0.0], device=device, dtype=torch.float)

    # ---------------- ray marching ---------------- #
    
    y_positions = torch.arange(cy * 2, device=device, dtype=torch.float)
    x_positions = torch.arange(cx * 2, device=device, dtype=torch.float)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions)
    z_positions = torch.ones_like(y_positions, dtype=torch.float)
    ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
    ray_positions = torch.einsum("mn,...n->...m", torch.inverse(camera_matrix),  ray_positions)
    ray_positions = torch.einsum("mn,...n->...m", camera_rotation, ray_positions) + camera_position
    ray_directions = F.normalize(ray_positions - camera_position, dim=-1)

    # ---------------- rendering ---------------- #
    print("sphere tracing")
    ray_positions = ray_positions.reshape((args.resolution*args.resolution,3))
    ray_directions = ray_directions.reshape((args.resolution*args.resolution,3))

    surface_positions, converged = renderers.sphere_tracing(
        signed_distance_function=SDF, 
        ray_positions=ray_positions, 
        ray_directions=ray_directions, 
        num_iterations=num_iterations, 
        convergence_threshold=convergence_threshold,
        bounding_radius=3
    )
    # M.mesh.save(M.mesh.from_arrays(surface_positions.detach().cpu().numpy()), "ray_surf.mesh")
    # surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))

    print("compute normals")
    surface_normals = renderers.compute_normal(
        signed_distance_function = SDF, 
        surface_positions=surface_positions,
    )
    surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals))

    print("phong shading")
    image = renderers.phong_shading(
        surface_normals=surface_normals, 
        view_directions=camera_position - surface_positions, 
        light_directions=light_directions, 
        light_ambient_color=torch.ones(1, 1, 3, device=device),
        light_diffuse_color=torch.ones(1, 1, 3, device=device), 
        light_specular_color=torch.ones(1, 1, 3, device=device), 
        material_ambient_color=torch.full((1, 1, 3), 0.2, device=device) + (torch.rand(1, 1, 3, device=device) * 2 - 1) * 0.3,
        material_diffuse_color=torch.full((1, 1, 3), 0.7, device=device) + (torch.rand(1, 1, 3, device=device) * 2 - 1) * 0.1,
        material_specular_color=torch.full((1, 1, 3), 0.1, device=device),
        material_emission_color=torch.zeros(1, 1, 3, device=device),
        material_shininess=64.0,
    )

    # shadowed = renderers.compute_shadows(
    #     signed_distance_function=SDF, 
    #     surface_positions=surface_positions, 
    #     surface_normals=surface_normals,
    #     light_directions=light_directions, 
    #     num_iterations=num_iterations, 
    #     convergence_threshold=convergence_threshold,
    #     foreground_masks=converged,
    # )
    # image = torch.where(shadowed, image * 0.5, image)

    image = torch.where(converged, image, torch.ones_like(image))
    image = image.reshape((args.resolution, args.resolution, 3))
    image = image.detach().cpu().numpy()
    plt.imsave("ray_marched.png", image)