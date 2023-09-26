from tqdm import tqdm
import torch
import numpy as np
import torch
import math
from pytorch3d.io import load_ply, IO
import pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import copy
import glob
# from utils import lookat, get3d_box_from_pcs, intrinsic_calibration, get_point_on_circle
import os
import copy

device = 'cuda'
pt3d_io = IO()

mesh_file = '/home/zelda/zh340/myzone/rocky/data/ChallengeDevelopmentSet/42445173/42445173_3dod_mesh.ply'

mesh = pt3d_io.load_mesh(mesh_file, device=device)
# vertices = mesh.verts_packed().cpu().numpy()

verts = mesh.verts_packed()
faces = mesh.faces_packed()
texture_tensor = mesh.textures.verts_features_packed()
a = verts
b = faces

save_path = "split_ply"

if not os.path.exists(save_path):
    os.mkdir(save_path)

num_subparts_x = 3  # Number of subparts in the X direction
num_subparts_y = 3  # Number of subparts in the Y direction

step_x = verts[:, 0].max() / num_subparts_x
step_y = verts[:, 1].max() / num_subparts_y

global_verts = verts.clone()
global_faces = faces.clone()
global_texture_tensor = texture_tensor
# Iterate through the subparts
for i in range(num_subparts_x):
    for j in range(num_subparts_y):
        # Define the bounding box for the subpart
        min_x = i * step_x
        max_x = (i + 1) * step_x
        min_y = j * step_y
        max_y = (j + 1) * step_y
        # Filter vertices within the bounding box
        idx =(global_verts[:, 0] >= min_x) & (global_verts[:, 0] <= max_x) & (global_verts[:, 1] >= min_y) & (global_verts[:, 1] <= max_y)
        # Crop the vertices and create an index map
        map_idx = torch.zeros_like(global_verts[:, 0], dtype=torch.long).cuda() - 1
        print(map_idx.shape, map_idx.device)
        map_idx[idx] = torch.arange(idx.sum()).cuda()
        a = global_verts[idx]
        texture_tensor = global_texture_tensor[idx]
        # Crop the triangle surface and update the indices
        b = global_faces[(idx[global_faces[:, 0]] & idx[global_faces[:, 1]] & idx[global_faces[:, 2]])]
        final_b = map_idx[b]
        converted_texture = pytorch3d.renderer.mesh.textures.TexturesVertex([texture_tensor])
        cropped_mesh = pytorch3d.structures.Meshes(verts=[a], faces=[final_b], textures= converted_texture).cuda()
        IO().save_mesh(cropped_mesh, f"{save_path}/{i}_{j}.ply", binary=False, colors_as_uint8=True)