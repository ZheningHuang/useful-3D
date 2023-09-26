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

import pyviz3d.visualizer as viz
import glob
# import pyviz3d.visualizer as viz
import open3d as o3d


PATH = "/home/zelda/zh340/myzone/rocky/data/ChallengeDevelopmentSet/*"
scene_path_list = glob.glob(PATH)

for scene in scene_path_list:
    scene_name = scene.split("/")[-1]
    device = 'cuda'
    pt3d_io = IO()

    mesh_file = f'/home/zelda/zh340/myzone/rocky/data/ChallengeDevelopmentSet/{scene_name}/{scene_name}_3dod_mesh.ply'

    mesh = pt3d_io.load_mesh(mesh_file, device=device)
    # vertices = mesh.verts_packed().cpu().numpy()

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    texture_tensor = mesh.textures.verts_features_packed()
    a = verts
    b = faces

    save_path = f"split_ply/{scene_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_subparts_x = 2  # Number of subparts in the X direction
    num_subparts_y = 2  # Number of subparts in the Y direction

    step_x = (verts[:, 0].max() - verts[:, 0].min())/ num_subparts_x
    step_y = (verts[:, 1].max() - verts[:, 1].min())/ num_subparts_y

    global_verts = verts.clone()
    global_faces = faces.clone()
    global_texture_tensor = texture_tensor
    # Iterate through the subparts
    for i in range(num_subparts_x):
        for j in range(num_subparts_y):
            # Define the bounding box for the subpart
            min_x = verts[:, 0].min() + i * step_x
            max_x = verts[:, 0].min() + (i + 1) * step_x
            min_y = verts[:, 1].min() + j * step_y
            max_y = verts[:, 1].min() + (j + 1) * step_y
            # Filter vertices within the bounding box
            idx =(global_verts[:, 0] >= min_x) & (global_verts[:, 0] <= max_x) & (global_verts[:, 1] >= min_y) & (global_verts[:, 1] <= max_y)
            # Crop the vertices and create an index map

            map_idx = torch.zeros_like(global_verts[:, 0], dtype=torch.long).cuda() - 1
            print(map_idx.shape, map_idx.device)
            map_idx[idx] = torch.arange(idx.sum()).cuda()
            a = global_verts[idx]
            if len(a) < 20:
                print("passed")
                continue
            texture_tensor = global_texture_tensor[idx]
            # Crop the triangle surface and update the indices
            b = global_faces[(idx[global_faces[:, 0]] & idx[global_faces[:, 1]] & idx[global_faces[:, 2]])]
            final_b = map_idx[b]
            converted_texture = pytorch3d.renderer.mesh.textures.TexturesVertex([texture_tensor])
            # print(a.shape)
            cropped_mesh = pytorch3d.structures.Meshes(verts=[a], faces=[final_b], textures= converted_texture).cuda()
            IO().save_mesh(cropped_mesh, f"{save_path}/{i}_{j}.ply", binary=False, colors_as_uint8=True)

    v = viz.Visualizer()
    split_list = glob.glob(save_path+ "/*")

    for split_path in split_list:
        split_name = split_path.split("/")[-1].split(".")[0]
        # print(split_name)
        pcd = o3d.io.read_point_cloud(split_path)
        v.add_points(f'{split_name}', np.asarray(pcd.points), np.asarray(pcd.colors)*255, point_size=20, visible=True)
    v.save(f'viz/{scene_name}')
