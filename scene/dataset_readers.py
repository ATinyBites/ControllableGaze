#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from roma.utils import rotvec_slerp,rotmat_slerp
from utils.camera_utils import loadCam
from scene.cameras import Camera
import torch
import copy
from utils.transform_utils import axis_angle_to_matrix,matrix_to_euler_angles,euler_angles_to_matrix,matrix_to_axis_angle
from scipy.spatial import ConvexHull, Delaunay

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud: Optional[BasicPointCloud]
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_train_meshes: dict = {}
    tgt_test_meshes: dict = {}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)): 
            file_path = frame["file_path"]
            if extension not in frame["file_path"]:
                file_path += extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None
            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, 
                image_path=image_path, image_name=image_name, 
                width=width, height=height, 
                timestep=timestep, camera_id=camera_id))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue
            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos
 
def interpolate_meshes(meshes:dict,interpolate_num,k=5):

    def sample_uniform_in_convexHull(X,size):
        hull = ConvexHull(X)
        hull_points = X[hull.vertices]  # 凸包顶点
        tri = Delaunay(hull_points)    # 三角剖分用于检查点是否在凸包内
        # 5. 在凸包内均匀采样
        def sample_in_hull(tri, n_samples=size):
            samples = []
            while len(samples) < n_samples:
                x = np.random.uniform(hull_points[:, 0].min(), hull_points[:, 0].max())
                y = np.random.uniform(hull_points[:, 1].min(), hull_points[:, 1].max())
                if tri.find_simplex((x, y)) >= 0:  # 判断点是否在凸包内
                    samples.append([x, y])
            return np.array(samples)
        def sample_in_gaussian(n_samples=size):
            mean = np.mean(X, axis=0)               # 均值
            cov = np.cov(X, rowvar=False)           # 协方差矩阵（行作为样本）

            print("Mean:", mean)
            print("Covariance Matrix:\n", cov)

            # 3. 根据高斯分布进行采样
            samples = np.random.multivariate_normal(mean, cov, n_samples)
            return samples


        # samples = sample_in_hull(tri, n_samples=size)
        samples = sample_in_gaussian(n_samples=size)
        return samples


    mesh_size = len(meshes)
    mesh_list = {}
    seed_size = k
    # seed_size = 1 
    # meshes_random_idx = np.random.randint(mesh_size,size = seed_size)
    # meshes_random_idx = [0,200,400]
    # eyes_pose_set = []
    # for mesh in meshes.values():
    #     eyes_pose_set.append(mesh['eyes_pose'])
    # eyes_pose_set = np.concatenate(eyes_pose_set,axis = 0).reshape(-1,2,3)
    # eyes_pose_matrix = axis_angle_to_matrix(torch.from_numpy(eyes_pose_set))
    # eyes_pose_euler = matrix_to_euler_angles(eyes_pose_matrix,convention='ZYX')
    # eyes_pose_euler_samples_left = sample_uniform_in_convexHull(eyes_pose_euler[:,0,1:].detach().cpu().numpy(),interpolate_num)
    # eyes_pose_euler_samples_right = sample_uniform_in_convexHull(eyes_pose_euler[:,1,1:].detach().cpu().numpy(),interpolate_num)
    # eyes_pose_euler_samples = np.stack([eyes_pose_euler_samples_left,eyes_pose_euler_samples_right],axis = 1)
    # eyes_pose_euler_samples = np.concatenate([np.zeros([eyes_pose_euler_samples.shape[0],2,1]),eyes_pose_euler_samples],axis = 2)
    # eyes_pose_matrix_samples = euler_angles_to_matrix(torch.from_numpy(eyes_pose_euler_samples),convention='ZYX')
    # eyes_pose_axis_angle_samples = matrix_to_axis_angle(eyes_pose_matrix_samples).detach().cpu().numpy().reshape(-1,6).astype(np.float32)


    for id in tqdm(range(interpolate_num)):
        meshes_choose = []
        meshes_random_idx = np.random.randint(mesh_size,size = seed_size)
        for _id in meshes_random_idx:
            __id = list(meshes.keys())[_id]
            meshes_choose.append(copy.deepcopy(meshes[__id]))

        # meshes_choose = meshes[meshes_random_idx]
        
        # weights = np.abs(np.random.randn(seed_size))
        weights = np.random.randint(1,1000,size=seed_size).astype(np.float32)
        weights = weights/np.sum(weights)
        # weights = [0,1,0]
        cur_weight = weights[0]
        new_flame_param = copy.deepcopy(meshes_choose[0])
        # new_flame_param = selected_mesh
        for i,mesh in enumerate(meshes_choose[1:]):
            for key,value in new_flame_param.items():
                if key == 'light':
                    new_value = mesh[key]
                    new_flame_param[key] = new_value
                    # new_flame_param[key] = selected_mesh[key]
                # elif key == 'eyes_pose':
                #     new_flame_param[key] = eyes_pose_axis_angle_samples[id][None]
                elif key not in ['rotation','neck_pose','jaw_pose','eyes_pose']:
                    new_value = (value*cur_weight+mesh[key]*weights[i+1])/(cur_weight+weights[i+1])
                    new_flame_param[key] = new_value
                else:
                    new_value = rotvec_slerp(torch.from_numpy(value).reshape(-1,3),torch.from_numpy(mesh[key]).reshape(-1,3),
                                             steps=torch.tensor([cur_weight/(cur_weight+weights[i+1]),],dtype=torch.float32)).reshape(1,-1).detach().cpu().numpy()
                    new_flame_param[key] = new_value
            cur_weight = cur_weight+weights[i+1]
            
        mesh_list[id] = new_flame_param
    return mesh_list

def interpolate_meshes_by_knearest(meshes:dict,interpolate_num,k=5):
    mesh_size = len(meshes)
    mesh_list = {}
    # meshes_random_idx = np.random.randint(mesh_size,size = seed_size)
    # meshes_random_idx = [0,200,400]
    rvec_list = [torch.from_numpy(meshes[key]['eyes_pose']) for key in meshes.keys()]
    rvec_list = torch.concat(rvec_list,dim=0)
    rvec_list = rvec_list.reshape(-1,2,3)
    rmat_list = axis_angle_to_matrix(rvec_list)
    gaze_vec_norm = torch.tensor([0,0,1],dtype=torch.float32)[None,None,None]
    gaze_vec = gaze_vec_norm@rmat_list.permute(0,1,3,2)
    gaze_vec = gaze_vec.reshape(-1,2,3)
    gaze_vec = torch.nn.functional.normalize(gaze_vec[:,0]+gaze_vec[:,1],p=2,dim=-1)

    for id in tqdm(range(interpolate_num)):
        seed_idx = torch.randint(0,mesh_size,(1,))
        gaze_vec_choosn = gaze_vec[seed_idx]
        # dist = torch.abs(torch.sum(gaze_vec_choosn[None]*gaze_vec,dim=-1)).squeeze()
        dist = torch.abs(gaze_vec_choosn[:,1]-gaze_vec[:,1])
        _,indices = torch.topk(dist,k=k,largest=False)
       
        meshes_choose = []
        meshes_random_idx = indices.detach().cpu().numpy()
        for _id in meshes_random_idx:
            meshes_choose.append(copy.deepcopy(meshes[_id]))
        # meshes_choose = meshes[meshes_random_idx]
        
        weights = np.abs(np.random.randn(k))
        weights = weights/np.sum(weights)
        # weights = [0,1,0]
        cur_weight = weights[0]
        new_flame_param = meshes_choose[0]
        for i,mesh in enumerate(meshes_choose[1:]):
            for key,value in new_flame_param.items():
                if key not in ['rotation','neck_pose','jaw_pose','eyes_pose']:
                    new_value = (value*cur_weight+mesh[key]*weights[i+1])/(cur_weight+weights[i+1])
                    new_flame_param[key] = new_value
                else:
                    new_value = rotvec_slerp(torch.from_numpy(value).reshape(-1,3),torch.from_numpy(mesh[key]).reshape(-1,3),
                                             steps=torch.tensor([cur_weight/(cur_weight+weights[i+1]),],dtype=torch.float32)).reshape(1,-1).detach().cpu().numpy()
                    new_flame_param[key] = new_value
            cur_weight = cur_weight+weights[i+1]
            
        mesh_list[id] = new_flame_param
    return mesh_list
       
def interpolate_data(cam_infos, resolution_scale, interpolate_num, meshes,args):
    cam_size = len(meshes)
    mesh_list = {}
    camera_list = []
    seed_size = 3
    for id in tqdm(range(interpolate_num)):
        cameras_choose = []
        meshes_choose = []
        camera_random_idx = np.random.randint(cam_size,size = seed_size)
        # camera_random_idx = [0] 
        for _id in camera_random_idx:
            cameras_choose.append(loadCam(args, _id, cam_infos[_id], resolution_scale))
            meshes_choose.append(copy.deepcopy(meshes[_id]))
            
        weights = np.abs(np.random.randn(seed_size))
        weights = weights/np.sum(weights)
        cur_weight = weights[0]
        new_R = copy.deepcopy(cameras_choose[0].R)
        new_T = copy.deepcopy(cameras_choose[0].T)
        new_image = cameras_choose[0].image
        new_flame_param = copy.deepcopy(meshes_choose[0])

        for i,cam in enumerate(cameras_choose[1:]):
            new_R = rotmat_slerp(torch.from_numpy(new_R[None]),torch.from_numpy(cam.R[None]),steps=torch.tensor([cur_weight/(cur_weight+weights[i+1]),],dtype=torch.float32)).squeeze().detach().cpu().numpy()
            new_T = (new_T*cur_weight+cam.T*weights[i+1])/(cur_weight+weights[i+1])
            # new_image = np.vstack([new_image,cam.image])
            
            mesh = meshes_choose[i+1]
            for key,value in new_flame_param.items():
                if key not in ['rotation','neck_pose','jaw_pose','eyes_pose']:
                    new_value = (value*cur_weight+mesh[key]*weights[i+1])/(cur_weight+weights[i+1])
                    new_flame_param[key] = new_value
                else:
                    new_value = rotvec_slerp(torch.from_numpy(value).reshape(-1,3),torch.from_numpy(mesh[key]).reshape(-1,3),
                                            steps=torch.tensor([cur_weight/(cur_weight+weights[i+1]),],dtype=torch.float32)).reshape(1,-1).detach().cpu().numpy()
                    new_flame_param[key] = new_value
            cur_weight = cur_weight+weights[i+1]
   
        cam_info = cam_infos[0]    
        new_cam = Camera(colmap_id=None, R=new_R, T=new_T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image_width=cameras_choose[0].image_width, image_height=cameras_choose[0].image_height,
                bg=cam_info.bg, 
                image=new_image, 
                image_path=cam_info.image_path,
                image_name=cam_info.image_name, uid=id, 
                timestep=id, data_device=args.data_device)
        camera_list.append(new_cam)
        mesh_list[id] = copy.deepcopy(new_flame_param)
    return camera_list,mesh_list

def readDynamicNerfInfo(path, white_background, eval, extension=".png", target_path=""):
    print("Reading Training Transforms")
    if target_path != "":
        train_cam_infos = readCamerasFromTransforms(target_path, "transforms_train.json", white_background, extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Training Meshes")
    train_mesh_infos = readMeshesFromTransforms(path, "transforms_train.json")
    if target_path != "":
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromTransforms(target_path, "transforms_train.json")
    else:
        tgt_train_mesh_infos = {}
     
    print("Reading Validation Transforms")
    if target_path != "":
        val_cam_infos = readCamerasFromTransforms(target_path, "transforms_val.json", white_background, extension)
    else:
        val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    
    print("Reading Test Transforms")
    if target_path != "":
        test_cam_infos = readCamerasFromTransforms(target_path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Test Meshes")
    test_mesh_infos = readMeshesFromTransforms(path, "transforms_test.json")
    if target_path != "":
        print("Reading Target Meshes (Test Division)")
        tgt_test_mesh_infos = readMeshesFromTransforms(target_path, "transforms_test.json")
    else:
        tgt_test_mesh_infos = {}
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    # nerf_normalization = getNerfppNorm(train_cam_infos) 
    nerf_normalization = {"radius":1} 

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "DynamicNerf" : readDynamicNerfInfo,
    "Blender" : readNerfSyntheticInfo,
}