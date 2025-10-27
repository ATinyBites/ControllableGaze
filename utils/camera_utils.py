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

from tqdm import tqdm
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from roma.utils import rotmat_slerp
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        image_width, image_height = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        image_width, image_height = (int(orig_w / scale), int(orig_h / scale))

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image_width=image_width, image_height=image_height,
                  bg=cam_info.bg, 
                  image=cam_info.image, 
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=id, 
                  timestep=cam_info.timestep, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        if args.select_camera_id != -1 and c.camera_id is not None:
            if c.camera_id != args.select_camera_id:
                continue
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list
 
def interpolate_camInfos(cam_infos, resolution_scale, interpolate_num, args):
    cam_size = len(cam_infos)
    camera_list = []
    seed_size = 3
    # seed_size = 1
    camera_idx = []
    camera_sets = {}
    # R_list = []
    # T_list = []
    for cam in cam_infos:
        if cam.camera_id not in camera_idx:
            camera_idx.append(cam.camera_id)
            camera_sets[cam.camera_id] = []
        camera_sets[cam.camera_id].append(cam)
    #     R_list.append(cam.R)
    #     T_list.append(cam.T)
        
    # R_list = np.stack(R_list,axis=0)
    # T_list = np.stack(T_list,axis=0)
    # T_mean = T_list.mean(axis=0)
    # T_std = T_list.std(axis=0)
    # T_min = T_list.min(axis=0)
    # T_max = T_list.max(axis=0)

    for id in tqdm(range(interpolate_num)):
        cameras_choose = []
        camera_random_idx = np.random.randint(cam_size,size = seed_size)
        camera_set_idx = np.random.randint(len(camera_idx),size=1)
        camera_set = camera_sets[camera_idx[camera_set_idx[0]]]
        # camera_random_idx = [0] 
        cam_first = loadCam(args, 0, camera_set[0], resolution_scale)
        for _id in camera_random_idx:
            cameras_choose.append(loadCam(args, _id, camera_set[_id%len(camera_set)], resolution_scale))
            # cameras_choose.append(cam_first)
        
        # weights = np.abs(np.random.randn(seed_size))
        weights = np.random.randint(1,1000,size=seed_size).astype(np.float32)
        weights = weights/np.sum(weights)
        cur_weight = weights[0]
        new_R = cameras_choose[0].R
        new_T = cameras_choose[0].T
        
        new_image = cameras_choose[0].image
        for i,cam in enumerate(cameras_choose[1:]):
            new_R = rotmat_slerp(torch.from_numpy(new_R[None]),torch.from_numpy(cam.R[None]),steps=torch.tensor([cur_weight/(cur_weight+weights[i+1]),],dtype=torch.float32)).squeeze().detach().cpu().numpy()
            new_T = (new_T*cur_weight+cam.T*weights[i+1])/(cur_weight+weights[i+1])
            new_image = np.vstack([new_image,cam.image])
            cur_weight = cur_weight+weights[i+1]
        
        # new_T = np.clip(np.random.randn(3),a_min=-3.0,a_max=3.0)*T_std + T_mean
        cam_info = cam_infos[0]    
        new_cam = Camera(colmap_id=None, R=new_R, T=new_T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image_width=cameras_choose[0].image_width, image_height=cameras_choose[0].image_height,
                bg=cam_info.bg, 
                image=cam_info.image, 
                image_path=cam_info.image_path,
                image_name=cam_info.image_name, uid=id, 
                timestep=id, data_device=args.data_device)
        camera_list.append(new_cam)
    
    return camera_list
        
        
def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
