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
from copy import deepcopy
import random
import json
from typing import Union, List
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks,interpolate_meshes,interpolate_data,interpolate_meshes_by_knearest
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.flame_gaussian_model import FlameGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, interpolate_camInfos
from utils.general_utils import PILtoTorch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # ---- from readCamerasFromTransforms() ----
            camera = deepcopy(self.cameras[idx])

            if camera.image is None:
                image = Image.open(camera.image_path)
            else:
                image = camera.image

            im_data = np.array(image.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + camera.bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # ---- from loadCam() and Camera.__init__() ----
            resized_image_rgb = PILtoTorch(image, (camera.image_width, camera.image_height))

            image = resized_image_rgb[:3, ...]

            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                image *= gt_alpha_mask
            
            camera.original_image = image.clamp(0.0, 1.0)
            return camera
        elif isinstance(idx, slice):
            return CameraDataset(self.cameras[idx])
        else:
            raise TypeError("Invalid argument type")

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : Union[GaussianModel, FlameGaussianModel], load_iteration=None, shuffle=True, interpolate_dataset=False,interpolte_num= 100,resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # load dataset
        assert os.path.exists(args.source_path), "Source path does not exist: {}".format(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "canonical_flame_param.npz")):
            print("Found FLAME parameter, assuming dynamic NeRF data set!")
            scene_info = sceneLoadTypeCallbacks["DynamicNerf"](args.source_path, args.white_background, args.eval, target_path=args.target_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # process cameras
        self.train_cameras = {}
        self.val_cameras = {}
        self.test_cameras = {}
        self.interpolate_cameras = {}
        if not self.loaded_iter:
            if gaussians.binding == None:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.val_cameras:
                camlist.extend(scene_info.val_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.val_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Validation Cameras")
            self.val_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.val_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        if interpolate_dataset:
            print("Interpolate Training Cameras")
            # self.interpolate_cameras[1.0],self.inter_meshes = interpolate_data(deepcopy(scene_info.train_cameras),1.0,interpolte_num,deepcopy(scene_info.train_meshes),args)
            self.interpolate_cameras[1.0] = interpolate_camInfos(deepcopy(scene_info.train_cameras),resolution_scale,interpolte_num,args)
            # self.interpolate_cameras[1.0] = self.train_cameras[1.0]

        # process meshes 
        self.train_meshes = scene_info.train_meshes
        self.test_meshes = scene_info.test_meshes
        if gaussians.binding != None: 
            self.gaussians.load_meshes(scene_info.train_meshes, scene_info.test_meshes, 
                                    scene_info.tgt_train_meshes, scene_info.tgt_test_meshes)
        # create gaussians
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply"),
                has_target=args.target_path != "",
            )
            if os.path.exists(os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),'checkpoint.pth')):
                state_dict = torch.load(os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),'checkpoint.pth'))
                self.gaussians.load_state_dict(state_dict,strict=False)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        if interpolate_dataset:
            # self.inter_meshes = interpolate_meshes_by_knearest(deepcopy(scene_info.train_meshes),interpolte_num,k=3)
            self.inter_meshes = interpolate_meshes(deepcopy(scene_info.train_meshes),interpolte_num,k=1)
            # self.inter_meshes = deepcopy(scene_info.train_meshes)
            self.gaussians.load_meshes(self.inter_meshes, self.inter_meshes, 
                    {},{})
            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        torch.save(self.gaussians.state_dict(),os.path.join(point_cloud_path,"checkpoint.pth"))

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale])
    
    def getValCameras(self, scale=1.0):
        return CameraDataset(self.val_cameras[scale])

    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale])
    
    def getInterpolateCameras(self,scale=1.0):
        return CameraDataset(self.interpolate_cameras[scale])