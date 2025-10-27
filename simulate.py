import torch
import numpy as np
import cv2
import h5py
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from scene import Scene
from gaussian_renderer import render
from torch.autograd import Variable
from utils.graphics_utils import fov2focal,focal2fov
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle,euler_angles_to_matrix,matrix_to_euler_angles
from flame_model.flame import FlameHead
from torch.utils.data import DataLoader
from utils.normalize_utils import normalizeData_face,estimateHeadPose
import math
import random
from scene.cameras import Camera
import matplotlib.pyplot as plt
import copy

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

class Simulator:

    def __init__(self,mp:ModelParams,pp:PipelineParams,datasize:int,dataset_type=None) -> None:
        self.mp = mp
        self.mp.eval = True
        self.pp = pp
        self.datasize = datasize
        self.dataset_type=dataset_type
        # construct gaussian avatar and scene
        self.gaussians = FlameGaussianModel(self.mp.sh_degree,z_rotate=False,load_light=args.load_light)
        self.scene = Scene(self.mp, self.gaussians, load_iteration=30000, shuffle=False,interpolate_dataset=True,interpolte_num=datasize)
        self.flame_model = self.gaussians.flame_model

        self.model_name = os.path.basename(self.mp.model_path)
        self.save_root = os.path.join(self.mp.model_path,"interpolate")
        self.render_path = os.path.join(self.save_root,"render")
        if args.render_eye:
            self.render_eye_path = os.path.join(self.save_root,"eye")
            self.render_eye_region_path = os.path.join(self.save_root,"eye_region")

        self.normal_path = os.path.join(self.save_root,"normal")
        if args.anno:
            self.normal_anno_path = os.path.join(self.save_root,"normal_anno")
        make_dir(self.save_root)
        make_dir(self.render_path)
        if args.render_eye:
            make_dir(self.render_eye_path)
            make_dir(self.render_eye_region_path)
        make_dir(self.normal_path)
        if args.anno:
            make_dir(self.normal_anno_path)
        
        # data buffer
        self.result_raw = None
        self.result_norm = None
        self.result_test = None

        if self.dataset_type == 'mpii':
            self.face_model = np.array([ -45.096768, -21.312858, 21.312858, 
                45.096768, -26.299577, 26.299577, -0.483773, 
                0.483773, 0.483773, -0.483773, 68.595035, 
                68.595035, 2.397030, -2.397030, -2.397030, 
                2.397030, -0.000000, -0.000000],dtype=np.float32).reshape(3,6).T
        else:
            face_model = np.loadtxt('data/face_model.txt')
            landmark_use = [20, 23, 26, 29, 15, 19]
            self.face_model = face_model[landmark_use]


        self.gaze_points = None
        self.eyes_center,self.face_center,self.landmark_3d,self.static_offset = self.get_neutral_face_points()

        # define view frustum params
        self.z_near = None
        self.z_far = None
        self.fovx = None
        self.fovy = None
    
    def create_view_frustum(self,z_near:float,z_far:float,fovx:float,fovy:float):
        """
            create a view frustum whose center is face center in Flame coords
            Args:
                z_near: Near clipping plane distance ,The unit is real-world millimeters
                z_far: Far clipping plane distance ,The unit is real-world millimeters
                fovx:The angle of the field of view in the x-axis direction, in angles
                fovy:The angle of the field of view in the y-axis direction, in angles
        """
        assert z_near < z_far and z_near>0 and fovx>0 and fovy>0
        scale_factor = 988.0130615234375
        self.z_near = z_near/scale_factor
        self.z_far = z_far/scale_factor
        self.fovx = fovx/180*np.pi
        self.fovy = fovy/180*np.pi

        self.fovx_limit_min = -self.fovx/2.0
        self.fovx_limit_max = self.fovx/2.0
        self.fovy_limit_min = -self.fovy/2.0
        self.fovy_limit_max = self.fovy/2.0
    
    def create_view_frustum_from_data(self,z_near:float,z_far:float):
        assert z_near>0 and z_near<z_far
        scale_factor = 988.0130615234375
        self.z_near = z_near/scale_factor
        self.z_far = z_far/scale_factor

        eyes_pose = []
        for key,value in self.scene.train_meshes.items():
            eyes_pose.append(value['eyes_pose'])
        
        eyes_pose = np.stack(eyes_pose,axis=0)
        eyes_pose = eyes_pose.reshape(-1,3)
        
        eyes_matrix = axis_angle_to_matrix(torch.from_numpy(eyes_pose))
        # euler_angles = matrix_to_euler_angles(eyes_matrix,convention="XYZ").detach().cpu().numpy()
        # fovx_limit = np.max(np.abs(euler_angles[:,0]))
        # fovy_limit = np.max(np.abs(euler_angles[:,1]))
        
        z_vec = torch.tensor([0,0,1],dtype=torch.float32,device=eyes_matrix.device)
        eye_vec = eyes_matrix@z_vec.reshape(1,3,1)
        eye_vec = eye_vec.reshape(-1,3)
        fovx_limit = torch.abs(torch.asin(eye_vec[:,0])).max().item()
        fovy_limit = torch.abs(torch.asin(eye_vec[:,1])).max().item()

        # fovx_limit_min = torch.asin(eye_vec[:,0]).min().item()
        # fovx_limit_max = torch.asin(eye_vec[:,0]).max().item()
        # fovy_limit_min = torch.asin(eye_vec[:,1]).min().item()
        # fovy_limit_max = torch.asin(eye_vec[:,1]).max().item()
        # fovx_limit = min(-fovx_limit_min,fovx_limit_max)
        # fovy_limit = min(-fovy_limit_min,fovy_limit_max)
        # self.fovx_limit_min = -self.fovx/2.0
        # self.fovy_limit_min = -self.fovy/2.0
        # self.fovx_limit_max = self.fovx/2.0
        # self.fovy_limit_max = self.fovy/2.0
        
        eye_vec = eye_vec/eye_vec[:,-1,None]
        self.fovx_limit_max = torch.atan(eye_vec[:,0]).max().item()
        self.fovy_limit_max = torch.atan(eye_vec[:,1]).max().item()
        self.fovx_limit_min = torch.atan(eye_vec[:,0]).min().item()
        self.fovy_limit_min = torch.atan(eye_vec[:,1]).min().item()
        
        # eye_dist = np.abs(self.eyes_center[0,0] - self.eyes_center[1,0])
        # self.fovx_limit_min = np.arctan((np.tan(self.fovx_limit_min)*self.z_near+eye_dist/2.0)/self.z_near)
        # self.fovx_limit_max = np.arctan((np.tan(self.fovx_limit_max)*self.z_near-eye_dist/2.0)/self.z_near)
        
        
        
        # self.fovx = (fovx_limit+5/180*np.pi)*2.0
        # self.fovy = (fovy_limit+5/180*np.pi)*2.0
        self.fovx_set = torch.asin(eye_vec[:,0]).detach().cpu().numpy()
        self.fovy_set = torch.asin(eye_vec[:,1]).detach().cpu().numpy()
        
        self.fovx = (fovx_limit)*2.0
        self.fovy = (fovy_limit)*2.0

    def generate_gaze_points_by_frustum(self):
        resolution = 10000

        h_near_size = 2*math.tan(self.fovy/2.0)*self.z_near
        w_near_size = 2*math.tan(self.fovx/2.0)*self.z_near
        random_xy = np.random.randint(0,resolution,size=(self.datasize,2)).astype(np.float32)/resolution-0.5 #(-0.5,0.5)
        
        shuffle_idx = np.arange(self.datasize)
        np.random.shuffle(shuffle_idx)
        random_xy[shuffle_idx[::2],:] = 0.0
        
        random_xy[:,0]*=w_near_size
        random_xy[:,1]*=h_near_size
        random_z = np.ones_like(random_xy[:,0])*self.z_near
        random_xyz = np.stack([random_xy[:,0],random_xy[:,1],random_z],axis=-1)
        ray_z = np.random.randint(0,resolution,size=(self.datasize,1)).astype(np.float32)/resolution #(0,1)
        random_xyz = random_xyz*ray_z+random_xyz*self.z_far/self.z_near*(1.0-ray_z)

        gaze_points = np.mean(self.eyes_center,axis=0).reshape(1,3)+random_xyz
        # gaze_points = self.nose_center+random_xyz
        # gaze_points = self.mouth_center+random_xyz

        gaze_points = gaze_points.astype(np.float32)
        self.update_eye_pose(gaze_points)
        self.gaze_points = gaze_points
        
        return gaze_points
        
    def get_neutral_face_points(self):
        for k in self.scene.train_meshes.keys():
            break

        mesh_info = self.scene.train_meshes[k]
        neutral_info = {}
        for key,value in mesh_info.items():
            if key == "shape":
                neutral_info[key] = torch.from_numpy(value).reshape(1,-1).cuda()
            elif key == 'light':
                continue
            else:
                if 'pose' in key: 
                    key = key.split('_')[0]
                if key !='static_offset':
                    neutral_info[key] = torch.zeros_like(torch.from_numpy(value)).reshape(1,-1).cuda()
                else:
                    neutral_info[key] = torch.from_numpy(value).cuda()
                    static_offset = torch.from_numpy(value).cuda()
        verts,landmarks,J = self.flame_model.forward(**neutral_info,zero_centered_at_root_node=False,return_J=True)
        pupil_centers = landmarks[0,[-1,-2]]
        eye_centers = J[0,-2:]

        rot_matrix = np.array([[ 0.99978507,  0.01133559,  0.01735884],
                                [ 0.00832352, -0.9863116,   0.1646822 ],
                                [ 0.01898799, -0.16450232, -0.98619395]],dtype=np.float32)
        translation = np.array([[-0.00102562, -0.01849389,  0.06861164]],dtype=np.float32)
        face_center = torch.mean(landmarks[0,[31,35]],dim=0).reshape(3).detach().cpu().numpy() # nose center

        scale_factor = 988.0130615234375
        one_cm = 10.0/scale_factor
        if self.dataset_type == 'mpii':
            face_center[1] += 5.0*one_cm # add 5cm to the nose
            
        landmark_face_txt_coords = (landmarks[0].detach().cpu().numpy()@rot_matrix.T+translation)*scale_factor
        return eye_centers.detach().cpu().numpy(),face_center,landmark_face_txt_coords,static_offset
    
    def update_eye_pose(self,gaze_points):
        gaze_points = gaze_points.reshape(-1,3)
        
        g_eyes = gaze_points[:,None] - self.eyes_center[None] # (data_size,2,3)

        g_eyes = g_eyes/np.linalg.norm(g_eyes,ord=2,axis=-1,keepdims=True)
        gaze_theta = np.arcsin(-g_eyes[...,1])
        gaze_phi = np.arctan2(g_eyes[...,0],g_eyes[...,2])
        
        euler_angles = np.concatenate([np.zeros_like(gaze_theta[...,None]),gaze_phi[...,None],gaze_theta[...,None]],axis=-1)
        rotation_matrices = euler_angles_to_matrix(torch.from_numpy(euler_angles),"ZYX")

        if args.add_kappa:
            # left_kappa_rotation = euler_angles_to_matrix(torch.from_numpy(np.array([0,self.kappa/180.0*np.pi,0])[None]), "ZYX").float()
            # right_kappa_rotation = euler_angles_to_matrix(torch.from_numpy(np.array([0,-self.kappa/180.0*np.pi,0])[None]), "ZYX").float()
            left_kappa_rotation = euler_angles_to_matrix(torch.from_numpy(np.array([0,0,-3.0/180.0*np.pi])[None]), "ZYX").float()
            right_kappa_rotation = euler_angles_to_matrix(torch.from_numpy(np.array([0,0,-3.0/180.0*np.pi])[None]), "ZYX").float()
            kappa_rotation = torch.cat([left_kappa_rotation,right_kappa_rotation],dim=0)
            rotation_matrices = rotation_matrices@kappa_rotation.reshape(1,2,3,3).to(rotation_matrices.device)
        
        rvec = matrix_to_axis_angle(rotation_matrices)

        for i,key in enumerate(self.scene.inter_meshes.keys()):
            self.scene.inter_meshes[key]['eyes_pose'] = rvec[i].reshape(1,6).detach().cpu().numpy()
        
        self.gaussians.flame_param['eyes_pose'][:] = rvec.reshape(-1,6).cuda()
        self.gaussians.flame_param_orig['eyes_pose'] = self.gaussians.flame_param['eyes_pose'].clone()
    

    def render_standard_pose(self,annotate=False):

        """get flame params"""
        for k in self.scene.train_meshes.keys():
            break
        # k=3
        mesh_info = self.scene.train_meshes[k]
        neutral_info = {}
        for key,value in mesh_info.items():
            if key == "shape":
                neutral_info[key] = torch.from_numpy(value).reshape(1,-1).cuda()
            else:
                if 'pose' in key:
                    key = key.split('_')[0]
                if key !='static_offset':
                    neutral_info[key] = torch.zeros_like(torch.from_numpy(value)).reshape(1,-1).cuda()
                else:
                    neutral_info[key] = torch.from_numpy(value).cuda()
        # set eyes pose to zeros
        mesh_info['eyes_pose'] = np.zeros_like(mesh_info['eyes_pose'])
        mesh_info['expr'] = np.zeros_like(mesh_info['expr'])
        
        # bind flame params to gaussianAvatars
        verts,vert_cano,landmarks = self.gaussians.flame_model(torch.from_numpy(mesh_info['shape']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['expr']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['rotation']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['neck_pose']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['jaw_pose']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['eyes_pose']).cuda().reshape(1,-1),
                                   torch.from_numpy(mesh_info['translation']).cuda().reshape(1,-1),
                                    zero_centered_at_root_node=False, 
                                    return_landmarks=True,
                                    return_verts_cano=True,
                                    static_offset = torch.from_numpy(mesh_info['static_offset']).cuda(),
                                    dynamic_offset= None)
        self.gaussians.select_mesh_by_timestep(0)
        self.gaussians.update_mesh_properties(verts,vert_cano)
        
        """create camera and render"""
        camera = self.scene.getInterpolateCameras()[0]
        R_new = np.eye(3,3)
        R_new[1:]*=-1

        T_new = camera.T
        T_new[:2] = 0
        
        fov_x,fov_y = camera.FoVx,camera.FoVy
        W,H = camera.image_width,camera.image_height
        fl_x,fl_y = fov2focal(fov_x,W),fov2focal(fov_y,H)
        cx,cy = W*0.5,H*0.5
        inmat = np.eye(3,3)
        inmat[[0,0,1,1],[0,2,1,2]] = np.array([fl_x,cx,fl_y,cy])
        
        camera_new = Camera(0,R_new,T_new,camera.FoVx,camera.FoVy,camera.bg,camera.image_width,camera.image,camera.image_height,camera.image_path,camera.image_name,camera.uid)
        bg = torch.ones((1,3),dtype=torch.float32,device='cuda')
        rendering = render(camera_new, self.gaussians, pipeline, bg)["render"]
        image = torch.clip((rendering[[2,1,0]].permute(1,2,0)*255),min=0,max=255).detach().cpu().numpy().astype(np.uint8).copy()

        if annotate:
            landmark_c = landmarks[0].detach().cpu().numpy()@camera_new.R.T+camera_new.T.reshape(1,3)
            landmark_uv = landmark_c@inmat.T
            landmark_uv = landmark_uv[:,:2]/(landmark_uv[:,2,None]+1e-6)
            
            for ldm in landmark_uv:
                image = cv2.circle(image,(int(ldm[0]),int(ldm[1])),1,(0,0,255),1)

        cv2.imwrite(os.path.join(self.save_root,"standard.png"),image)

    def parser_data(self):
        cam_infos,mesh_infos = self.scene.interpolate_cameras[1.0],self.scene.inter_meshes
        frame_num = len(cam_infos)
        flame_model = FlameHead(shape_params=300,expr_params=100)
        
        w2c_Rmats = []
        w2c_Tvecs = []
        inmats = []
        g_eyes_lst = []
        g_face_lst = []
        gc_lst = []
        landmark_3d_lst = []
        landmark_2d_lst = []
        for frame_idx in range(frame_num):
            cam = cam_infos[frame_idx]
            mesh = mesh_infos[frame_idx]
            # parser cam params
            fov_x,fov_y = cam.FoVx,cam.FoVy
            W,H = cam.image_width,cam.image_height
            fl_x,fl_y = fov2focal(fov_x,W),fov2focal(fov_y,H)
            cx,cy = W*0.5,H*0.5
            inmat = np.eye(3,3)
            inmat[[0,0,1,1],[0,2,1,2]] = np.array([fl_x,cx,fl_y,cy])
            w2c_Rmat = cam.R.T
            w2c_Tvec = cam.T
            w2c_matrix = np.eye(4,4)
            w2c_matrix[:3,:3] = w2c_Rmat
            w2c_matrix[:3,3] = w2c_Tvec.flatten()
            
            
            # parser face params


            if self.gaze_points is not None:
                g_eyes = self.gaze_points[frame_idx,None] - self.eyes_center # (2,3)
                g_eyes = g_eyes/np.linalg.norm(g_eyes,ord=2,axis=-1,keepdims=True)
            else:
                eye_rotation_matrices = axis_angle_to_matrix(torch.from_numpy(mesh['eyes_pose']).reshape(2,3)).reshape(2,3,3)
                g_eyes = torch.tensor([0,0,1],dtype=torch.float32)[None,None]@eye_rotation_matrices.permute(0,2,1)
                g_eyes = g_eyes.reshape(2,3)
                      
            if self.gaze_points is not None:
                g_face = self.gaze_points[frame_idx]-self.face_center
            else:
                g_face = g_eyes[0]+g_eyes[1]
            
            g_face = g_face/np.linalg.norm(g_face,ord=2,axis=-1,keepdims=True)
            
            vertices,landmark_3d  = flame_model.forward(shape=torch.from_numpy(mesh['shape']).reshape(1,-1),
                                            expr=torch.from_numpy(mesh['expr']).reshape(1,-1),
                                            rotation=torch.zeros((1,3)),
                                            neck=torch.zeros((1,3)),
                                            # neck = torch.from_numpy(mesh['neck_pose']).reshape(1,-1),
                                            jaw=torch.from_numpy(mesh['jaw_pose']),
                                            eyes=torch.from_numpy(mesh['eyes_pose']),
                                            translation=torch.zeros((1,3)),
                                            static_offset = torch.from_numpy(mesh['static_offset']),
                                            return_landmarks=True,
                                            return_J=False,
                                            zero_centered_at_root_node=False
                                        )
            
            # get a combined camera extrinsic matrix
            flame_translation = mesh['translation']
            flame_rotation = axis_angle_to_matrix(torch.from_numpy(mesh['rotation']))[0].detach().cpu().numpy()
            flame_neck_rotation = axis_angle_to_matrix(torch.from_numpy(mesh['neck_pose']))[0].detach().cpu().numpy()
            # flame_neck_rotation = np.eye(3,3) 
            flame_matrix = np.eye(4,4)
            flame_neck_matrix = np.eye(4,4)
            flame_matrix[:3,:3] = flame_rotation
            flame_matrix[:3,3] = flame_translation.flatten()
            flame_neck_matrix[:3,:3] = flame_neck_rotation
            w2c_matrix = w2c_matrix@flame_matrix@flame_neck_matrix
            w2c_Rmat = w2c_matrix[:3,:3]
            w2c_Tvec = w2c_matrix[:3,3]
            
            # convert flame space to the space as flame.txt
            scale_factor = 988.0130615234375
            rot_matrix = np.array([[ 0.99978507,  0.01133559,  0.01735884],
                                    [ 0.00832352, -0.9863116,   0.1646822 ],
                                    [ 0.01898799, -0.16450232, -0.98619395]],dtype=np.float32)
            translation = np.array([[-0.00102562, -0.01849389,  0.06861164]],dtype=np.float32)
            flame2faceModel_transform = np.eye(4,4)
            flame2faceModel_transform[:3,:3] = rot_matrix
            flame2faceModel_transform[:3,3] = translation.flatten()
            
            w2c_matrix = w2c_matrix@np.linalg.inv(flame2faceModel_transform)
            w2c_matrix[:3,3] *= scale_factor
            w2c_Rmat = w2c_matrix[:3,:3]
            w2c_Tvec = w2c_matrix[:3,3]
            
            g_eyes = g_eyes@rot_matrix.T
            g_face = g_face@rot_matrix.T
            
            landmark_3d_homo = np.concatenate([landmark_3d[0],np.ones((landmark_3d[0].shape[0],1))],axis=1)@flame2faceModel_transform.T
            landmark_3d = landmark_3d_homo[:,:3]*scale_factor 

            # landmark_3d_homo = np.concatenate([landmark_3d[0],np.ones((landmark_3d[0].shape[0],1))],axis=1)
            # landmark_3d = landmark_3d_homo[:,:3] 
                   
            landmark_3d_c = landmark_3d@w2c_Rmat.T+w2c_Tvec.reshape(1,3)
    
            landmark_2d_uv = landmark_3d_c@inmat.T 
            landmark_2d = landmark_2d_uv[:,:2]/(landmark_2d_uv[:,2,None]+1e-5)
            
            if self.gaze_points is not None:
                gc = self.gaze_points[frame_idx]
                # gc_homo = np.concatenate([gc,np.ones((1,))],axis=0)@flame2faceModel_transform.T
                # gc = gc_homo[:3]*scale_factor

                gc_homo = np.concatenate([gc,np.ones((1,))],axis=0)
                gc = gc_homo[:3]
                gc = gc@w2c_Rmat.T+w2c_Tvec.reshape(-1)
                gc_lst.append(gc)
            
            w2c_Rmats.append(w2c_Rmat)
            w2c_Tvecs.append(w2c_Tvec)
            inmats.append(inmat)
            g_eyes_lst.append(g_eyes)
            g_face_lst.append(g_face)
            landmark_3d_lst.append(landmark_3d)
            landmark_2d_lst.append(landmark_2d)
            
            
        w2c_Rmats = np.stack(w2c_Rmats,axis=0)
        w2c_Tvecs = np.stack(w2c_Tvecs,axis=0)
        inmats = np.stack(inmats,axis=0)
        g_eyes_lst = np.stack(g_eyes_lst,axis=0)
        g_face_lst = np.stack(g_face_lst,axis=0)
        if len(gc_lst)!=0:
            gc_lst = np.stack(gc_lst,axis=0)
        else:
            gc_lst = None
        landmark_3d_lst = np.stack(landmark_3d_lst,axis=0)
        landmark_2d_lst = np.stack(landmark_2d_lst,axis=0)
        
        self.result_raw = {'w2c_Rmats':w2c_Rmats,
                            'w2c_Tvecs':w2c_Tvecs,
                            'inmats':inmats,
                            'g_eyes':g_eyes_lst,
                            'g_face':g_face_lst,
                            'gc':gc_lst,
                            'landmark_3d':landmark_3d_lst,
                            'landmark_2d':landmark_2d_lst}
    
    def render_eye(self):
        if self.result_raw is None:
            self.parser_data()

        views_loader = DataLoader(self.scene.getInterpolateCameras(), batch_size=None, shuffle=False, num_workers=8)
        # max_threads = multiprocessing.cpu_count()
        max_threads = 1
        print('Max threads: ', max_threads)
        worker_args = []
        for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
            torch.cuda.empty_cache()
            if self.gaussians.binding != None:
                self.gaussians.select_mesh_by_timestep(view.timestep)
            # background = [1,1,1] if self.mp.white_background else [0,0,0]
            if args.white_background: 
                background = [1,1,1]
            else:
                background = np.random.randn(3)%1.0
                
            # background = np.random.randn(3)%1.0
            background = torch.tensor(background,dtype=torch.float32).cuda()
            # eye_gaussians = self.gaussians.get_sub_gaussian(type='eyeball')
            eye_gaussians = self.gaussians.get_sub_gaussian(type='half_eyeball') 

            rendering = render(view, eye_gaussians, pipeline, background)["render"]
            path2data = {}
            path2data[Path(self.render_eye_path) / f'{idx:05d}.png'] = rendering
            worker_args.append([path2data])

            if len(worker_args) == max_threads or idx == len(views_loader)-1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = [] 
            torch.cuda.empty_cache()
            
    def render_eye_region(self):
        if self.result_raw is None:
            self.parser_data()

        views_loader = DataLoader(self.scene.getInterpolateCameras(), batch_size=None, shuffle=False, num_workers=8)
        # max_threads = multiprocessing.cpu_count()
        max_threads = 1
        print('Max threads: ', max_threads)
        worker_args = []
        for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
            torch.cuda.empty_cache()
            if self.gaussians.binding != None:
                self.gaussians.select_mesh_by_timestep(view.timestep)
            # background = [1,1,1] if self.mp.white_background else [0,0,0]
            # background = [1,1,1]
            if args.white_background: 
                background = [1,1,1]
            else:
                background = np.random.randn(3)%1.0
            background = torch.tensor(background,dtype=torch.float32).cuda()
            eye_gaussians = self.gaussians.get_sub_gaussian(type='eye_region')
            rendering = render(view, eye_gaussians, pipeline, background)["render"]
            path2data = {}
            path2data[Path(self.render_eye_region_path) / f'{idx:05d}.png'] = rendering
            worker_args.append([path2data])

            if len(worker_args) == max_threads or idx == len(views_loader)-1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = []
            torch.cuda.empty_cache()
            

    @torch.no_grad()
    def render_raw(self):
        if self.result_raw is None:
            self.parser_data()

        views_loader = DataLoader(self.scene.getInterpolateCameras(), batch_size=None, shuffle=False, num_workers=8)
        max_threads = multiprocessing.cpu_count()
        print('Max threads: ', max_threads)
        worker_args = []
        for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
            if self.gaussians.binding != None:
                self.gaussians.select_mesh_by_timestep(view.timestep)
     
            background = np.random.randn(3)%1.0
            background = torch.tensor(background,dtype=torch.float32).cuda()
            rendering = render(view, self.gaussians, pipeline, background)["render"]
                
            path2data = {}
            path2data[Path(self.render_path) / f'{idx:05d}.png'] = rendering
            worker_args.append([path2data])

            if len(worker_args) == max_threads or idx == len(views_loader)-1:
                with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                    futures = [executor.submit(write_data, *args) for args in worker_args]
                    concurrent.futures.wait(futures)
                worker_args = []
            torch.cuda.empty_cache()
    
    def annotate_raw(self):
        params = self.result_raw
        if params is None:
            self.parser_data()
            self.render_raw()

        landmark_2ds = params['landmark_2d']
        render_list = os.listdir(self.render_path)
        render_list.sort()
        for i in tqdm(range(len(landmark_2ds)), desc="annotation raw images progress"):
            img = cv2.imread(os.path.join(self.render_path,render_list[i]))
            length = 0.5*min(img.shape[0],img.shape[1])
            if 'g_face' not in self.result_raw.keys():
                g_eyes = params['g_eyes'][i]
                g_eyes_c = g_eyes@params['w2c_Rmats'][i].T
                g = g_eyes_c[0]+g_eyes_c[1]
                g = g/np.linalg.norm(g,ord=2,axis=-1,keepdims=True)
            else:
                g = params['g_face'][i]@params['w2c_Rmats'][i].T
                g = g/np.linalg.norm(g,ord=2,axis=-1,keepdims=True)
                
            begin = params['landmark_2d'][i][30]
            end = (begin[0]+length*g[0],begin[1]+length*g[1])
            img = cv2.arrowedLine(img, (int(begin[0]),int(begin[1])), (int(end[0]), int(end[1])), (0, 0, 255), 5,8,0,0.3) 
            for point in landmark_2ds[i]:
                cv2.circle(img,(int(point[0]),int(point[1])),radius=1,color=(0,0,255),thickness=1)
            cv2.imwrite(os.path.join(self.annotation_path,render_list[i]),img)    
    
    def normalize_data(self,center_pos = 'face'):
        params = self.result_raw
        if params is None:
            self.parser_data()
            self.render_raw()

        landmark_2ds = params['landmark_2d']
        render_list = os.listdir(self.render_path)
        render_list.sort()
        landmark_2d_lst = []
        g_lst = []
        h_lst = []
        for i in tqdm(range(len(landmark_2ds)), desc="normalize images progress"):
            img = cv2.imread(os.path.join(self.render_path,render_list[i]))
            landmark_2d = params['landmark_2d'][i]
            cam = params['inmats'][i]

            if self.dataset_type == 'columbia':
                cam[[0,1],[0,1]] = 102400
           
            if self.dataset_type != 'mpii':
                hr, ht = estimateHeadPose(landmark_2d[[36,39,42,45,31,35]],self.face_model,cam,None)
            else:
                hr, ht = estimateHeadPose(landmark_2d[[36,39,42,45,48,54]],self.face_model,cam,None)
            if center_pos =='face':
                g = params['g_face'][i]@params['w2c_Rmats'][i].T
            elif center_pos == 'eyes':
                g = params['g_eyes'][i]@params['w2c_Rmats'][i].T
                g = g[0]+g[1]
              
            gc = g/np.linalg.norm(g,ord=2,axis=-1,keepdims=True)
            img_warped, hr_norm, gc_normalized, det_point_warped, R  = \
                normalizeData_face(img,self.face_model,landmark_2d,hr,ht,gc,cam,is_gaze_vector=True,center=center_pos)

            cv2.imwrite(os.path.join(self.normal_path,render_list[i]),img_warped)    
            

            hr_R,_ = cv2.Rodrigues(hr_norm)
            head_direction = hr_R@np.array([0,0,1],dtype=np.float32).reshape(3,1)
            head_direction = head_direction.reshape(-1)


            landmark_2d_lst.append(det_point_warped)
            g_lst.append(gc_normalized)
            h_lst.append(head_direction)
        
        landmark_2d_lst = np.stack(landmark_2d_lst,axis=0)
        g_lst = np.stack(g_lst,axis=0)
        h_lst = np.stack(h_lst,axis=0)
        self.result_norm = {
                    'landmark_2d':landmark_2d_lst,
                    'g':g_lst,
                    'h':h_lst
                    }
    
    def annotate_normal(self):
        params = self.result_norm
        if params is None:
            self.normalize_data()

        landmark_2ds = params['landmark_2d']
        render_list = os.listdir(self.render_path)
        render_list.sort()
        labels = []
        for i in tqdm(range(len(landmark_2ds)), desc="annotation normalized images progress"):
            img = cv2.imread(os.path.join(self.normal_path,render_list[i]))
            length = 1.5*min(img.shape[0],img.shape[1])
            g = params['g'][i]
            gaze_theta = np.arcsin((-1) * g[1])
            gaze_phi = np.arctan2((-1) * g[0], (-1) * g[2])
            gaze_pitchyaw = np.asarray([gaze_theta, gaze_phi])
            gaze_pitchyaw = gaze_pitchyaw * 180 / np.pi
     
            begin = (img.shape[1]//2,img.shape[0]//2)
                
            end = (begin[0]+length*g[0],begin[1]+length*g[1])
            img = cv2.arrowedLine(img, (int(begin[0]),int(begin[1])), (int(end[0]), int(end[1])), (0, 0, 255), 2,8,0,0.15) 
            for point in landmark_2ds[i]:
                cv2.circle(img,(int(point[0]),int(point[1])),radius=1,color=(0,255,0),thickness=1)
            # img = cv2.pyrUp(img)
            # img = cv2.pyrUp(img)
            # cv2.putText(img,"({:.1f},{:.1f})".format(gaze_pitchyaw[0],gaze_pitchyaw[1]),(0,20),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0))
            labels.append(gaze_pitchyaw)
            cv2.imwrite(os.path.join(self.normal_anno_path,render_list[i]),img)
 
    
    def output_h5(self):
        """ output synthetic dataset"""
        imglist = os.listdir(self.normal_path)
        imglist.sort()
        
        face_gazes = self.result_norm['g']
        N = len(face_gazes)
        assert N == len(face_gazes)
        resolution = cv2.imread(os.path.join(self.normal_path,imglist[0])).shape[:2]
        
        # output_h5_id = h5py.File(os.path.join(self.save_root,'train.h5'), "w")
        output_h5_id = h5py.File(os.path.join(args.output_h5_path,'{}.h5'.format(self.model_name)), "w")
        output_face_patch = output_h5_id.create_dataset(
            name="face_patch",
            shape=(N,resolution[0],resolution[1],3),
            compression="lzf",
            dtype=np.uint8,
            chunks= (1,resolution[0],resolution[1],3)
        )
        output_face_gaze = output_h5_id.create_dataset(
            name="face_gaze",
            shape=(N,2),
            compression="lzf",
            dtype=np.float32,
            chunks=(1,2)
        )
        output_head_pose = output_h5_id.create_dataset(
            name="head_pose",
            shape=(N,2),
            compression="lzf",
            dtype=np.float32,
            chunks=(1,2)
        )
        
        
        for i in tqdm(range(N),desc="output train set to h5 file"):
            imgfile = imglist[i]
            img = cv2.imread(os.path.join(self.normal_path,imgfile))
            img = cv2.resize(img,resolution)
            
            # convert gaze 3d to pitch yaw
            gc_normalized = face_gazes[i].reshape(-1)
            head_vec = self.result_norm['h'][i].reshape(-1)
            # if i == 0:
            #     print(gc_normalized)
            gaze_theta = np.arcsin((-1) * gc_normalized[1])
            gaze_phi = np.arctan2((-1) * gc_normalized[0], (-1) * gc_normalized[2])
            gaze_pitchyaw = np.asarray([gaze_theta, gaze_phi])

            head_vec = head_vec.reshape(-1)
            head_theta = np.arcsin((-1) * head_vec[1])
            head_phi = np.arctan2((-1) * head_vec[0], (-1) * head_vec[2])
            head_pitchyaw = np.asarray([head_theta, head_phi])

            
            output_face_patch[i] = img
            output_face_gaze[i] = gaze_pitchyaw
            output_head_pose[i] = head_pitchyaw
        
        output_h5_id.close()
 
if __name__ == '__main__':
    # 固定seed
    seed = 621
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description="Simulator script parameters")
    parser.add_argument('--data_size', type=int,default=100)
    parser.add_argument('--add_kappa', action="store_true")
    parser.add_argument('--load_light', action="store_true")
    parser.add_argument('--render_eye', action="store_true")
    parser.add_argument('--anno', action="store_true")
    parser.add_argument('--output_h5', action="store_true")
    parser.add_argument('--dataset_type', type=str,choices=['mpii','columbia','eve','xgaze'],required=True)
    parser.add_argument('--output_h5_path', type=str, required=True)

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    if os.path.exists(args.output_h5_path) == False:
        os.makedirs(args.output_h5_path)

    simulator = Simulator(model.extract(args),pipeline.extract(args),datasize=args.data_size,dataset_type=args.dataset_type)

    simulator.render_standard_pose(annotate=False)

    if args.dataset_type == 'mpii':
        simulator.create_view_frustum_from_data(0.35*1000,0.5*1000) # for MPII, place the screen at a distance of ​​0.35 to 0.5 meters in front of face
        # simulator.create_view_frustum_from_data(999*1000,1000*1000) # for MPII, place the screen at a distance of ​​0.35 to 0.5 meters in front of face
    elif args.dataset_type == 'xgaze':
        simulator.create_view_frustum_from_data(0.35*1000,0.5*1000) # for xgaze,0.35 to 0.5 meters
    elif args.dataset_type == 'columbia':
        simulator.create_view_frustum_from_data(2.5*1000,2.51*1000) # for Columbia,2.5 meters
    elif args.dataset_type == 'eve':
        simulator.create_view_frustum_from_data(0.95*1000,1.0*1000) # for EVE,0.95 to 1 meters
        
    simulator.generate_gaze_points_by_frustum()
    simulator.parser_data()
    
    if args.render_eye:
        simulator.render_eye()
        simulator.render_eye_region()
    
    simulator.render_raw()
    simulator.normalize_data(center_pos='face')
    if args.anno:
        simulator.annotate_normal()
    simulator.output_h5()  