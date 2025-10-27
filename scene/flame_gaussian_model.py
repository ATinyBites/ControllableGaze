# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from utils.transform_utils import matrix_to_quaternion,axis_angle_to_matrix,matrix_to_euler_angles,euler_angles_to_matrix,matrix_to_axis_angle,quaternion_to_matrix
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw,rotmat_to_unitquat
import pickle
import trimesh
import torch.nn as nn
from plyfile import PlyData, PlyElement

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class TinyGaussianModel:
    def setup_functions(self):
        def inverse_sigmoid(x):
            return torch.log(x/(1-x))
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log


        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        

    
    def __init__(self,sh_degree:int,xyz,quat,scale,opacity,feature) -> None:
        self.xyz = xyz
        self.quat = quat
        self.scale = scale
        self.opacity = opacity
        self.feature = feature
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.setup_functions()
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(3*((self.max_sh_degree+1))**2-3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.quat.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    @property
    def get_xyz(self):
        return self.xyz
    
    @property
    def get_scaling(self):
        return self.scale
    @property
    def get_rotation(self):
        return self.quat
    @property
    def get_opacity(self):
        return self.opacity
    
    @property
    def get_features(self):
        return self.feature
    
    def save_ply(self, path):
        opa_threshold = 0.001
        # mask = (self.get_opacity>opa_threshold).reshape(-1)
        mask = torch.ones((self.xyz.shape[0],),dtype=torch.bool)

        xyz = self.get_xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features = self.get_features

        f_dc = features[mask,0:1,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features[mask,1:,:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.inverse_opacity_activation(self.get_opacity[mask]).detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling[mask]).detach().cpu().numpy()
        rotation = self.get_rotation[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100,z_rotate=True, load_light=False):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr
 
        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda() 
        self.flame_param = None
        self.flame_param_orig = None

        self.zero_centered_at_root_node = False # when training gaussian Avatars Dataset,set False
        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

        with open("flame_model/assets/flame/FLAME_masks.pkl", "rb") as f:
            self.flame_mask_verts = pickle.load(f, encoding="latin1")
        
        self.flame_mask_faces = {}
        for key,value in self.flame_mask_verts.items():
            faces_idx = self.flame_model.faces.reshape(-1,1,3)==torch.tensor(value,device=self.flame_model.faces.device).reshape(1,-1,1)
            faces_idx = torch.any(faces_idx[:,:,0],dim=-1)
            faces_idx = torch.where(faces_idx)[0]
            self.flame_mask_faces[key] = faces_idx
            self.flame_mask_verts[key] = torch.tensor(value,device=self.flame_model.faces.device)
        self.eye_faces = torch.concat([self.flame_mask_faces['left_eyeball'],self.flame_mask_faces['right_eyeball']],dim=0)
        self.random_z_rotation = z_rotate
        
        self.faces_center_cano = None
        self.faces_orient_mat_cano = None
        self.faces_scaling_cano = None
        self.ldms_3d = None

        self.load_light = load_light
              
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        meshes = {**train_meshes, **test_meshes}
        tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes

        self.num_timesteps = max(pose_meshes) + 1  # required by viewers
    
        num_verts = self.flame_model.v_template.shape[0]

        for key in meshes.keys():
                break
            
        if not self.disable_flame_static_offset:
            
            static_offset = torch.from_numpy(meshes[key]['static_offset']).reshape(num_verts,3)
            if static_offset.shape[0] != num_verts:
                static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[key]['static_offset'].shape[1]))
        else:
            static_offset = torch.zeros([num_verts, 3])

        T = self.num_timesteps
        self.flame_param = {
            'shape': torch.from_numpy(meshes[key]['shape']),
            'expr': torch.zeros([T, meshes[key]['expr'].shape[1]]),
            'rotation': torch.zeros([T, 3]),
            'neck_pose': torch.zeros([T, 3]),
            'jaw_pose': torch.zeros([T, 3]),
            'eyes_pose': torch.zeros([T, 6]),
            'translation': torch.zeros([T, 3]),
            'static_offset': static_offset,
            'dynamic_offset': torch.zeros([T, num_verts, 3]),
            'light': torch.zeros([T, 9,3]),
        }

        has_light_key = False
        for i, mesh in pose_meshes.items():
            self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
            self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
            self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
            self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
            self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
            if self.load_light:
                if 'light' in mesh.keys():
                    has_light_key = True
                    self.flame_param['light'][i,:9] = torch.from_numpy(mesh['light'])
                else:
                    self.flame_param['light'][i,:9] = torch.from_numpy(torch.zeros([9,3])) # zero initialize
                

            # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
        
        if self.load_light and has_light_key:
            self.flame_param['light'][:,:9] = (self.flame_param['light'][:,:9]-self.flame_param['light'][:,:9].min(dim=0,keepdim=True)[0])/(self.flame_param['light'][:,:9].max(dim=0,keepdim=True)[0]-self.flame_param['light'][:,:9].min(dim=0,keepdim=True)[0])

        for k, v in self.flame_param.items():
            self.flame_param[k] = v.float().cuda()
        
        self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=self.zero_centered_at_root_node, # gaussian Avatars Dataset is False
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, z_rotate=False,original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param
        if self.load_light:
            self.light = flame_param['light'][timestep] #5

        if not z_rotate:
            verts, verts_cano,ldms_3d, J = self.flame_model(
                flame_param['shape'][None, ...],
                flame_param['expr'][[timestep]],
                flame_param['rotation'][[timestep]],
                flame_param['neck_pose'][[timestep]],
                flame_param['jaw_pose'][[timestep]],
                flame_param['eyes_pose'][timestep].reshape(1,6),
                flame_param['translation'][[timestep]],
                zero_centered_at_root_node=self.zero_centered_at_root_node, 
                return_landmarks=True,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'],
                dynamic_offset=flame_param['dynamic_offset'][[timestep]],
                return_J = True  
            )
            self.eyes_pose = flame_param['eyes_pose'][timestep].reshape(-1,6)
        else: # apply random symmetry rotation
            eyes_pose = flame_param['eyes_pose'][timestep].reshape(-1,3)
            eyes_rot_matrix = axis_angle_to_matrix(eyes_pose)
            eyes_euler = matrix_to_euler_angles(eyes_rot_matrix,convention="XYZ")
            delta = torch.randint(0,2000,(len(eyes_euler),)).cuda()/2000.0 *360* torch.pi/180
            eyes_euler[:,2] = delta # to roll component
            eyes_rot_matrix_new = euler_angles_to_matrix(eyes_euler,"XYZ")
            eyes_pose_new = matrix_to_axis_angle(eyes_rot_matrix_new).reshape(-1,6)
            
            verts, verts_cano , ldms_3d, J = self.flame_model(
                flame_param['shape'][None, ...],
                flame_param['expr'][[timestep]],
                flame_param['rotation'][[timestep]],
                flame_param['neck_pose'][[timestep]],
                flame_param['jaw_pose'][[timestep]],
                eyes_pose_new.reshape(-1,6),
                flame_param['translation'][[timestep]],
                zero_centered_at_root_node=self.zero_centered_at_root_node, 
                return_landmarks=True,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'],
                dynamic_offset=flame_param['dynamic_offset'][[timestep]],
                return_J = True
            )
            self.eyes_pose = eyes_pose_new
            
        self.pupil_centers = ldms_3d[0,-2:][[-1,-2]]
        self.eyeball_centers = J[0,-2:] # left and right eye ball center
        self.ldms_3d = ldms_3d[0]
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]
        self.triangles = triangles
        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
        
        # if self.faces_center_cano is None:
        triangles_cano = verts_cano[:,faces]
        self.triangles_cano = triangles_cano
        self.faces_center_cano = triangles_cano.mean(dim=-2).squeeze(0)
        self.faces_orient_mat_cano, self.faces_scaling_cano = compute_face_orientation(verts_cano.squeeze(0), faces.squeeze(0), return_scale=True)

        
        left_eye_points = self.face_center[self.flame_mask_faces['left_eyeball']]
        right_eye_points = self.face_center[self.flame_mask_faces['right_eyeball']]
        self.left_eye_center_w = torch.mean(left_eye_points,dim=0)
        self.right_eye_center_w = torch.mean(right_eye_points,dim=0)
        
        left_eye_points = self.faces_center_cano[self.flame_mask_faces['left_eyeball']]
        right_eye_points = self.faces_center_cano[self.flame_mask_faces['right_eyeball']]
        self.left_eye_center = torch.mean(left_eye_points,dim=0)
        self.right_eye_center = torch.mean(right_eye_points,dim=0)
        

        left_dist = torch.norm(left_eye_points-self.left_eye_center[None],p=2,dim=-1)
        right_dist = torch.norm(right_eye_points-self.right_eye_center[None],p=2,dim=-1)

        left_radius = left_dist.mean()
        right_radius = right_dist.mean()

        iris_radius = 0.5*0.0126/(left_radius+right_radius)*2.0

        left_iris_selector = torch.logical_and(left_eye_points[:,2]-self.left_eye_center[None,2]>0,torch.norm(left_eye_points[:,:2]-self.left_eye_center[None,:2],p=2,dim=-1)<left_radius*2*iris_radius)
        right_iris_selector = torch.logical_and(right_eye_points[:,2]-self.right_eye_center[None,2]>0,torch.norm(right_eye_points[:,:2]-self.right_eye_center[None,:2],p=2,dim=-1)<right_radius*2*iris_radius)

        left_iris_face_idx = torch.where(left_iris_selector)[0]
        right_iris_face_idx = torch.where(right_iris_selector)[0]
        left_sclera_face_idx = torch.where(torch.logical_not(left_iris_selector))[0]
        right_sclera_face_idx = torch.where(torch.logical_not(right_iris_selector))[0]

        self.left_iris_face_idx = self.flame_mask_faces['left_eyeball'][left_iris_face_idx]
        self.right_iris_face_idx = self.flame_mask_faces['right_eyeball'][right_iris_face_idx]
        self.left_sclera_face_idx = self.flame_mask_faces['left_eyeball'][left_sclera_face_idx]
        self.right_sclera_face_idx = self.flame_mask_faces['right_eyeball'][right_sclera_face_idx]

    def concat_new_gaussians(self,binding,xyz,scale,quat,opacity,feature_dc,feature_rest):
        
        self._xyz = nn.Parameter(torch.concat([self._xyz,xyz],dim=0))
        self._features_dc = nn.Parameter(torch.concat([self._features_dc,feature_dc],dim=0))
        self._features_rest = nn.Parameter(torch.concat([self._features_rest,feature_rest],dim=0))
        self._scaling = nn.Parameter(torch.concat([self._scaling,scale],dim=0))
        self._rotation = nn.Parameter(torch.concat([self._rotation,quat],dim=0))
        self._opacity = nn.Parameter(torch.concat([self._opacity,opacity],dim=0))
        self.binding = torch.concat([self.binding,binding],dim=0)

    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.not_finetune_flame_params:
            return
        
        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)
        
        # light
        if self.load_light:
            self.flame_param['light'].requires_grad = True
            param_light = {'params': [self.flame_param['light']], 'lr': 1e-4, "name": "light"}
            self.optimizer.add_param_group(param_light)


    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param['translation'] = flame_param['translation']
            self.flame_param['rotation'] = flame_param['rotation']
            self.flame_param['neck_pose'] = flame_param['neck_pose']
            self.flame_param['jaw_pose'] = flame_param['jaw_pose']
            self.flame_param['eyes_pose'] = flame_param['eyes_pose']
            self.flame_param['expr'] = flame_param['expr']
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]

    @property
    def get_xyz(self):
        if self.binding is None:
            xyz =  self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
            
            xyz = self._xyz
            xyz = torch.bmm(self.face_orien_mat[self.binding], xyz[..., None]).squeeze(-1)
            xyz = xyz * self.face_scaling[self.binding] + self.face_center[self.binding]

        return xyz

    @property
    def get_cano_xyz(self):
 
        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        if self.face_center is None:
            self.select_mesh_by_timestep(0)
        
        xyz = self._xyz
        xyz = torch.bmm(self.faces_orient_mat_cano[self.binding], xyz[..., None]).squeeze(-1)
        xyz = xyz * self.faces_scaling_cano[self.binding] + self.faces_center_cano[self.binding]
        return xyz

    @property
    def get_features(self):
        features_dc = self._features_dc+0.0 # [-1,1,3]
        features_rest = self._features_rest+0.0
        
        if self.load_light:
            light_embedding = self.light.flatten()
            pos_embedding = self._features_dynamic
            embedding = torch.concat([pos_embedding,light_embedding[None].repeat(len(pos_embedding),1)],dim=-1)
            features_dc_ = RGB2SH(self.ColorDecoder(embedding)+0.0)[:,None] # [N,3]
            features_dc = features_dc_
                
        features = torch.cat((features_dc, features_rest), dim=1)
        return features
     
    def get_sub_gaussian(self,type="eyeball",z_rotate=False):
        # assert type in ["eyeball","eye_region"]
        if self.face_center is None:
            self.select_mesh_by_timestep(0,z_rotate)
        elif self.timestep is not None:
            self.select_mesh_by_timestep(self.timestep,z_rotate)
        
        binding = self.get_eye_guassians_idx(type)

        xyz = self.get_xyz[binding]
        scaling = self.get_scaling[binding]
        quat = self.get_rotation[binding]
        features = self.get_features[binding]
        opacity = self.get_opacity[binding]
        return TinyGaussianModel(self.max_sh_degree,xyz,quat,scaling,opacity,features)
    
    @torch.no_grad()
    def get_eye_guassians_idx(self,type='eyeball'):
        if self.face_center is None:
            self.select_mesh_by_timestep(0)
        if type == 'eyeball':
            binding =  self.binding.reshape(-1,1) == self.eye_faces.reshape(1,-1)
        elif type == 'iris':
            binding =  self.binding.reshape(-1,1) == torch.concat([self.left_iris_face_idx,self.right_iris_face_idx],dim=0).reshape(1,-1)
        elif type == 'sclera':
            binding =  self.binding.reshape(-1,1) == torch.concat([self.left_sclera_face_idx,self.right_sclera_face_idx],dim=0).reshape(1,-1)
        elif type == 'eye_region':
            binding =  self.binding.reshape(-1,1) == torch.concat([self.flame_mask_faces['left_eye_region'],self.flame_mask_faces['right_eye_region']],dim=0).reshape(1,-1)
        elif type == 'eye_valid':
            binding = self.binding.reshape(-1,1) == torch.concat([self.eye_faces.flatten(),self.flame_mask_faces['left_eye_region'],self.flame_mask_faces['right_eye_region']],dim=0).reshape(1,-1)
        elif type == 'half_eyeball':
            verts_cano = self.verts_cano.squeeze()
            face_cano = verts_cano[self.flame_model.faces]
            face_center = torch.mean(face_cano,dim=-2)
            eyeball_face_center = face_center[self.eye_faces.flatten()].reshape(-1,2,3)
            eyeball_face_center = eyeball_face_center-self.eyeball_centers.reshape(1,2,3)
            eyeball_face_selector = eyeball_face_center[:,:,2].reshape(-1,)>0
            half_eyeball_faces = self.eye_faces.flatten()[eyeball_face_selector]
            binding = self.binding.reshape(-1,1) == half_eyeball_faces
        elif type == 'left_eyeball':
            binding =  self.binding.reshape(-1,1) == self.flame_mask_faces['left_eyeball'].reshape(1,-1)
        elif type == 'right_eyeball':
            binding =  self.binding.reshape(-1,1) == self.flame_mask_faces['right_eyeball'].reshape(1,-1)
        elif type == 'all':
            binding = torch.ones_like(self.binding.reshape(-1,1))


            
        binding = torch.any(binding,dim=-1)
        binding = torch.where(binding)[0]
            
        return binding
             
    def get_eye_params(self):
        center = self.eyeball_centers
        left_xyz = torch.mean(self.triangles[0,self.eye_faces[:len(self.eye_faces)//2]],dim=-2)-center[0]
        radius = torch.mean(torch.norm(left_xyz,p=2,dim=-1)).item()
        eyes_rotation = axis_angle_to_matrix(self.eyes_pose.reshape(2,3))
        
        return {'center':center,
                'radius':radius,
                'eyes_rotation':eyes_rotation}

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # do not prune the eye gaussians
        # eye_selector = self.get_eye_guassians_idx()
        # prune_mask[eye_selector] = False

        # if self.load_light:
        #     prune_mask = torch.zeros_like(prune_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()