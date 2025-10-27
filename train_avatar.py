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
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim,l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
import random
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
import numpy as np
import cv2 as cv
import copy
from piq import LPIPS
from utils.normalize_utils import normalizeData_Eye_tensor,estimateHeadPose,normalizeData_Eye
from flame_model.lbs import vertices2landmarks

def tensor2image(tensor_image):
    image= tensor_image.cpu().detach().numpy()*255
    if len(tensor_image.shape) == 3 and tensor_image.shape[0] ==3:
        image = image.transpose(1,2,0).astype(np.uint8)[...,[2,1,0]]
    else:
        image = image.astype(np.uint8)
    return image.copy()

def save_image(path,tensor_image,ldms_2d=None):
        image= tensor_image.cpu().detach().numpy()*255
        if len(tensor_image.shape) == 3 and tensor_image.shape[0] ==3:
            image = image.transpose(1,2,0).astype(np.uint8)[...,[2,1,0]]
        image = image.astype(np.uint8).copy()
        if ldms_2d is not None:
            for ldm in ldms_2d:
                image = cv.circle(image,center=ldm,radius=1,color=(0,0,255),thickness=1)
                
        cv.imwrite(path,image)
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    prepare_output(args)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,z_rotate=True,load_light=args.load_light)
    else: 
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    cameradataset = copy.deepcopy(scene.getTrainCameras())
    viewpoint_cam = cameradataset[0]

    loader_camera_train = DataLoader(cameradataset, batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    face_model = np.array([ -45.096768, -21.312858, 21.312858, 
            45.096768, -26.299577, 26.299577, -0.483773, 
            0.483773, 0.483773, -0.483773, 68.595035, 
            68.595035, 2.397030, -2.397030, -2.397030, 
            2.397030, -0.000000, -0.000000],dtype=np.float32).reshape(3,6).T
    
    lpips = LPIPS()
    
    for iteration in range(first_iter, opt.iterations + 1):       
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        
        
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep,True)

        # Render
        background = torch.randn((3,)).cuda()%1.0
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = torch.clamp(image,min=0,max=1.0)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        mask = torch.sum(gt_image,dim=0)<1e-5
        gt_image[:,mask] = background[:,None] 
        
        """calculate image loss"""
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
        
        """calculate eye geometry loss"""
        # get eye_mask gt by landmarks
        with torch.no_grad():
            ldms_3d = vertices2landmarks(
                gaussians.verts,
                gaussians.flame_model.faces,
                gaussians.flame_model.mp_full_lmk_faces_idx.repeat(1, 1),
                gaussians.flame_model.mp_full_lmk_bary_coords.repeat(1, 1, 1),
            ).cpu().numpy()
            le_ldms_idx = [20,33,23,24,25,26,27,22,34,28,29,30,31,32,35]
            re_ldms_idx = [36,49,39,40,41,42,43,50,44,45,46,47,48,51,37]
            ldms_2d = cv.projectPoints(ldms_3d,viewpoint_cam.rvec,viewpoint_cam.tvec,viewpoint_cam.inmat,distCoeffs=None)[0].reshape(-1,2)

            left_eye_mask = np.zeros((viewpoint_cam.image_height,viewpoint_cam.image_width),dtype=np.uint8)
            left_eye_mask = cv.fillPoly(left_eye_mask,[ldms_2d[le_ldms_idx].astype(np.int32)],color=(255,))
            right_eye_mask = np.zeros((viewpoint_cam.image_height,viewpoint_cam.image_width),dtype=np.uint8)
            right_eye_mask = cv.fillPoly(right_eye_mask,[ldms_2d[re_ldms_idx].astype(np.int32)],color=(255,))
            eye_mask_gt = np.logical_or(left_eye_mask>0,right_eye_mask>0)
            eye_mask_gt = torch.from_numpy(eye_mask_gt.astype(np.float32)).cuda()

        # render eye mask prediction
        eye_gaussians_idx = gaussians.get_eye_guassians_idx(type='half_eyeball')
        override_color = torch.zeros_like(gaussians._xyz)
        override_color[eye_gaussians_idx,:] = 1.0
        eye_render_pkg = render(viewpoint_cam, gaussians , pipe, torch.zeros_like(background),override_color=override_color)
        eye_mask_render = torch.mean(eye_render_pkg["render"],dim=0)
        
        # add eye geometry loss
        losses['eye_mask'] = l2_loss(eye_mask_render,eye_mask_gt)*opt.lambda_eye_mask

        """calculate eye appearance loss"""
        # crop eye patches
        ldms_3d = gaussians.ldms_3d.detach().cpu().numpy()
        ldms_2d = cv.projectPoints(ldms_3d,viewpoint_cam.rvec,viewpoint_cam.tvec,viewpoint_cam.inmat,distCoeffs=None)[0].reshape(-1,2)
        ldms_2d = np.concatenate([ldms_2d,ldms_2d+np.array([viewpoint_cam.image_width,0]).reshape(1,2)],axis=0)
        hr,ht = estimateHeadPose(ldms_2d[[36,39,42,45,48,54]],face_model,viewpoint_cam.inmat,None)
        
        # eye_normed_pred = normalizeData_Eye_tensor(image*eye_mask_gt[None], torch.from_numpy(face_model).cuda(),torch.from_numpy(hr).float().cuda(),torch.from_numpy(ht).float().cuda(),torch.from_numpy(viewpoint_cam.inmat).float().cuda()).permute(0,3,1,2)
        # eye_normed_gt = normalizeData_Eye_tensor(gt_image*eye_mask_gt[None], torch.from_numpy(face_model).cuda(),torch.from_numpy(hr).float().cuda(),torch.from_numpy(ht).float().cuda(),torch.from_numpy(viewpoint_cam.inmat).float().cuda()).permute(0,3,1,2)

        eye_normed_pred = normalizeData_Eye_tensor(image, torch.from_numpy(face_model).cuda(),torch.from_numpy(hr).float().cuda(),torch.from_numpy(ht).float().cuda(),torch.from_numpy(viewpoint_cam.inmat).float().cuda()).permute(0,3,1,2)
        eye_normed_gt = normalizeData_Eye_tensor(gt_image, torch.from_numpy(face_model).cuda(),torch.from_numpy(hr).float().cuda(),torch.from_numpy(ht).float().cuda(),torch.from_numpy(viewpoint_cam.inmat).float().cuda()).permute(0,3,1,2)
        
        # add eye appearance loss
        losses['l1'] = losses['l1'] + l1_loss(eye_normed_pred, eye_normed_gt) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = losses['ssim'] + (1.0-ssim(eye_normed_pred, eye_normed_gt)) * opt.lambda_dssim
        losses['lpips'] = lpips(eye_normed_pred, eye_normed_gt)*opt.lambda_lpips

        """calculate regularization loss"""
        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        
        losses['total'] = sum([v for k, v in losses.items()]) 
        losses['total'].backward()

        iter_end.record()


        # visualize training process
        if iteration%100==99: 
            print("gaussian nums:",len(gaussians._xyz))

            eye_image_gt = gt_image.clone()
            eye_image_render = image.clone()
            eye_log = torch.concat([eye_image_gt,eye_image_render,eye_mask_gt[None].repeat(3,1,1),eye_mask_render[None].repeat(3,1,1)],dim=2)
            
            ldms_3d = gaussians.ldms_3d.detach().cpu().numpy()
            ldms_2d = cv.projectPoints(ldms_3d,viewpoint_cam.rvec,viewpoint_cam.tvec,viewpoint_cam.inmat,distCoeffs=None)[0].reshape(-1,2)
            ldms_2d = np.concatenate([ldms_2d,ldms_2d+np.array([viewpoint_cam.image_width,0]).reshape(1,2)],axis=0)
            
            save_image('./view_face.png',eye_log)
            save_image('./view_eye.png',torch.concatenate([torch.concatenate([eye_normed_gt[0],eye_normed_gt[1]],dim=1),
                                        torch.concatenate([eye_normed_pred[0],eye_normed_pred[1]],dim=1),
                                        ],dim=2))

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{5}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{5}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{5}f}"    
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # save
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() 

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    # set seed
    seed = 621
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000,60000,90000,100000,300000,500000,600000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_light",action="store_true")
    parser.add_argument("--z_rotate",action="store_true")
    
    
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
