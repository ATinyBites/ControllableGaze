import numpy as np
import cv2
import torch
from pytorch3d.transforms import axis_angle_to_matrix
import torchvision

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

# normalization function for the face images
def normalizeData_face(img, face_model, landmarks, hr, ht, gc, cam, is_gaze_vector=False,center='face'):
    assert center in ['face','eyes']
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    # distance_norm = 300  # normalized distance between eye and camera,for eve
    distance_norm = 600 # for eth-xgaze
    roiSize = (224,224)
    # roiSize = (200,200)
    # roiSize = (448, 448)  # size of cropped eye image
    # roiSize = (386,386)  # size of cropped eye image
    # roiSize = (400,400)  # size of cropped eye image
    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

    ## ---------- normalize rotation ----------
    hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    ## ---------- normalize gaze vector ----------
    if not is_gaze_vector:
        if center == 'face':
            gc_normalized = gc - face_center  # gaze vector
        else:
            gc_normalized = gc - two_eye_center
    else:
        gc_normalized = gc
    gc_normalized = np.dot(R, gc_normalized)
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

    # warp the facial landmarks
    num_point, num_axis = landmarks.shape
    det_point = landmarks.reshape([num_point, 1, num_axis])
    det_point_warped = cv2.perspectiveTransform(det_point, W)
    det_point_warped = det_point_warped.reshape(num_point, num_axis)

    return img_warped, hr_norm, gc_normalized, det_point_warped, R

def normalizeData_Eye(img, face, hr, ht, gc, landmarks, cam,is_gaze_vector=False):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm = 600 # normalized distance between eye and camera
    roiSize = (60, 36) # size of cropped eye image

    # img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_u = img

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    if not is_gaze_vector:
        gc = gc.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face.T) + ht # 3D positions of facial landmarks
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    
    ## normalize each eye
    data = []
    for i,et in enumerate([re, le]):
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T # rotation matrix R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix
        img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
        # img_warped = cv2.equalizeHist(img_warped)
        
        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR) # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors
        
        ## ---------- normalize gaze vector ----------
        if not is_gaze_vector:
            gc_normalized = gc - et # gaze vector
        else:
            gc_normalized = gc[i].reshape(3,1)
        # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        # For original data normalization, here should be:
        # "M = np.dot(S,R)
        # gc_normalized = np.dot(R, gc_normalized)"
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)


        # warp the facial landmarks
        num_point, num_axis = landmarks.shape
        det_point = landmarks.reshape([num_point, 1, num_axis])
        det_point_warped = cv2.perspectiveTransform(det_point, W)
        det_point_warped = det_point_warped.reshape(num_point, num_axis)

        data.append([img_warped, hr_norm, gc_normalized,det_point_warped])
        
    return data


# normalization function for the eye images
def normalizeData(img, face_model, hr, ht, gc, cam):
    ## normalized camera parameters
    focal_norm = 1800  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (128, 128)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))  # center of left eye
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))  # center of right eye

    ## normalize each eye
    data = []
    for et in [re, le]:
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et)  # actual distance between eye and original camera

        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

        img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        ## ---------- normalize gaze vector ----------
        gc_normalized = gc - et  # gaze vector
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        data.append([img_warped, hr_norm, gc_normalized, R])

    return data

def getNormalizeAffineMatrix(face, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm = 600 # normalized distance between eye and camera
    roiSize = (60, 36) # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face.T) + ht # 3D positions of facial landmarks
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    
    ## normalize each eye
    data = []
    for i,et in enumerate([re, le]):
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T # rotation matrix R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix
        data.append(W)
    data = np.stack(data,axis=0)
    return data

def normalizeData_Eye_tensor(img_tensor, face, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm = 600 # normalized distance between eye and camera
    roiSize = (60, 36) # size of cropped eye image
        

    # img_u = img_tensor*torch.tensor([0.299,0.587,0.114],dtype=torch.float32,device=img_tensor.device).reshape(3,1,1)
    # img_u = torch.sum(img_u,dim=0) # [W,H]
    img_u = img_tensor

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    hR = axis_angle_to_matrix(hr.reshape(1,3))[0] # rotation matrix
    Fc = hR @ face.T + ht # 3D positions of facial landmarks
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    
    ## normalize each eye
    data = []
    for i,et in enumerate([re, le]):
        ## ---------- normalize image ----------
        distance = torch.norm(et,p=2,dim=0) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = torch.tensor([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ],dtype=torch.float32,device=img_tensor.device)
        S = torch.tensor([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ],dtype=torch.float32,device=img_tensor.device)
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = torch.cross(forward, hRx)
        down /= torch.norm(down,p=2,dim=0)
        right = torch.cross(down, forward)
        right /= torch.norm(right,p=2,dim=0)
        R = torch.stack([right,down,forward],dim=1).T # rotation matrix R
        W = cam_norm @ S@ R @ torch.linalg.inv(cam)# transformation matrix
        W = torch.linalg.inv(W)
        # img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
        # img_warped = cv2.equalizeHist(img_warped)
        
        _H, _W = img_tensor.shape[1], img_tensor.shape[2]

        grid_y, grid_x = torch.meshgrid(torch.arange(0, roiSize[1]), torch.arange(0, roiSize[0]))
        grid = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x.flatten())), dim=0).float().cuda()  # [3, H*W]
        warped_grid = W @ grid
        warped_grid = warped_grid/ warped_grid[2, :]  # Normalize by the third coordinate
        warped_grid = warped_grid[:2, :].reshape(2, roiSize[1], roiSize[0])  # [2, H, W]
        # warped_grid += torch.tensor(([_W//2,_H//2]),dtype=torch.float32,device=W.device).reshape(2,1,1)
        warped_grid = warped_grid.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 2]

        # Normalize grid to [-1, 1]
        warped_grid = 2 * warped_grid / torch.tensor([_W-1, _H-1], dtype=torch.float32).cuda() - 1
        imageWarped = torch.nn.functional.grid_sample(img_u[None], warped_grid.cuda()[...,[0,1]], align_corners=False).squeeze(0)
        # imageWarped = equalize_hist_pytorch(imageWarped)
        if i==0:
            # imageWarped = imageWarped[:,:,::-1] # flip right eye
            imageWarped = torchvision.transforms.functional.hflip(imageWarped)
        # data.append(imageWarped.permute(1,2,0)[224-18:224+18,224-30:224+30])
        data.append(imageWarped.permute(1,2,0).clone().squeeze(-1))
    data = torch.stack(data,dim=0)
    # data[0] =data[1]
    return data

# def normalizeData_face_tensor(img_tensor, face, hr, ht, cam):
#     ## normalized camera parameters
#     focal_norm = 960 # focal length of normalized camera
#     distance_norm = 600 # normalized distance between eye and camera
#     roiSize = (224, 224) # size of cropped eye image
        

#     img_u = img_tensor*torch.tensor([0.299,0.587,0.114],dtype=torch.float32,device=img_tensor.device).reshape(3,1,1)
#     img_u = torch.sum(img_u,dim=0) # [W,H]

#     ## compute estimated 3D positions of the landmarks
#     ht = ht.reshape((3,1))
#     hR = axis_angle_to_matrix(hr.reshape(1,3))[0] # rotation matrix
#     Fc = hR @ face.T + ht # 3D positions of facial landmarks
#     face_center = torch.mean(Fc,dim=-1).reshape(3,1)


#     ## ---------- normalize image ----------
#     distance = torch.norm(face_center,p=2,dim=0) # actual distance between eye and original camera
    
#     z_scale = distance_norm/distance
#     cam_norm = torch.tensor([
#         [focal_norm, 0, roiSize[0]/2],
#         [0, focal_norm, roiSize[1]/2],
#         [0, 0, 1.0],
#     ],dtype=torch.float32,device=img_tensor.device)
#     S = torch.tensor([ # scaling matrix
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#         [0.0, 0.0, z_scale],
#     ],dtype=torch.float32,device=img_tensor.device)
    
#     hRx = hR[:,0]
#     forward = (face_center/distance).reshape(3)
#     down = torch.cross(forward, hRx)
#     down /= torch.norm(down,p=2,dim=0)
#     right = torch.cross(down, forward)
#     right /= torch.norm(right,p=2,dim=0)
#     R = torch.stack([right,down,forward],dim=1).T # rotation matrix R
#     W = cam_norm @ S@ R @ torch.linalg.inv(cam)# transformation matrix
#     W = torch.linalg.inv(W)
    
#     _H, _W = img_tensor.shape[1], img_tensor.shape[2]

#     grid_y, grid_x = torch.meshgrid(torch.arange(0, roiSize[1]), torch.arange(0, roiSize[0]))
#     grid = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x.flatten())), dim=0).float().cuda()  # [3, H*W]
#     warped_grid = W @ grid
#     warped_grid = warped_grid/ warped_grid[2, :]  # Normalize by the third coordinate
#     warped_grid = warped_grid[:2, :].reshape(2, roiSize[1], roiSize[0])  # [2, H, W]
#     # warped_grid += torch.tensor(([_W//2,_H//2]),dtype=torch.float32,device=W.device).reshape(2,1,1)
#     warped_grid = warped_grid.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 2]

#     # Normalize grid to [-1, 1]
#     warped_grid = 2 * warped_grid / torch.tensor([_W-1, _H-1], dtype=torch.float32).cuda() - 1
#     imageWarped = torch.nn.functional.grid_sample(img_u[None,None], warped_grid.cuda()[...,[0,1]], align_corners=False).squeeze(0)

#     return imageWarped

def equalize_hist_pytorch(img):
    """
    PyTorch implementation of histogram equalization for images in the range [0, 1].
    :param img: Tensor of shape (C, H, W), image in the range [0, 1]
    :return: Tensor with equalized histogram, range [0, 1]
    """
    # Ensure the input image is in the range [0, 1]
    img = torch.clamp(img, 0, 1)

    # Convert to 0-255 range (as integer values)
    img_255 = (img * 255).to(torch.long)
    
    # Flatten the image to compute the histogram
    img_flat = img_255.view(-1)
    
    # Compute the histogram
    hist = torch.bincount(img_flat, minlength=256).float()

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum(dim=0)
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    # Map the pixel values to equalized values using CDF
    img_equalized_255 = torch.gather(cdf_normalized,dim=0,index=img_flat).reshape(*img.shape)
    
    # Convert back to 0-1 range
    img_equalized = img_equalized_255.float()
    
    return img_equalized