import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from gaze_estimation.model import gaze_network
import h5py
from gaze_estimation.data_loader import GazeDataset
from torch.utils.data import Dataset, DataLoader

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)

def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def calculate_angular_distance( a, b):
    a = pitchyaw_to_vector(a)
    b = pitchyaw_to_vector(b)

    sim = torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-8)
    sim = torch.nn.functional.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
    return torch.acos(sim) * 180/torch.pi


if __name__ == '__main__':
    # test mpii
    pre_trained_model_path = "ckpt/mpii/epoch_24_ckpt.pth.tar"
    test_dir = 'dataset/mpii_test.h5'
    
    # test columbia
    #pre_trained_model_path = "ckpt/columbia/epoch_24_ckpt.pth.tar"
    #test_dir = 'dataset/columbia_test.h5'

    print('gaze estimator:', pre_trained_model_path)
    print("test dataset:",test_dir)
    model = gaze_network()
    ckpt = torch.load(pre_trained_model_path)
    model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    model.cuda()

    dist_list = []
    hdf_file = h5py.File(test_dir, 'r', swmr=True)
    for subject in hdf_file.keys():
        dataset = GazeDataset(hdf_file[subject],transform=trans) 
        dataloader = DataLoader(dataset,100,shuffle=False,drop_last=False,num_workers=4)
        loss_list = []
        pred_list = []
        gt_list = []
        with torch.no_grad():
            for images,gt_gaze in dataloader:
                input_var = images.cuda()  # the input must be 4-dimension
                pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
                
                if len(pred_list)==0:
                    pred_list = pred_gaze.clone()
                    gt_list = gt_gaze.cuda().clone()
                else:
                    pred_list = torch.concat([pred_list,pred_gaze],dim=0)
                    gt_list = torch.concat([gt_list,gt_gaze.cuda()],dim=0)

            dist = calculate_angular_distance(gt_list[:],pred_list[:])
            dist_list.append(dist.mean().item())
            print("sub:{},acc:{:.4f}".format(subject,dist_list[-1]))
    # print("=====================")
    print("average acc:{:.4f}".format(np.array(dist_list).mean()))
