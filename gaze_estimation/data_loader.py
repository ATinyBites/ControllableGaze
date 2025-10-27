import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import cv2 as cv

trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    # with open(refer_list_file, 'r') as f:
    #     datastore = json.load(f)
    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    # sub_folder_use = './interpolate'
    sub_folder_use = './'
    # train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
    #                         transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    
    # train_set = GazeDataset(dataset_path=data_dir, keys_to_use=["train.h5"], sub_folder=sub_folder_use,
    #                         transform=trans, is_shuffle=is_shuffle, is_load_label=True)

    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=[f for f in os.listdir(os.path.join(data_dir,sub_folder_use)) if 'h5' in f], sub_folder=sub_folder_use,
                            transform=trans_train, is_shuffle=is_shuffle, is_load_label=True)
    

    print(train_set.selected_keys)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json') 
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader

def load_selected_idx(id):
        path = os.path.join('/data1/ltw/data/MPIIFaceGaze/selected2/','p{}'.format(str(id).zfill(2)))
        selected_idx = os.listdir(path)
        selected_idx.sort()
        selected_idx = [int(item.split('.')[0]) for item in selected_idx]
        return selected_idx
    
class GazeDataset(Dataset):
    def __init__(self, dataset_path, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True,is_mpii=False):
        if isinstance(dataset_path,str):
            self.path = dataset_path
            self.hdfs = {}
            self.sub_folder = sub_folder
            self.is_load_label = is_load_label
            self.selected_keys = [k for k in keys_to_use]
            assert len(self.selected_keys) > 0

            for num_i in range(0, len(self.selected_keys)):
                file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
                self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
                # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
                assert self.hdfs[num_i].swmr_mode

            # Construct mapping from full-data index to key and person-specific index
            if index_file is None:
                self.idx_to_kv = []
                for num_i in range(0, len(self.selected_keys)):
                    n = self.hdfs[num_i]["face_patch"].shape[0]
                    if not is_mpii:
                        self.idx_to_kv += [(num_i, i) for i in range(n)]
                    else:
                        selected_idx = load_selected_idx(int(keys_to_use[0].split('.')[0][-2:]))
                        self.idx_to_kv += [(num_i, i) for i in range(n) if i in selected_idx]

            else:
                print('load the file: ', index_file)
                self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

            for num_i in range(0, len(self.hdfs)):
                if self.hdfs[num_i]:
                    self.hdfs[num_i].close()
                    self.hdfs[num_i] = None

            if is_shuffle:
                random.shuffle(self.idx_to_kv)  # random the order to stable the training

            self.hdf = None
            self.transform = transform
        else:
            self.hdfs = dataset_path
            self.transform = transform
            self.is_load_label = is_load_label

    def __len__(self):
        if isinstance(self.hdfs,dict):
            return len(self.idx_to_kv)
        else:
            return self.hdfs['face_patch'].shape[0]

    def __del__(self):
        try:
            for num_i in range(0, len(self.hdfs)):
                if self.hdfs[num_i]:
                    self.hdfs[num_i].close()
                    self.hdfs[num_i] = None
        except:
            pass

    def __getitem__(self, idx):
        if isinstance(self.hdfs,dict):
            key, idx = self.idx_to_kv[idx]
            self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
            assert self.hdf.swmr_mode
        else:
            self.hdf = self.hdfs

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        image = cv.resize(image,(224,224))
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        # if is_filp:
            # image = cv.flip(image,1)
        image = self.transform(image)

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            # if is_filp:
                # gaze_label[1]*=-1

            return image, gaze_label
        else:
            return image




