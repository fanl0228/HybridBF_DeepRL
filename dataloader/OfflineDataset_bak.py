import os
import urllib.request
import warnings

import gym
from gym.utils import colorize
import h5py
from tqdm import tqdm


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pdb

# User custom
from .h5py_opts import read_h5py_file


class OfflineDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.h5py_samples = []

        for i in range(len(self.files)):
            if self.files[i].split('.')[-1] == 'h5':
                self.h5py_samples.append(self.files[i])


    def __len__(self):
        return len(self.h5py_samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.h5py_samples[idx])
        
        return sample_path



if __name__=="__main__":
    # test
    data_dir = "/home/hx/fanl/HybridBF_DeepRL/datasets/train"
    dataset = OfflineDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    pdb.set_trace()
    
    # 遍历dataloader
    for sample_path in dataloader:
        
        data_dict = read_h5py_file(sample_path[0])

        print(data_dict.keys())
        print('---> observationsRD shape:{}'.format(data_dict['observationsRD'].shape))
        print('---> observationsRA shape:{}'.format(data_dict['observationsRA'].shape))
        print('---> rewards_PSINR shape:{}'.format(data_dict['rewards_PSINR'].shape))
        print('---> rewards_Intensity shape:{}'.format(data_dict['rewards_Intensity'].shape))
        print('---> rewards_Phase shape:{}'.format(data_dict['rewards_Phase'].shape))

