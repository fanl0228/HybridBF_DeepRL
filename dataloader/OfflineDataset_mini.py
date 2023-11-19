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
import sys
sys.path.append("../")
from dataloader.h5py_opts import read_h5py_file_tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_dict_to_float32(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            output_dict[key] = convert_dict_to_float32(value) 
        elif isinstance(value, torch.Tensor):
            output_dict[key] = value.to(torch.float32)
        else:
            output_dict[key] = value
    return output_dict


class OfflineDataset_mini(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.h5py_paths = []
        self.data = []
        
        for i in range(len(self.files)):
            if self.files[i].split('.')[-1] == 'h5':
                self.data.append(read_h5py_file_tqdm(os.path.join(self.data_dir, self.files[i])))
                self.h5py_paths.append(self.files[i])
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data_dict = self.data[idx]
        return data_dict, self.h5py_paths[idx]


if __name__=="__main__":
    # test
    data_dir = "/data/mmWaveRL_Datasets/train_clear"
    dataset = OfflineDataset_mini(data_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    txbf_idx = 0    
    
    # 遍历dataloader
    for data_dict_batch in tqdm(dataloader, desc="load datafile"):
        print(data_dict_batch.keys())
        
        pdb.set_trace()
        # observationsRDA: [frame, range, Doppler, ant] = [b, 10, 64, 128, 16]
        observationsRDA = data_dict_batch['observationsRDA'][:, txbf_idx, ...].squeeze()
        observationsRDA = observationsRDA.permute(3, 1, 2, 4)
        
        rewards = data_dict_batch['Phase_estSINR'][:, txbf_idx, ...].squeeze()
        done = data_dict_batch['terminals'][:, txbf_idx, ...].squeeze()
        #hybridBFAction = 

        observationsRDA = observationsRDA.to(device, non_blocking=True) 
        #hybridBFAction = hybridBFAction.to(device, non_blocking=True)   
        rewards = rewards.to(device, non_blocking=True)   
        done = done.to(device, non_blocking=True) 

        
        print('---> observationsRD shape:{}'.format(observationsRDA.shape))


