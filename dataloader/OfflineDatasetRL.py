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
from dataloader.h5py_opts import read_h5py_file


class OfflineDatasetRL(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.h5py_samples = os.listdir(data_dir)
        # self.transform = transforms.Compose([
        #     #transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])

        for i in range(len(self.h5py_samples)):
            sample_path = os.path.join(self.data_dir, self.h5py_samples[idx])
            data_dict = read_h5py_file(sample_path[0])


    def __len__(self):
        return len(self.h5py_samples)

    def __getitem__(self, idx):

        sample_path = os.path.join(self.data_dir, self.h5py_samples[idx])
        
        return sample_path



if __name__=="__main__":
    # test
    data_dir = "/home/hx/fanl/HybridBF_DeepRL/datasets/train"
    dataset = OfflineDatasetRL(data_dir)
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

















# class OfflineDatasetEnv(gym.Env):
#     """
#         Base class for offline RL envs.
#     """

#     def __init__(self, dataset_url=None, **kwargs):
#         super(OfflineDatasetEnv, self).__init__(**kwargs)
#         self.dataset_url = dataset_url


#     @property
#     def dataset_filepath(self):
#         return self.dataset_url
    
#     def get_dataset(self, h5path=None):
#         if self._dataset_url is None:
#             if h5path is None:
#                 raise ValueError("Offline env not configured with a dataset URL.")

#         data_dict = {}
#         data_dict = read_h5py_file(h5path)
        
#         # Run a few quick sanity checks
#         for key in ['observationsRD', 'observationsRA', 'rewards_PSINR', 'rewards_Intensity', 'rewards_Phase']:
#             assert key in data_dict, 'Dataset is missing key %s' % key

#         N_TxBeams = data_dict['observationsRD'].shape[0]

#         return data_dict

#     def get_action_space(self,):




