import os.path

import h5py
from tqdm import tqdm
import numpy as np

import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# User custom
import sys
sys.path.append("../")
from dataloader.h5py_opts import read_h5py_file, write_h5py_file


import pdb

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

if __name__ == "__main__":
    
    dataset_path = "/data/mmWaveRL_Datasets/train/"
    res_dataset_path = "/data/mmWaveRL_Datasets/test_rewards_png/"
    make_dir(res_dataset_path)
    
    filenames =  os.listdir(dataset_path)
    h5py_samples  = []   
    for i in range(len(filenames)):
        if filenames[i].split('.')[-1] == 'h5':
            h5py_samples.append(filenames[i])

    for i in range(len(filenames)):
        src_file_paths = os.path.join(dataset_path, h5py_samples[i])
        
        res_file_paths = os.path.join(res_dataset_path, h5py_samples[i])
        
        print("Processing file...:{}".format(src_file_paths))
        # read test dataset
        data_dict = read_h5py_file(src_file_paths)

        print('---> observationsRDA shape:{}'.format(data_dict['observationsRDA'].shape))
        print('---> rewards_PSINR shape:{}'.format(data_dict['rewards_PSINR'].shape))
        print('---> rewards_Intensity shape:{}'.format(data_dict['rewards_Intensity'].shape))
        print('---> rewards_Phase shape:{}'.format(data_dict['rewards_Phase'].shape))
        print('---> terminals shape:{}, val:{}'.format(data_dict['terminals'].shape, data_dict['terminals']))
        
        # plt.figure()
        # plt.plot(data_dict['rewards_PSINR'], color='r', label="PSINR")
        # #plt.plot(data_dict['rewards_Intensity'], color='g', label="Intensity")
        # plt.plot(data_dict['rewards_Phase'], color='b', label="Phase")
        # plt.legend()
        # #plt.pause(0.01)
        # plt.savefig(res_file_paths.split('.')[0] + '_reward_log10.png')




        
        
        
        
    
        
        
        
        















