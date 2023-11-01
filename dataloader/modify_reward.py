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
    
    dataset_path = "/data/mmWaveRL_Datasets/train1/"
    res_dataset_path = "/data/mmWaveRL_Datasets/train_res/"
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
        
        data_dict['rewards_PSINR'] = data_dict['rewards_PSINR'] - np.mean(data_dict['rewards_PSINR'])
        data_dict['rewards_Intensity'] = data_dict['rewards_Intensity'] - np.mean(data_dict['rewards_Intensity'])
        data_dict['rewards_Phase'] = data_dict['rewards_Phase'] - np.mean(data_dict['rewards_Phase'])

        with h5py.File(res_file_paths, 'w') as dataset_file:
            dataset_file.create_dataset('observationsRDA',  data=data_dict['observationsRDA'],
                                    compression="gzip", compression_opts=6)
            dataset_file.create_dataset('rewards_PSINR',  data=data_dict['rewards_PSINR'],
                                        compression="gzip", compression_opts=6)
            dataset_file.create_dataset('rewards_Intensity',  data=data_dict['rewards_Intensity'],
                                        compression="gzip", compression_opts=6)
            dataset_file.create_dataset('rewards_Phase',  data=data_dict['rewards_Phase'],
                                        compression="gzip", compression_opts=6)
            dataset_file.create_dataset('terminals', data=data_dict['terminals'])


