import os.path

import h5py
from tqdm import tqdm
import numpy as np

import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

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
    res_dataset_path = "/data/mmWaveRL_Datasets/train_norm_v10/"
    PNG_Reward_path = "/data/mmWaveRL_Datasets/PNG_Reward_train_norm_v10/"
    make_dir(res_dataset_path)
    make_dir(PNG_Reward_path)
    
    filenames =  os.listdir(dataset_path)
    h5py_samples  = []   
    for i in range(len(filenames)):
        if filenames[i].split('.')[-1] == 'h5':
            h5py_samples.append(filenames[i])

    for i in range(len(filenames)):
        src_file_paths = os.path.join(dataset_path, h5py_samples[i])
        res_file_paths = os.path.join(res_dataset_path, h5py_samples[i])
        PNG_Reward_file_paths = os.path.join(PNG_Reward_path, h5py_samples[i])

        
        print("Processing file...:{}".format(src_file_paths))
        
        # read test dataset
        data_dict = read_h5py_file(src_file_paths)

        # 平滑 rewards_PSINR 数据
        data = data_dict['rewards_PSINR'].astype('float64')
        rewards_PSINR = savgol_filter(data, 3, 1, mode='nearest')
        if 1:
            plt.figure()
            plt.plot(data_dict['rewards_PSINR'], color='r', label="PSINR")
            plt.plot(rewards_PSINR, color='b', label="smooth")
            plt.legend()
            plt.savefig(PNG_Reward_file_paths.split('.')[0] + '_reward_smooth.png')

        data_dict['rewards_PSINR'] = rewards_PSINR


        # 归一化
        original_array = data_dict['rewards_PSINR'] #10**(data_dict['rewards_PSINR']/10)
        min_val = np.min(original_array)
        max_val = np.max(original_array)
        data_dict['rewards_PSINR'] = -1 + 2 * (original_array - min_val) / (max_val - min_val)
        print("rewards_PSINR reward: {} \n max: {} \n min:{}\n".format(data_dict['rewards_PSINR'], np.max(data_dict['rewards_PSINR']), np.min(data_dict['rewards_PSINR'])))
        
        original_array = data_dict['rewards_Intensity']#10**(data_dict['rewards_Intensity']/10)
        min_val = np.min(original_array)
        max_val = np.max(original_array)
        data_dict['rewards_Intensity'] = -1 + 2 * (original_array - min_val) / (max_val - min_val)
        print("rewards_Intensity reward: {} \n max: {} \n min:{}\n".format(data_dict['rewards_Intensity'], np.max(data_dict['rewards_Intensity']), np.min(data_dict['rewards_Intensity'])))

        original_array = data_dict['rewards_Phase']#10**(data_dict['rewards_Phase']/10)
        min_val = np.min(original_array)
        max_val = np.max(original_array)
        data_dict['rewards_Phase'] = -1 + 2 * (original_array - min_val) / (max_val - min_val)
        print("rewards_Phase reward: {} \n max: {} \n min:{}\n".format(data_dict['rewards_Phase'], np.max(data_dict['rewards_Phase']), np.min(data_dict['rewards_Phase'])))


        # 修改done？
        terminals_patch = np.zeros((121))
        max_reward = max(data_dict['rewards_PSINR'])
        max_idx = np.argmax(data_dict['rewards_PSINR'])
        if (max_idx < 5):
            search_low = max_idx
        else:
            search_low = max_idx - 5
        if (max_idx > len(data_dict['rewards_PSINR']) - 5):
            search_heigh = max_idx
        else:
            search_heigh = max_idx + 5

        for j in range(search_low, search_heigh):
            if data_dict['rewards_PSINR'][j] > 0.85 * max_reward:
                terminals_patch[j] = 1
        data_dict['terminals'] = np.array(terminals_patch).squeeze()


        if 1:
            plt.figure()
            plt.plot(data_dict['rewards_PSINR'], color='r', label="PSINR")
            plt.scatter(np.argmax(data_dict['terminals']), np.max(data_dict['terminals']), color='k', marker=10, label="Phase")
            plt.legend()
            plt.savefig(PNG_Reward_file_paths.split('.')[0] + '_reward.png')


        # 重新写回 h5 文件
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





