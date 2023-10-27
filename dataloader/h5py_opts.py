import os.path

import h5py
from tqdm import tqdm
import numpy as np

import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

# /home/hx/fanl/HybridBF_DeepRL/mmWaveRL/datasets/BeamAngle001_Sample00_State_RAObj.mat , "State_RAObj"
def get_file_data(filename, mat_name):
    data_dict = scipy.io.loadmat(filename)
    return data_dict[mat_name] 


# 'actions', 'infos/goal', 'infos/qpos', 'infos/qvel', 'observations', 'rewards', 'terminals', 'timeouts'
def write_h5py_file(result_filename, dataset_path, SampleNum):

    observations_RDA_patch = []
    observations_RA_patch = []
    rewards_PSINR_patch = []
    rewards_Intensity_patch = []
    rewards_Phase_patch = []
    for angle in range(1, 122):
        if angle < 10:
            observations_RDA_file = os.path.join(dataset_path, 'BeamAngle00' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RDA.mat')
            observations_RA_file = os.path.join(dataset_path, 'BeamAngle00' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RA.mat')
            rewards_PSINR_file = os.path.join(dataset_path, 'BeamAngle00' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_PSINR.mat')
            rewards_Intensity_file = os.path.join(dataset_path, 'BeamAngle00' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Intensity.mat')
            rewards_Phase_file = os.path.join(dataset_path, 'BeamAngle00' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Phase.mat')
        elif angle < 100:
            observations_RDA_file = os.path.join(dataset_path, 'BeamAngle0' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RDA.mat')
            observations_RA_file = os.path.join(dataset_path, 'BeamAngle0' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RA.mat')
            rewards_PSINR_file = os.path.join(dataset_path, 'BeamAngle0' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_PSINR.mat')
            rewards_Intensity_file = os.path.join(dataset_path, 'BeamAngle0' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Intensity.mat')
            rewards_Phase_file = os.path.join(dataset_path, 'BeamAngle0' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Phase.mat')
        else:
            observations_RDA_file = os.path.join(dataset_path, 'BeamAngle' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RDA.mat')
            observations_RA_file = os.path.join(dataset_path, 'BeamAngle' + str(angle) + '_Sample0' + str(SampleNum) + '_State_RA.mat')
            rewards_PSINR_file = os.path.join(dataset_path, 'BeamAngle' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_PSINR.mat')
            rewards_Intensity_file = os.path.join(dataset_path, 'BeamAngle' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Intensity.mat')
            rewards_Phase_file = os.path.join(dataset_path, 'BeamAngle' + str(angle) + '_Sample0' + str(SampleNum) + '_Reward_Phase.mat')

        # actions_file = ''
        # terminals_file
        # timeouts
        observations_RDA = get_file_data(observations_RDA_file, 'State_RDA').astype('complex64')
        # observations_RA = get_file_data(observations_RA_file, 'State_RA')#.astype('complex64')
        rewards_PSINR = get_file_data(rewards_PSINR_file, 'Reward_PSINR')
        rewards_Intensity = get_file_data(rewards_Intensity_file, 'Intensity_estSINR')
        rewards_Phase = get_file_data(rewards_Phase_file, 'Phase_estSINR')

        observations_RDA_patch.append(observations_RDA)
        # observations_RA_patch.append(observations_RA)
        rewards_PSINR_patch.append(rewards_PSINR)
        rewards_Intensity_patch.append(rewards_Intensity)
        rewards_Phase_patch.append(rewards_Phase)


    with h5py.File(result_filename, 'w') as dataset_file:
        dataset_file.create_dataset('observationsRDA',  data=observations_RDA_patch,
                                    compression="gzip", compression_opts=6)
        # dataset_file.create_dataset('observationsRA', data=observations_RA_patch,
        #                             compression="gzip", compression_opts=6)

        dataset_file.create_dataset('rewards_PSINR',  data=rewards_PSINR_patch,
                                    compression="gzip", compression_opts=6)
        dataset_file.create_dataset('rewards_Intensity',  data=rewards_Intensity_patch,
                                    compression="gzip", compression_opts=6)
        dataset_file.create_dataset('rewards_Phase',  data=rewards_Phase_patch,
                                    compression="gzip", compression_opts=6)

        # dataset_file.create_dataset('actions',  )
        # dataset_file.create_dataset('terminals',  )
        # dataset_file.create_dataset('timeouts',  )
        # dataset_file.create_dataset('infos/goal',  )
        # dataset_file.create_dataset('infos/qpos',  )
        # dataset_file.create_dataset('infos/qvel',  )


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def read_h5py_file(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict


if __name__ == "__main__":
    dataset_path = "I:\LOS_Scene5_DRL"
    for SampleNum in range(0, 1):

        filename = "LOS_Scene5_Sample" + str(SampleNum) +".h5"
        result_filename = os.path.join(dataset_path, filename)

        # write dataset
        write_h5py_file(result_filename, dataset_path, SampleNum)

        # read test dataset
        data_dict = read_h5py_file(result_filename)

        # log
        print(data_dict.keys())
        print('---> observationsRDA shape:{}'.format(data_dict['observationsRDA'].shape))
        #print('---> observationsRA shape:{}'.format(data_dict['observationsRA'].shape))
        print('---> rewards_PSINR shape:{}'.format(data_dict['rewards_PSINR'].shape))
        print('---> rewards_Intensity shape:{}'.format(data_dict['rewards_Intensity'].shape))
        print('---> rewards_Phase shape:{}'.format(data_dict['rewards_Phase'].shape))

        if True:
            N_samples = data_dict['observationsRDA'].shape[0]
            observationsRDA = data_dict['observationsRDA']
            #observationsRA = data_dict['observationsRA']

            fig, ax = plt.subplots()
            for b in range(10):
                stateRD = observationsRDA[b, 0, :, :, 0]

                x = np.arange(0, stateRD.shape[1])
                y = np.arange(0, stateRD.shape[0])
                X, Y = np.meshgrid(x, y)

                c = ax.pcolormesh(X, Y, 10*np.log10(abs(stateRD)), cmap=plt.cm.get_cmap('jet'))
                plt.pause(0.05)

            fig, _ = plt.subplots()
            ax = Axes3D(fig)
            for b in range(0, 10, 10):

                stateRA = observationsRDA[b, 0, :, :, :]

                stateRA = np.mean(stateRA, 1)

                stateRAfft = np.fft.fftshift(np.fft.fft(stateRA, 128, axis=-1), axes=-1)
                stateRAfft = np.flip(stateRAfft, axis=-1)

                x = np.arange(0, stateRAfft.shape[1])
                y = np.arange(0, stateRAfft.shape[0])
                X, Y = np.meshgrid(x, y)
                #c = ax.pcolormesh(X, Y, np.log10(np.abs(stateRAfft )), cmap='jet')
                surf = ax.plot_surface(X, Y, 10*np.log10(abs(np.abs(stateRAfft ))), rstride=1, cstride=1,
                                       cmap=plt.cm.get_cmap('jet'))
                plt.pause(0.05)

            # fig, _ = plt.subplots()
            # ax = Axes3D(fig)
            # for b in range(0, 10, 10):
            #     stateRA = observationsRA[b, 0, :, :]
            #
            #     x = np.arange(0, stateRA.shape[1])
            #     y = np.arange(0, stateRA.shape[0])
            #     X, Y = np.meshgrid(x, y)
            #     #c = ax.pcolormesh(X, Y, 10*np.log10(np.abs(stateRA)), cmap='jet')
            #     surf = ax.plot_surface(X, Y, 10 * np.log10(np.abs(stateRA)), rstride=1, cstride=1,
            #                            cmap=plt.cm.get_cmap('jet'))
            #     plt.pause(0.05)


    print("Done")





























