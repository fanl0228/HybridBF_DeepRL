import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
import argparse
from tqdm import tqdm
from tqdm import trange
import time
import json
import math

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V

from utils.log import Logger
from utils.common import make_dir, snapshot_src 
from dataloader import OfflineDataset
from dataloader.h5py_opts import read_h5py_file
from dataloader.ReplayBuffer import ReplayBuffer

from agents.IQLPolicy import IQLpolicy

from agents.eval_policy import eval_policy

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parsers():

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQLpolicy")                  # Policy name
    parser.add_argument("--train_dataset", default="/data/mmWaveRL_Datasets/train")
    parser.add_argument("--test_dataset", default="/data/mmWaveRL_Datasets/test")                    # dataset path
    parser.add_argument("--seed", default=3, type=int)              #  
    parser.add_argument("--train_max_steps", default=3200, type=int)   # Max time steps to run environment
    parser.add_argument("--train_episodes", default=100, type=int)
    parser.add_argument("--save_model", default=True, action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--eval_freq", default=1, type=int)       # How often (time steps) we evaluate
    parser.add_argument('--eval_max_steps', default=500, type=int)
    parser.add_argument('--eval_episodes', default=1, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=16, type=int)      # Batch size default is 1
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='tmp_debug', type=str)
    parser.add_argument('--model_dir', default='runs/savemodel_debug', type=str)
    args = parser.parse_args()

    # Build work dir
    base_dir = 'runs'
    make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, 'dataset')
    make_dir(args.work_dir)
    make_dir(args.model_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==========================================================")
    print(" Device: {}".format(device))
    print(" Args: {}".format(args))
    print("==========================================================")

    return args


def train(args):
    # Rxbf paramer
    mmWave_f0 = 7.7e10
    mmWave_d = 0.5034
    mmWave_D_BF = np.array([0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53])
    
    mmWave_doppler_dim = 128
    mmWave_frame_dim = 10
    mmWave_range_dim = 64
    mmWave_angle_dim = 128
    tx_rx_spacae_dim = 121   # txbf and rxbf search space [0:121] --> angle [-60, +60]

    state_shape = [args.batch_size, mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim] #  [batch, Doppler, frame, range, ant]
    action_shape = [args.batch_size, tx_rx_spacae_dim, tx_rx_spacae_dim] # [txbf, rxbf]
    kwargs = {
        "state_shape": state_shape,
        "action_shape": action_shape,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "expectile": args.expectile,
    }

    # Initialize policy
    torch.manual_seed(args.seed)
    hybridBFAction = np.random.randn(2, tx_rx_spacae_dim)
    observationsRDA = np.random.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim)
    next_observationsRDA = np.random.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim)
    rewards = np.zeros((1,))
    isDone = np.zeros((1,))
    
    replay_buffer = ReplayBuffer([mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim], [2, tx_rx_spacae_dim])

    if args.policy == 'IQLpolicy':
        policy = IQLpolicy(**kwargs) 
    else:
        raise NotImplementedError
    
    if args.debug:
        actor, critic, value = policy.get_model()
        print("--->actor model:{}\n --> critic model:{}\n --> value model{}".format(actor, critic, value))

    # Logger
    logger = Logger(args.work_dir, use_tb=True)

    # Dataset
    train_dataset = OfflineDataset(args.train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = OfflineDataset(args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    train_filenumbers = len(os.listdir(args.train_dataset))

    for t in trange(int(args.train_episodes), desc="Train Epoch"):
        
        sample_num = 0
        for data_dict_batch in tqdm(train_dataloader, desc="Loading train datafile"):

            for epoch in range(int(args.train_max_steps)):

                ''' Data preprocessing and to tensor
                '''
                txbf_idxs = np.argmax(hybridBFAction[0, :])  # [batch, txbf]
                rxbf_idxs = np.argmax(hybridBFAction[1, :])  # [batch, rxbf]
                
                # init data buffer 
                # 1. observationsRDA: [frame, range, Doppler, ant] = [b, 10, 64, 128, 16]
                next_observationsRDA = data_dict_batch['observationsRDA'][0, txbf_idxs, ...].numpy()
                
                # 2. action RXBF, rx antenna is 16
                mmWave_WX = math.sin(rxbf_idxs*math.pi/180)
                mmWave_RXBF_V = np.exp(-1j*2*math.pi*mmWave_f0*mmWave_d*mmWave_D_BF*mmWave_WX)
                next_observationsRDA = next_observationsRDA * mmWave_RXBF_V

                next_observationsRDA = np.fft.fft(next_observationsRDA, mmWave_angle_dim, axis=-1)

                next_observationsRDA = 10*np.log10(abs(next_observationsRDA) + 1e-8) # abs - 10*log10           
                
                # 3. Normalize
                next_observationsRDA_mean = next_observationsRDA.mean()
                next_observationsRDA_std = next_observationsRDA.std() + 1e-3
                next_observationsRDA = (next_observationsRDA - next_observationsRDA_mean)/next_observationsRDA_std

                rewards = data_dict_batch['rewards_PSINR'][0, txbf_idxs].numpy()
                isDone= data_dict_batch['terminals'][0, txbf_idxs].numpy()
                
                # [batch, frame, range, Doppler, ant] --> [batch, Doppler, frame, range, ant]
                next_observationsRDA = next_observationsRDA.transpose(2, 0, 1, 3)
 
                replay_buffer.add(observationsRDA, hybridBFAction, next_observationsRDA, rewards, isDone)
                
                hybridBFAction = policy.select_action(torch.FloatTensor(next_observationsRDA).unsqueeze(0).to(device)).cpu().data.numpy().squeeze(0)
                #hybridBFAction = hybridBFAction.numpy()
                observationsRDA = np.copy(next_observationsRDA)

                if (epoch+1) % args.batch_size == 0:
                    # train model
                    policy.train(replay_buffer, args.batch_size, logger=logger)

                    train_num = t*train_filenumbers*args.train_max_steps + sample_num*args.train_max_steps + epoch
                    # logger
                    logger.log('train/txbf_idxs[0]', txbf_idxs, train_num)
                    logger.log('train/rxbf_idxs[0]', rxbf_idxs, train_num)
            
            # # Evaluation number
            # eval_num = t*train_filenumbers + sample_num
            # eval_score = eval_policy(args, eval_num+1, logger, policy, test_dataloader, args.eval_episodes)
            # print("------------------Evaluation number: {} eval_score: {}".format(eval_num, eval_score))       
            
            sample_num += 1

        # Evaluation number
        eval_score = eval_policy(args, t+1, logger, policy, test_dataloader, args.eval_episodes)
        print("------------------Evaluation number: {} eval_score: {}".format(t, eval_score))
            
    if args.save_model:
        policy.save(args.model_dir)


if __name__=="__main__":
    args = get_parsers()

    train(args)


