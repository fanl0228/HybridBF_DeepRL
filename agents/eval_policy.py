import numpy as np
import torch
import argparse
import os
from tqdm import trange
from coolname import generate_slug
import time
import json
from torch.autograd import Variable as V


import utils
from utils.log import Logger
from dataloader.h5py_opts import read_h5py_file

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


# def eval_policy(args, iter, video: VideoRecorder, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
def eval_policy(args, iter, logger: Logger, policy, test_dataloader, eval_episodes):

    torch.manual_seed(args.seed)
    tx_rx_spacae_dim = 121   # txbf and rxbf search space [0:121] --> angle [-60, +60]
    #hybridBFAction_buffer = torch.rand(2, tx_rx_spacae_dim).cuda()
    
    hybridBFAction = np.zeros((2, tx_rx_spacae_dim))
    
    lengths = []
    returns = []
    avg_reward = 0.
    
    for _ in range(eval_episodes):
        
        for sample_path in test_dataloader:
            data_dict = read_h5py_file(sample_path[0])

            #pdb.set_trace()
            
            done = False
            steps = 0
            episode_return = 0
            #while not done:
            while steps < args.eval_sample_epochs:
                # Normalize
                #state = (np.array(state).reshape(1, -1) - mean)/std
                # video.init(enabled=(args.save_video and _ == 0))
                
                txbf_idx = np.argmax(hybridBFAction[0])#.item()
                observationsRDA = data_dict['observationsRDA'][txbf_idx]   # [frame, range, Doppler, ant] = [10, 64, 128, 16]
                # to tensor
                observationsRDA = torch.from_numpy(10*np.log10(abs(observationsRDA) ) )
                # [frame, range, Doppler, ant] --> [Doppler, frame, range, ant]
                observationsRDA = observationsRDA.permute(2, 0, 1, 3)
                # add batch dim,  [batch, Doppler, frame, range, ant]
                observationsRDA = torch.unsqueeze(observationsRDA, 0)
                observationsRDA = observationsRDA.to(device, non_blocking=True) 
                
                #
                #hybridBFAction = V(hybridBFAction_buffer) 
                
                hybridBFAction = policy.select_action(observationsRDA)
                
                #pdb.set_trace()

                # given action return state
                # txbf_idx = np.argmax(hybridBFAction[0])#.item()
                
                # observationsRDA = data_dict['observationsRDA'][txbf_idx] 
                
                reward = data_dict['rewards_PSINR'][txbf_idx] 
                #done = data_dict['done'][txbf_idx] 


                avg_reward += reward
                episode_return += reward
                steps += 1

                #print("----> reward:{}, episode_return:{}, txbf_idx:{} ".format(reward, episode_return, txbf_idx))
            lengths.append(steps)
            returns.append(episode_return)
            # video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

    avg_reward /= eval_episodes
    
    # TODO: eval socre
    eval_score = avg_reward

    logger.log('eval/lengths_mean', np.mean(lengths), iter)
    logger.log('eval/lengths_std', np.std(lengths), iter)
    logger.log('eval/returns_mean', np.mean(returns), iter)
    logger.log('eval/returns_std', np.std(returns), iter)
    logger.log('eval/eval_score', eval_score, iter)

    print("---------------------------------------")
    print("Evaluation over: {} episodes: {}".format(eval_episodes, eval_score))
    print("---------------------------------------")
    return eval_score