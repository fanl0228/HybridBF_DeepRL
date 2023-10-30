import numpy as np
import torch
import argparse
import os

from tqdm import tqdm
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
    
    hybridBFAction_buffer = torch.rand(args.batch_size, 2, tx_rx_spacae_dim).cuda()

    #hybridBFAction = np.rand((args.batch_size, 2, tx_rx_spacae_dim))
    
    lengths = []
    returns = []
    avg_reward = 0.
    eval_score = [] 
    for data_dict_batch in tqdm(test_dataloader, desc="Loading test datafile"):
        for _ in trange(int(eval_episodes), desc="Test Epoch"):
            
            steps = 0
            episode_return = 0
            isDone = torch.zeros((data_dict_batch['terminals'].size(0), 1))
            
            #while not done:
            while (isDone.mean().cpu().item() == 0):
                if (steps > args.eval_max_steps):
                    break
                # Normalize
                #state = (np.array(state).reshape(1, -1) - mean)/std
                
                hybridBFAction = V(hybridBFAction_buffer)
                txbf_idxs = torch.argmax(hybridBFAction[:, 0, :], -1).tolist()  # [batch, tx]
                #isDone = torch.zeros((data_dict_batch['terminals'].size(0), len(txbf_idxs)))
                rewards = torch.zeros((data_dict_batch['rewards_Phase'].size(0), 1))
                # init data buffer
                observationsRDA = torch.zeros((data_dict_batch['observationsRDA'].size(0), 
                                            1, 
                                            data_dict_batch['observationsRDA'].size(2), 
                                            data_dict_batch['observationsRDA'].size(3),
                                            data_dict_batch['observationsRDA'].size(4),
                                            data_dict_batch['observationsRDA'].size(5)
                                            ))

                for b in range(data_dict_batch['observationsRDA'].size(0)):
                    observationsRDA[b,...] = data_dict_batch['observationsRDA'][b, txbf_idxs[b], ...] # observationsRDA: [frame, range, Doppler, ant] = [b, 10, 64, 128, 16]
                    rewards[b] = data_dict_batch['rewards_Phase'][b, txbf_idxs[b]]
                    isDone[b]= data_dict_batch['terminals'][b, txbf_idxs[b]]
                
                observationsRDA = observationsRDA.squeeze()
                # [batch, frame, range, Doppler, ant] --> [batch, Doppler, frame, range, ant]
                observationsRDA = observationsRDA.permute(0, 3, 1, 2, 4)
                
                # to device
                rewards = rewards.to(device, non_blocking=True)  
                isDone = isDone.to(device, non_blocking=True)
                hybridBFAction = hybridBFAction.to(device, non_blocking=True)   
                observationsRDA = observationsRDA.to(device, non_blocking=True) 
                next_state = torch.rand(observationsRDA.shape).to(device, non_blocking=True)   # TODO torch.zeros()
                              
                hybridBFAction_buffer = policy.select_action(observationsRDA)
                
                avg_reward += rewards
                episode_return += rewards
                steps += 1

                #print("----> step:{}, reward:{}, isDone:{}, txbf_idx:{} ".format(steps, rewards, isDone, txbf_idxs))
            lengths.append(steps)
            returns.append(episode_return)
    

        avg_reward /= eval_episodes
        
        # TODO: eval socre
        eval_score.append(avg_reward)

        logger.log('eval/lengths_mean', np.mean(lengths), iter)
        logger.log('eval/lengths_std', np.std(lengths), iter)
        logger.log('eval/returns_mean', np.mean(returns[0].cpu().data.numpy()), iter)
        logger.log('eval/returns_std', np.std(returns[0].cpu().data.numpy()), iter)
        logger.log('eval/eval_score', np.mean(eval_score.cpu().data.numpy()), iter)

    eval_score = np.mean(eval_score)
    # print("---------------------------------------")
    # print("Evaluation over: {} episodes: {}".format(eval_episodes, eval_score))
    # print("---------------------------------------")
    return eval_score