import numpy as np
import torch
import argparse
import os

from tqdm import tqdm
from tqdm import trange
import time
import json
from torch.autograd import Variable as V
import math

import utils
from utils.log import Logger
from dataloader.h5py_opts import read_h5py_file

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

# def eval_policy(args, iter, video: VideoRecorder, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
def eval_policy(args, iter, logger: Logger, policy, test_dataloader, eval_episodes):
    # Rxbf paramer
    mmWave_f0 = torch.tensor(7.7e10).to(device, non_blocking=True)
    mmWave_d = torch.tensor(0.5034).to(device, non_blocking=True)
    mmWave_D_BF = torch.tensor([0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53]).to(device, non_blocking=True)
    
    torch.manual_seed(args.seed)
    mmWave_doppler_dim = 128
    mmWave_frame_dim = 3
    mmWave_range_dim = 64
    mmWave_angle_dim = 16
    tx_rx_spacae_dim = 121   # txbf and rxbf search space [0:121] --> angle [-60, +60]

    hybridBFAction_buffer = torch.randn(1, 2, tx_rx_spacae_dim).to(device, non_blocking=True)
    observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim).to(device, non_blocking=True)
    next_observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim).to(device, non_blocking=True)
    rewards_buffer = torch.zeros((1,)).to(device, non_blocking=True)
    isDone_buffer = torch.zeros((1,)).to(device, non_blocking=True)
    
    lengths_eval = []
    final_reward_eval = []  # eval_score = []
    avg_reward_eval = []
    
    test_filenumbers = len(os.listdir(args.test_dataset))
    
    #for epiod_ in range(int(eval_episodes)):
    
    for t in trange(int(args.eval_episodes), desc="Train Episodes"):  
       
        lengths_batch = []
        final_reward_batch = [] 
        
        avg_reward_batch = []
        final_txbfs_batch = []
        final_rxbfs_batch = []
    
        sample_num = 0
        for data_dict_batch in tqdm(test_dataloader, desc="Loading test datafile"):
      
            avg_reward_sample = 0
            final_reward_sample = 0.
            final_txbfs = 0
            final_rxbfs = 0

            isDone = torch.zeros((data_dict_batch['terminals'].size(0), 1))
            hybridBFAction_buffer = torch.rand(args.batch_size, 2, tx_rx_spacae_dim).cuda()

            #while not done:
            # while (isDone.mean().cpu().item() == 0):
            for step in range(int(args.eval_max_steps)):
                # convert to Variable
                hybridBFAction = V(hybridBFAction_buffer)
                observationsRDA = V(observationsRDA_buffer)
                next_observationsRDA = V(next_observationsRDA_buffer)
                rewards = V(rewards_buffer)
                isDone = V(isDone_buffer)
                                
                # get txbf, rxbf value
                txbf_idxs = torch.argmax(hybridBFAction[0, 0, :], -1) 
                rxbf_idxs = torch.argmax(hybridBFAction[0, 1, :], -1)
                
                 # 1. observationsRDA: [frame, range, Doppler, ant]
                rewards = data_dict_batch['rewards_PSINR'][0, txbf_idxs.item()]
                isDone= data_dict_batch['terminals'][0, txbf_idxs.item()]
                
                # break
                if (isDone.mean().cpu().item() != 0):
                    avg_reward_sample += rewards
                    final_reward_sample = rewards
                    break

                #     
                next_observationsRDA_buffer = torch.clone(data_dict_batch['observationsRDA'][0, txbf_idxs.item(), :mmWave_frame_dim,...]).to(device, non_blocking=True) 
                
                # 2. action RXBF, rx antenna is 16
                mmWave_WX = torch.sin(rxbf_idxs*torch.pi/180.0).to(device, non_blocking=True)
                real_part = torch.cos(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device, non_blocking=True)
                imag_part = torch.sin(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device, non_blocking=True)
                mmWave_RXBF_V = torch.exp(1j * (real_part + imag_part)).to('cuda')
                next_observationsRDA_buffer = next_observationsRDA_buffer * mmWave_RXBF_V
                next_observationsRDA_buffer = torch.fft.fft(next_observationsRDA_buffer, mmWave_angle_dim, axis=-1).to(device, non_blocking=True)
                
                next_observationsRDA_buffer = 10*torch.log10(torch.abs(next_observationsRDA_buffer) + 1e-8).to(device, non_blocking=True)          
                
                # 3. Normalize
                next_observationsRDA_buffer = (next_observationsRDA_buffer - torch.mean(next_observationsRDA_buffer)) / (next_observationsRDA_buffer.std() + 1e-3)
                
                next_observationsRDA_buffer = next_observationsRDA_buffer.permute(2, 0, 1, 3) # [F,R,D,A]] --> [D,F,R,A]
              
                # get action
                hybridBFAction_buffer = policy.select_action(next_observationsRDA.unsqueeze(0).to(device, non_blocking=True))
                            
                 # logger
                eval_num = iter*args.eval_episodes*test_filenumbers*args.eval_max_steps + t*test_filenumbers*args.eval_max_steps + sample_num*args.eval_max_steps + step
                logger.log('eval/step_reward', rewards.item(), eval_num)
                
                # 
                avg_reward_sample += rewards 
                final_reward_sample = rewards
            
                # print("----> step:{}, reward:{}, isDone:{}, txbf:{}, rxbf: {} ".format(eval_num, rewards, isDone, txbf_idxs, rxbf_idxs))
            
            avg_reward_sample /= step
            final_reward_sample = rewards
            final_txbfs = txbf_idxs
            final_rxbfs = txbf_idxs
            
            # logger
            eval_sample_num = iter*args.eval_episodes*test_filenumbers + t*test_filenumbers + sample_num
            #print("==================eval_sample_num: {}".format(eval_sample_num))
            
            logger.log('eval/avg_reward_sample', avg_reward_sample.item(), eval_sample_num)
            logger.log('eval/final_reward_sample', final_reward_sample.item(), eval_sample_num)
            logger.log('eval/final_txbfs_sample', final_txbfs.item(), eval_sample_num)
            logger.log('eval/final_rxbfs_sample', final_rxbfs.item(), eval_sample_num)
                
        
            lengths_batch.append(step)
            avg_reward_batch.append(avg_reward_sample)
            final_reward_batch.append(final_reward_sample)
            final_txbfs_batch.append(final_txbfs)
            final_rxbfs_batch.append(final_rxbfs)
            
        final_reward_eval.append(np.mean(final_reward_batch))
        lengths_eval.append(np.mean(lengths_batch))
        avg_reward_eval.append(np.mean(avg_reward_batch)) 
        #print("----> sample_num:{}, final_reward_eval:{}, lengths_eval:{}, avg_reward_eval: {} ".format(sample_num, final_reward_eval, lengths_eval, avg_reward_eval))
        sample_num += 1
    
    logger.log('eval/lengths_mean', np.mean(lengths_eval), iter)
    logger.log('eval/lengths_std', np.std(lengths_eval), iter)
    logger.log('eval/avg_reward_mean', np.mean(avg_reward_eval), iter)
    logger.log('eval/avg_reward_std', np.std(avg_reward_eval), iter)

    logger.log('eval/final_reward', np.mean(final_reward_eval), iter)

    final_reward_eval = np.mean(final_reward_eval)

    print("------------> Evaluation over iter: {}, avg_reward_eval:{}, final_reward_eval: {}".format(iter, avg_reward_eval, final_reward_eval))
    return final_reward_eval
