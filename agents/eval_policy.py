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
    mmWave_f0 = 7.7e10
    mmWave_d = 0.5034
    mmWave_D_BF = torch.tensor([0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53])

    torch.manual_seed(args.seed)
    tx_rx_spacae_dim = 121   # txbf and rxbf search space [0:121] --> angle [-60, +60]

    lengths_eval = []
    final_reward_eval = []  # eval_score = []
    avg_reward_eval = []
    
    test_filenumbers = len(os.listdir(args.test_dataset))
    
    batch_num = 0
    for data_dict_batch in tqdm(test_dataloader, desc="Loading test datafile"):
          
        lengths_batch = []
        final_reward_batch = [] 
        avg_reward_batch = []
        final_txbfs_batch = []
        final_rxbfs_batch = []
    
        for epiod_ in range(int(eval_episodes)):  
            steps = 0
            avg_reward_sample = 0.
            final_reward_sample = 0.
            final_txbfs = 0
            final_rxbfs = 0

            isDone = torch.zeros((data_dict_batch['terminals'].size(0), 1))
            hybridBFAction_buffer = torch.rand(args.batch_size, 2, tx_rx_spacae_dim).cuda()

            #while not done:
            while (isDone.mean().cpu().item() == 0):
                if (steps > args.eval_max_steps):
                    final_txbf_idxs = torch.argmax(hybridBFAction[:, 0, :], -1).tolist()  # [batch, tx]
                    final_rxbf_idxs = torch.argmax(hybridBFAction[:, 1, :], -1).tolist()  # [batch, rxbf]
                    break
                # Normalize
                #state = (np.array(state).reshape(1, -1) - mean)/std
                
                hybridBFAction = V(hybridBFAction_buffer)
                txbf_idxs = torch.argmax(hybridBFAction[:, 0, :], -1).tolist()  # [batch, tx]
                rxbf_idxs = torch.argmax(hybridBFAction[:, 1, :], -1).tolist()  # [batch, rxbf]

                #isDone = torch.zeros((data_dict_batch['terminals'].size(0), len(txbf_idxs)))
                rewards = torch.zeros((data_dict_batch['rewards_Phase'].size(0), 1))
                # init data buffer
                next_observationsRDA = torch.zeros((data_dict_batch['observationsRDA'].size(0), 
                                            1, 
                                            data_dict_batch['observationsRDA'].size(2), 
                                            data_dict_batch['observationsRDA'].size(3),
                                            data_dict_batch['observationsRDA'].size(4),
                                            data_dict_batch['observationsRDA'].size(5)
                                            ))

                for b in range(data_dict_batch['observationsRDA'].size(0)):
                    #next_observationsRDA[b,...] = data_dict_batch['observationsRDA'][b, txbf_idxs[b], ...] # observationsRDA: [frame, range, Doppler, ant] = [b, 10, 64, 128, 16]
                    
                    # 2. action RXBF, rx antenna is 16
                    mmWave_WX = torch.sin(torch.tensor(rxbf_idxs[b]*math.pi/180))
                    mmWave_RXBF_V = torch.exp(-1j*2*math.pi*mmWave_f0*mmWave_d*mmWave_D_BF*mmWave_WX)
                    
                    #next_observationsRDA[b,...] = data_dict_batch['observationsRDA'][b, txbf_idxs[b], ...] * mmWave_RXBF_V
                    next_observationsRDA[b,...] = 10*torch.log10(abs(data_dict_batch['observationsRDA'][b, txbf_idxs[b], ...] * mmWave_RXBF_V) + 1e-8) # abs - 10*log10           
                    
                    # 3. Normalize
                    next_observationsRDA_mean = next_observationsRDA[b,...].mean()
                    next_observationsRDA_std = next_observationsRDA[b,...].std()
                    next_observationsRDA[b,...] = (next_observationsRDA[b,...] - next_observationsRDA_mean)/next_observationsRDA_std

                    rewards[b] = data_dict_batch['rewards_Phase'][b, txbf_idxs[b]]
                    isDone[b]= data_dict_batch['terminals'][b, txbf_idxs[b]]
                
               
                next_observationsRDA = next_observationsRDA.squeeze(1)
                # [batch, frame, range, Doppler, ant] --> [batch, Doppler, frame, range, ant]
                next_observationsRDA = next_observationsRDA.permute(0, 3, 1, 2, 4)
                
                # to device
                rewards = rewards.to(device, non_blocking=True)  
                isDone = isDone.to(device, non_blocking=True)
                hybridBFAction = hybridBFAction.to(device, non_blocking=True)   
                next_observationsRDA = next_observationsRDA.to(device, non_blocking=True) 
                    
                hybridBFAction_buffer = policy.select_action(next_observationsRDA)
                
                avg_reward_sample += rewards
                final_reward_sample = rewards
                final_txbfs = txbf_idxs
                final_rxbfs = txbf_idxs
                steps += 1

                if args.debug:
                    print("----> step:{}, reward:{}, isDone:{}, txbf:{}, rxbf: {} ".format(steps, rewards, isDone, txbf_idxs, rxbf_idxs))
            
            
            lengths_batch.append(steps)
            avg_reward_batch.append(avg_reward_sample.cpu().data.numpy() / steps)
            final_reward_batch.append(final_reward_sample.cpu().data.numpy())
            final_txbfs_batch.append(final_txbfs)
            final_rxbfs_batch.append(final_rxbfs)
            
            if args.debug:
                print("----> epiod_:{}, lengths_batch:{}, "
                    "avg_reward_batch:{}, final_reward_batch:{}, "
                    "final_txbf_idxs:{}, final_rxbf_idxs: {} ".format(epiod_, np.mean(lengths_batch), 
                                                                        avg_reward_batch, final_reward_batch, 
                                                                        final_txbf_idxs, final_rxbf_idxs))
        
        #print(" num:{}".format(int(iter * test_filenumbers) + batch_num))
        # pdb.set_trace()
        logger.log('eval/final_txbf', np.mean(final_txbfs_batch), int(iter * test_filenumbers) + batch_num)
        logger.log('eval/final_rxbf', np.mean(final_rxbfs_batch), int(iter * test_filenumbers) + batch_num)

        #pdb.set_trace()
        final_reward_eval.append(np.mean(final_reward_batch))
        lengths_eval.append(np.mean(lengths_batch))
        avg_reward_eval.append(np.mean(avg_reward_batch)) 
        #print("----> batch_num:{}, final_reward_eval:{}, lengths_eval:{}, avg_reward_eval: {} ".format(batch_num, final_reward_eval, lengths_eval, avg_reward_eval))
        batch_num += 1
    
    #pdb.set_trace()
    logger.log('eval/lengths_mean', np.mean(lengths_eval), iter)
    logger.log('eval/lengths_std', np.std(lengths_eval), iter)
    logger.log('eval/avg_reward_mean', np.mean(avg_reward_eval), iter)
    logger.log('eval/avg_reward_std', np.std(avg_reward_eval), iter)

    logger.log('eval/final_reward', np.mean(final_reward_eval), iter)



    final_reward_eval = np.mean(final_reward_eval)

    print("------------> Evaluation over iter: {}, avg_reward_eval:{}, final_reward_eval: {}".format(iter, np.mean(avg_reward_eval), final_reward_eval))
    return final_reward_eval
