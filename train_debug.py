import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import numpy as np
import torch
import argparse
from tqdm import tqdm
from tqdm import trange
from coolname import generate_slug
import time
import json

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V

from utils.log import Logger
from utils.common import make_dir, snapshot_src 
from dataloader import OfflineDataset
from dataloader.h5py_opts import read_h5py_file

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
    parser.add_argument("--train_max_steps", default=1e3, type=int)   # Max time steps to run environment
    parser.add_argument("--train_episodes", default=100, type=int)
    parser.add_argument("--save_model", default=True, action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--eval_freq", default=1, type=int)       # How often (time steps) we evaluate
    parser.add_argument('--eval_max_steps', default=1000, type=int)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=8, type=int)      # Batch size default is 1
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='tmp', type=str)
    parser.add_argument('--model_dir', default='runs/savemodel', type=str)
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==========================================================")
    print(" Device: {}".format(device))
    print(" Args: {}".format(args))
    print("==========================================================")

    return args


def train(args):
    # Initialize policy

    torch.manual_seed(args.seed)
    tx_rx_spacae_dim = 121   # txbf and rxbf search space [0:121] --> angle [-60, +60]
    hybridBFAction_buffer = torch.rand(args.batch_size, 2, tx_rx_spacae_dim).cuda()

    state_shape = [args.batch_size, 128, 10, 64, 64] # [batch, frame, range, doppler, antenna]   
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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = OfflineDataset(args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    for t in trange(int(args.train_episodes), desc="Train Epoch"):

        for data_dict_batch in tqdm(train_dataloader, desc="Loading train datafile"):

            for epoch in trange(int(args.train_max_steps), desc="Train Epoch"):
                ''' Data preprocessing and to tensor
                '''
                #pdb.set_trace()
                hybridBFAction = V(hybridBFAction_buffer)
                txbf_idxs = torch.argmax(hybridBFAction[:, 0, :], -1).tolist()  # [batch, tx]
                
                # init data buffer
                observationsRDA = torch.zeros((data_dict_batch['observationsRDA'].size(0), 
                                                1, 
                                                data_dict_batch['observationsRDA'].size(2), 
                                                data_dict_batch['observationsRDA'].size(3),
                                                data_dict_batch['observationsRDA'].size(4),
                                                data_dict_batch['observationsRDA'].size(5)
                                                ))
                rewards = torch.zeros((data_dict_batch['rewards_Phase'].size(0), len(txbf_idxs)))
                isDone = torch.zeros((data_dict_batch['terminals'].size(0), len(txbf_idxs)))
                
                for b in range(data_dict_batch['observationsRDA'].size(0)):
                    # observationsRDA: [frame, range, Doppler, ant] = [b, 10, 64, 128, 16]
                    observationsRDA[b,...] = data_dict_batch['observationsRDA'][b, txbf_idxs[b], ...]
                
                    rewards[b, :] = data_dict_batch['rewards_Phase'][b, txbf_idxs[b]]
                    isDone[b, :]= data_dict_batch['terminals'][b, txbf_idxs[b]]
                
                observationsRDA = observationsRDA.squeeze(1)
                # [batch, frame, range, Doppler, ant] --> [batch, Doppler, frame, range, ant]
                observationsRDA = observationsRDA.permute(0, 3, 1, 2, 4)
                

                # to device
                hybridBFAction = hybridBFAction.to(device, non_blocking=True)   
                observationsRDA = observationsRDA.to(device, non_blocking=True) 
                rewards = rewards.to(device, non_blocking=True)  
                isDone = isDone.to(device, non_blocking=True)    
                next_state = torch.rand(observationsRDA.shape).to(device, non_blocking=True)   # TODO torch.zeros()

                # train
                hybridBFAction_buffer = policy.train(observationsRDA, hybridBFAction, rewards, next_state, isDone, batch_size=args.batch_size, logger=logger)
                
                if args.debug:
                    print("---->sample iter:{}, observationsRDA size:{}, txbf_idx: {}".format(t, observationsRDA.size(), txbf_idxs))
                    
    # Evaluation episode
    if (t+1) % args.eval_freq == 0: 

        #eval_episodes = 100 if t+1 == int(args.max_timesteps) else args.eval_episodes
        
        eval_score = eval_policy(args, t+1, logger, policy, test_dataloader, args.eval_episodes)

        print("------------------Evaluation over: {} episodes: {}".format(args.eval_episodes, eval_score))

            
    if args.save_model:
        policy.save(args.model_dir)


if __name__=="__main__":
    args = get_parsers()

    train(args)




    




    
















