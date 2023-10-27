import numpy as np
import torch
import argparse
import os

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

#from eval_policy import eval_policy

import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parsers():

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQLpolicy")                  # Policy name
    parser.add_argument("--train_dataset", default="/home/hx/fanl/HybridBF_DeepRL/datasets/train")
    parser.add_argument("--test_dataset", default="/home/hx/fanl/HybridBF_DeepRL/datasets/test")                    # dataset path
    parser.add_argument("--seed", default=3, type=int)              #  
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--sample_epochs", default=100, type=int)  
    parser.add_argument("--max_timesteps", default=1e3, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=False)        # Save model and optimizer parameters
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=1, type=int)      # Batch size default is 1
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='tmp', type=str)
    args = parser.parse_args()

    # Build work dir
    base_dir = 'runs'
    make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, 'dataset')
    make_dir(args.work_dir)

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
    hybridBFAction_buffer = torch.rand(2, tx_rx_spacae_dim).cuda()
    
    state_shape = [1, 128, 10, 64, 16] # [batch, frame, range, doppler, antenna]   
    action_shape = [121, 121] # [txbf, rxbf]
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
    dataset = OfflineDataset(args.train_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for t in trange(int(args.max_timesteps)):
    
        for sample_path in dataloader:
            data_dict = read_h5py_file(sample_path[0])

            for idx in range(args.sample_epochs):
                ''' Data preprocessing and to tensor
                '''
                hybridBFAction = V(hybridBFAction_buffer)

                txbf_idx = torch.argmax(hybridBFAction[0]).item()

                observationsRDA = data_dict['observationsRDA'][txbf_idx]   # [frame, range, Doppler, ant] = [10, 64, 128, 16]
                rewards = data_dict['rewards_PSINR'][txbf_idx] 

                # to tensor
                observationsRDA = torch.from_numpy(10*np.log10(abs(observationsRDA) ) )
                # hybridBFAction = torch.from_numpy(hybridBFAction)
                rewards = torch.from_numpy(rewards)

                # [frame, range, Doppler, ant] --> [Doppler, frame, range, ant]
                observationsRDA = observationsRDA.permute(2, 0, 1, 3)
                # add batch dim,  [batch, Doppler, frame, range, ant]
                observationsRDA = torch.unsqueeze(observationsRDA, 0)
                observationsRDA = observationsRDA.to(device, non_blocking=True) 
                hybridBFAction = hybridBFAction.to(device, non_blocking=True)   
                rewards = rewards.to(device, non_blocking=True)   

                next_state = torch.rand(observationsRDA.shape).to(device, non_blocking=True)   # TODO torch.zeros()
                not_done = 1
                
                hybridBFAction_buffer = policy.train(observationsRDA, hybridBFAction, rewards, next_state, not_done, batch_size=observationsRDA.size(0), logger=logger)
                

                if args.debug:
                    print("---->sample iter:{}, observationsRDA size:{}, txbf_idx: {}".format(idx, observationsRDA.size(), txbf_idx))
                
            
            # # Evaluation episode
            # if (t+1) % args.eval_greq == 0:
            #     eval_episodes = 100 if t+1 == int(args.max_timesteps) else args.eval_episodes
           
            #     eval_score = eval_policy(args, t+1, logger, policy, eval_episodes=eval_episodes)

            #     if args.save_model:
            #         policy.save(args.model_dir)


if __name__=="__main__":
    args = get_parsers()

    train(args)




    




    
















