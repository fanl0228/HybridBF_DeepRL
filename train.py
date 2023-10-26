import numpy as np
import torch
import argparse
import os

from tqdm import trange
from coolname import generate_slug
import time
import json
from torch.utils.data import Dataset, DataLoader


from utils.log import Logger
from utils.common import make_dir, snapshot_src 
from dataloader import OfflineDataset
from dataloader.h5py_opts import read_h5py_file
from eval_policy import eval_policy
from agents.IQLPolicy import IQLpolicy


import pdb


def get_parsers():

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQLpolicy")                  # Policy name
    parser.add_argument("--train_dataset", default="/home/hx/fanl/HybridBF_DeepRL/mmWaveRL/datasets/train")
    parser.add_argument("--test_dataset", default="/home/hx/fanl/HybridBF_DeepRL/mmWaveRL/datasets/test")                    # dataset path
    parser.add_argument("--seed", default=0, type=int)              #  
    parser.add_argument("--eval_freq", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e3, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=False)        # Save model and optimizer parameters
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
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
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, 'dataset')
    utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==========================================================")
    print(" Device: {}".format(device))
    print(" Args: {}".format(args))
    print("==========================================================")

    return args


def train(args):
    # Initialize policy
    
    # Set seeds
    # args.seed
    # env.action_space.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    state_dim = [10, 64, 128, 16] # [frame, range, doppler, ant]
    action_dim = [2, 121] # [txbf/rxbf, angle]


    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "expectile": args.expectile,
    }
    if args.policy == 'IQLpolicy':
        policy = IQLpolicy(**kwargs)
        pass
    else:
        raise NotImplementedError

    logger = Logger(args.work_dir, use_tb=True)

    dataset = OfflineDataset(args.train_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # dataloader
    for sample_path in dataloader:
        data_dict = read_h5py_file(sample_path[0])
        
        print(data_dict.keys())
        print('---> observationsRD shape:{}'.format(data_dict['observationsRD'].shape))
        print('---> observationsRA shape:{}'.format(data_dict['observationsRA'].shape))
        print('---> rewards_PSINR shape:{}'.format(data_dict['rewards_PSINR'].shape))
        print('---> rewards_Intensity shape:{}'.format(data_dict['rewards_Intensity'].shape))
        print('---> rewards_Phase shape:{}'.format(data_dict['rewards_Phase'].shape))

        for t in trange(int(args.max_timesteps)):
        
            policy.train(data_dict, args.batch_size, logger=logger)

            # # Evaluation episode
            # if (t+1) % args.eval_greq == 0:
            #     eval_episodes = 100 if t+1 == int(args.max_timesteps) else args.eval_episodes
           
            #     eval_score = eval_policy(args, t+1, logger, policy, eval_episodes=eval_episodes)

            #     if args.save_model:
            #         policy.save(args.model_dir)


if __name__=="__main__":
    args = get_parsers()

    train(args)




    




    
















