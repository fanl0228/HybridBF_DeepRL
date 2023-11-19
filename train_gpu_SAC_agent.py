import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from utils.common import make_dir, make_dirs, snapshot_src
from dataloader import OfflineDataset
from dataloader.OfflineDataset_mini import OfflineDataset_mini
from dataloader.h5py_opts import read_h5py_file
# from dataloader.ReplayBuffer_gpu import ReplayBuffer_gpu
#
# from agents.IQLPolicy import IQLpolicy
#
# from agents.eval_policy import eval_policy

from agent_continuous.sac import SAC_Agent
from agent_continuous.utils import ReplayPool, Transition, make_checkpoint
from agent_continuous.eval_policy import eval_policy

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parsers():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="SACpolicy")  # Policy name
    parser.add_argument("--train_dataset", default="/data/mmWaveRL_Datasets/v11_nologData/train") # train_nolog_norm_v11
    parser.add_argument("--test_dataset", default="/data/mmWaveRL_Datasets/v11_nologData/val")  #  val_nolog_norm_v11
    parser.add_argument("--seed", default=100, type=int)  #
    parser.add_argument("--train_max_steps", default=100, type=int)  # Max time steps to run environment
    parser.add_argument("--train_episodes", default=1000, type=int)
    parser.add_argument("--save_model", default=True, action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--eval_freq", default=1, type=int)  # How often (time steps) we evaluate
    parser.add_argument('--eval_max_steps', default=100, type=int)
    parser.add_argument('--eval_episodes', default=1, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=True, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=4, type=int)  # Batch size default is 16 (train faster)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='tmp_v11_norm_nolog_loss_final2', type=str)
    parser.add_argument('--model_dir', default='/data/mmWaveRL_Datasets/runs_gpu/tmp_v11_norm_nolog_loss_final2/savemodel', type=str)
    args = parser.parse_args()

    # Build work dir
    base_dir = 'runs_gpu'
    make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, 'dataset')
    make_dir(args.work_dir)
    make_dirs(args.model_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("==========================================================")
    print(" Device: {}".format(device))
    print(" Args: {}".format(args))
    print("==========================================================")

    return args


def train(args):
    # Rxbf paramer
    mmWave_f0 = torch.tensor(7.7e10).to(device, non_blocking=True)
    mmWave_d = torch.tensor(0.5034).to(device, non_blocking=True)
    mmWave_D_BF = torch.tensor([0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53]).to(device,
                                                                                                non_blocking=True)

    mmWave_doppler_dim = 64
    mmWave_frame_dim = 3
    mmWave_range_dim = 64
    mmWave_angle_dim = 1
    tx_rx_spacae_dim = 121  # txbf and rxbf search space [0:121] --> angle [-60, +60]

    step_penalty = 0.01

    state_shape = [args.batch_size, mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim]
    action_shape = [args.batch_size, tx_rx_spacae_dim, tx_rx_spacae_dim]  # [txbf, rxbf]
    kwargs = {
        "state_shape": state_shape,
        "action_shape": action_shape,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "expectile": args.expectile,
    }

    hybridBFAction_buffer = torch.randn(1, 2, tx_rx_spacae_dim).to(device, non_blocking=True)
    observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim).to(
        device, non_blocking=True)
    next_observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim,
                                              mmWave_angle_dim).to(device, non_blocking=True)
    rewards_buffer = torch.zeros((1,)).to(device, non_blocking=True)
    isDone_buffer = torch.zeros((1,)).to(device, non_blocking=True)

    replay_buffer = ReplayPool([mmWave_frame_dim, mmWave_doppler_dim,mmWave_range_dim],
                               2)

    if args.policy == 'SACpolicy':
        policy = SAC_Agent(args.seed, state_shape, action_dim=2, batchsize=args.batch_size)
    else:
        raise NotImplementedError

    # if args.debug:
    #     actor, critic, value = policy.get_model()
    #     print("--->actor model:{}\n --> critic model:{}\n --> value model{}".format(actor, critic, value))

    # Logger
    logger = Logger(args.work_dir, use_tb=True)

    # Dataset
    start = time.perf_counter()
    train_dataset = OfflineDataset_mini(args.train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = OfflineDataset_mini(args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    end = time.perf_counter()
    print("========>>>>>>>load all dataset time(s):{}".format(end - start))

    cnt = 0
    for t in trange(int(args.train_episodes), desc="Train Episodes"):

        time_start_epi = time.perf_counter()
        sample = 0
        step_list = []
        for data_dict_batch, _ in tqdm(train_dataloader, desc="Loading train datafile"):

            random_action = torch.randn(1, 2, tx_rx_spacae_dim).to(device, non_blocking=True)

            random_txbf_idxs = torch.argmax(random_action[0, 0, :], -1)
            random_rxbf_idxs = torch.argmax(random_action[0, 1, :], -1)

            observationsRDA_buffer = torch.clone(
                data_dict_batch['observationsRDA'][0, random_txbf_idxs.item(), :mmWave_frame_dim, :mmWave_range_dim,
                int(2 * mmWave_doppler_dim / 4):int(2 * mmWave_doppler_dim * 3 / 4), :]).to(device, non_blocking=True)

            mmWave_WX = torch.sin(random_rxbf_idxs * torch.pi / 180.0).to(device, non_blocking=True)
            real_part = torch.cos(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                     non_blocking=True)
            imag_part = torch.sin(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                     non_blocking=True)
            mmWave_RXBF_V = torch.exp(1j * (real_part + imag_part)).to('cuda')
            
            # pdb.set_trace()
            observationsRDA_buffer = torch.matmul(observationsRDA_buffer, mmWave_RXBF_V).unsqueeze(-1)
            #observationsRDA_buffer = torch.sum(observationsRDA_buffer,-1).unsqueeze(-1)
            # observationsRDA_buffer = torch.fft.fft(observationsRDA_buffer, mmWave_angle_dim, axis=-1).to(
            #     device, non_blocking=True)

            observationsRDA_buffer = 10 * torch.log10(torch.abs(observationsRDA_buffer) + 1e-8).to(device,
                                                                                                   non_blocking=True)

            observationsRDA_buffer = (observationsRDA_buffer -
                                      observationsRDA_buffer.min()) / (
                                             observationsRDA_buffer.max() - observationsRDA_buffer.min() + 1e-3)

            # observationsRDA_buffer = observationsRDA_buffer.permute(1, 2, 0,
            #                                                         3)  # [F,R,D,A]] --> [D,F,R,A]
            # print('shape:', observationsRDA_buffer.shape)
            observationsRDA_buffer = torch.squeeze(observationsRDA_buffer)
            # print('shape:',observationsRDA_buffer.shape)

            txbf_list = []
            rxbf_list = []
            reward_list = []

            for step in range(int(args.train_max_steps)):

                # convert to Variable

                observationsRDA = V(observationsRDA_buffer)

                # rewards = V(rewards_buffer)
                # isDone = V(isDone_buffer)

                # get action
                action = policy.get_action(
                    observationsRDA.unsqueeze(0).to(device, non_blocking=True))

                # hybridBFAction = V(hybridBFAction_buffer)

                # get txbf, rxbf value
                # txbf_idxs = torch.argmax(hybridBFAction[0, 0, :], -1)
                # rxbf_idxs = torch.argmax(hybridBFAction[0, 1, :], -1)
                # print('#####after')
                # print(hybridBFAction[0, 0, :])
                # print(hybridBFAction[0, 1, :])

                # print('#####')
                # print('action: ', action)

                txbf_idxs = int(action[0].item() * 60) + 60
                rxbf_idxs = int(action[1].item() * 60) + 60

                txbf_list.append(txbf_idxs)
                rxbf_list.append(rxbf_idxs)

                # 1. observationsRDA: [frame, range, Doppler, ant]
                rewards = data_dict_batch['rewards_PSINR'][0, txbf_idxs]

                # rewards -= step * step_penalty

                reward_list.append(rewards.item())

                isDone = data_dict_batch['terminals'][0, txbf_idxs]

                next_observationsRDA_buffer = torch.clone(
                    data_dict_batch['observationsRDA'][0, txbf_idxs, :mmWave_frame_dim, :mmWave_range_dim,
                    int(2 * mmWave_doppler_dim / 4):int(2 * mmWave_doppler_dim * 3 / 4), :]).to(device,
                                                                                                non_blocking=True)

                # 2. action RXBF, rx antenna is 16
                mmWave_WX = torch.sin(torch.tensor(rxbf_idxs) * torch.pi / 180.0).to(device, non_blocking=True)
                real_part = torch.cos(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                         non_blocking=True)
                imag_part = torch.sin(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                         non_blocking=True)
                mmWave_RXBF_V = torch.exp(1j * (real_part + imag_part)).to('cuda')
                # next_observationsRDA_buffer = next_observationsRDA_buffer * mmWave_RXBF_V
                # next_observationsRDA_buffer = torch.fft.fft(next_observationsRDA_buffer, mmWave_angle_dim, axis=-1).to(
                #     device, non_blocking=True)

                next_observationsRDA_buffer = torch.matmul(next_observationsRDA_buffer, mmWave_RXBF_V).unsqueeze(-1)

                next_observationsRDA_buffer = 10 * torch.log10(torch.abs(next_observationsRDA_buffer) + 1e-8).to(device,
                                                                                                                 non_blocking=True)

                # 3. Normalize
                next_observationsRDA_buffer = ((next_observationsRDA_buffer -
                                                next_observationsRDA_buffer.min()) /
                                               (next_observationsRDA_buffer.max() -
                                                next_observationsRDA_buffer.min() + 1e-3))

                # next_observationsRDA_buffer = next_observationsRDA_buffer.permute(1, 2, 0,
                #                                                                   3)  # [F,R,D,A]] --> [D,F,R,A]

                next_observationsRDA_buffer = torch.squeeze(next_observationsRDA_buffer)
                next_observationsRDA = V(next_observationsRDA_buffer)

                # print("========\n obs{}, \n nexobs:{}".format(observationsRDA[0,0,0,:].cpu().data.numpy(), next_observationsRDA[0,0,0,:].cpu().data.numpy(),))

                # add to replaybuffer
                # observationsRDA.cpu().numpy()
                # action.cpu().numpy()
                # next_observationsRDA.cpu().numpy()
                # rewards.cpu().numpy()
                # isDone.cpu().numpy()
                replay_buffer.push(Transition(observationsRDA.cpu().numpy(), action, rewards.cpu().numpy(),
                                              next_observationsRDA.cpu().numpy(),
                                              isDone.cpu().numpy()))

                observationsRDA_buffer = torch.clone(next_observationsRDA).to(device, non_blocking=True)

                if (cnt + 1) % 32 == 0:
                    # for _ in range(4):
                    # train model
                    q1_loss, q2_loss, pi_loss, a_loss = policy.optimize(replay_buffer, 4)
                    logger.log('train/Q-func_1', q1_loss, cnt)
                    logger.log('train/Q-func_2', q2_loss, cnt)
                    logger.log('train/policy', pi_loss, cnt)
                    logger.log('train/a_loss', a_loss, cnt)
                    logger.log('train/alpha', np.exp(policy.log_alpha.item()), cnt)
                    # print('training')
                    # print("========train_num:{}, txbf:{}, rxbf:{}".format(train_num, txbf_idxs.item(), rxbf_idxs.item()))

                    # logger.log_image("train/state_RD", next_observationsRDA[:,0,:,0].unsqueeze(0), cnt)
                    # logger.log_image("train/state_RA",  torch.mean(next_observationsRDA[:,0,:,:],dim=0).unsqueeze(0), cnt)

                # break
                if (isDone.cpu().item() != 0):
                    break

                # logger
                logger.log('train/txbf_idxs', txbf_idxs, cnt)
                logger.log('train/rxbf_idxs', rxbf_idxs, cnt)

                cnt += 1

            # print('\n######################################')
            # print(f'sample no: {sample}')
            # print(f'txbf: {txbf_list}')
            # print(f'rxbf: {rxbf_list}')
            # print(f'reward: {reward_list}')
            # print(f'step number: {step}')
            # print('######################################\n')
            # sample += 1
            step_list.append(step)
        logger.log('train/step_avg', np.mean(step_list), t)
        time_end_epi = time.perf_counter()
        print("========>>>>>>> train time (s):{}".format(time_end_epi - time_start_epi))

        if (t + 1) % args.eval_freq == 0:
            # Evaluation number
            start = time.perf_counter()
            eval_score = eval_policy(args, t, cnt,logger, policy, test_dataloader, mmWave_doppler_dim, mmWave_frame_dim,
                                     mmWave_range_dim, mmWave_angle_dim)
            end = time.perf_counter()
            print("========>>>>>>>> Evaluation number: {} eval_score: {}, time consumption(s):{}".format(t, eval_score,
                                                                                                         end - start))

        if (t + 1) % args.eval_freq == 0:
            make_checkpoint(policy, cnt, args.model_dir)
            # policy.save(args.model_dir)


if __name__ == "__main__":
    args = get_parsers()

    train(args)
