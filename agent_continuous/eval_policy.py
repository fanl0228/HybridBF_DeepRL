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
def eval_policy(args, iter, count, logger: Logger, policy, test_dataloader, mmWave_doppler_dim=128, mmWave_frame_dim=3,
                mmWave_range_dim=64, mmWave_angle_dim=16, ):
    # Rxbf paramer
    mmWave_f0 = torch.tensor(7.7e10).to(device, non_blocking=True)
    mmWave_d = torch.tensor(0.5034).to(device, non_blocking=True)
    mmWave_D_BF = torch.tensor([0, 1, 2, 3, 11, 12, 13, 14, 46, 47, 48, 49, 50, 51, 52, 53]).to(device,
                                                                                                non_blocking=True)

    torch.manual_seed(args.seed)
    mmWave_doppler_dim = mmWave_doppler_dim
    mmWave_frame_dim = mmWave_frame_dim
    mmWave_range_dim = mmWave_range_dim
    mmWave_angle_dim = mmWave_angle_dim
    tx_rx_spacae_dim = 121  # txbf and rxbf search space [0:121] --> angle [-60, +60]

    hybridBFAction_buffer = torch.randn(1, 2, tx_rx_spacae_dim).to(device, non_blocking=True)
    observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim, mmWave_angle_dim).to(
        device, non_blocking=True)
    next_observationsRDA_buffer = torch.randn(mmWave_doppler_dim, mmWave_frame_dim, mmWave_range_dim,
                                              mmWave_angle_dim).to(device, non_blocking=True)
    rewards_buffer = torch.zeros((1,)).to(device, non_blocking=True)
    isDone_buffer = torch.zeros((1,)).to(device, non_blocking=True)

    lengths_eval = []
    final_reward_eval = []  # eval_score = []
    avg_reward_eval = []

    test_filenumbers = len(os.listdir(args.test_dataset))

    # for epiod_ in range(int(eval_episodes)):

    for t in trange(int(args.eval_episodes), desc="Train Episodes"):

        lengths_batch = []
        final_reward_batch = []

        avg_reward_batch = []
        final_txbfs_batch = []
        final_rxbfs_batch = []
        step_batch = []

        sample_num = 0
        sample = 0
        for data_dict_batch, file_name in test_dataloader:  # tqdm(test_dataloader, desc="Loading test datafile"):

            avg_reward_sample = 0
            final_reward_sample = 0.
            final_txbfs = 0
            final_rxbfs = 0

            isDone = torch.zeros((data_dict_batch['terminals'].size(0), 1))
            random_action = torch.rand(args.batch_size, 2, tx_rx_spacae_dim).cuda()

            random_txbf_idxs = torch.argmax(random_action[0, 0, :], -1)
            random_rxbf_idxs = torch.argmax(random_action[0, 1, :], -1)

            # observationsRDA_buffer = torch.clone(
            #     data_dict_batch['observationsRDA'][0, random_txbf_idxs.item(), :mmWave_frame_dim, ...]).to(device,
            #                                                                                                non_blocking=True)
            observationsRDA_buffer = torch.clone(
                data_dict_batch['observationsRDA'][0, random_txbf_idxs.item(), :mmWave_frame_dim, :mmWave_range_dim,
                int(2 * mmWave_doppler_dim / 4):int(2 * mmWave_doppler_dim * 3 / 4), :]).to(device, non_blocking=True)

            mmWave_WX = torch.sin(random_rxbf_idxs * torch.pi / 180.0).to(device, non_blocking=True)
            real_part = torch.cos(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                     non_blocking=True)
            imag_part = torch.sin(-2 * torch.pi * mmWave_f0 * mmWave_d * mmWave_D_BF * mmWave_WX).to(device,
                                                                                                     non_blocking=True)
            mmWave_RXBF_V = torch.exp(1j * (real_part + imag_part)).to('cuda')

            # observationsRDA_buffer = observationsRDA_buffer * mmWave_RXBF_V
            # observationsRDA_buffer = torch.fft.fft(observationsRDA_buffer, mmWave_angle_dim, axis=-1).to(
            #     device, non_blocking=True)

            observationsRDA_buffer = torch.matmul(observationsRDA_buffer, mmWave_RXBF_V).unsqueeze(-1)

            observationsRDA_buffer = 10 * torch.log10(torch.abs(observationsRDA_buffer) + 1e-8).to(device,
                                                                                                   non_blocking=True)

            observationsRDA_buffer = (observationsRDA_buffer -
                                      observationsRDA_buffer.min()) / (
                                                 observationsRDA_buffer.max() - observationsRDA_buffer.min() + 1e-3)

            # observationsRDA_buffer = observationsRDA_buffer.permute(1, 2, 0,
            #                                                         3)  # [F,R,D,A]] --> [D,F,R,A]
            observationsRDA_buffer = torch.squeeze(observationsRDA_buffer)

            txbf_list = []
            rxbf_list = []
            reward_list = []

            # while not done:
            # while (isDone.mean().cpu().item() == 0):
            for step in range(int(args.eval_max_steps)):
                # convert to Variable

                observationsRDA = V(observationsRDA_buffer)
                # print('buffer:',observationsRDA_buffer[0, 0, 0, :])
                # print(observationsRDA[0, 0, 0, :])

                # rewards = V(rewards_buffer)
                # isDone = V(isDone_buffer)

                # print(observationsRDA)
                # get action
                action = policy.get_action(observationsRDA.unsqueeze(0).to(device, non_blocking=True))
                # hybridBFAction = V(hybridBFAction_buffer)

                # get txbf, rxbf value
                # txbf_idxs = torch.argmax(hybridBFAction[0, 0, :], -1)
                # rxbf_idxs = torch.argmax(hybridBFAction[0, 1, :], -1)

                txbf_idxs = int(action[0].item() * 60) + 60
                rxbf_idxs = int(action[1].item() * 60) + 60

                txbf_list.append(txbf_idxs)
                rxbf_list.append(rxbf_idxs)

                # txbf_list.append(txbf_idxs)
                # rxbf_list.append(rxbf_idxs)

                # 1. observationsRDA: [frame, range, Doppler, ant]
                rewards = data_dict_batch['rewards_PSINR'][0, txbf_idxs]

                reward_list.append(rewards.item())

                isDone = data_dict_batch['terminals'][0, txbf_idxs]

                # break
                if (isDone.cpu().item() != 0):
                    avg_reward_sample += rewards
                    final_reward_sample = rewards
                    break

                #     
                # next_observationsRDA_buffer = torch.clone(
                #     data_dict_batch['observationsRDA'][0, txbf_idxs.item(), :mmWave_frame_dim,...]).to(device, non_blocking=True)
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
                next_observationsRDA_buffer = (next_observationsRDA_buffer -
                                               next_observationsRDA_buffer.min()) / (
                                                          next_observationsRDA_buffer.max() - next_observationsRDA_buffer.min() + 1e-3)

                # next_observationsRDA_buffer = next_observationsRDA_buffer.permute(1, 2, 0,
                #                                                                   3)  # [F,R,D,A]] --> [D,F,R,A]
                next_observationsRDA_buffer = torch.squeeze(next_observationsRDA_buffer)

                next_observationsRDA = V(next_observationsRDA_buffer)
                # print('next_buffer:', next_observationsRDA_buffer[0,0,0,:])
                # print('next:',next_observationsRDA[0, 0, 0, :])

                observationsRDA_buffer = torch.clone(next_observationsRDA).to(device, non_blocking=True)

                # logger
                eval_num = iter * args.eval_episodes * test_filenumbers * args.eval_max_steps + t * test_filenumbers * args.eval_max_steps + sample_num * args.eval_max_steps + step
                logger.log('eval/step_reward', rewards.item(), eval_num)

                # 
                avg_reward_sample += rewards
                final_reward_sample = rewards

                # print("----> step:{}, reward:{}, isDone:{}, txbf:{}, rxbf: {} ".format(eval_num, rewards, isDone, txbf_idxs, rxbf_idxs))

            avg_reward_sample /= step
            final_reward_sample = rewards
            final_txbfs = txbf_idxs
            final_rxbfs = rxbf_idxs
            step_batch.append(step)

            print('\n######################################')
            print(f'sample name: {file_name}')
            print('total count:',count)
            print(f'txbf: {txbf_list}')
            print(f'rxbf: {rxbf_list}')
            print(f'reward: {reward_list}')
            print(f'step number: {step}')
            print('######################################\n')
            sample += 1

            # logger
            eval_sample_num = iter * args.eval_episodes * test_filenumbers + t * test_filenumbers + sample_num
            # print("==================eval_sample_num: {}".format(eval_sample_num))

            logger.log('eval/avg_reward_sample', avg_reward_sample.item(), eval_sample_num)
            logger.log('eval/final_reward_sample', final_reward_sample.item(), eval_sample_num)
            logger.log('eval/final_txbfs_sample', final_txbfs, eval_sample_num)
            logger.log('eval/final_rxbfs_sample', final_rxbfs, eval_sample_num)

            lengths_batch.append(step)
            avg_reward_batch.append(avg_reward_sample)
            final_reward_batch.append(final_reward_sample)
            final_txbfs_batch.append(final_txbfs)
            final_rxbfs_batch.append(final_rxbfs)

        final_reward_eval.append(np.mean(final_reward_batch))
        lengths_eval.append(np.mean(lengths_batch))
        avg_reward_eval.append(np.mean(avg_reward_batch))
        # print("----> sample_num:{}, final_reward_eval:{}, lengths_eval:{}, avg_reward_eval: {} ".format(sample_num, final_reward_eval, lengths_eval, avg_reward_eval))
        sample_num += 1

    logger.log('eval/lengths_mean', np.mean(lengths_eval), iter)
    logger.log('eval/lengths_std', np.std(lengths_eval), iter)
    logger.log('eval/avg_reward_mean', np.mean(avg_reward_eval), iter)
    logger.log('eval/avg_reward_std', np.std(avg_reward_eval), iter)

    logger.log('eval/final_reward', np.mean(final_reward_eval), iter)
    logger.log('eval/avg_steps', np.mean(step_batch), iter)

    final_reward_eval = np.mean(final_reward_eval)

    print("------------> Evaluation over iter: {}, avg_reward_eval:{}, final_reward_eval: {}".format(iter,
                                                                                                     avg_reward_eval,
                                                                                                     final_reward_eval))
    return final_reward_eval
