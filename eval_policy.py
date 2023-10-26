import numpy as np
import torch
import argparse
import os
from tqdm import trange
from coolname import generate_slug
import time
import json
from log import Logger

import utils


import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


# def eval_policy(args, iter, video: VideoRecorder, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
def eval_policy(args, iter, logger: Logger, policy, test_dataset, eval_episodes):

    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    lengths = []
    returns = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        # video.init(enabled=(args.save_video and _ == 0))
        state, done = eval_env.reset(), False

        steps = 0
        episode_return = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            episode_return += reward
            steps += 1
        lengths.append(steps)
        returns.append(episode_return)
        # video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

    avg_reward /= eval_episodes
    eval_score = eval_env.get_normalized_score(avg_reward)

    logger.log('eval/lengths_mean', np.mean(lengths), iter)
    logger.log('eval/lengths_std', np.std(lengths), iter)
    logger.log('eval/returns_mean', np.mean(returns), iter)
    logger.log('eval/returns_std', np.std(returns), iter)
    logger.log('eval/d4rl_score', eval_score, iter)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score