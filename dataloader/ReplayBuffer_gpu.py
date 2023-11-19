import numpy as np
import torch
from torch.autograd import Variable as V

import pdb

class ReplayBuffer_gpu(object):
    def __init__(self, state_dim, action_dim, max_size=int(128)):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        state_dim.insert(0, max_size)
        action_dim.insert(0, max_size)

        self.state_buf = torch.zeros(state_dim).to(self.device)
        self.action_buf = torch.zeros(action_dim).to(self.device)
        self.next_state_buf = torch.zeros(state_dim).to(self.device)
        self.reward_buf = torch.zeros((max_size, 1)).to(self.device)
        self.not_done_buf = torch.zeros((max_size, 1)).to(self.device)

        self.state = V(self.state_buf)
        self.action = V(self.action_buf)
        self.next_state = V(self.next_state_buf)
        self.reward = V(self.reward_buf)
        self.not_done = V(self.not_done_buf)

        

    def add(self, state, action, next_state, reward, done):
        
        

        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.next_state_buf[self.ptr] = next_state
        self.reward_buf[self.ptr] = reward
        self.not_done_buf[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        
        # pdb.set_trace()
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

        # return (
        #     torch.FloatTensor(self.state[ind]).to(self.device),
        #     torch.FloatTensor(self.action[ind]).to(self.device),
        #     torch.FloatTensor(self.next_state[ind]).to(self.device),
        #     torch.FloatTensor(self.reward[ind]).to(self.device),
        #     torch.FloatTensor(self.not_done[ind]).to(self.device)
        # )

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std
