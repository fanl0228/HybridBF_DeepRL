import copy
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V

from agents.actor import Actor
from agents.critic import ActionCritic, ValueCritic

from dataloader import OfflineDataset
from dataloader.h5py_opts import read_h5py_file
import utils
from utils.log import Logger

import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IQLpolicy(object):

    def __init__(
        self, 
        state_shape,
        action_shape,
        expectile,
        discount,
        tau,
        temperature,
    ): 
        if type(state_shape) != np.ndarray:
            self.state_shape = np.array(state_shape)
        else:
            self.state_shape = state_shape

        if type(action_shape) != np.ndarray:
            self.action_shape = np.array(action_shape)
        else:
            self.action_shape = action_shape
    
        self.actor = Actor(self.state_shape, self.action_shape).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        self.critic = ActionCritic(self.state_shape, self.action_shape).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = ValueCritic(self.state_shape).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.temperature = temperature
        self.total_it = 0
        self.expectile = expectile


    def get_model(self):
        return self.actor, self.critic, self.value


    def loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    
    def update_v(self, states, actions, logger=None):

        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2).detach()
        
        # use Advantage Function update
        v = self.value(states)
        
        value_loss = self.loss(q - v, expectile=self.expectile).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        logger.log('train/value_loss', value_loss, self.total_it)
        logger.log('train/v', v.mean(), self.total_it)
    
    
    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_v = self.value(next_states)
            # Compute the target Q value
            target_q = (rewards + self.discount * not_dones * next_v).detach()
        
        # Compute critic loss
        q1, q2 = self.critic(states, actions)
        # MSE loss
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logger.log('train/critic_loss', critic_loss, self.total_it)
        logger.log('train/q1', q1.mean(), self.total_it)
        logger.log('train/q2', q2.mean(), self.total_it)
    
    
    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def update_actor(self, states, actions=None, logger=None):
        
        with torch.no_grad():
            v = self.value(states)
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2)
            exp_a = torch.exp((q - v) * self.temperature)
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        act_txbf, act_rxbf = self.actor(states)
        
        act_txbf = act_txbf.unsqueeze(1)
        act_rxbf = act_rxbf.unsqueeze(1)
        HBFact = torch.cat((act_txbf, act_rxbf), axis=1)
        
        actor_loss = (exp_a.unsqueeze(-1) * ((HBFact - actions)**2).view(exp_a.size(0),-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        logger.log('train/actor_loss', actor_loss, self.total_it)
        logger.log('train/adv', (q - v).mean(), self.total_it)

        return HBFact

    
    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #pdb.set_trace()

        act_txbf, act_rxbf  = self.actor.get_action(state)#.cpu().data.numpy()#.flatten() 
        
        #act_txbf = self.softmax_txbf(act_txbf)
        #act_rxbf = self.softmax_rxbf(act_rxbf)

        act_txbf = act_txbf.unsqueeze(1)
        act_rxbf = act_rxbf.unsqueeze(1)
        HBFaction = torch.cat((act_txbf, act_rxbf), axis=1)

        return HBFaction

    
    def train(self, observations, actions, rewards, next_state, not_done, batch_size=1, logger=None):
        self.total_it += 1
        
        # Update
        self.update_v(observations, actions, logger)
        HBFact = self.update_actor(observations, actions, logger)
        self.update_q(observations, actions, rewards, next_state, not_done, logger)
        self.update_target()

        return HBFact

    
    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))
        torch.save(self.actor_scheduler.state_dict(), os.path.join(
            model_dir, f"actor_scheduler_s{str(self.total_it)}.pth"))

        torch.save(self.value.state_dict(), os.path.join(model_dir, f"value_s{str(self.total_it)}.pth"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            model_dir, f"value_optimizer_s{str(self.total_it)}.pth"))




if __name__=="__main__":

    policy = IQLpolicy(
        state_shape = [1, 128, 10, 64, 16],   # [batch, frame, range, doppler, antenna]
        action_shape = [121, 121],
        expectile=0.7,
        discount=0.99,
        tau=0.005,
        temperature=3.0,
    )    
    
    actor, critic, value = policy.get_model()
    print("-----------------------------------------")
    print(actor)
    print("-----------------------------------------")
    print(critic)
    print("-----------------------------------------")
    print(value)

    train_dataset = "/home/hx/fanl/HybridBF_DeepRL/datasets/train"
    dataset = OfflineDataset(train_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sample_epoch = 20


    seed = 3.0
    torch.manual_seed(seed)
    tx_rx_spacae_dim = 121

    hybridBFAction_buffer = torch.rand(2, tx_rx_spacae_dim).cuda()
    #hybridBFAction = V(torch.rand(2, 121).cuda())
    
    # Build work dir
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, 'tmp')
    utils.make_dir(base_dir)
    work_dir = os.path.join(base_dir, 'dataset')
    utils.make_dir(work_dir)
    
    logger = Logger(work_dir, use_tb=True)

    # dataloader
    for sample_path in dataloader:
        data_dict = read_h5py_file(sample_path[0])
        
        for idx in range(sample_epoch):
            
            '''
                Data preprocessing and to tensor
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
            

            # pdb.set_trace()
            print("---->sample iter:{}, observationsRDA size:{}, txbf_idx: {}".format(idx, observationsRDA.size(), txbf_idx))
            
            





