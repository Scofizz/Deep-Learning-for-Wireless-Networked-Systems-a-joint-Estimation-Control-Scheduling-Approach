#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


from model import Estimator, Actor, Critic, DQN


# In[ ]:


'''Tailored TD3 Algorithm'''
# select_action: generate control signal by actor
# select_q: generate Q-value by critic
# predict_state: generate predicted state by estimator
# schedule_transmission & control_transmission: generate transmission scheduler action based on DQN output
# train: network training of estimator, intelligent controller (actor, critic) and transmission scheduler (DQN)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transmission scheduler action: 1-transmit; 0-not transmit
TRANSMISSION_ACTION_LIST = np.array([0,1])

# Hyperparameters for NN training
TIMESTEPS_BEFORE_TRAIN = 25e3
PRE_TRAINING_TIMESTEPS = 2e5
SORT_FREQ = int(1e5)

class Tailored_TD3(object):
    def __init__(self, SCHEDULER_OW_DIM, CONTROLLER_OW_DIM, ESTIMATOR_INPUT_DIM, DQN_INPUT_DIM, ACTOR_INPUT_DIM, MAX_ACTION, 
                 CRITIC_INPUT_DIM, HIDDEN_SIZE, ESTIMATOR_OUTPUT_DIM, DQN_OUTPUT_DIM, ACTOR_OUTPUT_DIM, CRITIC_OUTPUT_DIM, 
                GAMMA, TAU, TARGET_ACTOR_NOISE_STD, TARGET_ACTOR_NOISE_CLIP, ACTOR_UPDATE_DELAY, LR_DYNAMICS, LR):
        self.estimator = Estimator(ESTIMATOR_INPUT_DIM, HIDDEN_SIZE, ESTIMATOR_OUTPUT_DIM).to(device)
        self.estimator_optimizer = torch.optim.Adam(self.estimator.parameters(), lr=LR_DYNAMICS)
        
        self.dqn = DQN(SCHEDULER_OW_DIM, DQN_INPUT_DIM, HIDDEN_SIZE, DQN_OUTPUT_DIM).to(device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=LR)

        self.actor = Actor(CONTROLLER_OW_DIM, ACTOR_INPUT_DIM, HIDDEN_SIZE, ACTOR_OUTPUT_DIM, MAX_ACTION).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR)

        self.critic = Critic(CONTROLLER_OW_DIM, CRITIC_INPUT_DIM, HIDDEN_SIZE, CRITIC_OUTPUT_DIM).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.max_action = MAX_ACTION
        self.discount = GAMMA
        self.tau = TAU
        self.policy_noise = TARGET_ACTOR_NOISE_STD
        self.noise_clip = TARGET_ACTOR_NOISE_CLIP
        self.policy_freq = ACTOR_UPDATE_DELAY

        self.total_it = 0


    def select_action(self, ow, state):
        ow = torch.FloatTensor(ow.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(ow, state).cpu().data.numpy().flatten()
    
    
    def select_q(self, ow, state, action):
        ow = torch.FloatTensor(ow.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.critic.Q1(ow, state, action).cpu().data.numpy().flatten()

    
    def predict_state(self, state, action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)
        return self.estimator(state, action).cpu().data.numpy().flatten()
    
    
    def schedule_transmission(self, ow, state):
        ow = torch.FloatTensor(ow.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        schedule_action_value = self.dqn(ow, state)
        schedule_action = torch.max(schedule_action_value, 1)[1].cpu().data.numpy()[0]
        return np.array([schedule_action])
    
    
    def control_transmission(self, schedule_action):
        transmission_action_idx = int(schedule_action)
        transmission_action = TRANSMISSION_ACTION_LIST[transmission_action_idx]
        return transmission_action

    
    def train(self, estimator_replay_buffer, scheduler_replay_buffer, controller_replay_buffer):
        # Sort memory every n steps
        if(self.total_it % SORT_FREQ == 0):
            estimator_replay_buffer.sort_priorities()
            scheduler_replay_buffer.sort_priorities()
            controller_replay_buffer.sort_priorities()
            
        self.total_it += 1
        
        # Sample replay buffer (Estimator update)
        state_E, action_E, reward_E, next_state_E, not_done_E, idxs_E, sampling_weights_E = estimator_replay_buffer.sample()
        sampling_weights_E_sqrt = sampling_weights_E.sqrt()

        # Compute estimator loss
        estimator_loss = F.mse_loss(self.estimator(state_E, action_E), next_state_E)

        # Optimize the estimator
        self.estimator_optimizer.zero_grad()
        estimator_loss.backward()
        self.estimator_optimizer.step()
        
        # Sample replay buffer (Actor-critic update)
        ow, state, action, reward, next_ow, next_state, not_done, idxs, sampling_weights = controller_replay_buffer.sample_ow()
        sampling_weights_sqrt = sampling_weights.sqrt()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_ow, next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_ow, next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(ow, state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(sampling_weights_sqrt*current_Q1, 
                                 sampling_weights_sqrt*target_Q) + F.mse_loss(sampling_weights_sqrt*current_Q2, 
                                                                              sampling_weights_sqrt*target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(ow, state, self.actor(ow, state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update sampling priority according to AoI-based experience replay
        update_idxs = idxs.cpu().numpy().astype(int)
        TD_error = ((current_Q1 - target_Q)**2 + (current_Q2 - target_Q)**2).detach().cpu().numpy().flatten().astype(float)
        controller_replay_buffer.update_priorities(update_idxs, TD_error)
        
        if(self.total_it >= PRE_TRAINING_TIMESTEPS - TIMESTEPS_BEFORE_TRAIN):
            # Sample replay buffer (DQN update)
            ow_S, state_S, action_S, reward_S, next_ow_S, next_state_S, not_done_S, idxs_S, sampling_weights_S = scheduler_replay_buffer.sample_ow()

            with torch.no_grad():
                # Compute the target Q value
                target_Q_S = self.dqn_target(next_ow_S, next_state_S)
                target_Q_S = reward_S + not_done_S * self.discount * target_Q_S.max(1)[0].unsqueeze(1)

            # Get current Q estimates
            action_S = action_S.long()
            current_Q_S = self.dqn(ow_S, state_S).gather(1, action_S)

            # Compute dqn loss
            dqn_loss = F.mse_loss(current_Q_S, target_Q_S)

            # Optimize the dqn
            self.dqn_optimizer.zero_grad()
            dqn_loss.backward()
            self.dqn_optimizer.step()
            
            if self.total_it % 100 == 0:
                # Update target networks
                for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
                    target_param.data.copy_(param.data)
                    
            # Update sampling priority according to AoI-based experience replay
            update_idxs_S = idxs_S.cpu().numpy().astype(int)
            TD_error_S = ((current_Q_S - target_Q_S)**2).detach().cpu().numpy().flatten().astype(float)
            scheduler_replay_buffer.update_priorities(update_idxs_S, TD_error_S)

