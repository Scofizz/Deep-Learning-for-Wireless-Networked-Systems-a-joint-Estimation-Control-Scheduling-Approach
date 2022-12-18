#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


'''Initialize NNs'''
# Estimator
# Intelligent controller (actor, critic) 
# Transmission scheduler (DQN) 


class Estimator(nn.Module):
    def __init__(self, ESTIMATOR_INPUT_DIM, HIDDEN_SIZE, ESTIMATOR_OUTPUT_DIM):
        super(Estimator, self).__init__()
        
        self.gru = nn.GRU(ESTIMATOR_INPUT_DIM, HIDDEN_SIZE)
        self.l1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, ESTIMATOR_OUTPUT_DIM)
        

    def forward(self, state, action):
        self.gru.flatten_parameters()
        sa = torch.cat([state, action], 1)
        
        s_gru, _ = self.gru(sa.view(len(sa), 1, -1))
        s = s_gru.view(len(sa), -1)
        
        s = F.relu(self.l1(s))
        
        return self.l2(s)
    
class Actor(nn.Module):
    def __init__(self, CONTROLLER_OW_DIM, ACTOR_INPUT_DIM, HIDDEN_SIZE, ACTOR_OUTPUT_DIM, MAX_ACTION):
        super(Actor, self).__init__()
        
        self.pre = nn.Linear(CONTROLLER_OW_DIM, HIDDEN_SIZE)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        
        self.l1 = nn.Linear(ACTOR_INPUT_DIM, HIDDEN_SIZE)
        
        self.l2 = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, ACTOR_OUTPUT_DIM)

        self.max_action = MAX_ACTION


    def forward(self, ow, state):
        self.gru.flatten_parameters()
        
        a_his = F.relu(self.pre(ow))
        a_gru, _ = self.gru(a_his.view(len(ow), 1, -1))
        a_his = a_gru.view(len(ow), -1)
        
        a_cur = F.relu(self.l1(state))
        
        a = torch.cat([a_his, a_cur], -1)
        a = F.relu(self.l2(a))
        
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, CONTROLLER_OW_DIM, CRITIC_INPUT_DIM, HIDDEN_SIZE, CRITIC_OUTPUT_DIM):
        super(Critic, self).__init__()

        # Q1
        self.pre1 = nn.Linear(CONTROLLER_OW_DIM, HIDDEN_SIZE)
        self.gru1 = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        
        self.l1 = nn.Linear(CRITIC_INPUT_DIM, HIDDEN_SIZE)
        
        self.l2 = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, CRITIC_OUTPUT_DIM)

        # Q2
        self.pre2 = nn.Linear(CONTROLLER_OW_DIM, HIDDEN_SIZE)
        self.gru2 = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        
        self.l4 = nn.Linear(CRITIC_INPUT_DIM, HIDDEN_SIZE)
        
        self.l5 = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.l6 = nn.Linear(HIDDEN_SIZE, CRITIC_OUTPUT_DIM)
        
        
    def forward(self, ow, state, action):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1_his = F.relu(self.pre1(ow))
        q1_gru, _ = self.gru1(q1_his.view(len(ow), 1, -1))
        q1_his = q1_gru.view(len(ow), -1)
        
        q1_cur = F.relu(self.l1(sa))
        
        q1 = torch.cat([q1_his, q1_cur], -1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        # Q2
        q2_his = F.relu(self.pre2(ow))
        q2_gru, _ = self.gru2(q2_his.view(len(ow), 1, -1))
        q2_his = q2_gru.view(len(ow), -1)
        
        q2_cur = F.relu(self.l4(sa))
        
        q2 = torch.cat([q2_his, q2_cur], -1)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2


    def Q1(self, ow, state, action):
        self.gru1.flatten_parameters()
        sa = torch.cat([state, action], 1)
        
        q1_his = F.relu(self.pre1(ow))
        q1_gru, _ = self.gru1(q1_his.view(len(ow), 1, -1))
        q1_his = q1_gru.view(len(ow), -1)
        
        q1_cur = F.relu(self.l1(sa))
        
        q1 = torch.cat([q1_his, q1_cur], -1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1
        

class DQN(nn.Module):
    def __init__(self, SCHEDULER_OW_DIM, DQN_INPUT_DIM, HIDDEN_SIZE, DQN_OUTPUT_DIM):
        super(DQN, self).__init__()
        
        self.pre = nn.Linear(SCHEDULER_OW_DIM, HIDDEN_SIZE)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        
        self.l1 = nn.Linear(DQN_INPUT_DIM, HIDDEN_SIZE)
        
        self.l2 = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, DQN_OUTPUT_DIM)


    def forward(self, ow, state):
        self.gru.flatten_parameters()
        
        q_his = F.relu(self.pre(ow))
        q_gru, _ = self.gru(q_his.view(len(ow), 1, -1))
        q_his = q_gru.view(len(ow), -1)
        
        q_cur = F.relu(self.l1(state))
        
        q = torch.cat([q_his, q_cur], -1)
        q = F.relu(self.l2(q))
        
        return self.l3(q)

