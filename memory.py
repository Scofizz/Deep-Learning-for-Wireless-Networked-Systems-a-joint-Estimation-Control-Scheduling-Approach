#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import namedtuple
import numpy as np
import random
import math
import torch


# In[ ]:


'''Initialize Experience Replay Buffer'''
# PrioritizedExperienceReplayBuffer: transitions in the format of [state, action, reward, next_state, done]
# PrioritizedExperienceReplayBuffer_OW: transitions in the format of [history, state, action, reward, next_history, next_state, done]

Experience = namedtuple("Experience", ("states", "actions", "rewards", "next_states", "dones"))
Experience_OW = namedtuple("Experience_OW", ("ows", "states", "actions", "rewards", "next_ows", "next_states", "dones")) 

class PrioritizedExperienceReplayBuffer:

    def __init__(self, alpha, beta, BATCH_SIZE, MEMORY_SIZE, MEMORY_CAPACITY):
      
        self._batch_size = BATCH_SIZE
        self._buffer_size = MEMORY_SIZE
        self._buffer_cap = MEMORY_CAPACITY
        self._buffer_length = 0 # current buffer length
        self._buffer = np.empty(self._buffer_cap, dtype=[("priority", np.float32), ("experience", Experience)])
        
        self._alpha = alpha
        self._beta = beta
        self._random_state = np.random.RandomState()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def buffer_len(self):
        return self._buffer_length
    
    def is_full(self) -> bool:
        return self._buffer_length == self._buffer_cap
    
    def add(self, experience: Experience, AoI):
        priority = -AoI
        if self.is_full():
            self._buffer = np.insert(self._buffer,self._buffer_cap,(priority, experience)) # push to end (newest)
            self._buffer = np.delete(self._buffer,0) # pop the first (oldest)
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def sample(self):
        # generate rank-based sampling probability sequence
        if(self._buffer_length >= self._buffer_size):
            rank_ps = np.arange(self._buffer_size)+1
        else:
            rank_ps = np.arange(self._buffer_length)+1
        sampling_probs = rank_ps**self._alpha / np.sum(rank_ps**self._alpha, dtype=np.int64)
        # sample transitions 
        idxs_ps = self._random_state.choice(np.arange(rank_ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        if(self._buffer_length >= self._buffer_size):
            idxs = idxs_ps + (self._buffer_length-self._buffer_size)
        else:
            idxs = idxs_ps
        # compute sampling weights
        experiences = self._buffer["experience"][idxs]
        batch = Experience(*zip(*experiences))
        
        if(self._buffer_length >= self._buffer_size):
            weights = (self._buffer_size * sampling_probs[idxs_ps])**-self._beta
        else:
            weights = (self._buffer_length * sampling_probs[idxs_ps])**-self._beta
        normalized_weights = weights / weights.max()
                        
        return (
            torch.FloatTensor(np.array(batch.states)).to(self.device),
            torch.FloatTensor(np.array(batch.actions)).to(self.device),
            torch.FloatTensor(np.array(batch.rewards)).to(self.device),
            torch.FloatTensor(np.array(batch.next_states)).to(self.device),
            torch.FloatTensor(np.array(batch.dones)).to(self.device),
            torch.FloatTensor(np.array(idxs)).to(self.device),
            torch.FloatTensor(np.array(normalized_weights)).to(self.device)
        )
    
    # Update the ranking value of transitions 
    def update_priorities(self, idxs: np.array, priorities: np.array):
        for i in idxs:
            self._buffer["priority"][i] = float(math.floor(self._buffer["priority"][i]))
        
        priorities_sigmoid = (1/(1 + np.exp(-priorities))).round(decimals=4) - 1e-4 # bias for rounding to avoid sigmoid≈1
        self._buffer["priority"][idxs] += priorities_sigmoid
    
    # Sort transitions in the memory based on ranking value
    def sort_priorities(self):
        if(self._buffer_length >= self._buffer_size):
            buffer_list = list(self._buffer[(self._buffer_length - self._buffer_size):self._buffer_length])
            buffer_list.sort(key=lambda x:x[0])

            self._buffer_length = self._buffer_size
        else:
            buffer_list = list(self._buffer[:self._buffer_length])
            buffer_list.sort(key=lambda x:x[0])
        
        self._buffer[:self._buffer_length] = np.array(buffer_list, 
                                                      dtype=[("priority", np.float32), ("experience_ow", Experience_OW)])
    

class PrioritizedExperienceReplayBuffer_OW:

    def __init__(self, alpha, beta, BATCH_SIZE, MEMORY_SIZE, MEMORY_CAPACITY):
      
        self._batch_size = BATCH_SIZE
        self._buffer_size = MEMORY_SIZE
        self._buffer_cap = MEMORY_CAPACITY
        self._buffer_length = 0 # current buffer length
        self._buffer = np.empty(self._buffer_cap, dtype=[("priority", np.float32), ("experience_ow", Experience_OW)])
        
        self._alpha = alpha
        self._beta = beta
        self._random_state = np.random.RandomState()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def buffer_len(self):
        return self._buffer_length
    
    def is_full(self) -> bool:
        return self._buffer_length == self._buffer_cap
    
    def add(self, experience_ow: Experience_OW, AoI):
        priority = -AoI
        if self.is_full():
            self._buffer = np.insert(self._buffer,self._buffer_cap,(priority, experience_ow)) # push to end (newest)
            self._buffer = np.delete(self._buffer,0) # pop the first (oldest)
        else:
            self._buffer[self._buffer_length] = (priority, experience_ow)
            self._buffer_length += 1

    def sample_ow(self):
        # generate rank-based sampling probability sequence
        if(self._buffer_length >= self._buffer_size):
            rank_ps = np.arange(self._buffer_size)+1
        else:
            rank_ps = np.arange(self._buffer_length)+1
        sampling_probs = rank_ps**self._alpha / np.sum(rank_ps**self._alpha, dtype=np.int64)
        # sample transitions
        idxs_ps = self._random_state.choice(np.arange(rank_ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        if(self._buffer_length >= self._buffer_size):
            idxs = idxs_ps + (self._buffer_length-self._buffer_size)
        else:
            idxs = idxs_ps
        # compute sampling weights
        experiences_ow = self._buffer["experience_ow"][idxs]
        batch = Experience_OW(*zip(*experiences_ow))
        
        if(self._buffer_length >= self._buffer_size):
            weights = (self._buffer_size * sampling_probs[idxs_ps])**-self._beta
        else:
            weights = (self._buffer_length * sampling_probs[idxs_ps])**-self._beta
        normalized_weights = weights / weights.max()
                        
        return (
            torch.FloatTensor(np.array(batch.ows)).to(self.device),
            torch.FloatTensor(np.array(batch.states)).to(self.device),
            torch.FloatTensor(np.array(batch.actions)).to(self.device),
            torch.FloatTensor(np.array(batch.rewards)).to(self.device),
            torch.FloatTensor(np.array(batch.next_ows)).to(self.device),
            torch.FloatTensor(np.array(batch.next_states)).to(self.device),
            torch.FloatTensor(np.array(batch.dones)).to(self.device),
            torch.FloatTensor(np.array(idxs)).to(self.device),
            torch.FloatTensor(np.array(normalized_weights)).to(self.device)
        )
    
    # Update the ranking value of transitions 
    def update_priorities(self, idxs: np.array, priorities: np.array):
        for i in idxs:
            self._buffer["priority"][i] = float(math.floor(self._buffer["priority"][i]))
        
        priorities_sigmoid = (1/(1 + np.exp(-priorities))).round(decimals=4) - 1e-4 # bias for rounding to avoid sigmoid≈1
        self._buffer["priority"][idxs] += priorities_sigmoid
    
    # Sort transitions in the memory based on ranking value
    def sort_priorities(self):
        if(self._buffer_length >= self._buffer_size):
            buffer_list = list(self._buffer[(self._buffer_length - self._buffer_size):self._buffer_length])
            buffer_list.sort(key=lambda x:x[0])

            self._buffer_length = self._buffer_size
        else:
            buffer_list = list(self._buffer[:self._buffer_length])
            buffer_list.sort(key=lambda x:x[0])
        
        self._buffer[:self._buffer_length] = np.array(buffer_list, 
                                                      dtype=[("priority", np.float32), ("experience_ow", Experience_OW)])

