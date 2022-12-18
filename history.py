#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import copy
from collections import deque


# In[ ]:


'''Initialize History Buffer'''
# Controller_ObservationAction_Window: state-action history of the intelligent controller
# Scheduler_ObservationAction_Window: state-action history of the transmission scheduler


class Controller_ObservationAction_Window:

    def __init__(self, LEN, ACTOR_INPUT_DIM, CONTROL_ACTION_DIM):
        self._buffer_size = LEN
        self._buffer = deque()
        # padding zeros
        for i in range(LEN):
            self._buffer.append(np.zeros(ACTOR_INPUT_DIM))
            self._buffer.append(np.zeros(CONTROL_ACTION_DIM))
    
    def add(self, obervation_OR_action):
        self._buffer.append(obervation_OR_action)
        self._buffer.popleft()
        
    def read(self):
        his_obs_act = self._buffer.copy()
        his_obs_act = np.array(his_obs_act)
        his_obs_act = np.concatenate(his_obs_act, axis=None)
        return his_obs_act
    

class Scheduler_ObservationAction_Window:

    def __init__(self, LEN, DQN_INPUT_DIM, SCHEDULE_ACTION_DIM):
        self._buffer_size = LEN
        self._buffer = deque()
        # padding zeros
        for i in range(LEN):
            self._buffer.append(np.zeros(DQN_INPUT_DIM))
            self._buffer.append(np.zeros(SCHEDULE_ACTION_DIM))
    
    def add(self, obervation_OR_action):
        self._buffer.append(obervation_OR_action)
        self._buffer.popleft()
        
    def read(self):
        his_obs_act = self._buffer.copy()
        his_obs_act = np.array(his_obs_act)
        his_obs_act = np.concatenate(his_obs_act, axis=None)
        return his_obs_act

