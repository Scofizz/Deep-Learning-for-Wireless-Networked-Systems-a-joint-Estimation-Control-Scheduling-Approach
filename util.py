#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random


# In[2]:


'''Functions for Implementation'''
# channel_state_transition: Markov fading channel state
# add_gaussian_noise: add sensor measurement noise
# concatenate_controller_state: generate input of intelligent controller
# concatenate_scheduler_state: generate input of transmission scheduler

def channel_state_transition(ch):
    if(ch == 0.05):
        next_ch = np.random.choice([0.05, 0.1], p=[0.3, 0.7])
    else:
        next_ch = np.random.choice([0.05, 0.1], p=[0.7, 0.3])
    return next_ch

def add_gaussian_noise(state, SENSOR_NOISE_STD):
    noise = np.random.randn(state.shape[0]) 
    sensor = state + SENSOR_NOISE_STD * noise
    return sensor

def concatenate_controller_state(plant_state, ch_sc, ch_ca, AoI_sc):
    ch_state = np.array([ch_sc, ch_ca, AoI_sc], dtype=np.float32)
    state = np.concatenate((plant_state,ch_state))
    return state

def concatenate_scheduler_state(plant_state, ch_sc, AoI_sc):
    ch_state = np.array([ch_sc, AoI_sc], dtype=np.float32)
    state = np.concatenate((plant_state,ch_state))
    return state

