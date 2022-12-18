#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import torch
import gym


# In[ ]:


from TailoredTD3 import Tailored_TD3
from history import Controller_ObservationAction_Window, Scheduler_ObservationAction_Window
from memory import PrioritizedExperienceReplayBuffer, PrioritizedExperienceReplayBuffer_OW
from evaluator import eval_policy
from util import *


# In[ ]:


'''ENV_NAME List of MuJoCo Tasks'''
# InvertedDoublePendulum-v2
# Hopper-v2
# HalfCheetah-v2


ENV_NAME = 'HalfCheetah-v2'
SEED = 0

# Set environment & seeds
env = gym.make(ENV_NAME)
env.seed(SEED)
env.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set state-action dimension
PLANT_STATE_DIM = env.observation_space.shape[0]
CHANNEL_STATE_DIM = 1
AoI_DIM = 1

CONTROL_ACTION_DIM = env.action_space.shape[0]
SCHEDULE_ACTION_DIM = 1 
MAX_ACTION = float(env.action_space.high[0])

# Transmission scheduler action: 1-transmit; 0-not transmit
TRANSMISSION_ACTION_LIST = np.array([0,1])

# Set NNs hyperparameters of estimator, intelligent controller (actor, critic) and transmission scheduler (DQN)
ACTOR_INPUT_DIM = PLANT_STATE_DIM + 2*CHANNEL_STATE_DIM + AoI_DIM
ACTOR_OUTPUT_DIM = CONTROL_ACTION_DIM

CRITIC_INPUT_DIM = (PLANT_STATE_DIM + 2*CHANNEL_STATE_DIM + AoI_DIM) + CONTROL_ACTION_DIM
CRITIC_OUTPUT_DIM = 1

ESTIMATOR_INPUT_DIM = PLANT_STATE_DIM + CONTROL_ACTION_DIM
ESTIMATOR_OUTPUT_DIM = PLANT_STATE_DIM

DQN_INPUT_DIM = PLANT_STATE_DIM + CHANNEL_STATE_DIM + AoI_DIM
DQN_OUTPUT_DIM = 2

HIDDEN_SIZE = 128

# Set history length
LEN = 3

SCHEDULER_OW_DIM = LEN*(DQN_INPUT_DIM+SCHEDULE_ACTION_DIM)
CONTROLLER_OW_DIM = LEN*(ACTOR_INPUT_DIM+CONTROL_ACTION_DIM)

# Set network training hyperparameters
TIMESTEPS_BEFORE_TRAIN = 25e3
PRE_TRAINING_TIMESTEPS = 2e5
MAX_TIMESTEPS = 2e6
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
LR_DYNAMICS = 1e-3
EPSILON = 0.9

EXPL_NOISE_STD = 0.1
TARGET_ACTOR_NOISE_STD = 0.2
TARGET_ACTOR_NOISE_CLIP = 0.5
ACTOR_UPDATE_DELAY = 2

MEMORY_CAPACITY = int(2e6)
BATCH_SIZE = 100

EVAL_EPISODES = 10
EVAL_FREQ = 5e3

# Set WNCS hyperparameters
LS = 0.05 # initial uplink (sensor-controller) channel loss rate
LA = 0.05 # initial downlink (controller-actuator) channel loss rate
SENSOR_NOISE_STD = 0.01 
COMM_COST = 10

# Set experience replay hyperparameters
ALPHA_IS = 1 # rank-based importance sampling
ALPHA_UN = 0 # uniform sampling
BETA = 1

MEMORY_SIZE = int(2e5)
SORT_FREQ = int(2e5)


# In[ ]:


'''Main: joint training of NNs in WNCS'''
# Estimator
# Intelligent controller (actor, critic) 
# Transmission scheduler (DQN) 


# Initialize policy
policy = Tailored_TD3()

# Initialize replay buffer
estimator_replay_buffer = PrioritizedExperienceReplayBuffer(ALPHA_UN, BETA, BATCH_SIZE, MEMORY_SIZE, MEMORY_CAPACITY)
scheduler_replay_buffer = PrioritizedExperienceReplayBuffer_OW(ALPHA_UN, BETA, BATCH_SIZE, MEMORY_SIZE, MEMORY_CAPACITY)
controller_replay_buffer = PrioritizedExperienceReplayBuffer_OW(ALPHA_IS, BETA, BATCH_SIZE, MEMORY_SIZE, MEMORY_CAPACITY)

# Evaluate untrained policy
eval_policy(policy, ENV_NAME, SEED, 0, EVAL_EPISODES, SENSOR_NOISE_STD, COMM_COST, CONTROL_ACTION_DIM, 
            PRE_TRAINING_TIMESTEPS, LEN, DQN_INPUT_DIM, ACTOR_INPUT_DIM, SCHEDULE_ACTION_DIM, CONTROL_ACTION_DIM)

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

INITIAL_OBS = add_gaussian_noise(state, SENSOR_NOISE_STD)

controller_obs = None
controller_est = None

controller_next_obs = INITIAL_OBS
controller_next_est = None

LS = LS # initial uplink (sensor-controller) channel loss rate
LA = LA # initial downlink (controller-actuator) channel loss rate

signal_sc = 1 # uplink dropout signal (0: drop, 1: receive)
AoI = 0

# Initialize key variables
SCHEDULER_OW = Scheduler_ObservationAction_Window(LEN, DQN_INPUT_DIM, SCHEDULE_ACTION_DIM)
CONTROLLER_OW = Controller_ObservationAction_Window(LEN, ACTOR_INPUT_DIM, CONTROL_ACTION_DIM)

scheduler_concatenated_est = concatenate_scheduler_state(INITIAL_OBS, LS, 1)
scheduler_concatenated_ow = SCHEDULER_OW.read()
schedule = np.array([1]) 
sc_transmission = 1

# Initialize temp variables
REC_estimator_current = None
REC_estimator_next = None

REC_scheduler_current = None
REC_scheduler_next = None

REC_scheduler_current_ow = None
REC_scheduler_next_ow = None

REC_controller_current = None
REC_controller_next = None

REC_controller_current_ow = None
REC_controller_next_ow = None

for t in range(int(MAX_TIMESTEPS)):

    episode_timesteps += 1
    
    '''Packet dropout signal'''
    if (signal_sc == 0):
        '''Uplink channel'''
        controller_est = controller_next_est
        
        controller_concatenated_est = concatenate_controller_state(controller_est, LS, LA, AoI)
        controller_concatenated_ow = CONTROLLER_OW.read()
        
        REC_estimator_current = controller_est
        
        REC_scheduler_current = scheduler_concatenated_est
        REC_scheduler_current_ow = scheduler_concatenated_ow
        
        REC_controller_current = controller_concatenated_est
        REC_controller_current_ow = controller_concatenated_ow
        
        # communication cost    
        if(sc_transmission == 0):
            comm_cost_sc = 0
        else:
            comm_cost_sc = COMM_COST
        
        '''Downlink channel'''
        if(np.random.rand() < LA):
            action = np.zeros(CONTROL_ACTION_DIM)
        else:
            # Select action randomly or according to policy
            if t < TIMESTEPS_BEFORE_TRAIN:
                action = env.action_space.sample()
            else:
                action = (policy.select_action(np.array(controller_concatenated_ow), 
                                               np.array(controller_concatenated_est)) 
                          + np.random.normal(0, MAX_ACTION * EXPL_NOISE_STD, 
                                           size=CONTROL_ACTION_DIM)).clip(-MAX_ACTION, MAX_ACTION)
        
        # Update history of transmission scheduler and intelligent controller  
        SCHEDULER_OW.add(scheduler_concatenated_est)
        SCHEDULER_OW.add(schedule)
        
        CONTROLLER_OW.add(controller_concatenated_est)
        CONTROLLER_OW.add(action)
        
        # Take 1 step
        controller_next_est = policy.predict_state(np.array(controller_est), np.array(action))
        
        next_state, oc_cost, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        reward = oc_cost - comm_cost_sc
        
        q_value = policy.select_q(controller_concatenated_ow, controller_concatenated_est, action)
        q_reward = float(q_value)
        
        '''Markov fading channel state'''
        LS = channel_state_transition(LS)
        LA = channel_state_transition(LA)
        
        '''Transmission scheduling'''
        scheduler_concatenated_est = concatenate_scheduler_state(controller_next_est, LS, AoI+1)
        scheduler_concatenated_ow = SCHEDULER_OW.read()
        
        if(t < PRE_TRAINING_TIMESTEPS):    
            schedule = np.array([1])
        else:
            # Select schedule action with e-greedy
            if(np.random.rand() < EPSILON):
                schedule = policy.schedule_transmission(np.array(scheduler_concatenated_ow), 
                                                        np.array(scheduler_concatenated_est))
            else:
                schedule = np.array([np.random.randint(0, 2)])
        
        sc_transmission = policy.control_transmission(schedule)
        
        '''Next time step'''
        REC_scheduler_next = scheduler_concatenated_est
        REC_scheduler_next_ow = scheduler_concatenated_ow
        
        if(np.random.rand() < LS) or (sc_transmission == 0):
            # Predict state
            controller_next_est = controller_next_est
            
            AoI += 1
            
            REC_estimator_next = controller_next_est
            REC_controller_next = concatenate_controller_state(controller_next_est, LS, LA, AoI)
            REC_controller_next_ow = CONTROLLER_OW.read()
            
            # Store transition in replay buffers
            scheduler_experience = Experience_OW(REC_scheduler_current_ow, REC_scheduler_current, schedule, [q_reward], 
                                       REC_scheduler_next_ow, REC_scheduler_next, [1. - done_bool])
            controller_experience = Experience_OW(REC_controller_current_ow, REC_controller_current, action, [reward], 
                                       REC_controller_next_ow, REC_controller_next, [1. - done_bool])
            
            aoi_label = 2*AoI - 1
            scheduler_replay_buffer.add(scheduler_experience, aoi_label) # transition AOI: n/n+1
            controller_replay_buffer.add(controller_experience, aoi_label)
        else:  
            # Receive noisy observation
            controller_next_obs = add_gaussian_noise(next_state, SENSOR_NOISE_STD)
            
            signal_sc = 1
            
            REC_estimator_next = controller_next_obs
            REC_controller_next = concatenate_controller_state(controller_next_obs, LS, LA, 0)
            REC_controller_next_ow = CONTROLLER_OW.read()
            
            # Store transition in replay buffers
            scheduler_experience = Experience_OW(REC_scheduler_current_ow, REC_scheduler_current, schedule, [q_reward], 
                                       REC_scheduler_next_ow, REC_scheduler_next, [1. - done_bool])
            controller_experience = Experience_OW(REC_controller_current_ow, REC_controller_current, action, [reward], 
                                          REC_controller_next_ow, REC_controller_next, [1. - done_bool])
            
            aoi_label = AoI
            scheduler_replay_buffer.add(scheduler_experience, aoi_label) # transition AOI: n/0
            controller_replay_buffer.add(controller_experience, aoi_label)
    
    else: 
        # Reset AoI
        AoI = 0
        
        '''Uplink channel'''
        controller_obs = controller_next_obs
        
        controller_concatenated_obs = concatenate_controller_state(controller_obs, LS, LA, 0)
        controller_concatenated_ow = CONTROLLER_OW.read()
        
        REC_estimator_current = controller_obs
        
        REC_scheduler_current = scheduler_concatenated_est
        REC_scheduler_current_ow = scheduler_concatenated_ow
        
        REC_controller_current = controller_concatenated_obs
        REC_controller_current_ow = controller_concatenated_ow
        
        # communication cost    
        if(sc_transmission == 0):
            comm_cost_sc = 0
        else:
            comm_cost_sc = COMM_COST
        
        '''Downlink channel'''
        if(np.random.rand() < LA):
            action = np.zeros(CONTROL_ACTION_DIM)
        else:
            # Select action randomly or according to policy
            if t < TIMESTEPS_BEFORE_TRAIN:
                action = env.action_space.sample()
            else:
                action = (policy.select_action(np.array(controller_concatenated_ow), 
                                               np.array(controller_concatenated_obs)) 
                          + np.random.normal(0, MAX_ACTION * EXPL_NOISE_STD, 
                                             size=CONTROL_ACTION_DIM)).clip(-MAX_ACTION, MAX_ACTION)
        
        # Update history of transmission scheduler and intelligent controller 
        SCHEDULER_OW.add(scheduler_concatenated_est)
        SCHEDULER_OW.add(schedule)
        
        CONTROLLER_OW.add(controller_concatenated_obs)
        CONTROLLER_OW.add(action)
        
        # Take 1 step
        controller_next_est = policy.predict_state(np.array(controller_obs), np.array(action))
        
        next_state, oc_cost, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        
        reward = oc_cost - comm_cost_sc
        
        q_value = policy.select_q(controller_concatenated_ow, controller_concatenated_obs, action)
        q_reward = float(q_value)
        
        '''Markov fading channel state'''
        LS = channel_state_transition(LS)
        LA = channel_state_transition(LA) 
        
        '''Transmission scheduling'''
        scheduler_concatenated_est = concatenate_scheduler_state(controller_next_est, LS, AoI+1)
        scheduler_concatenated_ow = SCHEDULER_OW.read()
        
        if(t < PRE_TRAINING_TIMESTEPS):    
            schedule = np.array([1])
        else:
            # Select schedule action with e-greedy
            if(np.random.rand() < EPSILON):
                schedule = policy.schedule_transmission(np.array(scheduler_concatenated_ow), 
                                                        np.array(scheduler_concatenated_est))
            else:
                schedule = np.array([np.random.randint(0, 2)])
        
        sc_transmission = policy.control_transmission(schedule)
        
        '''Next time step'''
        REC_scheduler_next = scheduler_concatenated_est
        REC_scheduler_next_ow = scheduler_concatenated_ow
        
        if(np.random.rand() < LS) or (sc_transmission == 0):
            # Predict state
            controller_next_est = controller_next_est
            
            signal_sc = 0
            AoI += 1
            
            REC_estimator_next = controller_next_est
            REC_controller_next = concatenate_controller_state(controller_next_est, LS, LA, 1)
            REC_controller_next_ow = CONTROLLER_OW.read()
            
            # Store transition in replay buffer
            scheduler_experience = Experience_OW(REC_scheduler_current_ow, REC_scheduler_current, schedule, [q_reward], 
                                       REC_scheduler_next_ow, REC_scheduler_next, [1. - done_bool])
            controller_experience = Experience_OW(REC_controller_current_ow, REC_controller_current, action, [reward], 
                                          REC_controller_next_ow, REC_controller_next, [1. - done_bool])
            
            aoi_label = 1
            scheduler_replay_buffer.add(scheduler_experience, aoi_label) # transition AOI: 0/1
            controller_replay_buffer.add(controller_experience, aoi_label)
        else:
            # Receive noisy observation
            controller_next_obs = add_gaussian_noise(next_state, SENSOR_NOISE_STD)
            
            REC_estimator_next = controller_next_obs
            REC_controller_next = concatenate_controller_state(controller_next_obs, LS, LA, 0)
            REC_controller_next_ow = CONTROLLER_OW.read()
            
            # Store transition in replay buffers
            estimator_experience = Experience(REC_estimator_current, action, [reward], 
                                              REC_estimator_next, [1. - done_bool])
            
            scheduler_experience = Experience_OW(REC_scheduler_current_ow, REC_scheduler_current, schedule, [q_reward], 
                                       REC_scheduler_next_ow, REC_scheduler_next, [1. - done_bool])
            controller_experience = Experience_OW(REC_controller_current_ow, REC_controller_current, action, [reward], 
                                          REC_controller_next_ow, REC_controller_next, [1. - done_bool])
            
            aoi_label = 0
            estimator_replay_buffer.add(estimator_experience, aoi_label) # transition AOI: 0/0 
            scheduler_replay_buffer.add(scheduler_experience, aoi_label) 
            controller_replay_buffer.add(controller_experience, aoi_label)

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= TIMESTEPS_BEFORE_TRAIN:
        policy.train(estimator_replay_buffer, scheduler_replay_buffer, controller_replay_buffer)

    if done: 
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        
        '''Reset variables'''
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 
        
        INITIAL_OBS = add_gaussian_noise(state, SENSOR_NOISE_STD)
        
        controller_obs = None
        controller_est = None
        
        controller_next_obs = INITIAL_OBS
        controller_next_est = None
        
        LS = 0.05
        LA = 0.05
        
        signal_sc = 1
        AoI = 0
        
        SCHEDULER_OW.__init__()
        CONTROLLER_OW.__init__()
        
        scheduler_concatenated_est = concatenate_scheduler_state(INITIAL_OBS, LS, 1)
        scheduler_concatenated_ow = SCHEDULER_OW.read()
        schedule = np.array([1]) 
        sc_transmission = 1
        
        REC_estimator_current = None
        REC_estimator_next = None
        
        REC_scheduler_current = None
        REC_scheduler_next = None

        REC_scheduler_current_ow = None
        REC_scheduler_next_ow = None

        REC_controller_current = None
        REC_controller_next = None

        REC_controller_current_ow = None
        REC_controller_next_ow = None

    '''Evaluate trained NNs'''
    if (t + 1) % EVAL_FREQ == 0:
        eval_policy(policy, ENV_NAME, SEED, t + 1, EVAL_EPISODES, SENSOR_NOISE_STD, COMM_COST, CONTROL_ACTION_DIM, 
                    PRE_TRAINING_TIMESTEPS, LEN, DQN_INPUT_DIM, ACTOR_INPUT_DIM, SCHEDULE_ACTION_DIM, CONTROL_ACTION_DIM)

