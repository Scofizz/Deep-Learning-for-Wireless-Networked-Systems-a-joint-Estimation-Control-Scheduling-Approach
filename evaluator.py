#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import statistics
import gym


# In[ ]:


from history import Controller_ObservationAction_Window, Scheduler_ObservationAction_Window
from util import *


# In[ ]:


'''Evaluation of trained NNs on MuJoCo tasks'''
# Estimator
# Intelligent controller (actor, critic) 
# Transmission scheduler (DQN) 


def eval_policy(policy, env_name, seed, training_timestep, eval_episodes, SENSOR_NOISE_STD, COMM_COST, 
                PRE_TRAINING_TIMESTEPS, LEN, DQN_INPUT_DIM, ACTOR_INPUT_DIM, SCHEDULE_ACTION_DIM, CONTROL_ACTION_DIM):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    eval_return_buffer = []
    eval_error_buffer = []
    
    eval_scheduler_OW = Scheduler_ObservationAction_Window(LEN, DQN_INPUT_DIM, SCHEDULE_ACTION_DIM)
    eval_controller_OW = Controller_ObservationAction_Window(LEN, ACTOR_INPUT_DIM, CONTROL_ACTION_DIM)
    
    for i in range(eval_episodes):
        eval_episode_reward = 0.
        eval_episode_comm_cost = 0.
        eval_episode_error = 0.
        
        eval_timesteps = 0
        
        eval_state, eval_done = eval_env.reset(), False
        EVAL_INITIAL_OBS = add_gaussian_noise(eval_state, SENSOR_NOISE_STD)
        
        eval_controller_input = EVAL_INITIAL_OBS
        
        eval_scheduler_OW.__init__(LEN, DQN_INPUT_DIM, SCHEDULE_ACTION_DIM)
        eval_controller_OW.__init__(LEN, ACTOR_INPUT_DIM, CONTROL_ACTION_DIM)
        
        eval_LS = 0.05 # initial uplink (sensor-controller) channel loss rate
        eval_LA = 0.05 # initial downlink (controller-actuator) channel loss rate
        
        eval_AoI = 0
        
        eval_scheduler_concatenated_input = concatenate_scheduler_state(EVAL_INITIAL_OBS, eval_LS, 1)
        eval_schedule = np.array([1])
        eval_sc_transmission = 1
        
        # check unusual env state
        nan_signal = 0
        
        while not eval_done:
            # check unusual env state
            if(nan_signal):
                break
        
            eval_controller_concatenated_input = concatenate_controller_state(eval_controller_input, eval_LS, eval_LA, eval_AoI)
            eval_controller_concatenated_ow = eval_controller_OW.read()
              
            if(eval_sc_transmission == 0):
                eval_comm_cost_sc = 0
            else:
                eval_comm_cost_sc = COMM_COST
             
            if(np.random.rand() < eval_LA):
                eval_action = np.zeros(CONTROL_ACTION_DIM)
            else:
                eval_action = policy.select_action(np.array(eval_controller_concatenated_ow), 
                                              np.array(eval_controller_concatenated_input))
            
            eval_scheduler_OW.add(eval_scheduler_concatenated_input)
            eval_scheduler_OW.add(eval_schedule)
            
            eval_controller_OW.add(eval_controller_concatenated_input)
            eval_controller_OW.add(eval_action)
            
            # check unusual env state
            if(True in np.isnan(eval_action)):
                nan_signal = 1
                break
            
            # take 1 step
            eval_state_est = policy.predict_state(np.array(eval_controller_input), np.array(eval_action))
            
            eval_state, eval_oc_cost, eval_done, _ = eval_env.step(eval_action)
            eval_state_obs = add_gaussian_noise(eval_state, SENSOR_NOISE_STD)
            
            eval_reward = eval_oc_cost - eval_comm_cost_sc
        
            eval_LS = channel_state_transition(eval_LS)
            eval_LA = channel_state_transition(eval_LA) 
           
            eval_scheduler_concatenated_input = concatenate_scheduler_state(eval_state_est, eval_LS, eval_AoI+1)
            eval_scheduler_concatenated_ow = eval_scheduler_OW.read()
            
            if(training_timestep > PRE_TRAINING_TIMESTEPS):
                eval_schedule = policy.schedule_transmission(np.array(eval_scheduler_concatenated_ow), 
                                                            np.array(eval_scheduler_concatenated_input))
            else:
                eval_schedule = np.array([1])
            
            eval_sc_transmission = policy.control_transmission(eval_schedule)
            
            # next step
            if(np.random.rand() < eval_LS) or (eval_sc_transmission == 0):
                eval_controller_input = eval_state_est
                eval_AoI += 1
            else:
                eval_controller_input = eval_state_obs
                eval_AoI = 0
            
            # Evaluate reward
            eval_episode_reward += eval_reward
            
            # Evaluate prediction error
            eval_timestep_error = np.square(eval_state_obs - eval_state_est).mean()
            eval_episode_error += eval_timestep_error
            eval_timesteps += 1
        
        if not(nan_signal):
            eval_return_buffer.append(eval_episode_reward)
            eval_error_buffer.append(eval_episode_error / eval_timesteps)
    
    eval_env.close()
    
    # Average over N eval episodes
    eval_return = statistics.mean(eval_return_buffer)
    eval_error = statistics.mean(eval_error_buffer)
   
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {eval_return:.3f}")
    print(f"Estimation over {eval_episodes} episodes: {eval_error:.3f}")
    print("---------------------------------------")

