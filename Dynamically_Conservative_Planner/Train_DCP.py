import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append("..")
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from tqdm import tqdm

from DCP_Agent.Agent import DCP_Agent

torch.set_printoptions(profile='short')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_EPISODES = 10000
LOAD_STEP = 0

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    agent = DCP_Agent(env, training=True)
    train_step = agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Loop over episodes
    for episode in tqdm(range(1, TRAIN_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        done = False
                
        # Loop over steps
        while True:
            action = agent.act(obs)
            new_obs, reward, done, collision_signal = env.step(action)       
            obs = new_obs      
                         
            agent.ensemble_transition_model.add_training_data(new_obs, done)
            agent.ensemble_transition_model.update_model()
            
            train_step += 1
            if self.train_step % 10000 == 0:
                agent.ensemble_transition_model.save(train_step)
            
            
            if done:
                agent.clear_buff()
                break
