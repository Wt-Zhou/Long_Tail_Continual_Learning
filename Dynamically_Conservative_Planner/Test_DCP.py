import glob
import os
import os.path as osp
import sys

sys.path.append("..")

import math
import random
import time

import carla
import gym
import matplotlib.pyplot as plt
import numpy as np
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from tqdm import tqdm

from DCP_Agent.Agent import DCP_Agent

TEST_EPISODES = 10000
LOAD_STEP = 0

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    agent = DCP_Agent(env)
    agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Loop over episodes
    for episode in tqdm(range(1, TEST_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        done = False
                
        # Loop over steps
        while True:
            action = agent.act(obs)
            new_obs, reward, done, collision_signal = env.step(action)   
                
            obs = new_obs                    
            
            if done:
                agent.clear_buff()
                break


    