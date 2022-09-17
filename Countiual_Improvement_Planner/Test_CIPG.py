import glob
import os
import os.path as osp
import sys

sys.path.append("..")

import copy
import math
import random
import time

import carla
import gym
import matplotlib.pyplot as plt
import numpy as np
from Test_Scenarios.TestScenario_Town02_OCRL import \
    CarEnv_02_Intersection_fixed
from tqdm import tqdm

from Agent.Agent_Fixed_Policy import CIPG_Agent

TRAIN_EPISODES = 250
LOAD_STEP = 0

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    agent = CIPG_Agent(env)
    agent.learning_by_driving(load_step=LOAD_STEP, train_episode=TRAIN_EPISODES)
            
            
            
            
            
            

    


    