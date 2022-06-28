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
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from tqdm import tqdm

from Agent.Agent import OCRL_Agent

TEST_EPISODES = 250
LOAD_STEP = 80000

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()
    empty_env = CarEnv_02_Intersection_fixed(empty=True)

    # Create Agent
    agent = OCRL_Agent(env, empty_env)
    agent.learning_by_driving()
            
            
            
            
            
            

    


    