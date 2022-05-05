import math
import os
import sys

import carla
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
            
            # plot
            if len(agent.rollout_trajectory_tuple)>0:
                for rollout_trajectory_head in agent.rollout_trajectory_tuple:
                    for rollout_trajectory in rollout_trajectory_head:                     
                        for i in range(len(rollout_trajectory[0])-1):
                            env.debug.draw_line(begin=carla.Location(x=rollout_trajectory[0][i],y=rollout_trajectory[1][i],z=env.ego_vehicle.get_location().z+1),
                                                end=carla.Location(x=rollout_trajectory[0][i+1],y=rollout_trajectory[1][i+1],z=env.ego_vehicle.get_location().z+1), 
                                                thickness=0.2,  color=carla.Color(255, 0, 0), life_time=0.2)
            
            if train_step % 10000 == 0:
                agent.ensemble_transition_model.save(train_step)
            train_step += 1

            
            if done:
                agent.clear_buff()
                break
