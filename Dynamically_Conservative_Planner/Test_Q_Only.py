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
from Test_Scenarios.TestScenario_Town02_Fixed_State import \
    CarEnv_02_Intersection_fixed_state
from Test_Scenarios.TestScenario_Town03_Waymo_long_tail import \
    CarEnv_03_Waymo_Long_Tail
from tqdm import tqdm

from DCP_Agent.Agent import DCP_Agent
from results import Results

TEST_EPISODES = 80000
LOAD_STEP = 80000
ROLLOUT_TIMES = 1

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_03_Waymo_Long_Tail()

    # Create Agent
    agent = DCP_Agent(env)
    # agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Result class
    # result = Results(agent.history_frame, create_new_train_file=False)
    # result.clear_old_test_data()
    
    # Loop over episodes
    for episode in tqdm(range(0, TEST_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')
        obs = env.reset()

        # trained_state = result.sampled_trained_state(episode)
        
        # obs = env.reset_with_state(trained_state)
        if obs is None:
            continue
        done = False
                
        # Loop over steps
        true_q_value = []

        for i in range(ROLLOUT_TIMES):
            obs = np.array(obs)
            agent.dynamic_map.update_map_from_list_obs(obs)
            candidate_trajectories_tuple = agent.trajectory_planner.generate_candidate_trajectories(agent.dynamic_map)
            # DCP process            
            agent.history_obs_list.append(obs)

            dcp_action = 5
        
            dcp_trajectory = agent.trajectory_planner.trajectory_update_CP(dcp_action)
            
            g_value = 0
            for i in range(50):
                control_action =  agent.controller.get_control(agent.dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
                action = [control_action.acc , control_action.steering]
                
                # reward calculation: assume that the ego vehicle will precisely follow trajectory
                trajectory = candidate_trajectories_tuple[dcp_action-1][0]
                Jp = trajectory.d_ddd[i]**2 
                Js = trajectory.s_ddd[i]**2
                ds = (30.0 / 3.6 - trajectory.s_d[i])**2 # target speed
                cd = 0.1 * Jp + 0.1 * 0.1 + 0.05 * trajectory.d[i]**2
                cv = 0.1 * Js + 0.1 * 0.1 + 0.05 * ds
                g_value -= 1.0 * cd + 1.0 * cv
                
                new_obs, reward, done, collision = env.step(action)   
                agent.dynamic_map.update_map_from_list_obs(new_obs)
                if done:
                    if collision == True:
                        g_value -= 500
                    break
                    
            time.sleep(0.1)
            true_q_value.append(g_value)
            agent.clear_buff()
            obs = env.reset()
            if obs is None:
                break

            
        true_q_value = np.mean(true_q_value)
        print("true_q_value", true_q_value)
        

         
            

    