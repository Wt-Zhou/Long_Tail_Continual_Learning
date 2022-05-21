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
from results import Results

TEST_EPISODES = 80000
LOAD_STEP = 80000
ROLLOUT_TIMES = 5

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    agent = DCP_Agent(env)
    agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Result class
    result = Results(agent.history_frame, create_new_train_file=False)
    result.clear_old_test_data()
    
    # Loop over episodes
    for episode in tqdm(range(170, TEST_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')
        # obs = env.reset()

        trained_state = result.sampled_trained_state(episode)
        
        obs = env.reset_with_state(trained_state)
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

            if len(agent.history_obs_list) >= agent.history_frame:
                worst_Q_list = agent.calculate_worst_Q_value(agent.history_obs_list, candidate_trajectories_tuple)
                dcp_action = np.where(worst_Q_list==np.max(worst_Q_list))[0] 
                
                # fixed action:
                dcp_action = np.array([8])
                
                estimated_q_lower_bound = worst_Q_list[dcp_action[0]]

                print("worst_Q_list",worst_Q_list)
                print("dcp_action",dcp_action)
                state = np.array(agent.history_obs_list).flatten().tolist() # for record
                agent.history_obs_list.pop(0)

            else:
                dcp_action = 0 # brake
        
            dcp_trajectory = agent.trajectory_planner.trajectory_update_CP(dcp_action)
            
            g_value = 0
            for i in range(agent.future_frame):
                control_action =  agent.controller.get_control(agent.dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
                action = [control_action.acc , control_action.steering]
                
                # reward calculation: assume that the ego vehicle will precisely follow trajectory
                trajectory = candidate_trajectories_tuple[dcp_action[0]-1][0]
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
            obs = env.reset_with_state(trained_state)
            if obs is None:
                break

            
        true_q_value = np.mean(true_q_value)
        print("estimated_q_lower_bound", estimated_q_lower_bound)
        print("true_q_value", true_q_value)
        result.estimated_q_lower_bound(state, dcp_action, estimated_q_lower_bound, true_q_value)
        

         
            

    