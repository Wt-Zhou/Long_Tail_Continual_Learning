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

from DCP_Agent.Agent import DCP_Agent
from results import Results

TEST_EPISODES = 250
LOAD_STEP = 80000

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
    for episode in tqdm(range(1, TEST_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        done = False
                
        # Loop over steps
        while True:
            obs = np.array(obs)
            agent.dynamic_map.update_map_from_list_obs(obs)
            candidate_trajectories_tuple = agent.trajectory_planner.generate_candidate_trajectories(agent.dynamic_map)
            # DCP process            
            agent.history_obs_list.append(obs)
            estimated_q_lower_bound = 0
            if len(agent.history_obs_list) >= agent.history_frame:
                worst_Q_list = agent.calculate_worst_Q_value(agent.history_obs_list, candidate_trajectories_tuple)
                dcp_action = np.where(worst_Q_list==np.max(worst_Q_list))[0] 
                               
                estimated_q_lower_bound = worst_Q_list[dcp_action[0]]

                print("worst_Q_list",worst_Q_list)
                print("dcp_action",dcp_action)
                state = np.array(agent.history_obs_list).flatten().tolist() # for record
                temp_obs = copy.deepcopy(agent.history_obs_list)
                agent.history_obs_list.pop(0)

            else:
                dcp_action = 0 # brake
        
            dcp_trajectory = agent.trajectory_planner.trajectory_update_CP(dcp_action)
            
            control_action =  agent.controller.get_control(agent.dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
            action = [control_action.acc , control_action.steering]
            
            # plot
            if len(agent.rollout_trajectory_tuple)>0:
                for rollout_trajectory_head in agent.rollout_trajectory_tuple:
                    for rollout_trajectory in rollout_trajectory_head:
                        for i in range(len(rollout_trajectory[0])-1):
                            env.debug.draw_line(begin=carla.Location(x=rollout_trajectory[0][i],y=rollout_trajectory[1][i],z=env.ego_vehicle.get_location().z+1),
                                                end=carla.Location(x=rollout_trajectory[0][i+1],y=rollout_trajectory[1][i+1],z=env.ego_vehicle.get_location().z+1), 
                                                thickness=0.2,  color=carla.Color(255, 0, 0), life_time=0.2)
            
            
            new_obs, reward, done, collision = env.step(action)  
            result.add_test_data(temp_obs, candidate_trajectories_tuple, dcp_trajectory.original_trajectory, collision, estimated_q_lower_bound)
 
            agent.dynamic_map.update_map_from_list_obs(new_obs)
            obs = new_obs
            if done:
                agent.clear_buff()
                break
            
    result.calculate_performance_metrics()
            
            
            
            
            
            

    


    