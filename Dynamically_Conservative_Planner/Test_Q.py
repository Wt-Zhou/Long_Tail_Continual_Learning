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

TEST_EPISODES = 10000
LOAD_STEP = 0

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    agent = DCP_Agent(env)
    agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Result class
    result = Result(agent.history_frame, create_new_train_file=False)
    
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

            if len(agent.history_obs_list) >= agent.history_frame:
                worst_Q_list = agent.calculate_worst_Q_value(agent.history_obs_list, candidate_trajectories_tuple)
                dcp_action = np.where(worst_Q_list==np.max(worst_Q_list))[0] 
                estimated_q_lower_bound = worst_Q_list[dcp_action]

                print("worst_Q_list",worst_Q_list)
                print("dcp_action",dcp_action)
                agent.history_obs_list.pop(0)

            else:
                dcp_action = 0 # brake
        
        
            dcp_trajectory = self.trajectory_planner.trajectory_update_CP(dcp_action)
            
            true_q_value = 0
            for i in range(agent.future_frame):
                control_action =  controller.get_control(dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
                action = [control_action.acc , control_action.steering]
                
                # reward calculation: assume that the ego vehicle will precisely follow trajectory
                trajectory = candidate_trajectories_tuple[dcp_action][0]
                Jp = trajectory.d_ddd[i]**2 
                Js = trajectory.s_ddd[i]**2
                ds = (30.0 / 3.6 - trajectory.s_d[i])**2 # target speed
                cd = 0.1 * Jp + 0.1 * 0.1 + 1.0 * trajectory.d[i]**2
                cv = 0.1 * Js + 0.1 * 0.1 + 1.0 * ds
                true_q_value += 1.0 * cd + 1.0 * cv
                
                new_obs, reward, done, _ = env.step(action)   
                agent.dynamic_map.update_map_from_list_obs(new_obs, env)
                if done:
                    true_q_value -= 500
                    break
                
            
            result.estimated_q_lower_bound(state, dcp_action, estimated_q_lower_bound, true_q_value)
            
            if done:
                agent.clear_buff()
                break

            # plot
            if len(agent.rollout_trajectory_tuple)>0:
                for rollout_trajectory_head in agent.rollout_trajectory_tuple:
                    for rollout_trajectory in rollout_trajectory_head:
                        for i in range(len(rollout_trajectory[0])-1):
                            env.debug.draw_line(begin=carla.Location(x=rollout_trajectory[0][i],y=rollout_trajectory[1][i],z=env.ego_vehicle.get_location().z+1),
                                                end=carla.Location(x=rollout_trajectory[0][i+1],y=rollout_trajectory[1][i+1],z=env.ego_vehicle.get_location().z+1), 
                                                thickness=0.2,  color=carla.Color(255, 0, 0), life_time=0.2)
         
            

    