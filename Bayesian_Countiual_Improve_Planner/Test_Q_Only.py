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

from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
# from Agent.zzz.JunctionTrajectoryPlanner_simple_predict import \
#     JunctionTrajectoryPlanner
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner

TEST_EPISODES = 80000
LOAD_STEP = 80000
ROLLOUT_TIMES = 1

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_03_Waymo_Long_Tail()
    
    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    dynamic_map.update_ref_path(env)
    
    # Loop over episodes
    all_q = []
    ego_velocity = []

    for action_id in tqdm(range(0, 9 + 1), unit='episodes'):
        
        print('Restarting episode')
        obs = env.reset()
        if obs is None:
            continue
        done = False
                
        # Loop over steps
        true_q_value = []
        
        ego_x_start = obs[0][0]
        ego_y_start = obs[0][1]

        for i in range(ROLLOUT_TIMES):
            obs = np.array(obs)
            dynamic_map.update_map_from_list_obs(obs)
            
            candidate_trajectories_tuple = trajectory_planner.generate_candidate_trajectories(dynamic_map)

            chosen_action_id = action_id
            chosen_trajectory = trajectory_planner.trajectory_update_CP(chosen_action_id)

            
            g_value = 0
            ego_velocity_record = 0
            for i in range(50):
                control_action =  controller.get_control(dynamic_map,  chosen_trajectory.trajectory, chosen_trajectory.desired_speed)
                action = [control_action.acc , control_action.steering]
                
                # reward calculation: assume that the ego vehicle will precisely follow trajectory
                trajectory = candidate_trajectories_tuple[chosen_action_id-1][0]
                Jp = trajectory.d_ddd[i]**2 
                Js = trajectory.s_ddd[i]**2
                ds = (30.0 / 3.6 - trajectory.s_d[i])**2 # target speed
                cd = 0.1 * Jp + 0.1 * 0.1 + 0.05 * trajectory.d[i]**2
                cv = 0.1 * Js + 0.1 * 0.1 + 0.05 * ds
                g_value -= 1.0 * cd + 1.0 * cv
                
                new_obs, reward, done, collision = env.step(action)   
                
                print("delta",new_obs[0][0] - ego_x_start, new_obs[0][1] - ego_y_start, 
                      new_obs[1][0] - ego_x_start, new_obs[1][1] - ego_y_start)
                
                ego_velocity_record += math.sqrt(new_obs[0][2] ** 2 + new_obs[0][3] ** 2)
                dynamic_map.update_map_from_list_obs(new_obs)

                if done:
                    if collision == True:
                        g_value -= 200
                    break
                    
            time.sleep(0.1)
            true_q_value.append(g_value)
            ego_velocity.append(ego_velocity_record/50)
            trajectory_planner.clear_buff()
            obs = env.reset()
            if obs is None:
                break


        true_q_value = np.mean(true_q_value)
        # ego_velocity.append(sum(chosen_trajectory.desired_speed)/len(chosen_trajectory.desired_speed))
        all_q.append(true_q_value)
    print("true_q_value", all_q)
    print("ego_velocity", ego_velocity)


         
            

    