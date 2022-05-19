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
from Test_Scenarios.TestScenario_Town02_Fixed_State import \
    CarEnv_02_Intersection_fixed_state
from tqdm import tqdm

from DCP_Agent.Agent import DCP_Agent
from results import Results

TEST_EPISODES = 300
LOAD_STEP = 160000
ROLLOUT_TIMES = 10

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed_state()

    # Create Agent
    agent = DCP_Agent(env)
    agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Result class
    result = Results(agent.history_frame, create_new_train_file=False)
    result.clear_old_test_data()
    last_train_state = None
    # Loop over episodes
    for episode in tqdm(range(0, TEST_EPISODES), unit='episodes'):
        
        print('Restarting episode')
        obs = env.reset()
        # trained_state = result.sampled_trained_state(episode)
        # if trained_state == last_train_state:
        #     continue
        # last_train_state = trained_state

        # obs = env.reset_with_state(trained_state)
        if obs is None:
            continue
        done = False
                
        # Loop over steps
        true_q_value = []
        safety = []
        efficiency = []

        for i in range(ROLLOUT_TIMES):

            obs = np.array(obs)
            agent.dynamic_map.update_map_from_list_obs(obs)
            candidate_trajectories_tuple = agent.trajectory_planner.generate_candidate_trajectories(agent.dynamic_map)
            # DCP process            
            agent.history_obs_list.append(obs)

            if len(agent.history_obs_list) >= agent.history_frame:
                worst_Q_list, used_worst_Q_list = agent.calculate_worst_Q_value(agent.history_obs_list, candidate_trajectories_tuple)
                dcp_action = np.where(used_worst_Q_list==np.max(used_worst_Q_list))[0] 
                
                potential_dcp_action = np.where(worst_Q_list==np.max(worst_Q_list))[0] 
                estimated_q_lower_bound = worst_Q_list[potential_dcp_action[0]]

                # print("used_worst_Q_list",used_worst_Q_list)
                print("obs",agent.history_obs_list)
                print("worst_Q_list",worst_Q_list)
                print("dcp_action",dcp_action)
                state = np.array(agent.history_obs_list).flatten().tolist() # for record
                agent.history_obs_list.pop(0)

            else:
                dcp_action = 0 # brake
        
            dcp_trajectory = agent.trajectory_planner.trajectory_update_CP(dcp_action)
            
            #plot
            if len(agent.rollout_trajectory_tuple)>0:
                for rollout_trajectory_head in agent.rollout_trajectory_tuple:
                    for rollout_trajectory in rollout_trajectory_head:                     
                        for i in range(len(rollout_trajectory[0])-1):
                            env.debug.draw_line(begin=carla.Location(x=rollout_trajectory[0][i],y=rollout_trajectory[1][i],z=env.ego_vehicle.get_location().z+1),
                                                end=carla.Location(x=rollout_trajectory[0][i+1],y=rollout_trajectory[1][i+1],z=env.ego_vehicle.get_location().z+1), 
                                                thickness=0.2,  color=carla.Color(255, 0, 0), life_time=4)
            
            # for trajectory in agent.trajectory_planner.all_trajectory:
            #     for i in range(len(trajectory[0].x)-1):
    
            #         # env.debug.draw_point(carla.Location(x=trajectory[0].x[i],y=trajectory[0].y[i],z=env.ego_vehicle.get_location().z+1),
            #         #                      size=0.04, color=carla.Color(r=0,g=0,b=255), life_time=0.11)
            #         env.debug.draw_line(begin=carla.Location(x=trajectory[0].x[i],y=trajectory[0].y[i],z=env.ego_vehicle.get_location().z+0.1),
            #                             end=carla.Location(x=trajectory[0].x[i+1],y=trajectory[0].y[i+1],z=env.ego_vehicle.get_location().z+0.1), 
            #                             thickness=0.1, color=carla.Color(r=0,g=0,b=255), life_time=4)
            
            g_value = 0
            non_colli = 1
            trajectory = candidate_trajectories_tuple[dcp_action[0]-1][0]
            if dcp_action == 0:
                efficiency = 0
            else:
                efficiency = np.mean(trajectory.s_d)

            for i in range(agent.future_frame):
                control_action =  agent.controller.get_control(agent.dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
                action = [control_action.acc , control_action.steering]
                
                # reward calculation: assume that the ego vehicle will precisely follow trajectory
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
                        non_colli = 0
                    break
                    
            time.sleep(0.1)
            true_q_value.append(g_value)
            safety.append(non_colli)
            
            agent.clear_buff()
            # obs = env.reset_with_state(trained_state)
            obs = env.reset()
            if obs is None:
                break
            
        true_q_value = np.mean(true_q_value)
        safety = np.mean(safety)
        
        result.record_dcp_performance(state, dcp_action, estimated_q_lower_bound, true_q_value, 
                                      safety, efficiency)
        

         
            

    
            
            
            
            
            
            

    


    