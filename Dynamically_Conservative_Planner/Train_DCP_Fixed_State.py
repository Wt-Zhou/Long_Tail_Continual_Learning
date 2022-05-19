import math
import os
import sys

import carla
import numpy as np
import torch
import torch.nn as nn

sys.path.append("..")
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Test_Scenarios.TestScenario_Town02_Fixed_State import \
    CarEnv_02_Intersection_fixed_state
from tqdm import tqdm

from DCP_Agent.Agent import DCP_Agent
from results import Results

torch.set_printoptions(profile='short')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_EPISODES = 10000
LOAD_STEP = 0

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_02_Intersection_fixed_state(training=True)

    # Create Agent
    agent = DCP_Agent(env, training=True)
    train_step = agent.ensemble_transition_model.load(LOAD_STEP)
    
    # Result class
    result = Results(agent.history_frame, create_new_train_file=True)
    
    # Loop over episodes
    for episode in tqdm(range(1, TRAIN_EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        result.add_training_data(obs)

        done = False
        
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
            # print("worst_Q_list",worst_Q_list)
            print("dcp_action",dcp_action)
            state = np.array(agent.history_obs_list).flatten().tolist() # for record
            agent.history_obs_list.pop(0)

        else:
            dcp_action = 0 # brake
    
        dcp_trajectory = agent.trajectory_planner.trajectory_update_CP(dcp_action)
                
        # Loop over steps
        for i in range(agent.future_frame++agent.history_frame+1):

            control_action =  agent.controller.get_control(agent.dynamic_map,  dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
            action = [control_action.acc , control_action.steering]

            new_obs, reward, done, collision_signal = env.step(action)       
            agent.dynamic_map.update_map_from_list_obs(new_obs)

            if i == agent.future_frame+agent.history_frame: # the env will not send done in fixed horizon
                done = True
                         
            agent.ensemble_transition_model.add_training_data(obs, done)
            agent.ensemble_transition_model.update_model()
            obs = new_obs      

            
            # plot
            if len(agent.rollout_trajectory_tuple)>0:
                for rollout_trajectory_head in agent.rollout_trajectory_tuple:
                    for rollout_trajectory in rollout_trajectory_head:                     
                        for i in range(len(rollout_trajectory[0])-1):
                            env.debug.draw_line(begin=carla.Location(x=rollout_trajectory[0][i],y=rollout_trajectory[1][i],z=env.ego_vehicle.get_location().z+1),
                                                end=carla.Location(x=rollout_trajectory[0][i+1],y=rollout_trajectory[1][i+1],z=env.ego_vehicle.get_location().z+1), 
                                                thickness=1.0,  color=carla.Color(255, 0, 0), life_time=0.2)
            
            if train_step % 2500 == 0:
                agent.ensemble_transition_model.save(train_step)
            train_step += 1

            
            if done:
                agent.clear_buff()
                break
            
    result.calculate_training_distribution()
