import glob
import math
import os
import os.path as osp
import random
import sys
import time

import carla
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from results import Results
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from tqdm import tqdm

from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from DCP_Agent.transition_model.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from DCP_Agent.transition_model.predmlp import TrajPredGaussion

EPISODES=62

class DCP_Agent():
    def __init__(self, env, training=False):
        self.env = env
        
        # basic component
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.dynamic_map.update_ref_path(self.env)
        
        # transition model parameter        
        self.ensemble_num = 1
        self.history_frame = 1
        self.future_frame = 1
        self.obs_scale = 5
        self.action_scale = 5
        self.agent_dimension = 5  # x,y,vx,vy,yaw
        self.agent_num = 4
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)
        
        self.ensemble_transition_model = DCP_Transition_Model(self.ensemble_num, self.history_frame, 
                                                              self.future_frame, self.agent_dimension, self.agent_num,
                                                              self.action_scale, self.device, training)
        
    def act(self, state):
        
        obs = np.array(state)
        self.dynamic_map.update_map_from_list_obs(obs)
        candidate_trajectories_tuple = self.trajectory_planner.generate_candidate_trajectories(self.dynamic_map)
        
        # DCP process
        sorted_tuple = sorted(candidate_trajectories_tuple, key=lambda candidate_trajectories_tuple: candidate_trajectories_tuple[1])
        dcp_action = sorted_tuple[0][2] + 1
        dcp_trajectory = self.trajectory_planner.trajectory_update_CP(dcp_action)
        
        control_action =  self.controller.get_control(self.dynamic_map, dcp_trajectory.trajectory, dcp_trajectory.desired_speed)
        action = [control_action.acc, control_action.steering]
        
        return action
    
    def clear_buff(self):
        self.trajectory_planner.clear_buff(clean_csp=False)

    def calculate_worst_Q_value(self, state, candidate_trajectories_tuple):
        worst_Q_list = []
        for action in candidate_trajectories_tuple:
            imagination = self.imaginaton(state, action[0])
            q_value_of_heads = 1
            worst_Q_value = 1
            worst_Q_list.append(worst_Q_value)
        
        return worst_Q_list

    def q_value_for_a_head(self, state, action, ensemble_index):
        g_value_list = []
        for i in range(self.rollout_times):
            g_value = self.calculate_g_value(self.rollout(state, action, ensemble_index))
            g_value_list.append(g_value)
            
        q_value = np.mean(g_value_list)
        
    def calculate_g_value(self, rollout_trajectory):
        g_colli = 0
        
        for state in rollout_trajectory:
            if self.collision_checking(state):
                g_colli = -50
                break
            
        g_value = g_colli + g_ego
        return g_value
    
    def colli_check(self, state):
        
        return False
    
       
class DCP_Transition_Model():
    def __init__(self, ensemble_num, history_frame, future_frame, agent_dimension, agent_num, action_scale, device, training):
        super(DCP_Transition_Model, self).__init__()
        
        self.ensemble_num = ensemble_num
        self.history_frame = history_frame
        self.future_frame = future_frame
        # self.obs_scale = 5
        self.action_scale = action_scale
        self.agent_dimension = agent_dimension  # x,y,vx,vy,yaw
        self.agent_num = agent_num
        
        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.device = device
        
        for i in range(self.ensemble_num):
            env_transition = TrajPredGaussion(self.history_frame * self.agent_dimension * self.agent_num, self.future_frame * 2 * self.agent_num, hidden_unit=128)
            env_transition.to(self.device)
            env_transition.apply(self.weight_init)
            if training:
                env_transition.train()
  
            self.ensemble_models.append(env_transition)
            self.ensemble_optimizer.append(torch.optim.Adam(
                env_transition.parameters(), lr=0.001))
            
        # transition vehicle model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(30)
        self.dt = 0.1
        self.c_r = 0.01
        self.c_a = 0.05
        self.kbm = KinematicBicycleModel(
            self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        
        # dataset
        self.data = []
        self.trained_data = []
        self.one_trajectory = []
        self.infer_obs_list = []
    
        self.rollout_times = 30   

    def rollout(self, state, candidate_trajectory, ensemble_index):
        if ensemble_index > self.ensemble_num-1:
            print("[Warning]: Ensemble Index out of index!")
            return None
        
        rollout_trajectory = []
        
        history_obs = state
        vehicle_num = 0
        for i in range(len(history_obs[0])):
            if history_obs[0][i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1

        history_obs = np.array(history_obs).flatten().tolist()
        history_obs = torch.tensor(history_obs).to(self.device)

        predict_action, sigma = self.ensemble_models[ensemble_index](history_data)
        predict_action = predict_action.cpu().detach().numpy()
        
        for j in range(1, vehicle_num):  # exclude ego vehicle
            one_path = Frenet_path()
            one_path.t = [t for t in np.arange(0.0, 0.1 * self.future_frame, 0.1)]
            one_path.c = j  # use the c to indicate which vehicle
            one_path.cd = ensemble_index  # use the cd to indicate which ensemble model
            one_path.cf = self.ensemble_num  # use the cf to indicate heads num
            x = obs[j][0]
            y = obs[j][1]
            velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
            yaw = obs[j][4]
            for k in range(0, self.future_frame):
                throttle = predict_action[0][j][2*k] * self.action_scale
                delta = predict_action[0][j][2*k+1] * self.action_scale
                x, y, yaw, velocity, _, _ = self.kbm.kinematic_model(
                    x, y, yaw, velocity, throttle, delta)
                one_path.x.append(x)
                one_path.y.append(y)
            # paths_of_one_model.append(one_path)
            rollout_trajectory.append(one_path)
    
        ego_states = candidate_trajectory
        
        return rollout_trajectory

        
    # transition_training functions
    def add_training_data(self, obs, done):
        trajectory_length = self.history_frame + self.future_frame
        if not done:
            self.one_trajectory.append(obs)
            if len(self.one_trajectory) > trajectory_length:
                self.data.append(self.one_trajectory[0:trajectory_length])
                self.one_trajectory.pop(0)
        else:
            self.one_trajectory = []
    
    def update_model(self):
        if len(self.data) > 0:
            # take data
            one_trajectory = self.data[0]
            
            history_obs = one_trajectory[0:self.history_frame] 
            history_obs = np.array(history_obs).flatten().tolist()
            history_obs = torch.tensor(history_obs).to(self.device)

            # target: output action
            target_action = self.get_target_action_from_obs(one_trajectory)
            target_action = np.array(target_action).flatten().tolist()
            target_action = torch.tensor(
                target_action).to(self.device).unsqueeze(0)

            for i in range(self.ensemble_num):
                # compute loss
                predict_action, sigma = self.ensemble_models[i](history_obs)
                print("target_action",target_action)
                print("predict_action",predict_action)
                # print("sigma",sigma)
                diff = (predict_action - target_action) / sigma
                loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
                
                print("------------loss", loss)

                # train
                self.ensemble_optimizer[i].zero_grad()
                loss.backward()

                self.ensemble_optimizer[i].step()

            # closed loop test
            # self.predict_future_paths(one_trajectory[0], done=False)
            # self.predict_future_paths(one_trajectory[1], done=False)
            # self.predict_future_paths(one_trajectory[2], done=False)
            # self.predict_future_paths(one_trajectory[3], done=False)
            # paths_of_all_models = self.predict_future_paths(
            #     one_trajectory[4], done=False)

            # dx = paths_of_all_models[0].x[-1] - one_trajectory[-1][1][0]
            # dy = paths_of_all_models[0].y[-1] - one_trajectory[-1][1][1]
            # fde = math.sqrt(dx**2 + dy**2)

            # print("fde", fde)

            self.trained_data.append(one_trajectory)
            self.data.pop(0)

        # if self.train_step % 10000 == 0:
        #     self.save_prediction_model(self.train_step)

        return None
 
    def get_target_action_from_obs(self, one_trajectory):
        action_list = []
        vehicle_num = 0
        for i in range(len(one_trajectory[0])):
            if one_trajectory[0][i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1
        action_list = []

        for j in range(0, vehicle_num):
            vehicle_action = []
            for i in range(0, self.future_frame):
                x1 = one_trajectory[self.history_frame-1+i][j][0]
                y1 = one_trajectory[self.history_frame-1+i][j][1]
                yaw1 = one_trajectory[self.history_frame-1+i][j][4]
                v1 = math.sqrt(one_trajectory[self.history_frame-1+i][j][2]
                               ** 2 + one_trajectory[self.history_frame-1+i][j][3] ** 2)
                x2 = one_trajectory[self.history_frame+i][j][0]
                y2 = one_trajectory[self.history_frame+i][j][1]
                yaw2 = one_trajectory[self.history_frame+i][j][4]
                v2 = math.sqrt(one_trajectory[self.history_frame+i][j][2]
                               ** 2 + one_trajectory[self.history_frame+i][j][3] ** 2)
                throttle, delta = self.kbm.calculate_a_from_data(
                    x1, y1, yaw1, v1, x2, y2, yaw2, v2)

                vehicle_action.append(throttle/self.action_scale)
                vehicle_action.append(delta/self.action_scale)
                action_list.append(vehicle_action)
        for k in range (self.agent_num - vehicle_num):
            vehicle_action = []
            for i in range(0, self.future_frame):
                vehicle_action.append(0) 
                vehicle_action.append(0) 
            action_list.append(vehicle_action)

        return action_list 
 
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load(self, load_step):
        try:
            for i in range(self.ensemble_num):

                self.ensemble_models[i].load_state_dict(
                    torch.load('DCP_models/ensemble_models_%s_%s.pt' %
                               (load_step, i))
                )
            print("[DCP] : Load Learned Model, Step=", load_step)
        except:
            load_step = 0
            print("[DCP] : No Learned Model, Creat New Model")
        return load_step
   
    def save(self):
        for i in range(self.ensemble_num):
            torch.save(
                self.ensemble_models[i].state_dict(),
                'DCP_models/ensemble_models_%s_%s.pt' % (step, i)
            )

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    target_speed = 30/3.6 
    
    # results = Results(trajectory_planner.obs_prediction.gnn_predictin_model.history_frame)

    pass_time = 0
    task_time = 0
    
    fig, ax = plt.subplots()

    # Loop over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        episode_reward = 0
        done = False
        decision_count = 0
        
        his_obs_frames = []
        
        # Loop over steps
        while True:
            obs = np.array(obs)
            dynamic_map.update_map_from_list_obs(obs, env)
            rule_trajectory, rule_action = trajectory_planner.trajectory_update(dynamic_map)
            rule_trajectory = trajectory_planner.trajectory_update_CP(rule_action, rule_trajectory)
            # Control
            control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
            action = [control_action.acc, control_action.steering]
            new_obs, reward, done, collision_signal = env.step(action)   
            
            his_obs_frames.append(obs)
            # if len(his_obs_frames) > trajectory_planner.obs_prediction.gnn_predictin_model.history_frame-1:
            #     results.add_data_for_real_time_metrics(his_obs_frames, trajectory_planner.all_trajectory[int(rule_action - 1)][0], collision_signal)

            #     his_obs_frames.pop(0)
                
            obs = new_obs
            episode_reward += reward  
            
            # draw debug signal

            # for trajectory in trajectory_planner.all_trajectory:
            #     for i in range(len(trajectory[0].x)-1):
    
            #         # env.debug.draw_point(carla.Location(x=trajectory[0].x[i],y=trajectory[0].y[i],z=env.ego_vehicle.get_location().z+1),
            #         #                      size=0.04, color=carla.Color(r=0,g=0,b=255), life_time=0.11)
            #         env.debug.draw_line(begin=carla.Location(x=trajectory[0].x[i],y=trajectory[0].y[i],z=env.ego_vehicle.get_location().z+0.1),
            #                             end=carla.Location(x=trajectory[0].x[i+1],y=trajectory[0].y[i+1],z=env.ego_vehicle.get_location().z+0.1), 
            #                             thickness=0.1, color=carla.Color(r=0,g=0,b=255), life_time=0.2)
            
            for ref_point in env.ref_path.central_path:
                env.debug.draw_point(carla.Location(x=ref_point.position.x,y=ref_point.position.y,z=env.ego_vehicle.get_location().z+0.1),
                                        size=0.05, color=carla.Color(r=0,g=0,b=0), life_time=0.3)
                # env.debug.draw_line(begin=carla.Location(x=ref_point.position.x,y=ref_point.position.y,z=env.ego_vehicle.get_location().z+1),
                #     end=carla.Location(x=ref_point.position.x,y=ref_point.position.y,z=env.ego_vehicle.get_location().z+1), 
                #     thickness=0.1, color=(0,0,0), life_time=0.1)
            

            for predict_trajectory in trajectory_planner.obs_prediction.predict_paths:
                for i in range(len(predict_trajectory.x)-1):
                    # env.debug.draw_point(carla.Location(x=predict_trajectory.x[i],y=predict_trajectory.y[i],z=env.ego_vehicle.get_location().z+0.5),
                    #                      size=0.04, color=carla.Color(r=255,g=0,b=0), life_time=0.11)  
                    env.debug.draw_line(begin=carla.Location(x=predict_trajectory.x[i],y=predict_trajectory.y[i],z=env.ego_vehicle.get_location().z+0.1),
                                        end=carla.Location(x=predict_trajectory.x[i+1],y=predict_trajectory.y[i+1],z=env.ego_vehicle.get_location().z+0.1), 
                                        thickness=0.1,  color=carla.Color(255, 0, 0), life_time=0.2)
                    
            
            if done:
                his_obs_frames = []
                trajectory_planner.clear_buff(clean_csp=False)
                task_time += 1
                if reward > 0:
                    pass_time += 1
                break
            
    # Calculate Experiment Results
    # results.calculate_all_state_visited_time()     

    