import copy
import glob
import math
import os
import os.path as osp
import random
import sys
import time
from math import atan2

import carla
import gym
import matplotlib.pyplot as plt
import numba
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from carla import BoundingBox, Location, Rotation, Vector3D
from gym import spaces
from joblib import Parallel, delayed
from numba import jit
from numpy import clip, cos, sin, tan
from tqdm import tqdm

from Agent.drl_library.dqn.replay_buffer import (Ensemble_PrioritizedBuffer,
                                                 Ensemble_Replay_Buffer,
                                                 Replay_Buffer)
from Agent.transition_model.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from Agent.transition_model.predmlp import TrajPredGaussion, TrajPredMLP
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner

USE_CUDA = True
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


@jit(nopython=True)
def _kinematic_model(vehicle_num, obs, future_frame, action_list, throttle_scale, steer_scale, dt):
    # vehicle model parameter
    wheelbase = 2.96
    max_steer = np.deg2rad(60)
    c_r = 0.01
    c_a = 0.05

    path_list = []

    for j in range(1, vehicle_num):
        x = obs[j][0]
        y = obs[j][1]
        velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
        yaw = obs[j][4]
        
        path = []
        x_list = [x]
        y_list = [y]
        yaw_list = [yaw]
    
        for k in range(future_frame):
            throttle = action_list[j*2*future_frame + 2*k] * throttle_scale
            delta = action_list[j*2*future_frame + 2*k+1] * steer_scale
            f_load = velocity * (c_r + c_a * velocity)

            velocity += (dt) * (throttle - f_load)
            if velocity <= 0:
                velocity = 0
            
            # Compute the radius and angular velocity of the kinematic bicycle model
            if delta >= max_steer:
                delta = max_steer
            elif delta <= -max_steer:
                delta = -max_steer
            # Compute the state change rate
            x_dot = velocity * cos(yaw)
            y_dot = velocity * sin(yaw)
            omega = velocity * tan(delta) / wheelbase

            # Compute the final state using the discrete time model
            x += x_dot * dt
            y += y_dot * dt
            yaw += omega * dt
            yaw = atan2(sin(yaw), cos(yaw))
                
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)

        path.append(x_list)
        path.append(y_list)
        path.append(yaw_list)
        path_list.append(path)
        
    return path_list

@jit(nopython=True)
def _colli_check_acc(ego_x_list, ego_y_list, ego_yaw_list, rollout_trajectory, future_frame, move_gap, check_radius, time_expansion_rate):
    
    for i in range(future_frame):
        ego_x = ego_x_list[i]
        ego_y = ego_y_list[i]
        ego_yaw = ego_yaw_list[i]
        
        ego_front_x = ego_x+np.cos(ego_yaw)*move_gap
        ego_front_y = ego_y+np.sin(ego_yaw)*move_gap
        ego_back_x = ego_x-np.cos(ego_yaw)*move_gap
        ego_back_y = ego_y-np.sin(ego_yaw)*move_gap
        
        for j in range(len(rollout_trajectory)):
            one_vehicle_path = rollout_trajectory[j]
            obst_x = one_vehicle_path[0][i]
            obst_y = one_vehicle_path[1][i]
            obst_yaw = one_vehicle_path[2][i]
            
            obst_front_x = obst_x+np.cos(obst_yaw)*move_gap
            obst_front_y = obst_y+np.sin(obst_yaw)*move_gap
            obst_back_x = obst_x-np.cos(obst_yaw)*move_gap
            obst_back_y = obst_y-np.sin(obst_yaw)*move_gap
            d = (ego_front_x - obst_front_x)**2 + (ego_front_y - obst_front_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_front_x - obst_back_x)**2 + (ego_front_y - obst_back_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_back_x - obst_front_x)**2 + (ego_back_y - obst_front_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_back_x - obst_back_x)**2 + (ego_back_y - obst_back_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            
    return False
    

class CIPG_Agent():
    def __init__(self, env, training=False):
        self.env = env
        
        # transition model parameter        
        self.ensemble_num = 10
        self.used_ensemble_num = 1
        self.history_frame = 1
        self.future_frame = 1 
        self.discrete_action_num = 10
        self.obs_scale = 1
        self.obs_bias_x = 133
        self.obs_bias_y = 189
        self.throttle_scale = 2
        self.steer_scale = 1
        self.agent_dimension = 5  # x,y,vx,vy,yaw
        self.agent_num = 4
        
        self.rollout_times = 10000
        self.rollout_length = 50
        self.dt = 0.1
        self.gamma = 0.95
        self.q_batch_size = 64
        
        self.test_times = 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)
        
        self.ensemble_transition_model = OCRL_Transition_Model(self.ensemble_num, self.history_frame, 
                                                              self.future_frame, self.agent_dimension, self.agent_num,
                                                              self.obs_bias_x, self.obs_bias_y, self.obs_scale, 
                                                              self.throttle_scale, self.steer_scale, 
                                                              self.device, training, self.dt)
        
        # self.worst_confidence_Q_network = Q_network(self.history_frame * self.agent_dimension * self.agent_num, self.discrete_action_num).to(self.device)
        # self.Q_optimizer = optim.Adam(self.worst_confidence_Q_network.parameters(), lr=0.001)

        self.collect_data = []
        
        self.trained_replay_buffer = Replay_Buffer(obs_shape=env.observation_space.shape,
                                            action_shape=env.action_space.shape, # discrete, 1 dimension!
                                            capacity= 1000000,
                                            batch_size= 1,
                                            device=self.device)
        
        self.history_obs_list = []
        self.rollout_trajectory_tuple = []
        
        # basic component
        self.trajectory_planner = JunctionTrajectoryPlanner()
        assert self.future_frame * self.trajectory_planner.dt <= self.trajectory_planner.mint,"The trajectory is shorter than the agents' horizon!"
        assert self.dt == self.trajectory_planner.dt,"Trajectory planner dt is not equal to DCP agent!"
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.dynamic_map.update_ref_path(self.env)
        
        # collision checking parameter
        self.robot_radius = 2.0 # 
        self.move_gap = 2.5
        self.time_expansion_rate = 0.05
        self.check_radius = self.robot_radius
        
        # reward
        self.r_c = -10
      
    def act(self, obs):
        
        obs = np.array(obs)
        self.dynamic_map.update_map_from_list_obs(obs)
        candidate_trajectories_tuple = self.trajectory_planner.generate_candidate_trajectories(self.dynamic_map)
        
        # OCRL process            
        self.history_obs_list.append(obs)

        worst_confidence_q_list = []
        for ocrl_action, ego_trajectory in enumerate(candidate_trajectories_tuple):
            worst_confidence_q_list.append(self.estimate_worst_confidence_value(obs, ocrl_action, ego_trajectory))
        print("worst_confidence_q_list",worst_confidence_q_list)
     
        ocrl_action = np.where(worst_confidence_q_list==np.min(worst_confidence_q_list))[0]

        ocrl_trajectory = self.trajectory_planner.trajectory_update_CP(ocrl_action)

        return ocrl_action, ocrl_trajectory, candidate_trajectories_tuple
    
    def learning_by_driving(self, load_step, train_episode):
        # Create environment 
        env = self.env

        # Create Agent
        self.ensemble_transition_model.load(load_step)
        
        # Loop over episodes
        for episode in tqdm(range(1, train_episode + 1), unit='episodes'):
            
            print('Restarting episode')
            obs = env.reset()
            obs = np.array(obs)
            ocrl_action, ocrl_trajectory = self.act(obs)
            
            # Update Transition Model
            self.update_ensemble_transition_model()

            # Loop over steps
            step = 0
            while True: 
                # Reset environment and get initial state
                obs = env.reset()
                done = False        
                for i in range(self.test_times):
                    g_value = 0
                    for i in range(self.future_frame):
                        control_action =  self.controller.get_control(self.dynamic_map, ocrl_trajectory.trajectory, ocrl_trajectory.desired_speed)
                        action = [control_action.acc , control_action.steering]
                        
                        # reward calculation: assume that the ego vehicle will precisely follow trajectory
                        
                        g_value = self.estimate_ego_value(ocrl_trajectory.trajectory)
                        new_obs, reward, done, collision = env.step(action)   
                        self.collect_data.append([obs, action, reward, new_obs, done])

                        self.dynamic_map.update_map_from_list_obs(new_obs)
                        if done:
                            if collision == True:
                                g_value -= 10
                            break
                            
                    time.sleep(0.1)
                    true_q_value.append(g_value)
                    
                    self.clear_buff()
                    obs = env.reset()
                    if obs is None:
                        break
                
                if ocrl_action == 0:
                    true_q_value = -1
                else:
                    true_q_value = np.mean(true_q_value)

                print("True_Performance", true_q_value)
                print("Worst_Confidence_Value", worst_confidence_value)
                      
        return None

    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1
        epsilon_final = 0.01
        epsilon_decay = self.rollout_times / 2
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
           
    def update_ensemble_transition_model(self):
        print("[OCRL] Start Update Transition Model!")
        data_amount = len(self.collect_data)
        for i in range(data_amount):
            obs = self.collect_data[0][0]
            action = self.collect_data[0][1]
            reward = self.collect_data[0][2]
            new_obs = self.collect_data[0][3]
            done = self.collect_data[0][4]
            self.ensemble_transition_model.update_model(obs, new_obs)

            self.trained_replay_buffer.add(np.array(obs).flatten(), action, reward, np.array(new_obs).flatten(), done)
            self.collect_data.pop(0)
        return None
     
    def estimate_worst_confidence_value(self, s_0, ocrl_action, ego_trajectory):
        print("[OCRL] Start Estimate Worst Confidence Value!",ocrl_action)        
        
        self.worst_confidence_Q_network = Q_network(self.history_frame * self.agent_dimension * self.agent_num, self.discrete_action_num).to(self.device)
        self.target_Q_network = Q_network(self.history_frame * self.agent_dimension * self.agent_num, self.discrete_action_num).to(self.device)
        self.Q_optimizer = optim.Adam(self.worst_confidence_Q_network.parameters(), lr=0.0001)
        self.imagine_replay_buffer = Ensemble_PrioritizedBuffer(capacity=1000000)

        # Update_Collision_Value
        for i in range(self.rollout_times):
            
            obs = np.array(s_0)
            for steps in range(self.rollout_length+1):
                               
                collision = self.colli_check(obs)
                if collision:
                    reward = self.r_c
                    done = True
                    # print("collision!",steps)
                elif steps == self.rollout_length:
                    reward = 0
                    done = True
                else:
                    reward = 0
                    done = False
                
                new_ego_obs = self.get_ego_state_from_trajectory(ego_trajectory, steps)
                # print("debug new_ego_obs",new_ego_obs, steps)
                # use transition model to imagine surrounding agents
                new_obs_list = self.ensemble_transition_model.rollout(obs, ocrl_action, new_ego_obs)          
                # print("debug new_obs_list",new_obs_list[0], steps)
  
                self.imagine_replay_buffer.add(obs, ocrl_action, reward, new_obs_list, done)
                
                worst_confidence_q_list = []
                for new_obs in new_obs_list:
                    new_obs = torch.tensor(self.ensemble_transition_model.normalize_state(new_obs)).to(self.device)
                    worst_confidence_q = self.worst_confidence_Q_network.forward(new_obs)[ocrl_action].cpu().detach().numpy()
                    worst_confidence_q_list.append(worst_confidence_q)     
                
                new_worst_case_obs_idx = np.where(worst_confidence_q_list==np.min(worst_confidence_q_list))[0][0]
                new_worst_imgaine_obs = new_obs_list[new_worst_case_obs_idx]
                
                new_obs_idx = random.randint(0, len(new_obs_list)-1)
                new_imgaine_obs = new_obs_list[new_obs_idx]    
                
                epsilon = self.epsilon_by_frame(steps)    

                if random.random() > epsilon:
                    obs = new_worst_imgaine_obs
                else:
                    obs = new_imgaine_obs   
    
                steps += 1

                if done:
                    self.update_Q_network_with_buffer()
                    if i % 10 == 0:
                        self.update_target(self.worst_confidence_Q_network, self.target_Q_network)
                    if i % 50 == 0:
                        init_obs = torch.tensor(self.ensemble_transition_model.normalize_state(np.array(s_0))).to(self.device)
                        q_env = self.worst_confidence_Q_network.forward(init_obs)[ocrl_action].cpu().detach().numpy()
                        print("q_env",q_env)
                    break
                           
        q_ego = self.estimate_ego_value(ego_trajectory)
        
        obs = torch.tensor(self.ensemble_transition_model.normalize_state(obs)).to(self.device)
        q_env = self.worst_confidence_Q_network.forward(obs)[ocrl_action].cpu().detach().numpy()
        print("q_env",q_env)
        worst_q = q_ego + q_env
        return worst_q
    
    def get_ego_state_from_trajectory(self, ego_trajectory, steps):
        ego_state = [ego_trajectory[0].x[steps], ego_trajectory[0].y[steps], 0, 0, 0]
        # print("debug",ego_trajectory[0].x[-1])
        # print("debug",ego_trajectory[0].y[-1])
        # print("debug",ego_trajectory[0].yaw)
        return ego_state
    
    def draw_debug(self, env, obs):
        vehicle_num = 0
        for i in range(len(obs)):
            if obs[i][0] != -999: # use -999 as signal, very unstable
                vehicle_num += 1
        for j in range(1, vehicle_num):
            rotation = Rotation(0,obs[j][4]/3.14*180)
            location = Location(x=obs[j][0], y=obs[j][1],z=0.5)
            box = BoundingBox(location, Vector3D(3,1,0.1))
            
            # env.debug.draw_box(box, rotation, thickness=0.2,  color=carla.Color(255, 0, 0), life_time=0.11)
            env.debug.draw_box(box, rotation, thickness=0.05,  color=carla.Color(255, 0, 0), life_time=20)
            
    def clear_buff(self):
        self.trajectory_planner.clear_buff(clean_csp=False)
        self.history_obs_list = []

    def estimate_ego_value(self, ego_trajectory):
        q_ego = 0
        for i in range(self.rollout_length):
            # reward calculation: assume that the ego vehicle will precisely follow trajectory
            Jp = ego_trajectory[0].d_ddd[i]**2 
            Js = ego_trajectory[0].s_ddd[i]**2
            ds = (30.0 / 3.6 - ego_trajectory[0].s_d[i])**2 # target speed
            cd = 0.1 * Jp + 0.1 * 0.1 + 0.05 * ego_trajectory[0].d[i]**2
            cv = 0.1 * Js + 0.1 * 0.1 + 0.05 * ds
            q_ego -= 0.02 * cd + 0.02 * cv
            
        return q_ego
  
    def colli_check(self, obs):
        ego_x = obs[0][0]
        ego_y = obs[0][1]
        ego_yaw = obs[0][4]


        for i in range(1,self.agent_num):
            obst_x = obs[i][0]
            obst_y = obs[i][1]
            obst_yaw = obs[i][4]
            
            if self.colli_between_vehicle(ego_x, ego_y, ego_yaw, obst_x, obst_y, obst_yaw):
                return True
                
        return False
    
    def colli_between_vehicle(self, ego_x, ego_y, ego_yaw, obst_x, obst_y, obst_yaw):
        ego_front_x = ego_x+np.cos(ego_yaw)*self.move_gap
        ego_front_y = ego_y+np.sin(ego_yaw)*self.move_gap
        ego_back_x = ego_x-np.cos(ego_yaw)*self.move_gap
        ego_back_y = ego_y-np.sin(ego_yaw)*self.move_gap
        
        obst_front_x = obst_x+np.cos(obst_yaw)*self.move_gap
        obst_front_y = obst_y+np.sin(obst_yaw)*self.move_gap
        obst_back_x = obst_x-np.cos(obst_yaw)*self.move_gap
        obst_back_y = obst_y-np.sin(obst_yaw)*self.move_gap

        d = (ego_front_x - obst_front_x)**2 + (ego_front_y - obst_front_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_front_x - obst_back_x)**2 + (ego_front_y - obst_back_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_back_x - obst_front_x)**2 + (ego_back_y - obst_front_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_back_x - obst_back_x)**2 + (ego_back_y - obst_back_y)**2
        if d <= self.check_radius**2: 
            return True
        
        return False
        
    def update_Q_network_with_buffer(self):
        
        # can it accerlarate not using for? the buffer can directly sample n buffers
        for i in range(self.q_batch_size):
            obs, ocrl_action, reward, new_obs_list, done, indices, weights = self.imagine_replay_buffer.sample(1)
            worst_confidence_q_list = []
            for new_obs in new_obs_list:
                new_obs = torch.tensor(self.ensemble_transition_model.normalize_state(new_obs)).to(self.device)
                worst_confidence_q = self.target_Q_network.forward(new_obs).cpu().detach().numpy()[ocrl_action] # FIXME
                worst_confidence_q_list.append(worst_confidence_q)
            worst_next_q_value = np.min(worst_confidence_q_list)
            
            obs = torch.tensor(self.ensemble_transition_model.normalize_state(obs[0])).to(self.device)
            q_values         = self.worst_confidence_Q_network(obs)
            q_value          = q_values.gather(0, torch.tensor(int(ocrl_action[0])).to(self.device))
            next_q_value     = worst_next_q_value
            
            expected_q_value = torch.tensor(reward[0] + self.gamma * next_q_value * (1 - done[0])).to(self.device)
            loss  = (q_value - expected_q_value).pow(2)* torch.tensor(weights).to(self.device)
            prios = loss + 1e-5
            loss  = loss.mean()
                
            self.Q_optimizer.zero_grad()
            loss.backward()
            self.Q_optimizer.step()
            self.imagine_replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
                
    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
        
          
class OCRL_Transition_Model():
    def __init__(self, ensemble_num, history_frame, future_frame, agent_dimension, agent_num, obs_bias_x, obs_bias_y, 
                 obs_scale, throttle_scale, steer_scale, device, training, dt):
        super(OCRL_Transition_Model, self).__init__()
        
        self.ensemble_num = ensemble_num
        self.history_frame = history_frame
        self.future_frame = future_frame
        
        self.obs_bias_x = obs_bias_x
        self.obs_bias_y = obs_bias_y
        self.obs_scale = obs_scale
        self.throttle_scale = throttle_scale
        self.steer_scale = steer_scale
        self.agent_dimension = agent_dimension  # x,y,vx,vy,yaw
        self.agent_num = agent_num

        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.device = device
        
        for i in range(self.ensemble_num):
            env_transition = TrajPredGaussion(self.history_frame * self.agent_dimension * self.agent_num,
                                              self.future_frame * 2 * self.agent_num, hidden_unit=128)
            env_transition.to(self.device)
            env_transition.apply(self.weight_init)
            if training:
                env_transition.train()
  
            self.ensemble_models.append(env_transition)
            self.ensemble_optimizer.append(torch.optim.Adam(env_transition.parameters(), lr=0.005, weight_decay=0))
            
        # transition vehicle model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(30)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 0.05
        self.kbm = KinematicBicycleModel(
            self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        
        # dataset
        self.data = []
        self.trained_data = []
        self.one_trajectory = []
        self.infer_obs_list = []
    
    def rollout(self, obs, action, new_ego_obs):
            
        vehicle_num = 0
        for i in range(len(obs)):
            if obs[i][0] != -999: # use -999 as signal, very unstable
                vehicle_num += 1

        
        torch_obs = torch.tensor(self.normalize_state(obs)).to(self.device)   

        next_obs_list = []

        for ensemble_index in range(self.ensemble_num):
            predict_action = self.ensemble_models[ensemble_index].sample_prediction(torch_obs)
            predict_action = predict_action.cpu().detach().numpy()
                        
            next_obs = [copy.deepcopy(new_ego_obs)]
            for j in range(1, vehicle_num):  # exclude ego vehicle

                x = obs[j][0]
                y = obs[j][1]
                velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
                yaw = obs[j][4]
                throttle = predict_action[j*2*self.future_frame] * self.throttle_scale
                delta = predict_action[j*2*self.future_frame+1] * self.steer_scale

                x, y, yaw, velocity, _, _ = self.kbm.kinematic_model(x, y, yaw, velocity, throttle, delta)
                next_obs.append([x,y,velocity*math.cos(yaw),velocity*math.sin(yaw),yaw])

            for i in range(0,self.agent_num-vehicle_num):
                next_obs.append([-999,-999,0,0,0])
            next_obs_list.append(next_obs)
        
        return next_obs_list
  
    def normalize_state(self, obs):
        normalize_state = []
        normalize_obs = copy.deepcopy(obs)
        obs_length = self.agent_num
        for i in range(self.agent_num):
            if obs[i][0] == -999:
                normalize_obs[i][0] = 20
                normalize_obs[i][1] = 0
            else:
                normalize_obs[i][0] = obs[i][0] - self.obs_bias_x
                normalize_obs[i][1] = obs[i][1] - self.obs_bias_y
            
        return (np.array(normalize_obs).flatten()/self.obs_scale) # flatten to list
    
    def update_model(self, obs, new_obs):
       
        target_action = self.get_target_action_from_obs(obs, new_obs) # Run before normalize!
        torch_obs = torch.tensor(self.normalize_state(obs)).to(self.device) 
      
        # target: output action
        target_action = np.array(target_action).flatten().tolist()
        target_action = torch.tensor(target_action).to(self.device)

        for i in range(self.ensemble_num):
            # compute loss

            predict_action, sigma = self.ensemble_models[i](torch_obs)
            diff = (predict_action - target_action) / sigma
            loss = torch.mean(0.5 * torch.pow(diff, 2) + torch.log(sigma))  
            # print("-- transition loss", loss)
            # print("------------loss", predict_action)
            # print("------------loss", target_action)

            # train
            self.ensemble_optimizer[i].zero_grad()
            loss.backward()
            self.ensemble_optimizer[i].step()
            
        return None
 
    def get_target_action_from_obs(self, obs, new_obs):

        vehicle_num = 0
        for i in range(len(obs)):
            if obs[i][0] != -999: # use -999 as signal, very unstable
                vehicle_num += 1
                
        action_list = []
        for j in range(0, vehicle_num):
            vehicle_action = []
            x1 = obs[j][0]
            y1 = obs[j][1]
            yaw1 = obs[j][4]
            v1 = math.sqrt(obs[j][2]** 2 + obs[j][3] ** 2)
            x2 = new_obs[j][0]
            y2 = new_obs[j][1]
            yaw2 = new_obs[j][4]
            v2 = math.sqrt(new_obs[j][2] ** 2 + new_obs[j][3] ** 2)
            throttle, delta = self.kbm.calculate_a_from_data(
                x1, y1, yaw1, v1, x2, y2, yaw2, v2)


            vehicle_action.append(throttle/self.throttle_scale)
            vehicle_action.append(delta/self.steer_scale)
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
            nn.init.uniform_(m.weight, a=-0.3, b=0.3)
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
   
    def save(self, train_step):
        for i in range(self.ensemble_num):
            torch.save(
                self.ensemble_models[i].state_dict(),
                'DCP_models/ensemble_models_%s_%s.pt' % (train_step, i)
            )


class Rtree_Q_Store(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Rtree_Q_Store, self).__init__()
        
        
    def update_all_values(self):
        return None
    
    def add(self, obs, action, next_obs_list, reward, done):
        return None
        
    def forward(self, obs, action):
        
        q_ego = 0        # Related To Trajectory
        q_collision = 0  # Related To Collision
        
        q_value = q_ego + q_collision
        
        return q_value
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            # state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(0)[1].data.cpu().detach().numpy()
        else:
            action = random.randrange(self.num_actions)
        return action


class Q_network(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q_network, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            # state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(0)[1].data.cpu().detach().numpy()
        else:
            action = random.randrange(self.num_actions)
        return action



if __name__ == '__main__':
    a=1
