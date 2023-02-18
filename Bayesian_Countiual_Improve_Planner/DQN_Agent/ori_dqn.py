
import math
import random

import gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import NaivePrioritizedBuffer, Replay_Buffer

USE_CUDA = False#torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


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
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = random.randrange(self.num_actions)
        return action, q_value
    
class DQN():
    def __init__(self, env, batch_size):
        self.env = env
        self.current_model = Q_network(1, env.action_num)
        self.target_model  = Q_network(1, env.action_num)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.05)
        self.replay_buffer = NaivePrioritizedBuffer(1000000)
        
    def compute_td_loss(self, batch_size, beta, gamma):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta) 

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        q_values      = self.current_model(state)
        next_q_values = self.target_model(next_state)
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        # print("q_value",q_value, expected_q_value)
        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()
        
        return loss
    
    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 20
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    
    def beta_by_frame(self, frame_idx):
        beta_start = 0.4
        beta_frames = 10  
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    def train(self, num_frames, gamma, true_Q, ego_velocity):
       
        obs = self.env.reset()
        for frame_idx in range(0, num_frames + 1):

            obs = np.array(obs)

            epsilon = self.epsilon_by_frame(frame_idx)
            action, q_value = self.current_model.act(obs, epsilon)

            new_obs, reward, done, _ = self.env.step(action)
            
            # Calculate Data
            
            true_q = true_Q[action]
            collision = 0
            if true_q < -200:
                collision = 1
            random_id = random.randint(0,3)
            velocity = ego_velocity[random_id][action]
            
            print("------action_id", action, q_value[0][action].detach().numpy()*100, true_q, collision, velocity)

            # self.replay_buffer.add(obs, np.array([dqn_action]), np.array([reward]), new_obs, np.array([done]))
            self.replay_buffer.push(obs, action, reward, new_obs, done)
            
            obs = new_obs
            
            if done:
                obs = self.env.reset()

                
            if (frame_idx) > self.batch_size:
                beta = self.beta_by_frame(frame_idx)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                
            if (frame_idx) % 2 == 0:
                self.update_target(self.current_model, self.target_model)
                # self.save(frame_idx)

    def save(self, step):
        torch.save(
            self.current_model.state_dict(),
            'saved_model/current_model_%s.pt' % (step)
        )
        torch.save(
            self.target_model.state_dict(),
            'saved_model/target_model_%s.pt' % (step)
        )
        torch.save(
            self.replay_buffer,
            'saved_model/replay_buffer_%s.pt' % (step)
        )
        
    def load(self, load_step):
        try:
            self.current_model.load_state_dict(
            torch.load('saved_model/current_model_%s.pt' % (load_step))
            )

            self.target_model.load_state_dict(
            torch.load('saved_model/target_model_%s.pt' % (load_step))
            )
            
            self.replay_buffer = torch.load('saved_model/replay_buffer_%s.pt' % (load_step))
        
            print("[DQN] : Load learned model successful, step=",load_step)
        except:
            load_step = 0
            print("[DQN] : No learned model, Creat new model")
        return load_step


class known_transition_env:
    def __init__(self):
        self.action_num = 10
        self.q_transition_1 = [-200, -146.69015381981995, -319.81442211277, -296.0436107734644, -148.05297451384564, 
                      -320.8900389496694, -295.3669524361717, -151.725738843112, -128.27559202715858, -301.0827401681554] #(x=-3, y=2.5, 110)
        self.q_transition_2 = [-200, -246.69015381981995, -123.24000700386648, -104.35303753244723, -248.05297451384564, 
                        -124.60282769789222, -105.71585822647289, -231.725738843112, -325.2578958291856, -298.8163051876744] # (x=-0.6, y=3, 100)
        self.q_transition_3 = [-200, -146.69015381981995, -123.24000700386648, -104.35303753244723, -148.05297451384564, 
                        -124.60282769789222, -105.71585822647289, -151.725738843112, -128.27559202715858, -109.38862255573929] # stop
        self.q_transition = [self.q_transition_1, self.q_transition_2, self.q_transition_3]

        self.probability = [0.95,0,0.05]
        
        
        
    def reset(self):
        return [0]
    
    def step(self,action):
        new_obs = [1]
        done = True
        sample_transition = np.random.choice(3, 1, p=self.probability)
        reward = self.q_transition[sample_transition[0]][action]/100
        return new_obs, reward, done, 1

if __name__ == '__main__':
    q_transition_1 = [-200, -146.69015381981995, -319.81442211277, -296.0436107734644, -148.05297451384564, 
                      -320.8900389496694, -295.3669524361717, -151.725738843112, -128.27559202715858, -301.0827401681554] #(x=-3, y=2.5, 110)
    q_transition_2 = [-200, -246.69015381981995, -123.24000700386648, -104.35303753244723, -248.05297451384564, 
                      -124.60282769789222, -105.71585822647289, -231.725738843112, -325.2578958291856, -298.8163051876744] # (x=-0.6, y=3, 100)
    q_transition_3 = [-200, -146.69015381981995, -123.24000700386648, -104.35303753244723, -148.05297451384564, 
                      -124.60282769789222, -105.71585822647289, -151.725738843112, -128.27559202715858, -109.38862255573929] # stop
    q_transition = [q_transition_1, q_transition_2, q_transition_3]
   
    
    q_x = [i * 0.95 for i in q_transition_1]
    q_y = [i * 0.05 for i in q_transition_3]
    true_Q = [x + y for x, y in zip(q_x, q_y)]
    
    ego_velocity1= [5.4449475883232696e-08, 1.1938640330044978, 3.3530382280990043, 5.347253331405253, 1.1936368125261834, 3.355947608500486, 5.346057338512974, 1.1934681948139654, 3.1177073762368956, 3.7799043334725355]
    ego_velocity2= [5.4449475883232696e-08, 1.1940300305162699, 3.3519309116070106, 5.214032353380392, 1.1933670728770647, 3.3524816313367523, 5.32455334581238, 1.1923701222458096, 3.118730485634846, 3.776360729271646]
    ego_velocity3= [5.4449475883232696e-08, 1.1939027589790452, 3.362001795496397, 5.375778462795443, 1.1934883910806569, 3.352561741720073, 5.341210034170485, 1.192539293486069, 3.1177109520071413, 3.776158657951352]
    ego_velocity4= [5.4449475883232696e-08, 1.1939027589790452, 3.354240695079671, 5.349979047848185, 1.1933893744984403, 3.352480411725109, 5.322576561970025, 1.1922777431086553, 3.117728846474652, 3.776604650549351]
    ego_velocity = [ego_velocity1, ego_velocity2, ego_velocity3, ego_velocity4]

    
    env = known_transition_env()
    agent = DQN(env, batch_size=20)
    agent.train(100, gamma=0.99, true_Q=true_Q, ego_velocity=ego_velocity)
