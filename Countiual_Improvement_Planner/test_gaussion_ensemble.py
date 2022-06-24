
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agent.transition_model.predmlp import TrajPredGaussion


class Gaussion_Data_Generate():
    def __init__(self):

        self.mean = 0.5
        self.sigma = 0.25

    def sample_data(self):
        return np.random.normal(loc=self.mean, scale=self.sigma)
    
    
class Ensemble_Guassion_Transition():
    
    def __init__(self, ensemble_num):
        super(Ensemble_Guassion_Transition, self).__init__()
        
        self.ensemble_num = ensemble_num
        
        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(self.ensemble_num):
            env_transition = TrajPredGaussion(1, 1, hidden_unit=8)
            env_transition.to(self.device)
            env_transition.apply(self.weight_init)
            env_transition.train()
  
            self.ensemble_models.append(env_transition)
            self.ensemble_optimizer.append(torch.optim.Adam(env_transition.parameters(), lr=0.001, weight_decay=0))
    
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-1, b=1)
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

        
    def update_model(self, data):
        
        state = torch.tensor([1.5,]).to(self.device)
        data = torch.tensor(data).to(self.device)
        print("data",data)

        for i in range(self.ensemble_num):
            # compute loss
            
            predict_action, sigma = self.ensemble_models[i](state)
            diff = (predict_action - data) / sigma
            loss = torch.mean(0.5 * torch.pow(diff, 2) + torch.log(sigma))  
            # print("------------loss", loss)
            print("------------", i)
            print("predict_mean",predict_action)
            print("predict_sigma",sigma)
            
            predict_action, sigma = self.ensemble_models[i](torch.tensor([2.5,]).to(self.device))
            print("2.5",predict_action, sigma)
            predict_action, sigma = self.ensemble_models[i](torch.tensor([-1.0,]).to(self.device))
            print("-1",predict_action, sigma)

            # train
            self.ensemble_optimizer[i].zero_grad()
            loss.backward()
            self.ensemble_optimizer[i].step()


        return None


if __name__ == '__main__':
    distribution = Gaussion_Data_Generate()
    agent = Ensemble_Guassion_Transition(ensemble_num=5)
    
    training_steps = 500
    for i in range(training_steps):
        print("train_steps",i)
        data = distribution.sample_data()
        agent.update_model(data)
