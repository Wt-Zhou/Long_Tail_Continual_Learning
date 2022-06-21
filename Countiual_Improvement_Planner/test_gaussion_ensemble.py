
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agent.transition_model.predmlp import TrajPredGaussion


class Gaussion_Data_Generate():
    def __init__(self):

        self.mean = 1.0
        self.sigma = 1.0

    def sample_data(self):
        return normal(loc=self.mean, scale=self.sigma)
    
    
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
            if training:
                env_transition.train()
  
            self.ensemble_models.append(env_transition)
            self.ensemble_optimizer.append(torch.optim.Adam(env_transition.parameters(), lr=0.005, weight_decay=0))
    
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

        
    def update_model(self, data):
        
        input = torch.tensor(1)
        data = torch.tensor(target_action).to(self.data)

        for i in range(self.ensemble_num):
            # compute loss
            predict_action, sigma = self.ensemble_models[i](input)
            # print("target_action",target_action[40:80])
            # print("predict_action",predict_action[40:80])
            # print("sigma", sigma)
            diff = (predict_action - target_action) / sigma
            loss = torch.mean(0.5 * torch.pow(diff, 2) + torch.log(sigma))  
            print("------------loss", loss)

            # train
            self.ensemble_optimizer[i].zero_grad()
            loss.backward()
            self.ensemble_optimizer[i].step()


        return None


if __name__ == '__main__':
    distribution = Gaussion_Data_Generate()
    agent = Ensemble_Guassion_Transition(1)
    
    training_steps = 100
    for i in range(training_steps):
        data = distribution.sample_data()
        agent.update_model(data)
