import random

import numpy as np
from matplotlib import pyplot as plt

from dirichlet import dirichlet


class Agent:
    def __init__(self):
        self.dimension = 3
        self.dirichlet = dirichlet(dimension=self.dimension)
        self.action_num = 9
        
        
    
    def infer_possible_q(self, data, q_transition):
        possible_q = []
        x = self.dirichlet.generate_x(num_each_dimension=1000)
        x, y, u, s = self.dirichlet.dirichlet(x, data)
        sumy = sum(y)
        # print("y",sumy/len(x))
        
        nor_y = [item / (sumy) for item in y]
        sampled_x = self.dirichlet.sample_dirichlet(x, nor_y, sample_num=50)
        # print("sample_x",sampled_x)
        for action_id in range(self.action_num+1):
            q_action = []
            for px in sampled_x:
                q = 0
                for i, q_t in enumerate(q_transition):
                    q += px[i]*q_t[action_id]
                q_action.append(q)
            possible_q.append(q_action)
        return possible_q
    
    def choose_action_id(self, possible_q):
        chosen_action_id = 0
        max_q = -9999
        for action_id, q_action in enumerate(possible_q):
            # print("q_action",q_action,action_id)
            q = min(q_action)
            if q > max_q:
                max_q = q
                chosen_action_id = action_id
                
        return chosen_action_id, max_q
    
    
        


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

    agent = Agent()
    
    data = [1,1,1]
    
    for i in range(0, 100):
        
        # Make Decision
        possible_q = agent.infer_possible_q(data, q_transition)
        action_id, max_q = agent.choose_action_id(possible_q)
        
        # Calculate Data
        true_q = true_Q[action_id]
        collision = 0
        if true_q < -200:
            collision = 1
        random_id = random.randint(0,3)
        velocity = ego_velocity[random_id][action_id]
        
        print("------action_id", action_id, max_q, data, true_q, collision, velocity)
        
        # Collect Data
        add_data = np.random.choice(3, 1, p=[0.95,0,0.05])
        data[add_data[0]] += 1
    
    
    # Draw beta
    # alpha = [1,0,1]
    # x, y, u, s = agent.dirichlet.marginal_beta(2, alpha)
    # plt.plot(x, y, label=r'Trained episodes=%d' % (0))
    # alpha = [17,1,2]
    # x, y, u, s = agent.dirichlet.marginal_beta(2, alpha)
    # plt.plot(x, y, label=r'Trained episodes=%d' % (sum(alpha)))
    # alpha = [37,1,2]
    # x, y, u, s = agent.dirichlet.marginal_beta(2, alpha)
    # plt.plot(x, y, label=r'Trained episodes=%d' % (sum(alpha)))
    # alpha = [57,1,2]
    # x, y, u, s = agent.dirichlet.marginal_beta(2, alpha)
    # plt.plot(x, y, label=r'Trained episodes=%d' % (sum(alpha)))

    # plt.legend()
    # plt.show()
        

    
    
    
