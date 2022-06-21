import os
import pdb

# from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            # nn.Linear(hidden_unit, hidden_unit),
            # nn.LayerNorm(hidden_unit),
            # nn.LeakyReLU(),

            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        # print("in mlp",x, self.mlp(x))
        return self.mlp(x)
    
    
class TrajPredGaussion(nn.Module):
    """Predict gaussion trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit, max_sigma=0.5, min_sigma=1e-4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            # nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(hidden_unit, out_channels)
        self.fc_sigma = nn.Linear(hidden_unit, out_channels)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)

    def forward(self, x):
        x = self.fc(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        # sigma = self.fc_sigma(x)  
        # scaled of action should not be too large (i.e., max_sigma=10), it will never converge!
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
