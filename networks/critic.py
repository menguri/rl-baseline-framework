import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

from .mlp import MLP

class CriticNetwork(nn.Module):
    """MLP 기반 Critic 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 has_continuous_action_space=True, action_std_init=0.6):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        
        # Critic (가치 함수) 네트워크
        self.critic = MLP(state_dim, 1, hidden_dims, activation='tanh')
    
    def evaluate(self, state):
        """주어진 상태의 가치를 계산합니다."""
        return self.critic(state)