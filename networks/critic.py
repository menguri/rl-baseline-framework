import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

from .mlp import MLP

class CriticNetwork(nn.Module):
    """MLP 기반 Critic 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 has_continuous_action_space=True, action_std_init=0.6,
                 activation_function='tanh'):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        
        # Critic (가치 함수) 네트워크
        self.critic = MLP(state_dim, 1, hidden_dims, activation=activation_function)
    
    def evaluate(self, state):
        """주어진 상태의 가치를 계산합니다."""
        return self.critic(state)


class QCriticNetwork(nn.Module):
    """MLP-based Q-critic: Q(s, a) -> R"""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), activation_function='relu'):
        super(QCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic = MLP(state_dim, action_dim, hidden_dims, activation=activation_function)
        self.q = MLP(state_dim + action_dim, 1, hidden_dims, activation_function)

        # (선택) 가벼운 초기화
        for m in self.q.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        """
        state:  (B, state_dim)
        action: (B, action_dim)
        return: (B, 1)  # Q(s,a)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        q = self.q(x)
        return q

    def evaluate(self, state, action):
        """Alias for forward."""
        return self.forward(state, action)