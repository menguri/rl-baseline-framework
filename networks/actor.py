import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

from .mlp import MLP

class ActorNetwork(nn.Module):
    """MLP 기반 Actor 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64],
                 has_continuous_action_space=True, action_std_init=0.6, 
                 clamp_logits=False, logits_clip_range=10, activation_function='tanh'):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        self.clamp_logits = clamp_logits
        self.logits_clip_range = logits_clip_range

        # 안전한 CUDA 체크
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except RuntimeError:
            self.device = torch.device('cpu')
        
        # Actor (정책) 네트워크
        self.action_dim = action_dim
        self.action_var = torch.full(
                                    (self.action_dim,),
                                    action_std_init ** 2,
                                    dtype=torch.float32,
                                    device=self.device
                                )
        
        self.actor = MLP(state_dim, action_dim, hidden_dims, activation=activation_function)
        
    def set_action_std(self, new_action_std):
        """연속 행동 공간에서 행동 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                                        (self.action_dim,),
                                        new_action_std ** 2,
                                        dtype=torch.float32,
                                        device=self.device
                                    )
        else:
            print("WARNING: Calling set_action_std() on discrete action space policy")
    
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        
        """주어진 상태에서 행동을 선택합니다."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = Normal(action_mean, torch.sqrt(self.action_var))
        else:
            logits = self.actor(state)
            logits = torch.clamp(logits, -10, 10)  # or [-20, 20] 도 가능
            dist = Categorical(logits=logits)
        
        action = dist.sample()
        if self.has_continuous_action_space:
            action_logprob = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob
    
    def evaluate(self, state, action):
        """주어진 상태와 행동에 대한 평가를 수행합니다."""

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            dist = Normal(action_mean, torch.sqrt(self.action_var))

            # 단일 행동 환경을 위한 처리
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            logits = self.actor(state)
            logits = torch.clamp(logits, -10, 10)  # or [-20, 20] 도 가능

            if torch.isnan(logits).any():
                print("🚨 [DEBUG] logits에 NaN 있음!")
                print("after softmax:", logits)
                print("state:", state)
                raise ValueError("logits contains NaN")
            dist = Categorical(logits=logits)

            # 🔧 중요: action shape flatten
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)  # [batch_size, 1] → [batch_size]

        # 로그 확률, 엔트로피, 가치 함수
        if self.has_continuous_action_space:
            action_logprobs = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()                       # shape: [batch_size]

        return action_logprobs, dist_entropy
    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            dist = Normal(action_mean, torch.sqrt(self.action_var))
            return dist.log_prob(action).sum(dim=-1)
        else:
            logits = self.actor(state)
            logits = torch.clamp(logits, -10, 10)  # or [-20, 20] 도 가능
            dist = Categorical(logits=logits)
            return dist.log_prob(action)