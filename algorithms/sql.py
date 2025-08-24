import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.distributions import Normal, Categorical

from .base import BaseOffPolicyAlgorithm
from networks.ddpg_critic import DDPGCritic
from utils.buffer import ReplayBuffer


class SQLDiscreteActor(nn.Module):
    """SQL을 위한 이산 행동공간 Actor (정책 네트워크)"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], temperature=1.0):
        super(SQLDiscreteActor, self).__init__()
        
        self.temperature = temperature
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, state):
        x = self.backbone(state)
        logits = self.policy_head(x)
        return logits
    
    def get_action_probs(self, state):
        """행동 확률 분포 반환"""
        logits = self.forward(state)
        return F.softmax(logits / self.temperature, dim=-1)
    
    def sample(self, state):
        """행동 샘플링"""
        action_probs = self.get_action_probs(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, action_probs


class SQLContinuousActor(nn.Module):
    """SQL을 위한 연속 행동공간 Actor (정책 네트워크)"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 log_std_min=-20, log_std_max=2, temperature=1.0):
        super(SQLContinuousActor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.temperature = temperature
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


class SQL(BaseOffPolicyAlgorithm):
    """Soft Q-Learning (SQL) 알고리즘
    
    Maximum entropy reinforcement learning 프레임워크에서 
    soft Q-function을 학습하는 알고리즘입니다.
    """
    
    def __init__(self, state_dim, action_dim, device='cpu',
                 hidden_dims=[256, 256], lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=256,
                 temperature=1.0, has_continuous_action_space=True,
                 stable_update_size=10000):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.stable_update_size = stable_update_size
        self.device = device
        self.temperature = temperature
        self.has_continuous_action_space = has_continuous_action_space

        # Actor Network (정책)
        if has_continuous_action_space:
            self.actor = SQLContinuousActor(
                state_dim, action_dim, hidden_dims, temperature=temperature
            ).to(device)
        else:
            self.actor = SQLDiscreteActor(
                state_dim, action_dim, hidden_dims, temperature=temperature
            ).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)

        # Q-function (Critic)
        if has_continuous_action_space:
            self.q_function = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        else:
            # 이산 행동공간에서는 state만 입력받는 Q-function
            self.q_function = self._create_discrete_q_function(state_dim, action_dim, hidden_dims).to(device)
        self.q_optim = Adam(self.q_function.parameters(), lr=lr_critic)

        # Target Q-function
        if has_continuous_action_space:
            self.q_target = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        else:
            self.q_target = self._create_discrete_q_function(state_dim, action_dim, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_function.state_dict())

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def _create_discrete_q_function(self, state_dim, action_dim, hidden_dims):
        """이산 행동공간을 위한 Q-function 생성"""
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        return nn.Sequential(*layers)

    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.has_continuous_action_space:
                if evaluate:
                    _, _, action = self.actor.sample(state)
                else:
                    action, _, _ = self.actor.sample(state)
                return action.detach().cpu().numpy().flatten()
            else:
                if evaluate:
                    # 평가 시에는 greedy 선택
                    q_values = self.q_function(state)
                    action = q_values.argmax(dim=-1)
                else:
                    # 학습 시에는 soft policy에 따라 샘플링
                    action, _, _ = self.actor.sample(state)
                return action.detach().cpu().numpy().item()

    def update(self):
        if len(self.buffer) < self.stable_update_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        if self.has_continuous_action_space:
            return self._update_continuous(states, actions, rewards, next_states, dones)
        else:
            return self._update_discrete(states, actions, rewards, next_states, dones)

    def _update_continuous(self, states, actions, rewards, next_states, dones):
        """연속 행동공간에서의 SQL 업데이트"""
        
        # Target Q-value 계산
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q_next = self.q_target(next_states, next_actions)
            # Soft Q-learning: Q + temperature * entropy
            soft_q_next = q_next + self.temperature * (-next_log_probs)
            target_q = rewards + (1 - dones.float()) * self.gamma * soft_q_next

        # Q-function 업데이트
        current_q = self.q_function(states, actions)
        q_loss = self.mse_loss(current_q, target_q)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # Policy 업데이트
        sampled_actions, log_probs, _ = self.actor.sample(states)
        q_values = self.q_function(states, sampled_actions)
        
        # Soft policy improvement: maximize Q + temperature * entropy
        policy_loss = -(q_values + self.temperature * (-log_probs)).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Soft update of target network
        self._soft_update(self.q_target, self.q_function)

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def _update_discrete(self, states, actions, rewards, next_states, dones):
        """이산 행동공간에서의 SQL 업데이트"""
        
        # Target Q-value 계산
        with torch.no_grad():
            next_q_values = self.q_target(next_states)  # [batch_size, action_dim]
            next_action_probs = self.actor.get_action_probs(next_states)  # [batch_size, action_dim]
            
            # Soft value function: V = sum_a π(a|s) * (Q(s,a) + temperature * H(π))
            next_log_probs = torch.log(next_action_probs + 1e-8)
            soft_v_next = torch.sum(next_action_probs * (next_q_values + self.temperature * (-next_log_probs)), dim=-1, keepdim=True)
            target_q = rewards + (1 - dones.float()) * self.gamma * soft_v_next

        # Q-function 업데이트
        current_q_values = self.q_function(states)  # [batch_size, action_dim]
        actions_long = actions.long().squeeze(-1)
        current_q = current_q_values.gather(1, actions_long.unsqueeze(-1))  # [batch_size, 1]
        
        q_loss = self.mse_loss(current_q, target_q)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # Policy 업데이트 (현재 Q-function 사용)
        current_q_values = self.q_function(states).detach()  # gradient 차단
        action_probs = self.actor.get_action_probs(states)
        log_probs = torch.log(action_probs + 1e-8)
        
        # Soft policy improvement: maximize sum_a π(a|s) * (Q(s,a) + temperature * H(π))
        policy_objective = torch.sum(action_probs * (current_q_values + self.temperature * (-log_probs)), dim=-1)
        policy_loss = -policy_objective.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Soft update of target network
        self._soft_update(self.q_target, self.q_function)

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'q_function': self.q_function.state_dict(),
            'q_target': self.q_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'q_optim': self.q_optim.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q_function.load_state_dict(checkpoint['q_function'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.q_optim.load_state_dict(checkpoint['q_optim'])

    def set_eval_mode(self):
        self.actor.eval()
        self.q_function.eval()
        self.q_target.eval()

    def set_train_mode(self):
        self.actor.train()
        self.q_function.train()
        self.q_target.train()