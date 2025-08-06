import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base import BaseOnPolicyAlgorithm
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork


class A2C(BaseOnPolicyAlgorithm):
    """Advantage Actor-Critic (A2C) 알고리즘
    
    Actor-Critic 방법으로, 정책 네트워크(Actor)와 가치 함수(Critic)를 동시에 학습합니다.
    Advantage function을 사용하여 정책 그래디언트의 분산을 줄입니다.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, 
                 has_continuous_action_space=False, action_std_init=0.6,
                 entropy_coef=0.01, value_loss_coef=0.5, device="cpu"):
        super().__init__(state_dim, action_dim, has_continuous_action_space, device)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.action_std_init = action_std_init
        
        # Actor (정책 네트워크)
        self.policy = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        
        # Critic (가치 함수)
        self.value_function = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=lr_critic)
        
        # 경험 버퍼 (n-step buffer)
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def select_action(self, state):
        """행동을 선택합니다."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 가치 함수 평가
            value = self.value_function(state)
            
            # 정책에서 행동 선택
            if self.has_continuous_action_space:
                action, log_prob = self.policy.act(state)
                action_np = action.cpu().numpy().flatten()
            else:
                action_probs = self.policy(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_np = action.cpu().numpy().item()
        
        # 버퍼에 저장
        self.states.append(state.cpu().numpy().flatten())
        self.actions.append(action_np)
        self.values.append(value.cpu().numpy().item())
        self.log_probs.append(log_prob.cpu().numpy().item())
        
        return action_np
    
    def store_transition(self, reward, done):
        """전환 정보를 저장합니다."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, next_state=None):
        """정책과 가치 함수를 업데이트합니다."""
        if len(self.rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # 다음 상태의 가치 계산 (에피소드가 끝나지 않은 경우)
        if next_state is not None and not self.dones[-1]:
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.value_function(next_state).cpu().numpy().item()
        else:
            next_value = 0.0
        
        # Returns와 Advantages 계산
        returns, advantages = self._compute_gae(next_value)
        
        # 텐서로 변환
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device) if self.has_continuous_action_space else torch.LongTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # 정규화
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 정책 손실 계산
        if self.has_continuous_action_space:
            _, log_probs = self.policy.act(states, actions)
            dist = self.policy.get_distribution(states)
            entropy = dist.entropy().mean()
        else:
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 가치 함수 손실 계산
        values = self.value_function(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # 전체 손실
        total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
        
        # 그래디언트 업데이트
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        # 버퍼 초기화
        self.clear_buffer()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def _compute_gae(self, next_value, gae_lambda=0.95):
        """Generalized Advantage Estimation (GAE)를 계산합니다."""
        returns = []
        advantages = []
        
        gae = 0
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
        
        return returns, advantages
    
    def clear_buffer(self):
        """버퍼를 초기화합니다."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save(self, filepath):
        """모델을 저장합니다."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_function.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """모델을 로드합니다."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_function.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
    
    def get_action_std(self):
        """연속 행동 공간에서 액션 표준편차를 반환합니다."""
        if self.has_continuous_action_space:
            return self.policy.action_var.item()
        return None
    
    def set_action_std(self, new_action_std):
        """연속 행동 공간에서 액션 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.policy.set_action_std(new_action_std)