import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base import BaseOnPolicyAlgorithm
from networks.actor import ActorNetwork


class REINFORCE(BaseOnPolicyAlgorithm):
    """REINFORCE (Williams, 1992) - Monte Carlo Policy Gradient 알고리즘
    
    순수한 정책 그래디언트 방법으로, 에피소드가 끝난 후에 전체 궤적에 대해 학습합니다.
    베이스라인 없이 순수하게 return을 사용하여 그래디언트를 계산합니다.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 lr=1e-3, gamma=0.99, has_continuous_action_space=False, 
                 action_std_init=0.6, device="cpu"):
        super().__init__(state_dim, action_dim, has_continuous_action_space, device)
        
        self.gamma = gamma
        self.action_std_init = action_std_init
        
        # 정책 네트워크만 사용 (value function 없음)
        self.policy = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 에피소드 버퍼 (REINFORCE는 에피소드 단위로 학습)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
    def select_action(self, state):
        """행동을 선택합니다."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.has_continuous_action_space:
                action, log_prob = self.policy.act(state)
                action = action.cpu().numpy().flatten()
            else:
                action_probs = self.policy(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.cpu().numpy().item()
        
        # 학습을 위해 저장
        self.episode_states.append(state.cpu().numpy().flatten())
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob.cpu().numpy().item())
        
        return action
    
    def store_reward(self, reward):
        """보상을 저장합니다."""
        self.episode_rewards.append(reward)
    
    def update(self):
        """에피소드가 끝난 후 정책을 업데이트합니다."""
        if len(self.episode_rewards) == 0:
            return {'policy_loss': 0.0}
        
        # Monte Carlo returns 계산
        returns = []
        discounted_sum = 0
        
        # 역방향으로 discounted return 계산
        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        # 정규화 (선택적)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 정책 손실 계산
        policy_loss = []
        
        for i in range(len(self.episode_rewards)):
            state = torch.FloatTensor(self.episode_states[i]).unsqueeze(0).to(self.device)
            
            if self.has_continuous_action_space:
                action = torch.FloatTensor([self.episode_actions[i]]).to(self.device)
                _, log_prob = self.policy.act(state, action)
            else:
                action_probs = self.policy(state)
                dist = torch.distributions.Categorical(action_probs)
                action = torch.tensor([self.episode_actions[i]]).to(self.device)
                log_prob = dist.log_prob(action)
            
            # REINFORCE 업데이트: -log_prob * Return
            policy_loss.append(-log_prob * returns[i])
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # 그래디언트 업데이트
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 에피소드 버퍼 초기화
        self.clear_episode_buffer()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': 0.0,  # REINFORCE는 value function 없음
            'entropy_loss': 0.0
        }
    
    def clear_episode_buffer(self):
        """에피소드 버퍼를 초기화합니다."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def save(self, filepath):
        """모델을 저장합니다."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """모델을 로드합니다."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_action_std(self):
        """연속 행동 공간에서 액션 표준편차를 반환합니다."""
        if self.has_continuous_action_space:
            return self.policy.action_var.item()
        return None
    
    def set_action_std(self, new_action_std):
        """연속 행동 공간에서 액션 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.policy.set_action_std(new_action_std)