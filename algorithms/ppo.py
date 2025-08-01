import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal

from .base import BaseOnPolicyAlgorithm
from networks.mlp import MLPActorCritic
from utils.buffer import RolloutBuffer


class PPO(BaseOnPolicyAlgorithm):
    """Proximal Policy Optimization (PPO) 알고리즘"""
    
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, target_kl=0.01,
                 train_policy_iters=80, train_value_iters=80, lam=0.97,
                 has_continuous_action_space=True, action_std_init=0.6,
                 hidden_dims=[64, 64], device='cpu'):
        
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_policy_iters = train_policy_iters
        self.train_value_iters = train_value_iters
        self.lam = lam
        self.has_continuous_action_space = has_continuous_action_space
        
        # 네트워크 초기화
        self.policy = MLPActorCritic(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)

        self.policy = MLPActorCritic(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': float(lr_actor)},
            {'params': self.policy.critic.parameters(), 'lr': float(lr_critic)}
        ])
        
        # 이전 정책 (PPO에서 필요)
        self.policy_old = MLPActorCritic(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 버퍼
        self.buffer = RolloutBuffer(device)
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()
        
        # 연속 행동 공간에서의 행동 표준편차
        if has_continuous_action_space:
            self.action_std = action_std_init
    
    def set_action_std(self, new_action_std):
        """연속 행동 공간에서 행동 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING: Calling set_action_std() on discrete action space policy")
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """행동 표준편차를 감소시킵니다."""
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)
        else:
            print("WARNING: Calling decay_action_std() on discrete action space policy")
    
    def select_action(self, state):
        """주어진 상태에서 행동을 선택합니다."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        # 버퍼에 저장
        self.buffer.add(state, action, action_logprob, 0, state_val, False)
        
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()
    
    def update(self, batch=None):
        """PPO 업데이트를 수행합니다."""
        if batch is None:
            # 버퍼에서 배치 가져오기
            batch = self.buffer.get_batch()
        
        states = batch['states']
        actions = batch['actions']
        old_logprobs = batch['logprobs']
        rewards = batch['rewards']
        state_values = batch['state_values']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # 이전 정책을 현재 정책으로 복사
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 정책 업데이트
        # old_logprobs는 detach()로 고정 - PPO에선 이전 정책의 로그 확률을 사용
        for _ in range(self.train_policy_iters):
            # 현재 정책으로 평가
            logprobs, _, dist_entropy = self.policy.evaluate(states, actions, ignore_critic=True)
            
            # 비율 계산
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # PPO 클립 목적 함수
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            
            # 정책 손실
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 엔트로피 손실
            entropy_loss = -dist_entropy.mean()
            
            # 전체 손실
            loss = policy_loss + 0.01 * entropy_loss
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # KL 발산 체크
            # Clip된 KL 발산이 목표 KL 발산을 초과하면 중단 - 논문 실험상으론 Clip 시행 시 KL 발산이 목표(0.02-3)를 초과하지 않음
            kl = (old_logprobs - logprobs).mean()
            if kl > self.target_kl:
                break
        
        # 가치 함수 업데이트
        for _ in range(self.train_value_iters):
            _, state_values, _ = self.policy.evaluate(states, actions)
            value_loss = self.mse_loss(state_values, returns)
            
            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()
        
        # 버퍼 초기화
        self.buffer.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_divergence': kl.item()
        }
    
    def save(self, path):
        """모델을 저장합니다."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_std': self.action_std if self.has_continuous_action_space else None
        }, path)
    
    def load(self, path):
        """모델을 로드합니다."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.has_continuous_action_space and checkpoint['action_std'] is not None:
            self.action_std = checkpoint['action_std']
            self.set_action_std(self.action_std)
    
    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        self.policy.eval()
        self.policy_old.eval()
    
    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        self.policy.train()
        self.policy_old.train()
    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            return self.policy.get_action_logprob(state, action)
    
    def get_value(self, state):
        """주어진 상태의 가치를 계산합니다."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.policy.get_value(state)
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """Generalized Advantage Estimation (GAE)를 계산합니다."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        # 정규화
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns 