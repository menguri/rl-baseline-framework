from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseAlgorithm(ABC):
    """모든 강화학습 알고리즘의 기본 클래스"""
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 학습 관련 변수들
        self.total_steps = 0
        self.episode_count = 0
        
    @abstractmethod
    def select_action(self, state):
        """주어진 상태에서 행동을 선택합니다."""
        pass
    
    @abstractmethod
    def update(self, batch):
        """배치 데이터로 네트워크를 업데이트합니다."""
        pass
    
    @abstractmethod
    def save(self, path):
        """모델을 저장합니다."""
        pass
    
    @abstractmethod
    def load(self, path):
        """모델을 로드합니다."""
        pass
    
    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        pass
    
    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        pass
    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다."""
        pass
    
    def get_value(self, state):
        """주어진 상태의 가치를 계산합니다."""
        pass
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """행동 표준편차를 감소시킵니다 (연속 행동 공간에서만 사용)."""
        pass


class BaseOnPolicyAlgorithm(BaseAlgorithm):
    """온-정책 알고리즘의 기본 클래스 (PPO, TRPO 등)"""
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        super().__init__(state_dim, action_dim, device)
        
        # 온-정책 알고리즘 관련 변수들
        self.buffer = None
        self.policy_old = None
    
    def collect_rollout(self, env, max_steps=2048):
        """롤아웃을 수집합니다."""
        states, actions, logprobs, rewards, state_values, is_terminals = [], [], [], [], [], []
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 행동 선택
            action, logprob, state_value = self.select_action(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, done, info = env.step(action)
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            state_values.append(state_value)
            is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                state, _ = env.reset()
                self.episode_count += 1
                episode_reward = 0
                episode_length = 0
        
        return {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'logprobs': torch.FloatTensor(logprobs).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'state_values': torch.FloatTensor(state_values).to(self.device),
            'is_terminals': torch.BoolTensor(is_terminals).to(self.device)
        }


class BaseOffPolicyAlgorithm(BaseAlgorithm):
    """오프-정책 알고리즘의 기본 클래스 (DQN, SAC 등)"""
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        super().__init__(state_dim, action_dim, device)
        
        # 오프-정책 알고리즘 관련 변수들
        self.buffer = None
        self.target_network = None
    
    def store_transition(self, state, action, reward, next_state, done):
        """전이를 버퍼에 저장합니다."""
        if self.buffer is not None:
            self.buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self, tau=0.005):
        """타겟 네트워크를 소프트 업데이트합니다."""
        if self.target_network is not None:
            for target_param, param in zip(self.target_network.parameters(), self.policy.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data) 