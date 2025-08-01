import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """경험 리플레이 버퍼 (DQN 등에서 사용)"""
    
    def __init__(self, capacity, state_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # 버퍼 초기화
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
    
    def push(self, state, action, reward, next_state, done):
        """새로운 경험을 버퍼에 추가합니다."""
        self.states[self.position] = torch.FloatTensor(state)
        self.actions[self.position] = torch.FloatTensor(action)
        self.rewards[self.position] = torch.FloatTensor([reward])
        self.next_states[self.position] = torch.FloatTensor(next_state)
        self.dones[self.position] = torch.BoolTensor([done])
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """배치 크기만큼 샘플을 추출합니다."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size


class RolloutBuffer:
    """롤아웃 버퍼 (PPO, TRPO 등에서 사용)"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.clear()
    
    def clear(self):
        """버퍼를 초기화합니다."""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, logprob, reward, state_value, is_terminal):
        """새로운 경험을 버퍼에 추가합니다."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)
    
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """리턴과 어드밴티지를 계산합니다."""
        advantages = []
        returns = []
        
        # GAE 계산
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.state_values[i + 1]
            
            delta = self.rewards[i] + gamma * next_value * (1 - self.is_terminals[i]) - self.state_values[i]
            gae = delta + gamma * gae_lambda * (1 - self.is_terminals[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.state_values[i])
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 정규화
        if len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_batch(self):
        """배치 데이터를 반환합니다."""
        return {
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'logprobs': torch.stack(self.logprobs),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=self.device),
            'state_values': torch.stack(self.state_values),
            'advantages': self.advantages,
            'returns': self.returns
        }
    
    def __len__(self):
        return len(self.states)


class PrioritizedReplayBuffer:
    """우선순위 기반 리플레이 버퍼"""
    
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4, device='cpu'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.position = 0
        self.size = 0
        
        # 버퍼 초기화
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
        
        # 우선순위 저장
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """새로운 경험을 버퍼에 추가합니다."""
        self.states[self.position] = torch.FloatTensor(state)
        self.actions[self.position] = torch.FloatTensor(action)
        self.rewards[self.position] = torch.FloatTensor([reward])
        self.next_states[self.position] = torch.FloatTensor(next_state)
        self.dones[self.position] = torch.BoolTensor([done])
        
        # 최대 우선순위로 설정
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """우선순위 기반으로 샘플을 추출합니다."""
        if self.size == 0:
            return None
        
        # 우선순위 기반 샘플링
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs.cpu().numpy())
        
        # 중요도 가중치 계산
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            torch.FloatTensor(weights).to(self.device),
            indices
        )
    
    def update_priorities(self, indices, priorities):
        """우선순위를 업데이트합니다."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size 