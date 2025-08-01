import numpy as np
from .base_env import GymEnv


class CartPoleEnv(GymEnv):
    """CartPole 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__("CartPole-v1", seed)
    
    def _setup_env(self):
        """CartPole 환경을 설정합니다."""
        super()._setup_env()
        
        # CartPole은 이산 행동 공간
        self.has_continuous_action_space = False
        
        # 상태 정규화를 위한 통계
        self.state_mean = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_std = np.array([2.4, 3.2, 0.42, 3.2])
    
    def normalize_state(self, state):
        """상태를 정규화합니다."""
        return (state - self.state_mean) / self.state_std
    
    def reset(self):
        """환경을 리셋하고 정규화된 상태를 반환합니다."""
        state = super().reset()
        return self.normalize_state(state)
    
    def step(self, action):
        """환경에서 한 스텝을 진행하고 정규화된 상태를 반환합니다."""
    
        result = super().step(action)

        # Gymnasium 또는 gym>=0.26일 경우 5개
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # 구버전 gym
            next_state, reward, done, info = result
        
        return self.normalize_state(next_state), reward, done, info 