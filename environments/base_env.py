import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """환경의 기본 클래스"""
    
    def __init__(self, env_name, seed=None):
        self.env_name = env_name
        self.seed = seed
        self.env = None
        self.state_dim = None
        self.action_dim = None
        self.has_continuous_action_space = None
        
        self._setup_env()
    
    @abstractmethod
    def _setup_env(self):
        """환경을 설정합니다."""
        pass
    
    def reset(self):
        """환경을 리셋합니다."""
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        """환경에서 한 스텝을 진행합니다."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
    
    def close(self):
        """환경을 닫습니다."""
        if self.env is not None:
            self.env.close()
    
    def get_state_dim(self):
        """상태 차원을 반환합니다."""
        return self.state_dim
    
    def get_action_dim(self):
        """행동 차원을 반환합니다."""
        return self.action_dim
    
    def has_continuous_actions(self):
        """연속 행동 공간인지 확인합니다."""
        return self.has_continuous_action_space
    
    def get_action_bounds(self):
        """행동의 범위를 반환합니다."""
        if self.has_continuous_action_space:
            return self.env.action_space.low, self.env.action_space.high
        else:
            return 0, self.action_dim - 1
    
    def render(self, mode='human'):
        """환경을 렌더링합니다."""
        return self.env.render(mode=mode)
    
    def set_seed(self, seed):
        """시드를 설정합니다."""
        self.seed = seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


class GymEnv(BaseEnv):
    """Gymnasium 환경의 기본 래퍼"""
    
    def _setup_env(self):
        """Gymnasium 환경을 설정합니다."""
        self.env = gym.make(self.env_name)
        
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
        
        # 상태와 행동 차원 설정
        self.state_dim = self.env.observation_space.shape[0]
        
        # base_env.py > _setup_env() 내부
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.has_continuous_action_space = True
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.has_continuous_action_space = False
        else:
            raise NotImplementedError("지원하지 않는 action_space 타입입니다.")

        