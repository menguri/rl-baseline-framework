import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .base_env import GymEnv


class MuJoCoEnvironment(GymEnv):
    """MuJoCo 환경을 위한 기본 래퍼 클래스"""
    
    def __init__(self, env_name, seed=None):
        super().__init__(env_name, seed)
        self.max_episode_steps = 1000


class HalfCheetahEnv(MuJoCoEnvironment):
    """HalfCheetah-v4 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__("HalfCheetah-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """HalfCheetah에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # HalfCheetah는 forward reward와 energy cost로 구성
        # 기본 구현이 적절하므로 그대로 사용
        return observation, reward, done, info


class Walker2dEnv(MuJoCoEnvironment):
    """Walker2d-v4 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__("Walker2d-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """Walker2d에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # Walker2d는 forward reward + alive bonus - control cost
        # 기본 구현이 적절하므로 그대로 사용
        return observation, reward, done, info


class HumanoidEnv(MuJoCoEnvironment):
    """Humanoid-v4 환경 (로봇팔 대신 전신 휴머노이드)"""
    
    def __init__(self, seed=None):
        super().__init__("Humanoid-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """Humanoid에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # Humanoid는 forward reward + alive bonus - control cost
        # 더 복잡한 보상 구조를 가지고 있음
        return observation, reward, done, info


class AntEnv(MuJoCoEnvironment):
    """Ant-v4 환경 래퍼 (사족보행 로봇)"""
    
    def __init__(self, seed=None):
        super().__init__("Ant-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """Ant에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # Ant는 forward reward + alive bonus - control cost
        return observation, reward, done, info


class SwimmerEnv(MuJoCoEnvironment):
    """Swimmer-v4 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__("Swimmer-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """Swimmer에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # Swimmer는 forward reward - control cost
        return observation, reward, done, info


class HopperEnv(MuJoCoEnvironment):
    """Hopper-v4 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__("Hopper-v4", seed)
        self.max_episode_steps = 1000
    
    def step(self, action):
        """Hopper에 맞는 보상 함수 적용"""
        observation, reward, done, info = super().step(action)
        
        # Hopper는 forward reward + alive bonus - control cost  
        return observation, reward, done, info