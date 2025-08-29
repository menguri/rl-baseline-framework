import gymnasium as gym
import gymnasium_robotics
import numpy as np
from .base_env import BaseEnv


class FetchReachEnv(BaseEnv):
    """FetchReach-v5 환경 래퍼 클래스
    
    Fetch 로봇팔이 목표 위치에 end-effector를 도달시키는 task
    - Action space: Box(4,) - [dx, dy, dz, gripper_control]
    - Observation space: Dict with observation, achieved_goal, desired_goal
    - Episode length: 50 steps (기본값)
    """
    
    def _setup_env(self):
        """FetchReach-v5 환경을 설정합니다."""
        # gymnasium-robotics 환경 등록
        gym.register_envs(gymnasium_robotics)
        
        # FetchReach-v5 환경 생성
        self.env = gym.make('FetchReach-v5')
        
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
        
        # FetchReach는 Dict observation space를 가지므로 특별 처리
        # observation space: Dict with keys 'observation', 'achieved_goal', 'desired_goal'
        obs_space = self.env.observation_space['observation']
        goal_space = self.env.observation_space['achieved_goal'] 
        desired_goal_space = self.env.observation_space['desired_goal']
        
        # 전체 상태 차원: observation + achieved_goal + desired_goal
        self.state_dim = (obs_space.shape[0] + 
                         goal_space.shape[0] + 
                         desired_goal_space.shape[0])
        
        # Action space: Box(4,) - continuous
        self.action_dim = self.env.action_space.shape[0]  # 4 (dx, dy, dz, gripper)
        self.has_continuous_action_space = True
        
    def reset(self):
        """환경을 리셋하고 flattened observation을 반환합니다."""
        obs_dict, _ = self.env.reset()
        
        # Dict observation을 flatten
        state = self._flatten_observation(obs_dict)
        return state
    
    def step(self, action):
        """환경에서 한 스텝을 진행합니다."""
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Dict observation을 flatten
        next_state = self._flatten_observation(obs_dict)
        
        return next_state, reward, done, info
    
    def _flatten_observation(self, obs_dict):
        """Dict observation을 1D array로 변환합니다."""
        observation = obs_dict['observation']
        achieved_goal = obs_dict['achieved_goal']  
        desired_goal = obs_dict['desired_goal']
        
        # Concatenate all observations
        state = np.concatenate([observation, achieved_goal, desired_goal])
        return state
    
    def get_observation_info(self):
        """observation의 구성 정보를 반환합니다."""
        obs_space = self.env.observation_space['observation']
        goal_space = self.env.observation_space['achieved_goal']
        desired_goal_space = self.env.observation_space['desired_goal']
        
        return {
            'observation_dim': obs_space.shape[0],  # 10 (gripper position, velocity etc.)
            'achieved_goal_dim': goal_space.shape[0],  # 3 (x, y, z)
            'desired_goal_dim': desired_goal_space.shape[0],  # 3 (x, y, z)
            'total_dim': self.state_dim
        }
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Goal-conditioned reward를 계산합니다 (HER에서 사용 가능)."""
        return self.env.compute_reward(achieved_goal, desired_goal, info)
    
    def set_seed(self, seed):
        """시드를 설정합니다."""
        self.seed = seed
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        np.random.seed(seed)