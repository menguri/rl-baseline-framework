import numpy as np
from .base_env import BaseEnv
try:
    import gym_pybullet_drones
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.envs.TakeoffAviary import TakeoffAviary
    PYBULLET_DRONES_AVAILABLE = True
except ImportError:
    PYBULLET_DRONES_AVAILABLE = False
    print("Warning: gym-pybullet-drones not available. Please install with: pip install gym-pybullet-drones")


class DroneHoverEnv(BaseEnv):
    """Drone Hover 환경 래퍼 클래스
    
    단일 드론이 z=1.0 높이에서 호버링하는 가장 간단한 task
    - Action space: Box(4,) - normalized RPMs [0,1] for 4 motors  
    - Observation space: kinematics (position, velocity, orientation)
    - Episode length: 8000 steps (기본값, ~27초 @ 300Hz)
    """
    
    def __init__(self, env_name="hover", seed=None):
        self.env_name = env_name
        if not PYBULLET_DRONES_AVAILABLE:
            raise ImportError("gym-pybullet-drones not installed. Please install with: pip install gym-pybullet-drones")
        super().__init__(env_name, seed)
    
    def _setup_env(self):
        """HoverAviary 환경을 설정합니다."""
        if self.env_name == "hover":
            # Single drone hovering at z=1.0
            self.env = HoverAviary(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                neighbourhood_radius=10,
                initial_xyzs=None,
                initial_rpys=None,
                physics=Physics.PYB,
                freq=240,  # 240Hz control frequency
                aggregate_phy_steps=1,
                gui=False,  # Headless training
                record=False,
                obstacles=False,
                user_debug_gui=False
            )
        elif self.env_name == "takeoff":
            # Single drone takeoff task
            self.env = TakeoffAviary(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                neighbourhood_radius=10,
                initial_xyzs=None,
                initial_rpys=None,
                physics=Physics.PYB,
                freq=240,
                aggregate_phy_steps=1,
                gui=False,
                record=False,
                obstacles=False,
                user_debug_gui=False
            )
        else:
            raise ValueError(f"Unsupported drone environment: {self.env_name}")
        
        if self.seed is not None:
            self.env.seed(self.seed)
        
        # Get observation and action dimensions
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
            
        # Handle multi-agent case (we use single drone but framework returns dict)
        if isinstance(obs, dict):
            # Single drone case: obs is {0: observation_array}
            sample_obs = list(obs.values())[0]
            self.state_dim = sample_obs.shape[0]
        else:
            self.state_dim = obs.shape[0]
        
        # Action space: normalized RPMs for 4 motors
        self.action_dim = self.env.action_space[0].shape[0] if isinstance(self.env.action_space, dict) else self.env.action_space.shape[0]
        self.has_continuous_action_space = True
        
    def reset(self):
        """환경을 리셋하고 observation을 반환합니다."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
            
        # Convert multi-agent obs to single agent
        if isinstance(obs, dict):
            # Single drone: return obs[0]
            state = list(obs.values())[0]
        else:
            state = obs
            
        return state
    
    def step(self, action):
        """환경에서 한 스텝을 진행합니다."""
        # Convert single agent action to multi-agent format if needed
        if isinstance(self.env.action_space, dict):
            # Multi-agent format: {0: action}
            action_dict = {0: action}
            obs, reward, done, truncated, info = self.env.step(action_dict)
        else:
            obs, reward, done, truncated, info = self.env.step(action)
        
        # Handle done/truncated (new gym API)
        if isinstance(done, dict):
            done = list(done.values())[0]
        if isinstance(truncated, dict):
            truncated = list(truncated.values())[0]
        
        # Convert multi-agent obs/reward to single agent
        if isinstance(obs, dict):
            next_state = list(obs.values())[0]
        else:
            next_state = obs
            
        if isinstance(reward, dict):
            reward = list(reward.values())[0]
        
        # Combine done and truncated
        done = done or truncated
        
        return next_state, reward, done, info
    
    def get_action_bounds(self):
        """행동의 범위를 반환합니다."""
        if self.has_continuous_action_space:
            if isinstance(self.env.action_space, dict):
                action_space = list(self.env.action_space.values())[0]
            else:
                action_space = self.env.action_space
            return action_space.low, action_space.high
        else:
            return 0, self.action_dim - 1
    
    def render(self, mode='rgb_array'):
        """환경을 렌더링합니다."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        else:
            return None
    
    def close(self):
        """환경을 닫습니다."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def set_seed(self, seed):
        """시드를 설정합니다."""
        self.seed = seed
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        np.random.seed(seed)


class DroneTakeoffEnv(DroneHoverEnv):
    """Drone Takeoff 환경 래퍼"""
    
    def __init__(self, seed=None):
        super().__init__(env_name="takeoff", seed=seed)