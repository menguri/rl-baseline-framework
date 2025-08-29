import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from utils.logger import ExperimentLogger, set_seed
from environments.cartpole_env import CartPoleEnv
from environments import (
    LunarLanderContinuousEnv, HalfCheetahEnv, Walker2dEnv, 
    HumanoidEnv, AntEnv, SwimmerEnv, HopperEnv, FetchReachEnv, FetchPushEnv
)
from algorithms import PPO, TRPO, DDPG, REINFORCE, A2C, TD3, SAC


class Trainer:
    """강화학습 학습기"""
    
    def __init__(self, config_path, seed=None):
        self.config = self._load_config(config_path)
        self.seed = seed                                     #or self.config['environment']['seed']
        
        # 시드 설정
        set_seed(self.seed)
        
        # 디바이스 설정 (자동 감지, 안전한 CUDA 체크)
        requested_device = self.config['experiment']['device']
        
        def safe_cuda_check():
            """CUDA 드라이버 에러를 방지하는 안전한 CUDA 체크"""
            try:
                return torch.cuda.is_available()
            except RuntimeError:
                return False
        
        if requested_device == 'auto':
            # 자동 감지: GPU 있으면 cuda, 없으면 cpu
            if safe_cuda_check():
                self.device = torch.device('cuda')
                print("Auto-detected device: CUDA")
            else:
                self.device = torch.device('cpu')
                print("Auto-detected device: CPU")
        elif requested_device == 'cuda' and not safe_cuda_check():
            print("Warning: CUDA requested but not available. Using CPU instead.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(requested_device)
        
        # 환경 설정
        self.env = self._setup_environment()
        
        # 알고리즘 설정
        self.algorithm = self._setup_algorithm()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 학습 변수들
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -np.inf
        
    def _load_config(self, config_path):
        """설정 파일을 로드합니다."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_environment(self):
        """환경을 설정합니다."""
        env_name = self.config['environment']['name']
        
        if env_name == "CartPole-v1":
            return CartPoleEnv(seed=self.seed)
        elif env_name == "LunarLanderContinuous-v3":
            return LunarLanderContinuousEnv(seed=self.seed)
        elif env_name == "HalfCheetah-v5":
            return HalfCheetahEnv(seed=self.seed)
        elif env_name == "Walker2d-v5":
            return Walker2dEnv(seed=self.seed)
        elif env_name == "Humanoid-v5":
            return HumanoidEnv(seed=self.seed)
        elif env_name == "Ant-v5":
            return AntEnv(seed=self.seed)
        elif env_name == "Swimmer-v5":
            return SwimmerEnv(seed=self.seed)
        elif env_name == "Hopper-v5":
            return HopperEnv(seed=self.seed)
        elif env_name == "FetchReach-v3":
            return FetchReachEnv(seed=self.seed)
        elif env_name == "FetchPush-v3":
            return FetchPushEnv(seed=self.seed)
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
    
    def _setup_algorithm(self):
        """알고리즘을 설정합니다."""
        alg_config = self.config['algorithm']
        alg_name = alg_config['name']
        
        state_dim = self.env.get_state_dim()
        action_dim = self.env.get_action_dim()
        has_continuous_action_space = self.env.has_continuous_actions()
        if has_continuous_action_space is None:
            has_continuous_action_space = False
            
        device = self.device
        print(f"Using device: {device}")
        
        if alg_name == "PPO":
            return PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                lr_actor=alg_config['lr_actor'],
                lr_critic=alg_config['lr_critic'],
                gamma=alg_config['gamma'],
                gae_lambda=alg_config['gae_lambda'],
                clip_ratio=alg_config['clip_ratio'],
                target_kl=alg_config['target_kl'],
                train_policy_iters=alg_config['train_policy_iters'],
                train_value_iters=alg_config['train_value_iters'],
                lam=alg_config['lam'],
                has_continuous_action_space=has_continuous_action_space,
                action_std_init=alg_config['action_std_init'],
                hidden_dims=self.config['network']['hidden_dims'],
                entropy_coef=alg_config.get('entropy_coef', 0.01),
                device=device
            )
        elif alg_name == "SAC":
            return SAC(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=self.config['network']['hidden_dims'],
                lr_actor=alg_config.get('lr_actor', 3e-4),
                lr_critic=alg_config.get('lr_critic', 1e-3),
                lr_alpha=alg_config.get('lr_alpha', 3e-4),
                gamma=alg_config['gamma'],
                tau=alg_config['tau'],
                buffer_size=alg_config['buffer_size'],
                batch_size=alg_config['batch_size'],
                alpha=alg_config.get('alpha', 0.2),
                automatic_entropy_tuning=alg_config.get('automatic_entropy_tuning', True),
                target_entropy=alg_config.get('target_entropy', -action_dim),
                stable_update_size=alg_config.get('stable_update_size', 10000),
                device=device
            )
        elif alg_name == "TRPO":
            return TRPO(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=alg_config['lr'],
                gamma=alg_config['gamma'],
                gae_lambda=alg_config['gae_lambda'],
                max_kl=alg_config['max_kl'],
                damping=alg_config['damping'],
                has_continuous_action_space=has_continuous_action_space,
                action_std_init=alg_config['action_std_init'],
                hidden_dims=self.config['network']['hidden_dims'],
                critic_iters=alg_config.get('critic_iters', 80),
                device=device
            )
        elif alg_name == "DDPG":
            return DDPG(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                hidden1=alg_config.get('hidden1', 400),
                hidden2=alg_config.get('hidden2', 300),
                init_w=alg_config.get('init_w', 3e-3),
                lr_actor=alg_config['lr_actor'],
                lr_critic=alg_config['lr_critic'],
                gamma=alg_config['gamma'],
                tau=alg_config['tau'],
                buffer_size=alg_config['buffer_size'],
                batch_size=alg_config['batch_size'],
                ou_theta=alg_config.get('ou_theta', 0.15),
                ou_mu=alg_config.get('ou_mu', 0.0),
                ou_sigma=alg_config.get('ou_sigma', 0.2),
                noise_type=alg_config.get('noise_type', 'ou'),
                exploration_std=alg_config.get('exploration_std', 0.1),
                stable_update_size=alg_config.get('stable_update_size', 10000),
                bn_use = alg_config.get('bn_use', True)
            )
        elif alg_name == "TD3":
            return TD3(
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                hidden1=alg_config.get('hidden1', 400),
                hidden2=alg_config.get('hidden2', 300),
                init_w=alg_config.get('init_w', 3e-3),
                lr_actor=alg_config['lr_actor'],
                lr_critic=alg_config['lr_critic'],
                gamma=alg_config['gamma'],
                tau=alg_config['tau'],
                buffer_size=alg_config['buffer_size'],
                batch_size=alg_config['batch_size'],
                policy_noise=alg_config.get('policy_noise', 0.2),
                noise_clip=alg_config.get('noise_clip', 0.5),
                exploration_std=alg_config.get('exploration_std', 0.1),
                policy_freq= alg_config.get('policy_freq', 2),
            )
        elif alg_name == "REINFORCE":
            return REINFORCE(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=self.config['network']['hidden_dims'],
                lr=alg_config.get('lr', 1e-3),
                gamma=alg_config['gamma'],
                has_continuous_action_space=has_continuous_action_space,
                action_std_init=alg_config.get('action_std_init', 0.6),
                device=device
            )
        elif alg_name == "A2C":
            return A2C(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=self.config['network']['hidden_dims'],
                lr_actor=alg_config.get('lr_actor', 3e-4),
                lr_critic=alg_config.get('lr_critic', 1e-3),
                gamma=alg_config['gamma'],
                has_continuous_action_space=has_continuous_action_space,
                action_std_init=alg_config.get('action_std_init', 0.6),
                entropy_coef=alg_config.get('entropy_coef', 0.01),
                value_loss_coef=alg_config.get('value_loss_coef', 0.5),
                device=device
            )
        else:
            raise ValueError(f"Unsupported algorithm: {alg_name}")
    
    def _setup_logger(self):
        """로거를 설정합니다."""
        save_dir = self.config['experiment']['save_dir']
        experiment_name = self.config['experiment']['name']
        
        # 시드별 로그 디렉토리
        log_dir = os.path.join(save_dir, f"seed_{self.seed}")
        os.makedirs(log_dir, exist_ok=True)
        
        # wandb 설정
        logging_config = self.config.get('logging', {})
        use_wandb = logging_config.get('use_wandb', False)
        wandb_project = logging_config.get('wandb_project', 'rl-framework')
        wandb_entity = logging_config.get('wandb_entity')
        
        # 스텝 기반 로깅 설정
        enable_step_logging = logging_config.get('enable_step_logging', False)
        step_log_interval = logging_config.get('step_log_interval', 1000)
        
        return ExperimentLogger(
            log_dir, 
            f"{experiment_name}_seed_{self.seed}",
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            enable_step_logging=enable_step_logging,
            step_log_interval=step_log_interval
        )
    
    def collect_rollout(self, max_steps):
        """롤아웃을 수집합니다."""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0  # agent-steps 기준(의사결정 스텝)
        env_steps_in_episode = 0  # (선택) 물리/env 스텝 카운트

        # 한번만 가져오자
        action_low, action_high = (None, None)
        if self.env.has_continuous_actions():
            action_low, action_high = self.env.get_action_bounds()

        alg_name = self.config['algorithm']['name']
        k = int(self.config['algorithm'].get('action_repeat', 1))  # e.g., 3
        k = max(1, k)

        for step in tqdm(range(max_steps), desc="Interacting with Environment & Update"):
            # 1) 정책에서 action 샘플
            action = self.algorithm.select_action(state)
            if self.env.has_continuous_actions():
                action = np.clip(action, action_low, action_high)

            # 2) action repeat: 같은 action을 k번 적용
            acc_reward = 0.0
            done = False
            info_final = {}
            for j in range(k):
                next_state, reward, done, info = self.env.step(action)
                acc_reward += float(reward)
                env_steps_in_episode += 1
                info_final = info  # 마지막 info를 보존
                if done:
                    break

            # 3) (중요) agent-전이: 마지막 관측으로 next_state 결정
            #    알고리즘/버퍼는 한 번의 의사결정에 대해 한 건의 전이만 받는다.
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

            if alg_name in ['DDPG', 'TD3', 'SAC']:
                # 오프폴리시: (s, a, sum_r, s', done) 1건 저장
                self.algorithm.store_transition(state, action, acc_reward, next_state_tensor, done)
                self.algorithm.update()  # DDPG/TD3/SAC는 매 agent-step 업데이트
            else:
                # 온폴리시: 액션-리피트 후의 누적 reward를 한 번 기록
                if len(self.algorithm.buffer.rewards) > 0:
                    self.algorithm.buffer.rewards[-1] = acc_reward
                if done and len(self.algorithm.buffer.is_terminals) > 0:
                    self.algorithm.buffer.is_terminals[-1] = True

            # 4) 로깅 및 카운트 (agent-step 기준)
            episode_reward += acc_reward
            episode_length += 1
            self.total_steps += 1  # agent-step 총합 (원하면 env 총합도 따로 둬)
            self.logger.check_step_logging(self.total_steps, acc_reward)

            state = next_state_tensor
            if done:
                # 에피소드 종료 로그 (원하면 env_steps_in_episode도 같이 기록)
                self.logger.log_episode(self.episode_count, episode_reward, episode_length, self.total_steps)
                self.episode_count += 1
                state = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                env_steps_in_episode = 0

        # 온폴리시만 returns/advantages 계산
        if alg_name not in ['DDPG', 'TD3', 'SAC'] and hasattr(self.algorithm.buffer, 'compute_returns_and_advantages'):
            self.algorithm.buffer.compute_returns_and_advantages(
                gamma=self.config['algorithm']['gamma'],
                gae_lambda=self.config['algorithm']['gae_lambda']
            )
    
    def train(self):
        """학습을 수행합니다."""
        training_config = self.config['training']
        max_episodes = training_config['max_episodes']
        update_interval = training_config['update_interval']
        eval_interval = training_config['eval_interval']
        save_interval = training_config['save_interval']
        self.logger.log_hyperparameters(self.config)
        
        print(f"Starting training with seed {self.seed}")
        print(f"Environment: {self.config['environment']['name']}")
        print(f"Algorithm: {self.config['algorithm']['name']}")
        print(f"Max episodes: {max_episodes}")
        print(f"device: {self.device}")


        alg_name = self.config['algorithm']['name']
        
        for episode in tqdm(range(max_episodes), desc="Training"):
            self.collect_rollout(update_interval)
            update_info = self.algorithm.update()
            

            if alg_name in ['DDPG', 'TD3', 'SAC'] and update_info is not None:
                self.logger.log_losses(
                    update_info['actor_loss'],
                    update_info['critic_loss'],
                    update_info.get('alpha_loss', 0),
                    episode
                )
            elif alg_name in ['PPO', 'TRPO', 'A2C', 'REINFORCE'] and update_info is not None:
                self.logger.log_losses(
                    update_info['policy_loss'],
                    update_info['value_loss'],
                    update_info.get('entropy_loss', 0),
                    episode
                )

            if episode % eval_interval == 0:
                eval_reward = self.evaluate()
                self.logger.log_evaluation(eval_reward, episode, self.total_steps)
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.save_model("best_model.pth")

            if episode % save_interval == 0:
                self.save_model(f"model_episode_{episode}.pth")

        self.save_model("final_model.pth")
        self.logger.save_metrics()
        self.logger.close()
        print(f"Training completed. Best reward: {self.best_reward:.2f}")
    
    def evaluate(self, num_episodes=10):
        """평가를 수행합니다."""
        self.algorithm.set_eval_mode()
        
        total_reward = 0
        action_low, action_high = None, None
        if self.env.has_continuous_actions():
            action_low, action_high = self.env.get_action_bounds()
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if self.config['algorithm']['name'] in ['DDPG', 'TD3', 'SAC']:
                    action = self.algorithm.select_action(state, noise=False, evaluate=True)
                else:
                    action = self.algorithm.select_action(state)
                if self.env.has_continuous_actions():
                    action = np.clip(action, action_low, action_high)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        self.algorithm.set_train_mode()
        return total_reward / num_episodes
    
    def save_model(self, filename):
        """모델을 저장합니다."""
        save_dir = self.config['experiment']['save_dir']
        model_path = os.path.join(save_dir, f"seed_{self.seed}", filename)
        self.algorithm.save(model_path)
    
 