import os
import logging
import torch
import numpy as np
from datetime import datetime


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def setup_logging(log_dir, name="experiment"):
    """로깅 설정을 초기화합니다."""
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 핸들러 설정
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 로거 설정
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name="experiment"):
    """기존 로거를 가져옵니다."""
    return logging.getLogger(name)


class ExperimentLogger:
    """실험 로깅을 위한 클래스"""
    
    def __init__(self, log_dir, experiment_name, use_wandb=False, wandb_project=None, wandb_entity=None, 
                 enable_step_logging=False, step_log_interval=1000):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.logger = setup_logging(log_dir, experiment_name)
        
        
        # wandb 설정
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "rl-framework",
                entity=wandb_entity,
                name=experiment_name,
                config={},
                dir=log_dir
            )
            # GPU 추적 활성화 (안전한 CUDA 체크)
            try:
                if torch.cuda.is_available():
                    wandb.watch_called = False  # 중복 호출 방지
            except RuntimeError:
                # CUDA 드라이버 없으면 GPU 추적 생략
                pass
        
        # 메트릭 저장
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            # 스텝 기반 메트릭 추가
            'step_metrics': {
                'steps': [],
                'step_rewards': [],
                'step_running_average': [],
                'step_episode_counts': []
            }
        }
        
        # 실시간 평균 계산을 위한 변수들
        self.reward_history = []
        self.length_history = []
        self.episode_count = 0
        
        # 스텝 기반 로깅을 위한 변수들
        self.step_logging_enabled = enable_step_logging
        self.step_log_interval = step_log_interval
        self.last_logged_step = 0
        self.step_episode_rewards = []  # 스텝 간격 동안의 에피소드 보상들
        self.step_episode_count_at_interval = 0  # 스텝 간격에서의 에피소드 수
    
    def log_episode(self, episode, reward, length, step):
        """에피소드 결과를 로깅합니다."""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        
        # 실시간 평균 계산을 위한 히스토리 업데이트
        self.reward_history.append(reward)
        self.length_history.append(length)
        self.episode_count += 1
        
        # 스텝 기반 로깅을 위한 에피소드 보상 저장
        if self.step_logging_enabled:
            self.step_episode_rewards.append(reward)
            self.step_episode_count_at_interval += 1
            
            # 스텝 간격에 도달했으면 스텝 메트릭 로깅
            if step >= self.last_logged_step + self.step_log_interval:
                self._log_step_metrics(step)
        
        
        # wandb에 기록
        if self.use_wandb:
            log_data = {
                'episode/reward': reward,
                'episode/length': length,
                'episode': episode,
                'step': step
            }
            
            # 실시간 평균 계산 (최근 10개 에피소드)
            if len(self.reward_history) >= 10:
                recent_rewards = self.reward_history[-10:]
                recent_lengths = self.length_history[-10:]
                
                log_data.update({
                    'episode/avg_reward_10': np.mean(recent_rewards),
                    'episode/avg_length_10': np.mean(recent_lengths),
                    'episode/std_reward_10': np.std(recent_rewards),
                    'episode/std_length_10': np.std(recent_lengths)
                })
            
            # 전체 평균
            if len(self.reward_history) > 0:
                log_data.update({
                    'episode/avg_reward_all': np.mean(self.reward_history),
                    'episode/avg_length_all': np.mean(self.length_history),
                    'episode/max_reward_so_far': np.max(self.reward_history),
                    'episode/min_reward_so_far': np.min(self.reward_history)
                })
            
            # GPU 메트릭 추가 (안전한 CUDA 체크)
            try:
                if torch.cuda.is_available():
                    # 기본 GPU 메모리 정보만 수집 (MB 단위)
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**2

                    gpu_metrics = {
                        f"gpu/memory_allocated_mb": memory_allocated,
                        f"gpu/memory_reserved_mb": memory_reserved,
                        }
                    log_data.update(gpu_metrics)
            except RuntimeError:
                # CUDA 드라이버 없으면 GPU 메트릭 생략
                pass
            
            wandb.log(log_data)
        
        # 콘솔에 출력
        if episode % 10 == 0:
            avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
            self.logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}, Avg Reward (last 10)={avg_reward:.2f}")
    
    def _log_step_metrics(self, current_step):
        """내부적으로 스텝 기반 메트릭을 로깅합니다."""
        if not self.step_episode_rewards:
            return
        
        # 현재 스텝 간격에서의 평균 보상 계산 (실제 스텝별 보상들)
        interval_avg_reward = np.mean(self.step_episode_rewards)
        interval_sum_reward = np.sum(self.step_episode_rewards)
        steps_in_interval = len(self.step_episode_rewards)
        
        # 전체 히스토리에서의 러닝 평균 계산
        if self.reward_history:
            running_average = np.mean(self.reward_history)
        else:
            running_average = interval_avg_reward
        
        # 스텝 메트릭 저장
        self.metrics['step_metrics']['steps'].append(current_step)
        self.metrics['step_metrics']['step_rewards'].append(interval_avg_reward)
        self.metrics['step_metrics']['step_running_average'].append(running_average)
        self.metrics['step_metrics']['step_episode_counts'].append(self.step_episode_count_at_interval)
        
        
        # wandb에 기록
        if self.use_wandb:
            wandb.log({
                'step_metrics/episode_count': self.step_episode_count_at_interval,
                'step_metrics/avg_reward_per_step': interval_avg_reward,
                'step_metrics/sum_reward_in_interval': interval_sum_reward,
                'step_metrics/steps_in_interval': steps_in_interval,
                'step_metrics/running_average': running_average,
                'training_step': current_step
            }, step=current_step)
        
        # 콘솔에 출력
        # self.logger.info(f"Step {current_step}: Interval Avg={interval_avg_reward:.4f} (sum={interval_sum_reward:.2f}, steps={steps_in_interval}), Running Avg={running_average:.2f}")
        
        # 다음 간격을 위해 리셋
        self.step_episode_rewards = []
        self.step_episode_count_at_interval = 0
        self.last_logged_step = current_step
    
    def log_losses(self, policy_loss, value_loss, entropy_loss, step):
        """손실 함수들을 로깅합니다."""
        self.metrics['policy_losses'].append(policy_loss)
        self.metrics['value_losses'].append(value_loss)
        self.metrics['entropy_losses'].append(entropy_loss)
        
        
        # wandb에 기록
        if self.use_wandb:
            wandb.log({
                'loss/policy': policy_loss,
                'loss/value': value_loss,
                'loss/entropy': entropy_loss,
                'step': step
            })
    
    def log_hyperparameters(self, config):
        """하이퍼파라미터를 로깅합니다."""
        self.logger.info("Hyperparameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # wandb에 하이퍼파라미터 기록
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_evaluation(self, eval_reward, eval_episode, step):
        """평가 결과를 로깅합니다."""
        if self.use_wandb:
            wandb.log({
                'evaluation/reward': eval_reward,
                'evaluation/episode': eval_episode,
                'step': step
            })
    
    def close(self):
        """로거를 종료합니다."""
        if self.use_wandb:
            wandb.finish()
        self.logger.info("Experiment completed.")
    
    def save_metrics(self):
        """메트릭을 파일로 저장합니다."""
        import json
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def check_step_logging(self, current_step, reward):
        """매 스텝마다 호출되어 스텝 로깅을 체크합니다."""
        if not self.step_logging_enabled:
            return
        
        # 현재 스텝 간격의 보상들을 수집
        self.step_episode_rewards.append(reward)
        
        # 스텝 간격에 도달했는지 체크
        if current_step >= self.last_logged_step + self.step_log_interval:
            self._log_step_metrics(current_step)
    
    def _get_gpu_metrics(self):
        """GPU 메모리 사용량만 간단히 로깅합니다."""
        try:
            if not torch.cuda.is_available():
                return {}
                
            gpu_metrics = {}
            for i in range(torch.cuda.device_count()):
                device_name = f"gpu_{i}"
                
                # 기본 GPU 메모리 정보만 수집 (MB 단위)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                
                gpu_metrics.update({
                    f"gpu/{device_name}/memory_allocated_mb": memory_allocated,
                    f"gpu/{device_name}/memory_reserved_mb": memory_reserved,
                })
            
            return gpu_metrics
        except Exception as e:
            print(f"Warning: Failed to collect GPU metrics: {str(e)}")
            return {}

def set_seed(seed):
    """랜덤 시드를 설정합니다."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)