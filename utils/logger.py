import os
import logging
import torch
import numpy as np
from datetime import datetime

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

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
        
        # TensorBoard writer
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(log_dir, "tensorboard")
            self.writer = SummaryWriter(tb_dir)
        
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
        
        # TensorBoard에 기록
        if self.writer is not None:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Length', length, episode)
        
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
            
            wandb.log(log_data)
        
        # 콘솔에 출력
        if episode % 10 == 0:
            avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
            self.logger.info(f"Episode {episode}: Reward={reward:.2f}, Length={length}, Avg Reward (last 10)={avg_reward:.2f}")
    
    def _log_step_metrics(self, current_step):
        """내부적으로 스텝 기반 메트릭을 로깅합니다."""
        if not self.step_episode_rewards:
            return
        
        # 현재 스텝 간격에서의 평균 보상 계산
        interval_avg_reward = np.mean(self.step_episode_rewards)
        
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
        
        # TensorBoard에 기록
        if self.writer is not None:
            self.writer.add_scalar('Step/AverageReward', interval_avg_reward, current_step)
            self.writer.add_scalar('Step/RunningAverage', running_average, current_step)
            self.writer.add_scalar('Step/EpisodeCount', self.step_episode_count_at_interval, current_step)
        
        # wandb에 기록
        if self.use_wandb:
            wandb.log({
                'step_metrics/avg_reward': interval_avg_reward,
                'step_metrics/running_average': running_average,
                'step_metrics/episode_count': self.step_episode_count_at_interval,
                'training_step': current_step
            })
        
        # 콘솔에 출력
        self.logger.info(f"Step {current_step}: Avg Reward={interval_avg_reward:.2f}, Running Avg={running_average:.2f}, Episodes={self.step_episode_count_at_interval}")
        
        # 다음 간격을 위해 리셋
        self.step_episode_rewards = []
        self.step_episode_count_at_interval = 0
        self.last_logged_step = current_step
    
    def log_step_metrics(self, step, avg_reward=None, episode_count=None):
        """외부에서 직접 스텝 메트릭을 로깅할 때 사용합니다."""
        if not self.step_logging_enabled:
            return
            
        if avg_reward is not None and episode_count is not None:
            # 전체 히스토리에서의 러닝 평균 계산
            if self.reward_history:
                running_average = np.mean(self.reward_history)
            else:
                running_average = avg_reward
            
            # 스텝 메트릭 저장
            self.metrics['step_metrics']['steps'].append(step)
            self.metrics['step_metrics']['step_rewards'].append(avg_reward)
            self.metrics['step_metrics']['step_running_average'].append(running_average)
            self.metrics['step_metrics']['step_episode_counts'].append(episode_count)
            
            # TensorBoard에 기록
            if self.writer is not None:
                self.writer.add_scalar('Step/AverageReward', avg_reward, step)
                self.writer.add_scalar('Step/RunningAverage', running_average, step)
                self.writer.add_scalar('Step/EpisodeCount', episode_count, step)
            
            # wandb에 기록
            if self.use_wandb:
                wandb.log({
                    'step_metrics/avg_reward': avg_reward,
                    'step_metrics/running_average': running_average,
                    'step_metrics/episode_count': episode_count,
                    'training_step': step
                })
            
            self.logger.info(f"Step {step}: Avg Reward={avg_reward:.2f}, Running Avg={running_average:.2f}, Episodes={episode_count}")
    
    def log_losses(self, policy_loss, value_loss, entropy_loss, step):
        """손실 함수들을 로깅합니다."""
        self.metrics['policy_losses'].append(policy_loss)
        self.metrics['value_losses'].append(value_loss)
        self.metrics['entropy_losses'].append(entropy_loss)
        
        # TensorBoard에 기록
        if self.writer is not None:
            self.writer.add_scalar('Loss/Policy', policy_loss, step)
            self.writer.add_scalar('Loss/Value', value_loss, step)
            self.writer.add_scalar('Loss/Entropy', entropy_loss, step)
        
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
            if self.writer is not None:
                self.writer.add_text('Hyperparameters', f"{key}: {value}", 0)
        
        # wandb에 하이퍼파라미터 기록
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_model(self, model, step):
        """모델을 로깅합니다."""
        if self.use_wandb:
            # 모델 파일 저장
            model_path = os.path.join(self.log_dir, f"model_step_{step}.pth")
            torch.save(model.state_dict(), model_path)
            
            # wandb에 모델 아티팩트로 업로드
            artifact = wandb.Artifact(f"model-{self.experiment_name}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
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
        if self.writer is not None:
            self.writer.close()
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


def set_seed(seed):
    """랜덤 시드를 설정합니다."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 