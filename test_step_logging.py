#!/usr/bin/env python3
"""
스텝 기반 로깅 기능 테스트 스크립트
"""

import os
import sys
import yaml
import json
import tempfile
import shutil
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import ExperimentLogger
from utils.config_validator import validate_step_logging_consistency

def test_step_based_logging():
    """스텝 기반 로깅이 제대로 작동하는지 테스트합니다."""
    
    print("=" * 60)
    print("스텝 기반 로깅 기능 테스트")
    print("=" * 60)
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 로거 초기화
        logger = ExperimentLogger(
            log_dir=temp_dir,
            experiment_name="test_step_logging",
            use_wandb=False,
            enable_step_logging=True,
            step_log_interval=1000
        )
        
        # 가짜 훈련 시뮬레이션
        episode_count = 0
        step_count = 0
        
        print("\n가짜 훈련 시뮬레이션 시작...")
        
        for episode in range(50):  # 50개 에피소드
            episode_reward = 0
            episode_length = 0
            
            # 에피소드 길이는 20-100 스텝 사이에서 랜덤
            import random
            episode_len = random.randint(20, 100)
            
            for step_in_episode in range(episode_len):
                # 매 스텝마다 보상 추가
                reward = random.uniform(-1, 2)  # -1에서 2 사이 보상
                episode_reward += reward
                episode_length += 1
                step_count += 1
            
            # 에피소드 종료 시 로깅
            logger.log_episode(episode, episode_reward, episode_length, step_count)
            episode_count += 1
            
            if episode % 10 == 0:
                print(f"에피소드 {episode}: 보상={episode_reward:.2f}, 길이={episode_length}, 총 스텝={step_count}")
        
        print(f"\n훈련 완료! 총 {episode_count}개 에피소드, {step_count}개 스텝")
        
        # 메트릭 저장
        logger.save_metrics()
        logger.close()
        
        # 결과 검증
        metrics_file = os.path.join(temp_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print(f"\n결과 검증:")
            print(f"- 에피소드 기반 메트릭: {len(metrics['episode_rewards'])}개 에피소드")
            
            if 'step_metrics' in metrics and metrics['step_metrics']['steps']:
                step_metrics = metrics['step_metrics']
                print(f"- 스텝 기반 메트릭: {len(step_metrics['steps'])}개 스텝 포인트")
                print(f"- 로깅된 스텝들: {step_metrics['steps']}")
                print(f"- 스텝별 평균 보상: {[f'{r:.2f}' for r in step_metrics['step_rewards']]}")
                print(f"- 러닝 평균: {[f'{r:.2f}' for r in step_metrics['step_running_average']]}")
                print(f"- 스텝별 에피소드 수: {step_metrics['step_episode_counts']}")
                
                print(f"\n✅ 스텝 기반 로깅이 성공적으로 작동했습니다!")
                return True
            else:
                print(f"\n❌ 스텝 기반 메트릭이 저장되지 않았습니다.")
                return False
        else:
            print(f"\n❌ 메트릭 파일이 생성되지 않았습니다.")
            return False

def test_step_based_plotting():
    """스텝 기반 플롯 함수가 제대로 작동하는지 테스트합니다."""
    
    print("\n" + "=" * 60)
    print("스텝 기반 플롯 기능 테스트")
    print("=" * 60)
    
    # 기존 결과 디렉토리가 있는지 확인
    results_dir = "results/cartpole_ppo_wandb"
    if not os.path.exists(results_dir):
        print(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        print("실제 실험을 먼저 실행해주세요:")
        print("python -m main multi --config config/cartpole_ppo.yaml --seeds 0 1")
        return False
    
    # 스텝 기반 메트릭이 있는지 확인
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith("seed_")]
    if not seed_dirs:
        print(f"시드 디렉토리를 찾을 수 없습니다.")
        return False
    
    # 첫 번째 시드의 메트릭 파일 확인
    first_seed_dir = os.path.join(results_dir, seed_dirs[0])
    metrics_file = os.path.join(first_seed_dir, "metrics.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if 'step_metrics' in metrics and metrics['step_metrics']['steps']:
            print("✅ 스텝 기반 메트릭이 발견되었습니다.")
            print("플롯 함수를 테스트하려면 다음 명령어를 실행하세요:")
            print(f"python -m main plot --results_dir {results_dir} --plot_type step_learning_curves")
            return True
        else:
            print("❌ 스텝 기반 메트릭이 없습니다. enable_step_logging: true로 실험을 다시 실행해주세요.")
            return False
    else:
        print(f"메트릭 파일을 찾을 수 없습니다: {metrics_file}")
        return False

if __name__ == "__main__":
    # 1. 설정 파일 일관성 검증
    print("1단계: 설정 파일 일관성 검증")
    config_valid = validate_step_logging_consistency()
    
    # 2. 스텝 기반 로깅 테스트
    print("\n2단계: 스텝 기반 로깅 기능 테스트")
    logging_success = test_step_based_logging()
    
    # 3. 플롯 기능 테스트
    if logging_success:
        print("\n3단계: 플롯 기능 테스트")
        test_step_based_plotting()
    
    print("\n" + "=" * 60)
    if config_valid and logging_success:
        print("모든 테스트가 완료되었습니다! 🎉")
        print("\n다음 단계:")
        print("1. 실제 알고리즘으로 스텝 기반 로깅 테스트:")
        print("   python -m main multi --config config/cartpole_ppo.yaml --seeds 0 1")
        print("2. 스텝 기반 학습 곡선 플롯:")
        print("   python -m main plot --results_dir results/cartpole_ppo_wandb --plot_type step_learning_curves")
        print("3. 알고리즘 간 스텝 기반 비교:")
        print("   python -m main plot --plot_type step_comparison --comparison_dirs results/cartpole_ppo_wandb results/cartpole_trpo_wandb --labels PPO TRPO")
        print("\n💡 팁: wandb에서 그래프를 볼 때")
        print("   - X축을 'training_step'으로 설정")
        print("   - 'step_metrics/avg_reward'와 'step_metrics/running_average' 메트릭 확인")
        print("   - 모든 알고리즘이 같은 스텝에서 데이터 포인트를 가져야 함")
    else:
        print("테스트에서 문제가 발생했습니다.")
        if not config_valid:
            print("  ❌ 설정 파일 일관성 문제")
        if not logging_success:
            print("  ❌ 스텝 기반 로깅 문제")
    print("=" * 60)