#!/usr/bin/env python3
"""
멀티 시드 강화학습 실험 실행 스크립트
"""

import argparse
import os
import sys
import yaml
import numpy as np
import json
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.trainer import Trainer
from utils.logger import set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def parse_args():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="멀티 시드 강화학습 실험 실행")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="설정 파일 경로"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="디바이스 (cpu/cuda, 기본값: 설정 파일의 디바이스 사용)"
    )
    
    parser.add_argument(
        "--max_episodes", 
        type=int, 
        default=None,
        help="최대 에피소드 수 (기본값: 설정 파일의 값 사용)"
    )
    
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=min(4, mp.cpu_count()),  # CPU 코어 수에 따라 자동 설정 (최대 4)
        help=f"병렬 실행할 워커 수 (기본값: {min(4, mp.cpu_count())}, 최대: {mp.cpu_count()})"
    )
    
    return parser.parse_args()


def aggregate_step_metrics(successful_results, save_dir):
    """여러 시드의 스텝 기반 메트릭을 집계합니다."""
    step_metrics_list = []
    
    # 각 시드의 스텝 메트릭 로드
    for result in successful_results:
        seed = result['seed']
        seed_metrics_file = os.path.join(save_dir, f"seed_{seed}", "metrics.json")
        
        if os.path.exists(seed_metrics_file):
            try:
                with open(seed_metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if 'step_metrics' in metrics and metrics['step_metrics']['steps']:
                        step_metrics_list.append(metrics['step_metrics'])
            except Exception as e:
                print(f"Warning: 시드 {seed}의 스텝 메트릭을 로드할 수 없습니다: {e}")
    
    if not step_metrics_list:
        return None
    
    # 모든 시드에서 공통된 스텝 찾기
    all_steps_sets = [set(metrics['steps']) for metrics in step_metrics_list]
    common_steps = set.intersection(*all_steps_sets) if all_steps_sets else set()
    
    if not common_steps:
        return None
    
    common_steps = sorted(list(common_steps))
    
    # 공통 스텝에서 메트릭 집계
    aggregated_metrics = {
        'steps': common_steps,
        'mean_rewards': [],
        'std_rewards': [],
        'mean_running_averages': [],
        'std_running_averages': [],
        'mean_episode_counts': [],
        'std_episode_counts': []
    }
    
    for step in common_steps:
        step_rewards = []
        step_running_averages = []
        step_episode_counts = []
        
        for metrics in step_metrics_list:
            try:
                step_idx = metrics['steps'].index(step)
                step_rewards.append(metrics['step_rewards'][step_idx])
                step_running_averages.append(metrics['step_running_average'][step_idx])
                step_episode_counts.append(metrics['step_episode_counts'][step_idx])
            except (ValueError, IndexError):
                continue
        
        if step_rewards:
            aggregated_metrics['mean_rewards'].append(np.mean(step_rewards))
            aggregated_metrics['std_rewards'].append(np.std(step_rewards))
            aggregated_metrics['mean_running_averages'].append(np.mean(step_running_averages))
            aggregated_metrics['std_running_averages'].append(np.std(step_running_averages))
            aggregated_metrics['mean_episode_counts'].append(np.mean(step_episode_counts))
            aggregated_metrics['std_episode_counts'].append(np.std(step_episode_counts))
    
    return aggregated_metrics


def run_single_seed(config_path, seed, device=None, max_episodes=None):
    """단일 시드로 실험을 실행합니다."""
    try:
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if max_episodes is not None:
            config['training']['max_episodes'] = max_episodes
        
        # 시드 설정
        config['environment']['seed'] = seed
        set_seed(seed)
        

        # 학습기 생성 및 학습 실행
        trainer = Trainer(config_path, seed=seed)
        trainer.train()
        
        # 결과 수집
        results = {
            'seed': seed,
            'best_reward': trainer.best_reward,
            'final_episode': trainer.episode_count,
            'total_steps': trainer.total_steps,
            'metrics': trainer.logger.metrics if hasattr(trainer.logger, 'metrics') else {}
        }
        
        print(f"시드 {seed} 실험 완료. 최고 보상: {trainer.best_reward:.2f}")
        
        return results
        
    except Exception as e:
        import traceback
        print(f"시드 {seed} 실험 중 오류 발생:")
        traceback.print_exc()
        return {
            'seed': seed,
            'error': str(e),
            'best_reward': -np.inf
        }


def run_multi_seed_experiment(config_path, device=None, max_episodes=None, num_workers=1):
    """여러 시드로 실험을 실행합니다."""
    
    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # wandb 설정 확인
    use_wandb = config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE
    wandb_project = config.get('logging', {}).get('wandb_project', 'rl-framework')
    wandb_entity = config.get('logging', {}).get('wandb_entity')
    
    # 시드 설정
    seeds = config['experiment'].get('seeds', [0, 1, 2, 3, 4])
    
    print("=" * 60)
    print("멀티 시드 강화학습 실험 시작")
    print("=" * 60)
    print(f"설정 파일: {config_path}")
    print(f"환경: {config['environment']['name']}")
    print(f"알고리즘: {config['algorithm']['name']}")
    print(f"시드들: {config['environment']['seed']} (실험 시드: {seeds})")
    print(f"디바이스: {device or config['experiment']['device']}")
    print(f"최대 에피소드: {max_episodes or config['training']['max_episodes']}")
    print(f"워커 수: {num_workers}")
    print(f"wandb 사용: {use_wandb}")
    print("=" * 60)
    
    results = []
    
    if num_workers == 1:
        # 순차 실행
        for seed in seeds:

            if use_wandb:
                # wandb 초기화 (멀티 시드 실험용)
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f"{config['experiment']['name']}_multi_{seed}",
                    config=config,
                    group="multi_seed_experiment"
                )
            result = run_single_seed(config_path, seed, device, max_episodes)
            results.append(result)

            if use_wandb:
                    wandb.finish()
            
    else:
        # 병렬 실행
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 작업 제출
            future_to_seed = {
                executor.submit(run_single_seed, config_path, seed, device, max_episodes): seed
                for seed in seeds
            }
            
            # 결과 수집
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    print(f"시드 {seed} 실험 중 예외 발생: {e}")
                    results.append({
                        'seed': seed,
                        'error': str(e),
                        'best_reward': -np.inf
                    })
    
    # 결과 정렬 (시드 순서대로)
    results.sort(key=lambda x: x['seed'])
    
    # 결과 요약
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)
    
    if successful_results:
        rewards = [r['best_reward'] for r in successful_results]
        steps = [r['total_steps'] for r in successful_results]
        episodes = [r['final_episode'] for r in successful_results]
        
        print(f"성공한 실험: {len(successful_results)}/{len(seeds)}")
        print(f"평균 최고 보상: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"최고 보상: {np.max(rewards):.2f}")
        print(f"최저 보상: {np.min(rewards):.2f}")
        print(f"평균 총 스텝: {np.mean(steps):.0f} ± {np.std(steps):.0f}")
        print(f"평균 에피소드: {np.mean(episodes):.0f} ± {np.std(episodes):.0f}")
        
        # 시드별 결과
        print("\n시드별 결과:")
        for result in results:
            if 'error' in result:
                print(f"  시드 {result['seed']}: 오류 - {result['error']}")
            else:
                print(f"  시드 {result['seed']}: {result['best_reward']:.2f}")
    
    if failed_results:
        print(f"\n실패한 실험: {len(failed_results)}")
        for result in failed_results:
            print(f"  시드 {result['seed']}: {result['error']}")
    
    # 결과 저장
    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 스텝 기반 메트릭 집계
    step_based_summary = None
    if successful_results:
        step_based_summary = aggregate_step_metrics(successful_results, save_dir)
    
    results_file = os.path.join(save_dir, "multi_seed_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'seeds': seeds,
            'results': results,
            'summary': {
                'total_experiments': len(seeds),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(failed_results),
                'mean_reward': np.mean([r['best_reward'] for r in successful_results]) if successful_results else None,
                'std_reward': np.std([r['best_reward'] for r in successful_results]) if successful_results else None,
                'max_reward': np.max([r['best_reward'] for r in successful_results]) if successful_results else None,
                'min_reward': np.min([r['best_reward'] for r in successful_results]) if successful_results else None
            },
            'step_based_summary': step_based_summary
        }, f, indent=2)
    
    print(f"\n결과가 저장되었습니다: {results_file}")
    print("=" * 60)
    
    # wandb 종료
    if use_wandb:
        wandb.finish()
    
    return results


def main():
    """메인 함수"""
    args = parse_args()
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):
        print(f"Error: 설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    # 실험 실행
    try:
        run_multi_seed_experiment(
            config_path=args.config,
            device=args.device,
            max_episodes=args.max_episodes,
            num_workers=args.num_workers
        )
    except KeyboardInterrupt:
        print("\n실험이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"실험 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 