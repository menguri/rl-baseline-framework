#!/usr/bin/env python3
"""
학습 결과 시각화 스크립트
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_metrics(log_dir):
    """메트릭 파일을 로드합니다."""
    metrics_file = os.path.join(log_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def plot_step_based_learning_curves(results_dir, save_path=None):
    """여러 시드의 스텝 기반 학습 곡선을 플롯합니다."""
    
    # 시드별 결과 디렉토리 찾기
    seed_dirs = glob.glob(os.path.join(results_dir, "seed_*"))
    seed_dirs.sort()
    
    if not seed_dirs:
        print(f"시드 디렉토리를 찾을 수 없습니다: {results_dir}")
        return
    
    # 데이터 수집
    all_step_metrics = []
    seeds = []
    
    for seed_dir in seed_dirs:
        seed = int(seed_dir.split("_")[-1])
        metrics = load_metrics(seed_dir)
        
        if metrics and 'step_metrics' in metrics and metrics['step_metrics']['steps']:
            step_metrics = metrics['step_metrics']
            all_step_metrics.append(step_metrics)
            seeds.append(seed)
    
    if not all_step_metrics:
        print("스텝 기반 메트릭 데이터를 찾을 수 없습니다.")
        return
    
    # 모든 시드에서 공통된 스텝 찾기
    all_steps_sets = [set(metrics['steps']) for metrics in all_step_metrics]
    common_steps = sorted(list(set.intersection(*all_steps_sets)))
    
    if not common_steps:
        print("공통된 스텝이 없습니다.")
        return
    
    # 플롯 설정
    plt.figure(figsize=(12, 8))
    
    # 개별 시드 플롯 (투명하게)
    colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
    
    for i, (step_metrics, seed, color) in enumerate(zip(all_step_metrics, seeds, colors)):
        # 공통 스텝에서만 데이터 추출
        seed_steps = []
        seed_rewards = []
        for step in common_steps:
            if step in step_metrics['steps']:
                step_idx = step_metrics['steps'].index(step)
                seed_steps.append(step)
                seed_rewards.append(step_metrics['step_rewards'][step_idx])
        
        if seed_steps:
            plt.plot(seed_steps, seed_rewards, color=color, alpha=0.3, 
                    linewidth=1, label=f'Seed {seed}')
    
    # 평균과 표준편차 계산
    mean_rewards = []
    std_rewards = []
    
    for step in common_steps:
        step_rewards = []
        for metrics in all_step_metrics:
            if step in metrics['steps']:
                step_idx = metrics['steps'].index(step)
                step_rewards.append(metrics['step_rewards'][step_idx])
        
        if step_rewards:
            mean_rewards.append(np.mean(step_rewards))
            std_rewards.append(np.std(step_rewards))
    
    # 평균과 표준편차 플롯
    plt.plot(common_steps, mean_rewards, color='red', linewidth=3, label='Mean')
    plt.fill_between(common_steps, 
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     color='red', alpha=0.2, label='±1 Std')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Step-based Learning Curves (Multiple Seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"스텝 기반 플롯이 저장되었습니다: {save_path}")
    
    plt.show()


def plot_step_based_comparison(results_dirs, labels=None, save_path=None):
    """여러 알고리즘의 스텝 기반 성능을 비교합니다."""
    
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # 데이터 수집
    all_algorithms_data = []
    
    for results_dir in results_dirs:
        # 멀티 시드 결과 파일에서 스텝 기반 데이터 찾기
        multi_seed_file = os.path.join(results_dir, "multi_seed_results.json")
        
        if os.path.exists(multi_seed_file):
            with open(multi_seed_file, 'r') as f:
                data = json.load(f)
            
            if 'step_based_summary' in data and data['step_based_summary'] is not None:
                all_algorithms_data.append(data['step_based_summary'])
            else:
                all_algorithms_data.append(None)
        else:
            all_algorithms_data.append(None)
    
    # 유효한 데이터만 필터링
    valid_data = [(data, label) for data, label in zip(all_algorithms_data, labels) if data is not None]
    
    if not valid_data:
        print("비교할 스텝 기반 데이터를 찾을 수 없습니다.")
        return
    
    # 모든 알고리즘에서 공통된 스텝 찾기
    all_steps_sets = [set(data['steps']) for data, _ in valid_data]
    common_steps = sorted(list(set.intersection(*all_steps_sets)))
    
    if not common_steps:
        print("알고리즘 간 공통된 스텝이 없습니다.")
        return
    
    # 플롯 설정
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_data)))
    
    for i, (data, label) in enumerate(valid_data):
        # 공통 스텝에서만 데이터 추출
        alg_steps = []
        alg_mean_rewards = []
        alg_std_rewards = []
        
        for step in common_steps:
            if step in data['steps']:
                step_idx = data['steps'].index(step)
                alg_steps.append(step)
                alg_mean_rewards.append(data['mean_rewards'][step_idx])
                alg_std_rewards.append(data['std_rewards'][step_idx])
        
        if alg_steps:
            color = colors[i]
            plt.plot(alg_steps, alg_mean_rewards, color=color, linewidth=2, label=label)
            plt.fill_between(alg_steps, 
                           np.array(alg_mean_rewards) - np.array(alg_std_rewards),
                           np.array(alg_mean_rewards) + np.array(alg_std_rewards),
                           color=color, alpha=0.2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Step-based Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"스텝 기반 비교 플롯이 저장되었습니다: {save_path}")
    
    plt.show()


def plot_learning_curves(results_dir, save_path=None):
    """여러 시드의 학습 곡선을 플롯합니다."""
    
    # 시드별 결과 디렉토리 찾기
    seed_dirs = glob.glob(os.path.join(results_dir, "seed_*"))
    seed_dirs.sort()
    
    if not seed_dirs:
        print(f"시드 디렉토리를 찾을 수 없습니다: {results_dir}")
        return
    
    # 데이터 수집
    all_rewards = []
    all_episodes = []
    seeds = []
    
    for seed_dir in seed_dirs:
        seed = int(seed_dir.split("_")[-1])
        metrics = load_metrics(seed_dir)
        
        if metrics and 'episode_rewards' in metrics:
            rewards = metrics['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            all_rewards.append(rewards)
            all_episodes.append(episodes)
            seeds.append(seed)
    
    if not all_rewards:
        print("메트릭 데이터를 찾을 수 없습니다.")
        return
    
    # 플롯 설정
    plt.figure(figsize=(12, 8))
    
    # 개별 시드 플롯 (투명하게)
    colors = plt.cm.viridis(np.linspace(0, 1, len(seeds)))
    
    for i, (rewards, episodes, seed, color) in enumerate(zip(all_rewards, all_episodes, seeds, colors)):
        plt.plot(episodes, rewards, color=color, alpha=0.3, linewidth=1, label=f'Seed {seed}')
    
    # 평균과 표준편차 계산
    max_episodes = max(len(ep) for ep in all_episodes)
    mean_rewards = []
    std_rewards = []
    episode_range = []
    
    for episode in range(1, max_episodes + 1):
        episode_rewards = []
        for rewards, episodes in zip(all_rewards, all_episodes):
            if episode <= len(episodes):
                episode_rewards.append(rewards[episode - 1])
        
        if episode_rewards:
            mean_rewards.append(np.mean(episode_rewards))
            std_rewards.append(np.std(episode_rewards))
            episode_range.append(episode)
    
    # 평균과 표준편차 플롯
    plt.plot(episode_range, mean_rewards, color='red', linewidth=3, label='Mean')
    plt.fill_between(episode_range, 
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     color='red', alpha=0.2, label='±1 Std')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Learning Curves (Multiple Seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"플롯이 저장되었습니다: {save_path}")
    
    plt.show()


def plot_comparison(results_dirs, labels=None, save_path=None):
    """여러 알고리즘의 성능을 비교합니다."""
    
    if labels is None:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # 데이터 수집
    all_data = []
    
    for results_dir in results_dirs:
        # 멀티 시드 결과 파일 찾기
        multi_seed_file = os.path.join(results_dir, "multi_seed_results.json")
        
        if os.path.exists(multi_seed_file):
            with open(multi_seed_file, 'r') as f:
                data = json.load(f)
            
            if 'summary' in data and data['summary']['mean_reward'] is not None:
                all_data.append({
                    'mean': data['summary']['mean_reward'],
                    'std': data['summary']['std_reward'],
                    'min': data['summary']['min_reward'],
                    'max': data['summary']['max_reward']
                })
        else:
            # 개별 시드 결과 수집
            seed_dirs = glob.glob(os.path.join(results_dir, "seed_*"))
            rewards = []
            
            for seed_dir in seed_dirs:
                metrics = load_metrics(seed_dir)
                if metrics and 'episode_rewards' in metrics:
                    rewards.append(max(metrics['episode_rewards']))
            
            if rewards:
                all_data.append({
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards)
                })
    
    if not all_data:
        print("비교할 데이터를 찾을 수 없습니다.")
        return
    
    # 플롯 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 바 차트
    x = np.arange(len(labels))
    means = [d['mean'] for d in all_data]
    stds = [d['std'] for d in all_data]
    
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Algorithm Comparison (Mean ± Std)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 박스플롯 스타일 (최소, 최대, 평균)
    for i, data in enumerate(all_data):
        ax2.plot([i, i], [data['min'], data['max']], 'k-', linewidth=2)
        ax2.plot(i, data['mean'], 'ro', markersize=8)
        ax2.plot(i, data['min'], 'k_', markersize=10)
        ax2.plot(i, data['max'], 'k^', markersize=10)
    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Reward')
    ax2.set_title('Algorithm Comparison (Min-Max-Mean)')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"비교 플롯이 저장되었습니다: {save_path}")
    
    plt.show()


def plot_single_algorithm_analysis(results_dir, save_path=None):
    """단일 알고리즘의 상세 분석을 플롯합니다."""
    
    # 멀티 시드 결과 파일 찾기
    multi_seed_file = os.path.join(results_dir, "multi_seed_results.json")
    
    if not os.path.exists(multi_seed_file):
        print(f"멀티 시드 결과 파일을 찾을 수 없습니다: {multi_seed_file}")
        return
    
    with open(multi_seed_file, 'r') as f:
        data = json.load(f)
    
    # 데이터 추출
    results = data['results']
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("성공한 실험 결과가 없습니다.")
        return
    
    # 플롯 설정
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 시드별 최고 보상
    seeds = [r['seed'] for r in successful_results]
    rewards = [r['best_reward'] for r in successful_results]
    
    ax1.bar(seeds, rewards, alpha=0.7)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Best Reward')
    ax1.set_title('Best Reward by Seed')
    ax1.grid(True, alpha=0.3)
    
    # 2. 보상 분포
    ax2.hist(rewards, bins=10, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    ax2.set_xlabel('Best Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 시드별 총 스텝 수
    steps = [r['total_steps'] for r in successful_results]
    ax3.bar(seeds, steps, alpha=0.7)
    ax3.set_xlabel('Seed')
    ax3.set_ylabel('Total Steps')
    ax3.set_title('Total Steps by Seed')
    ax3.grid(True, alpha=0.3)
    
    # 4. 보상 vs 스텝 수
    ax4.scatter(steps, rewards, alpha=0.7)
    ax4.set_xlabel('Total Steps')
    ax4.set_ylabel('Best Reward')
    ax4.set_title('Reward vs Steps')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"분석 플롯이 저장되었습니다: {save_path}")
    
    plt.show()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="학습 결과 시각화")
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True,
        help="결과 디렉토리 경로"
    )
    
    parser.add_argument(
        "--plot_type", 
        type=str, 
        choices=['learning_curves', 'comparison', 'analysis', 'step_learning_curves', 'step_comparison'],
        default='learning_curves',
        help="플롯 타입"
    )
    
    parser.add_argument(
        "--comparison_dirs", 
        type=str, 
        nargs='+',
        help="비교할 결과 디렉토리들 (comparison 플롯용)"
    )
    
    parser.add_argument(
        "--labels", 
        type=str, 
        nargs='+',
        help="알고리즘 라벨들 (comparison 플롯용)"
    )
    
    parser.add_argument(
        "--save_path", 
        type=str, 
        help="저장할 파일 경로"
    )
    
    args = parser.parse_args()
    
    if args.plot_type == 'learning_curves':
        plot_learning_curves(args.results_dir, args.save_path)
    elif args.plot_type == 'comparison':
        if not args.comparison_dirs:
            print("Error: comparison 플롯을 위해서는 --comparison_dirs가 필요합니다.")
            sys.exit(1)
        plot_comparison(args.comparison_dirs, args.labels, args.save_path)
    elif args.plot_type == 'analysis':
        plot_single_algorithm_analysis(args.results_dir, args.save_path)
    elif args.plot_type == 'step_learning_curves':
        plot_step_based_learning_curves(args.results_dir, args.save_path)
    elif args.plot_type == 'step_comparison':
        if not args.comparison_dirs:
            print("Error: step_comparison 플롯을 위해서는 --comparison_dirs가 필요합니다.")
            sys.exit(1)
        plot_step_based_comparison(args.comparison_dirs, args.labels, args.save_path)


if __name__ == "__main__":
    main() 