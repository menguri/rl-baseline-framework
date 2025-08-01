#!/usr/bin/env python3
"""
결과 분석 스크립트
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_results(results_dir):
    """실험 결과를 분석합니다."""
    
    # 멀티 시드 결과 파일 찾기
    multi_seed_file = os.path.join(results_dir, "multi_seed_results.json")
    
    if not os.path.exists(multi_seed_file):
        print(f"멀티 시드 결과 파일을 찾을 수 없습니다: {multi_seed_file}")
        return None
    
    with open(multi_seed_file, 'r') as f:
        data = json.load(f)
    
    # 결과 추출
    results = data['results']
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if not successful_results:
        print("성공한 실험 결과가 없습니다.")
        return None
    
    # 통계 계산
    rewards = [r['best_reward'] for r in successful_results]
    steps = [r['total_steps'] for r in successful_results]
    episodes = [r['final_episode'] for r in successful_results]
    
    analysis = {
        'total_experiments': len(results),
        'successful_experiments': len(successful_results),
        'failed_experiments': len(failed_results),
        'success_rate': len(successful_results) / len(results),
        'rewards': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'median': np.median(rewards)
        },
        'steps': {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'min': np.min(steps),
            'max': np.max(steps)
        },
        'episodes': {
            'mean': np.mean(episodes),
            'std': np.std(episodes),
            'min': np.min(episodes),
            'max': np.max(episodes)
        }
    }
    
    return analysis


def print_analysis(analysis):
    """분석 결과를 출력합니다."""
    if analysis is None:
        return
    
    print("=" * 60)
    print("실험 결과 분석")
    print("=" * 60)
    
    print(f"총 실험 수: {analysis['total_experiments']}")
    print(f"성공한 실험: {analysis['successful_experiments']}")
    print(f"실패한 실험: {analysis['failed_experiments']}")
    print(f"성공률: {analysis['success_rate']:.2%}")
    
    print("\n보상 통계:")
    rewards = analysis['rewards']
    print(f"  평균: {rewards['mean']:.2f} ± {rewards['std']:.2f}")
    print(f"  중간값: {rewards['median']:.2f}")
    print(f"  최소: {rewards['min']:.2f}")
    print(f"  최대: {rewards['max']:.2f}")
    
    print("\n스텝 통계:")
    steps = analysis['steps']
    print(f"  평균: {steps['mean']:.0f} ± {steps['std']:.0f}")
    print(f"  최소: {steps['min']:.0f}")
    print(f"  최대: {steps['max']:.0f}")
    
    print("\n에피소드 통계:")
    episodes = analysis['episodes']
    print(f"  평균: {episodes['mean']:.0f} ± {episodes['std']:.0f}")
    print(f"  최소: {episodes['min']:.0f}")
    print(f"  최대: {episodes['max']:.0f}")
    
    print("=" * 60)


def save_analysis(analysis, save_path):
    """분석 결과를 파일로 저장합니다."""
    if analysis is None:
        return
    
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"분석 결과가 저장되었습니다: {save_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="실험 결과 분석")
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True,
        help="결과 디렉토리 경로"
    )
    
    parser.add_argument(
        "--save_path", 
        type=str,
        help="분석 결과를 저장할 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 결과 분석
    analysis = analyze_results(args.results_dir)
    
    # 결과 출력
    print_analysis(analysis)
    
    # 결과 저장
    if args.save_path:
        save_analysis(analysis, args.save_path)


if __name__ == "__main__":
    main() 