#!/usr/bin/env python3
"""
모듈화된 강화학습 프레임워크 메인 실행 파일
"""

import argparse
import os
import sys
import multiprocessing as mp
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# 직접 import
from train.run_experiment import run_experiment
from train.multi_seed_trainer import run_multi_seed_experiment



def parse_args():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="모듈화된 강화학습 프레임워크")
    
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 단일 실험 실행
    train_parser = subparsers.add_parser('train', help='단일 실험 실행')
    train_parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    train_parser.add_argument('--seed', type=int, default=None, help='랜덤 시드')
    train_parser.add_argument('--device', type=str, default=None, help='디바이스 (auto/cpu/cuda, 기본값: auto)')
    train_parser.add_argument('--max_episodes', type=int, default=None, help='최대 에피소드 수')
    
    # 멀티 시드 실험 실행
    multi_parser = subparsers.add_parser('multi', help='멀티 시드 실험 실행')
    multi_parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    multi_parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='실험할 시드들')
    multi_parser.add_argument('--device', type=str, default=None, help='디바이스 (auto/cpu/cuda, 기본값: auto)')
    multi_parser.add_argument('--max_episodes', type=int, default=None, help='최대 에피소드 수')
    multi_parser.add_argument('--num_workers', type=int, default=min(4, mp.cpu_count()), help=f'병렬 실행할 워커 수 (기본값: {min(4, mp.cpu_count())})')
    
    # 결과 시각화
    plot_parser = subparsers.add_parser('plot', help='결과 시각화')
    plot_parser.add_argument('--results_dir', type=str, required=True, help='결과 디렉토리 경로')
    plot_parser.add_argument('--plot_type', type=str, choices=['learning_curves', 'comparison', 'analysis'], 
                           default='learning_curves', help='플롯 타입')
    plot_parser.add_argument('--comparison_dirs', type=str, nargs='+', help='비교할 결과 디렉토리들')
    plot_parser.add_argument('--labels', type=str, nargs='+', help='알고리즘 라벨들')
    plot_parser.add_argument('--save_path', type=str, help='저장할 파일 경로')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    if args.command == 'train':
        # 단일 실험 실행
        run_experiment(
            config_path=args.config,
            seed=args.seed,
            device=args.device,
            max_episodes=args.max_episodes
        )
    
    elif args.command == 'multi':
        # 멀티 시드 실험 실행
        run_multi_seed_experiment(
            config_path=args.config,
            device=args.device,
            max_episodes=args.max_episodes,
            num_workers=args.num_workers
        )

if __name__ == "__main__":
    main() 