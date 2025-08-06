#!/usr/bin/env python3
"""
강화학습 실험 실행 스크립트
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.trainer import Trainer
from utils.logger import set_seed


def parse_args():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description="강화학습 실험 실행")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="설정 파일 경로"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="랜덤 시드 (기본값: 설정 파일의 시드 사용)"
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
    
    return parser.parse_args()


def run_experiment(config_path, seed=None, device=None, max_episodes=None):
    """실험을 실행합니다."""
    
    # 설정 파일 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 인수로 전달된 값들로 설정 덮어쓰기
    if seed is not None:
        config['environment']['seed'] = seed
    
    if device is not None:
        # GPU 자동 감지 및 CPU 폴백 (안전한 CUDA 체크)
        def safe_cuda_check():
            try:
                return torch.cuda.is_available()
            except RuntimeError:
                return False
        
        if device == 'auto':
            config['experiment']['device'] = 'auto'
        elif device == 'cuda' and not safe_cuda_check():
            print("Warning: CUDA requested but not available. Using CPU instead.")
            config['experiment']['device'] = 'cpu'
        else:
            config['experiment']['device'] = device
    
    if max_episodes is not None:
        config['training']['max_episodes'] = max_episodes
    
    # 시드 설정
    experiment_seed = config['environment']['seed']
    set_seed(experiment_seed)
    
    print("=" * 60)
    print("강화학습 실험 시작")
    print("=" * 60)
    print(f"설정 파일: {config_path}")
    print(f"환경: {config['environment']['name']}")
    print(f"알고리즘: {config['algorithm']['name']}")
    print(f"시드: {experiment_seed}")
    print(f"디바이스: {config['experiment']['device']}")
    print(f"최대 에피소드: {config['training']['max_episodes']}")
    print("=" * 60)
    
    # 학습기 생성 및 학습 실행
    trainer = Trainer(config_path, seed=experiment_seed)
    trainer.train()
    
    print("=" * 60)
    print("실험 완료!")
    print("=" * 60)


def main():
    """메인 함수"""
    args = parse_args()
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):
        print(f"Error: 설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    # 실험 실행
    try:
        run_experiment(
            config_path=args.config,
            seed=args.seed,
            device=args.device,
            max_episodes=args.max_episodes
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