#!/bin/bash

# CartPole PPO 실험 실행 스크립트

echo "CartPole PPO 실험 시작..."

# 단일 시드 실험
echo "단일 시드 실험 실행 중..."
python main.py train --config config/cartpole_ppo.yaml --seed 42

# 멀티 시드 실험
echo "멀티 시드 실험 실행 중..."
python main.py multi --config config/cartpole_ppo.yaml --seeds 0 1 2 3 4

# 결과 시각화
echo "결과 시각화 중..."
python main.py plot --results_dir results/cartpole_ppo --plot_type learning_curves --save_path results/cartpole_ppo/learning_curves.png

echo "CartPole PPO 실험 완료!" 