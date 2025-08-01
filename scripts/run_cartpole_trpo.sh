#!/bin/bash

# CartPole TRPO 실험 실행 스크립트

echo "CartPole TRPO 실험 시작..."

# 단일 시드 실험
echo "단일 시드 실험 실행 중..."
python main.py train --config config/cartpole_trpo.yaml --seed 42

# 멀티 시드 실험
echo "멀티 시드 실험 실행 중..."
python main.py multi --config config/cartpole_trpo.yaml --seeds 0 1 2 3 4

# 결과 시각화
echo "결과 시각화 중..."
python main.py plot --results_dir results/cartpole_trpo --plot_type learning_curves --save_path results/cartpole_trpo/learning_curves.png

echo "CartPole TRPO 실험 완료!" 