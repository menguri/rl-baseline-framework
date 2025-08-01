#!/bin/bash

# ---------------------
# 시뮬레이션 설정
# ---------------------
ALGO=$1                      # ddpg, ppo, trpo 중 하나
SEED=$2                      # seed_0, seed_1, seed_2 중 하나
EPISODES=${3:-5}             # 평가 에피소드 수 (기본값: 5)
RECORD_DIR="./video_output" # 영상 저장 경로
ENV="LunarLanderContinuous-v3"

# ---------------------
# 모델 경로 구성
# ---------------------
MODEL_PATH="results/lunarlandercontinuous_${ALGO}/$SEED/best_model.pth"

# ---------------------
# PYTHONPATH 설정
# ---------------------
export PYTHONPATH=.

# ---------------------
# 시뮬레이션 실행
# ---------------------
echo "▶️ Simulating: algo=$ALGO | seed=$SEED | episodes=$EPISODES"
echo "📁 Model: $MODEL_PATH"
echo "🎥 Video: $RECORD_DIR"

python ./eval/simulate.py \
  --env $ENV \
  --model-path $MODEL_PATH \
  --algorithm $ALGO \
  --episodes $EPISODES \
  --record-dir $RECORD_DIR
