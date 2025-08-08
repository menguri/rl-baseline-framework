#!/bin/bash
# scripts/run_overnight_experiments.sh
# 실행 전: chmod +x scripts/run_overnight_experiments.sh

# GPU 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=1
    echo "GPU detected. Using GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPU detected. Using CPU."
fi

echo "=========================================="
echo "밤새 강화학습-halfcheetah 실험 시작"
echo "시작 시간: $(date)"
echo "=========================================="

# PYTHONPATH 설정 (상대경로 import 오류 방지)
export PYTHONPATH=$(pwd)

# wandb 로그인 확인
export WANDB_API_KEY=0fccac088b1041eeae51eaad8941a490bf71592c
export WANDB_CONFIG_DIR=/home/mlic/mingukang/.wandb_config
echo "wandb 로그인 상태 확인 중..."
wandb status >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "wandb 로그인이 필요합니다. 다음 명령어로 로그인하세요:"
  echo "wandb login"
  exit 1
fi

echo "wandb 로그인 확인 완료!"

# # PPO 실험 실행
# echo
# echo "=========================================="
# echo "Halfcheetah PPO 실험 시작"
# echo "=========================================="
# python -m main multi --config config/halfcheetah_ppo.yaml --num_workers 1

# # PPO 완료 후 10초 대기
# echo "PPO 실험 완료. 10초 후 TRPO 실험 시작..."
# sleep 10

# TRPO 실험 실행
echo
echo "=========================================="
echo "Halfcheetah TRPO 실험 시작"
echo "=========================================="
python -m main multi --config config/halfcheetah_trpo.yaml --num_workers 2

# TRPO 완료 후 10초 대기
echo "TRPO 실험 완료. 10초 후 DDPG 실험 시작..."
sleep 10

# DDPG 실험 실행
echo
echo "=========================================="
echo "Halfcheetah DDPG 실험 시작"
echo "=========================================="
python -m main multi --config config/halfcheetah_ddpg.yaml --num_workers 2

# DDPG 완료 후 10초 대기
echo "DDPG 실험 완료..."
sleep 10

echo
echo "=========================================="
echo "모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
echo "PPO 결과: results/halfcheetah_ppo/multi_seed_results.json"
echo "TRPO 결과: results/halfcheetah_trpo/multi_seed_results.json"
echo "TRPO 결과: results/halfcheetah_ddpg/multi_seed_results.json"
echo
echo "wandb에서 실시간으로 결과를 확인할 수 있습니다!"
