#!/bin/bash
# scripts/run_overnight_experiments.sh
# 실행 전: chmod +x scripts/run_overnight_experiments.sh

# 자동으로 메모리 사용이 적은 GPU를 선택
# export CUDA_DEVICE=$(python -c "
# import pynvml
# pynvml.nvmlInit()
# gpus = [(i, pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used)
#         for i in range(pynvml.nvmlDeviceGetCount())]
# print(sorted(gpus, key=lambda x: x[1])[0][0])
# ")
# export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

echo "=========================================="
echo "밤새 강화학습-lunarlander 실험 시작"
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
# echo "CartPole PPO 실험 시작"
# echo "=========================================="
# python -m main multi --config config/cartpole_ppo.yaml --num_workers 1

# # PPO 완료 후 10초 대기
# echo "PPO 실험 완료. 10초 후 TRPO 실험 시작..."
# sleep 10

# TRPO 실험 실행
echo
echo "=========================================="
echo "CartPole TRPO 실험 시작"
echo "=========================================="
python -m main multi --config config/cartpole_trpo.yaml --num_workers 1

echo
echo "=========================================="
echo "모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
echo "PPO 결과: results/cartpole_ppo_wandb/multi_seed_results.json"
echo "TRPO 결과: results/cartpole_trpo_wandb/multi_seed_results.json"
echo
echo "wandb에서 실시간으로 결과를 확인할 수 있습니다!"
