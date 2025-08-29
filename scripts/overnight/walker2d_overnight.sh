#!/bin/bash
# Walker2d-v4 Overnight 실험 스크립트
# 실행 예: chmod +x scripts/overnight/walker2d_overnight.sh && \
#          bash scripts/overnight/walker2d_overnight.sh

echo "=========================================="
echo "Walker2d-v4 Overnight 실험 시작"
echo "시작 시간: $(date)"
echo "=========================================="

# PYTHONPATH 설정 (현재 프로젝트 루트)
export PYTHONPATH=$(pwd)

# --- 내 전용 W&B 설정 경로 ---
export WANDB_DIR=/home/mlic/mingukang/wandb_config
export WANDB_CONFIG_DIR=/home/mlic/mingukang/wandb_config

mkdir -p $WANDB_DIR
mkdir -p $WANDB_CONFIG_DIR

# GPU 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=1
    echo "GPU detected. Using GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPU detected. Using CPU."
fi

# 알고리즘 목록 (Walker2d-v4 연속 행동공간)
algorithms=("sac" "td3" "ppo" "ddpg" "trpo")

for algo in "${algorithms[@]}"; do
    echo
    echo "=========================================="
    echo "Walker2d-v4 ${algo^^} 실험 시작"
    echo "=========================================="
    
    # 내 wandb 세션을 강제 사용하여 실행
    WANDB_DIR=$WANDB_DIR WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
    python -m main multi \
        --config config/walker2d/${algo}.yaml \
        --num_workers 5
    
    if [ $? -eq 0 ]; then
        echo "${algo^^} 실험 완료 성공!"
    else
        echo "${algo^^} 실험 실패!"
    fi
    
    echo "10초 후 다음 실험 시작..."
    sleep 10
done

echo
echo "=========================================="
echo "Walker2d-v4 모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
for algo in "${algorithms[@]}"; do
    result_file="results/walker2d-v4_${algo}/multi_seed_results.json"
    if [ -f "$result_file" ]; then
        echo "✅ ${algo^^}: $result_file"
    else
        echo "❌ ${algo^^}: 결과 파일 없음"
    fi
done

echo
echo "wandb 대시보드에서 결과를 확인하세요."