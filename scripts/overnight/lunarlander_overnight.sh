#!/bin/bash
# LunarLander 환경 Overnight 실험 스크립트
# 실행: chmod +x scripts/overnight/lunarlander_overnight.sh && bash scripts/overnight/lunarlander_overnight.sh

echo "=========================================="
echo "LunarLander Overnight 실험 시작"
echo "시작 시간: $(date)"
echo "=========================================="

# PYTHONPATH 설정
export PYTHONPATH=$(pwd)

# GPU 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    echo "GPU detected. Using GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPU detected. Using CPU."
fi

# wandb 로그인 확인
echo "wandb 로그인 상태 확인 중..."
wandb status >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "wandb 로그인이 필요합니다. 다음 명령어로 로그인하세요:"
    echo "wandb login"
    exit 1
fi
echo "wandb 로그인 확인 완료!"

# 알고리즘 목록 (LunarLander - 연속 행동공간)
algorithms=("ppo" "trpo" "a2c" "reinforce" "ddpg" "td3" "sac" "sql")

for algo in "${algorithms[@]}"; do
    echo
    echo "=========================================="
    echo "LunarLander ${algo^^} 실험 시작"
    echo "=========================================="
    
    python -m main multi --config config/lunarlander/${algo}.yaml --num_workers 1
    
    if [ $? -eq 0 ]; then
        echo "${algo^^} 실험 완료 성공!"
    else
        echo "${algo^^} 실험 실패!"
    fi
    
    # 다음 실험 전 10초 대기
    echo "10초 후 다음 실험 시작..."
    sleep 10
done

echo
echo "=========================================="
echo "LunarLander 모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
for algo in "${algorithms[@]}"; do
    result_file="results/lunarlander_${algo}/multi_seed_results.json"
    if [ -f "$result_file" ]; then
        echo "✅ ${algo^^}: $result_file"
    else
        echo "❌ ${algo^^}: 결과 파일 없음"
    fi
done
echo
echo "wandb에서 실시간으로 결과를 확인할 수 있습니다!"