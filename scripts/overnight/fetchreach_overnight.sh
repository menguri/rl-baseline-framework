#!/bin/bash
# FetchReach-v4 Overnight 실험 스크립트
# 실행 예: chmod +x scripts/overnight/fetchreach_overnight.sh && \
#          bash scripts/overnight/fetchreach_overnight.sh

echo "=========================================="
echo "FetchReach-v4 Overnight 실험 시작"
echo "시작 시간: $(date)"
echo "로봇팔 manipulation task - End-effector 목표 위치 도달"
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

# 알고리즘 목록 (FetchReach-v4 로봇팔 manipulation, SAC/TD3가 robotics에 효과적)
algorithms=("sac" "td3" "ddpg" "ppo" "trpo")

# gymnasium-robotics 설치 확인
echo "Checking gymnasium-robotics installation..."
python -c "import gymnasium_robotics; print('gymnasium-robotics available')" 2>/dev/null || {
    echo "❌ gymnasium-robotics not installed. Installing..."
    pip install gymnasium-robotics
}

for algo in "${algorithms[@]}"; do
    echo
    echo "=========================================="
    echo "FetchReach-v4 ${algo^^} 실험 시작"
    echo "Action space: Box(4,) - [dx, dy, dz, gripper]"
    echo "Episode length: 50 steps"
    echo "=========================================="
    
    # 내 wandb 세션을 강제 사용하여 실행
    WANDB_DIR=$WANDB_DIR WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
    python -m main multi \
        --config config/fetchreach/${algo}.yaml \
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
echo "FetchReach-v4 모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
for algo in "${algorithms[@]}"; do
    result_file="results/fetchreach-v4_${algo}/multi_seed_results.json"
    if [ -f "$result_file" ]; then
        echo "✅ ${algo^^}: $result_file"
    else
        echo "❌ ${algo^^}: 결과 파일 없음"
    fi
done

echo
echo "wandb 대시보드에서 결과를 확인하세요."
echo "예상: SAC/TD3가 robotics manipulation에서 최고 성능"
echo "참고: 에피소드당 50 step, 총 100K steps (2000 episodes)"