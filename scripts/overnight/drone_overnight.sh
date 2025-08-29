#!/bin/bash
# DroneHover-v0 Overnight 실험 스크립트
# 실행 예: chmod +x scripts/overnight/drone_overnight.sh && \
#          bash scripts/overnight/drone_overnight.sh

echo "=========================================="
echo "DroneHover-v0 Overnight 실험 시작"
echo "시작 시간: $(date)"
echo "드론 hovering task - z=1.0 높이 호버링"
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

# 알고리즘 목록 (드론 hover, 연구에서 TD3가 가장 일관된 성능, SAC/PPO가 decent)
algorithms=("td3" "sac" "ppo" "ddpg" "trpo")

# gym-pybullet-drones 설치 확인
echo "Checking gym-pybullet-drones installation..."
python -c "import gym_pybullet_drones; print('gym-pybullet-drones available')" 2>/dev/null || {
    echo "❌ gym-pybullet-drones not installed. Installing..."
    pip install gym-pybullet-drones
}

# PyBullet 헤드리스 모드 설정 (GUI 없이 실행)
export DISPLAY=""

for algo in "${algorithms[@]}"; do
    echo
    echo "=========================================="
    echo "DroneHover-v0 ${algo^^} 실험 시작"
    echo "Action space: Box(4,) - Normalized RPMs [0,1]"
    echo "Episode length: 8000 steps (~27초 @ 240Hz)"
    echo "Task: Single drone hover at z=1.0m"
    echo "=========================================="
    
    # 내 wandb 세션을 강제 사용하여 실행
    WANDB_DIR=$WANDB_DIR WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
    python -m main multi \
        --config config/drone/${algo}.yaml \
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
echo "DroneHover-v0 모든 실험 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

# 결과 요약
echo
echo "결과 요약:"
for algo in "${algorithms[@]}"; do
    result_file="results/drone_hover_${algo}/multi_seed_results.json"
    if [ -f "$result_file" ]; then
        echo "✅ ${algo^^}: $result_file"
    else
        echo "❌ ${algo^^}: 결과 파일 없음"
    fi
done

echo
echo "wandb 대시보드에서 결과를 확인하세요."
echo "예상: TD3 > SAC/PPO > DDPG/TRPO 순으로 성능 예상"
echo "주의: 모든 알고리즘이 시드에 민감하므로 5개 시드 결과 확인 필요"
echo "참고: 에피소드당 8000 step, 총 24M steps (3000 episodes)"