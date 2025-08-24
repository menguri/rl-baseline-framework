#!/bin/bash
# TD3 하이퍼파라미터 튜닝 스크립트
# 실행: chmod +x scripts/tuning/td3_tuning.sh && bash scripts/tuning/td3_tuning.sh [environment]

# 환경 인자 확인
if [ $# -eq 0 ]; then
    echo "사용법: $0 [environment]"
    echo "예시: $0 lunarlander"
    echo "      $0 halfcheetah"
    echo "주의: TD3는 연속 행동공간 환경에만 적합합니다."
    exit 1
fi

ENVIRONMENT=$1

# CartPole 환경 경고
if [ "$ENVIRONMENT" = "cartpole" ]; then
    echo "⚠️  경고: TD3는 연속 행동공간 알고리즘입니다."
    echo "   CartPole은 이산 행동공간이므로 적합하지 않습니다."
    echo "   LunarLander 또는 HalfCheetah 환경을 사용하세요."
    exit 1
fi

echo "=========================================="
echo "TD3 하이퍼파라미터 튜닝 시작 (환경: $ENVIRONMENT)"
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

# 기본 config 파일 확인
BASE_CONFIG="config/${ENVIRONMENT}/td3.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ 기본 config 파일이 존재하지 않습니다: $BASE_CONFIG"
    exit 1
fi

# 튜닝할 하이퍼파라미터 설정
lr_actors=(1e-4 3e-4 1e-3)
policy_noises=(0.1 0.2 0.3)
policy_freqs=(1 2 4)

TUNING_COUNT=0
TOTAL_EXPERIMENTS=$((${#lr_actors[@]} * ${#policy_noises[@]} * ${#policy_freqs[@]}))

echo "총 ${TOTAL_EXPERIMENTS}개의 하이퍼파라미터 조합을 테스트합니다."
echo

for lr in "${lr_actors[@]}"; do
    for noise in "${policy_noises[@]}"; do
        for freq in "${policy_freqs[@]}"; do
            TUNING_COUNT=$((TUNING_COUNT + 1))
            
            # 실험 이름 생성
            EXP_NAME="${ENVIRONMENT}_td3_tune_lr${lr}_noise${noise}_freq${freq}"
            
            echo "=========================================="
            echo "실험 ${TUNING_COUNT}/${TOTAL_EXPERIMENTS}: ${EXP_NAME}"
            echo "lr_actor=${lr}, policy_noise=${noise}, policy_freq=${freq}"
            echo "=========================================="
            
            # 임시 config 파일 생성
            TEMP_CONFIG="config/temp_${EXP_NAME}.yaml"
            cp "$BASE_CONFIG" "$TEMP_CONFIG"
            
            # 하이퍼파라미터 수정
            sed -i "s/lr_actor: .*/lr_actor: ${lr}/" "$TEMP_CONFIG"
            sed -i "s/policy_noise: .*/policy_noise: ${noise}/" "$TEMP_CONFIG"
            sed -i "s/policy_freq: .*/policy_freq: ${freq}/" "$TEMP_CONFIG"
            sed -i "s/name: .*/name: \"${EXP_NAME}\"/" "$TEMP_CONFIG"
            sed -i "s/save_dir: .*/save_dir: \"results\/${EXP_NAME}\"/" "$TEMP_CONFIG"
            
            # 실험 실행
            python -m main train --config "$TEMP_CONFIG" --seed 42
            
            # 결과 확인
            if [ $? -eq 0 ]; then
                echo "✅ 실험 ${TUNING_COUNT} 완료 성공!"
            else
                echo "❌ 실험 ${TUNING_COUNT} 실패!"
            fi
            
            # 임시 파일 정리
            rm "$TEMP_CONFIG"
            
            # 다음 실험 전 5초 대기
            if [ $TUNING_COUNT -lt $TOTAL_EXPERIMENTS ]; then
                echo "5초 후 다음 실험 시작..."
                sleep 5
            fi
        done
    done
done

echo
echo "=========================================="
echo "TD3 하이퍼파라미터 튜닝 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

echo
echo "결과 요약:"
echo "총 ${TOTAL_EXPERIMENTS}개 실험 완료"
echo "결과는 results/ 디렉토리와 wandb에서 확인 가능합니다."
echo
echo "최적 하이퍼파라미터 분석을 위해 wandb dashboard를 확인하세요!"