#!/bin/bash
# SAC 하이퍼파라미터 튜닝 스크립트
# 실행: chmod +x scripts/tuning/sac_tuning.sh && bash scripts/tuning/sac_tuning.sh [environment]

# 환경 인자 확인
if [ $# -eq 0 ]; then
    echo "사용법: $0 [environment]"
    echo "예시: $0 cartpole"
    echo "      $0 lunarlander"
    echo "      $0 halfcheetah"
    exit 1
fi

ENVIRONMENT=$1
echo "=========================================="
echo "SAC 하이퍼파라미터 튜닝 시작 (환경: $ENVIRONMENT)"
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
BASE_CONFIG="config/${ENVIRONMENT}/sac.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ 기본 config 파일이 존재하지 않습니다: $BASE_CONFIG"
    exit 1
fi

# 튜닝할 하이퍼파라미터 설정
learning_rates=(1e-4 3e-4 1e-3)
batch_sizes=(128 256 512)
tau_values=(0.001 0.005 0.01)

TUNING_COUNT=0
TOTAL_EXPERIMENTS=$((${#learning_rates[@]} * ${#batch_sizes[@]} * ${#tau_values[@]}))

echo "총 ${TOTAL_EXPERIMENTS}개의 하이퍼파라미터 조합을 테스트합니다."
echo

for lr in "${learning_rates[@]}"; do
    for batch in "${batch_sizes[@]}"; do
        for tau in "${tau_values[@]}"; do
            TUNING_COUNT=$((TUNING_COUNT + 1))
            
            # 실험 이름 생성
            EXP_NAME="${ENVIRONMENT}_sac_tune_lr${lr}_batch${batch}_tau${tau}"
            
            echo "=========================================="
            echo "실험 ${TUNING_COUNT}/${TOTAL_EXPERIMENTS}: ${EXP_NAME}"
            echo "lr_actor=${lr}, batch_size=${batch}, tau=${tau}"
            echo "=========================================="
            
            # 임시 config 파일 생성
            TEMP_CONFIG="config/temp_${EXP_NAME}.yaml"
            cp "$BASE_CONFIG" "$TEMP_CONFIG"
            
            # 하이퍼파라미터 수정
            sed -i "s/lr_actor: .*/lr_actor: ${lr}/" "$TEMP_CONFIG"
            sed -i "s/lr_critic: .*/lr_critic: ${lr}/" "$TEMP_CONFIG"
            sed -i "s/batch_size: .*/batch_size: ${batch}/" "$TEMP_CONFIG"
            sed -i "s/tau: .*/tau: ${tau}/" "$TEMP_CONFIG"
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
echo "SAC 하이퍼파라미터 튜닝 완료!"
echo "종료 시간: $(date)"
echo "=========================================="

echo
echo "결과 요약:"
echo "총 ${TOTAL_EXPERIMENTS}개 실험 완료"
echo "결과는 results/ 디렉토리와 wandb에서 확인 가능합니다."
echo
echo "최적 하이퍼파라미터 분석을 위해 wandb dashboard를 확인하세요!"