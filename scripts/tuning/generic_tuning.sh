#!/bin/bash
# 범용 하이퍼파라미터 튜닝 스크립트
# 실행: chmod +x scripts/tuning/generic_tuning.sh && bash scripts/tuning/generic_tuning.sh [algorithm] [environment] [parameter_file]

# 인자 확인
if [ $# -lt 2 ]; then
    echo "사용법: $0 [algorithm] [environment] [parameter_file(optional)]"
    echo "예시: $0 ppo cartpole"
    echo "      $0 sac lunarlander custom_params.txt"
    echo
    echo "지원 알고리즘: ppo, trpo, ddpg, td3, sac, sql, a2c, reinforce"
    echo "지원 환경: cartpole, lunarlander, halfcheetah"
    exit 1
fi

ALGORITHM=$1
ENVIRONMENT=$2
PARAM_FILE=${3:-""}

echo "=========================================="
echo "${ALGORITHM^^} 하이퍼파라미터 튜닝 시작 (환경: $ENVIRONMENT)"
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

# 기본 config 파일 확인
BASE_CONFIG="config/${ENVIRONMENT}/${ALGORITHM}.yaml"
if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ 기본 config 파일이 존재하지 않습니다: $BASE_CONFIG"
    exit 1
fi

# 파라미터 파일이 제공된 경우
if [ -n "$PARAM_FILE" ] && [ -f "$PARAM_FILE" ]; then
    echo "사용자 정의 파라미터 파일 사용: $PARAM_FILE"
    
    TUNING_COUNT=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        # 빈 줄이나 주석 건너뛰기
        [[ -z "$line" || "$line" =~ ^#.* ]] && continue
        
        TUNING_COUNT=$((TUNING_COUNT + 1))
        
        # 실험 이름 생성
        EXP_NAME="${ENVIRONMENT}_${ALGORITHM}_tune_custom_${TUNING_COUNT}"
        
        echo "=========================================="
        echo "실험 ${TUNING_COUNT}: ${EXP_NAME}"
        echo "파라미터: $line"
        echo "=========================================="
        
        # 임시 config 파일 생성
        TEMP_CONFIG="config/temp_${EXP_NAME}.yaml"
        cp "$BASE_CONFIG" "$TEMP_CONFIG"
        
        # 실험 이름 설정
        sed -i "s/name: .*/name: \"${EXP_NAME}\"/" "$TEMP_CONFIG"
        sed -i "s/save_dir: .*/save_dir: \"results\/${EXP_NAME}\"/" "$TEMP_CONFIG"
        
        # 파라미터 적용 (format: param_name=value param_name2=value2)
        IFS=' ' read -ra PARAMS <<< "$line"
        for param in "${PARAMS[@]}"; do
            if [[ $param == *"="* ]]; then
                param_name=$(echo $param | cut -d'=' -f1)
                param_value=$(echo $param | cut -d'=' -f2)
                sed -i "s/${param_name}: .*/${param_name}: ${param_value}/" "$TEMP_CONFIG"
                echo "  - ${param_name}: ${param_value}"
            fi
        done
        
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
        echo "5초 후 다음 실험 시작..."
        sleep 5
        
    done < "$PARAM_FILE"
    
    echo
    echo "=========================================="
    echo "사용자 정의 하이퍼파라미터 튜닝 완료!"
    echo "종료 시간: $(date)"
    echo "총 ${TUNING_COUNT}개 실험 완료"
    echo "=========================================="
    
else
    # 기본 그리드 서치
    echo "기본 그리드 서치를 수행합니다."
    echo "더 정교한 튜닝을 위해 algorithm별 전용 스크립트를 사용하세요:"
    echo "  - scripts/tuning/ppo_tuning.sh"
    echo "  - scripts/tuning/sac_tuning.sh"
    echo "  - scripts/tuning/ddpg_tuning.sh"
    echo
    
    # 간단한 학습률 그리드 서치
    learning_rates=(1e-4 3e-4 1e-3)
    
    for lr in "${learning_rates[@]}"; do
        EXP_NAME="${ENVIRONMENT}_${ALGORITHM}_tune_lr${lr}"
        
        echo "=========================================="
        echo "실험: ${EXP_NAME}"
        echo "learning_rate=${lr}"
        echo "=========================================="
        
        # 임시 config 파일 생성
        TEMP_CONFIG="config/temp_${EXP_NAME}.yaml"
        cp "$BASE_CONFIG" "$TEMP_CONFIG"
        
        # 파라미터 수정
        if grep -q "lr_actor:" "$TEMP_CONFIG"; then
            sed -i "s/lr_actor: .*/lr_actor: ${lr}/" "$TEMP_CONFIG"
        fi
        if grep -q "lr:" "$TEMP_CONFIG"; then
            sed -i "s/lr: .*/lr: ${lr}/" "$TEMP_CONFIG"
        fi
        sed -i "s/name: .*/name: \"${EXP_NAME}\"/" "$TEMP_CONFIG"
        sed -i "s/save_dir: .*/save_dir: \"results\/${EXP_NAME}\"/" "$TEMP_CONFIG"
        
        # 실험 실행
        python -m main train --config "$TEMP_CONFIG" --seed 42
        
        # 결과 확인
        if [ $? -eq 0 ]; then
            echo "✅ 실험 완료 성공!"
        else
            echo "❌ 실험 실패!"
        fi
        
        # 임시 파일 정리
        rm "$TEMP_CONFIG"
        
        # 다음 실험 전 5초 대기
        echo "5초 후 다음 실험 시작..."
        sleep 5
    done
    
    echo
    echo "=========================================="
    echo "기본 하이퍼파라미터 튜닝 완료!"
    echo "종료 시간: $(date)"
    echo "=========================================="
fi

echo
echo "결과는 results/ 디렉토리와 wandb에서 확인 가능합니다."
echo "최적 하이퍼파라미터 분석을 위해 wandb dashboard를 확인하세요!"