#!/bin/bash

# FetchPush-v3 환경 overnight 실험 스크립트
# 로봇팔 manipulation 환경 - 박스 밀기 태스크

echo "=========================================="
echo "FetchPush-v3 Overnight Experiments"
echo "=========================================="

# 실험 시작 시간 기록
start_time=$(date)
echo "실험 시작: $start_time"

# GPU 메모리 확인
echo ""
echo "GPU 상태 확인:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# 실험 디렉토리 생성
mkdir -p results/fetchpush_overnight
cd results/fetchpush_overnight

echo ""
echo "=== 1. SAC on FetchPush-v3 ==="
echo "시작 시간: $(date)"
python -m main multi --config ../../config/fetchpush/sac.yaml --seeds 0 1 2
echo "완료 시간: $(date)"
echo ""

echo "=== 2. PPO on FetchPush-v3 ==="
echo "시작 시간: $(date)"
python -m main multi --config ../../config/fetchpush/ppo.yaml --seeds 0 1 2
echo "완료 시간: $(date)"
echo ""

echo "=== 3. TD3 on FetchPush-v3 ==="
echo "시작 시간: $(date)"
python -m main multi --config ../../config/fetchpush/td3.yaml --seeds 0 1 2
echo "완료 시간: $(date)"
echo ""

echo "=== 4. DDPG on FetchPush-v3 ==="
echo "시작 시간: $(date)"
python -m main multi --config ../../config/fetchpush/ddpg.yaml --seeds 0 1 2
echo "완료 시간: $(date)"
echo ""

echo "=== 5. TRPO on FetchPush-v3 ==="
echo "시작 시간: $(date)"
python -m main multi --config ../../config/fetchpush/trpo.yaml --seeds 0 1 2
echo "완료 시간: $(date)"
echo ""

# 실험 종료
end_time=$(date)
echo "=========================================="
echo "모든 FetchPush-v3 실험 완료!"
echo "시작 시간: $start_time"
echo "종료 시간: $end_time"
echo "=========================================="

# 결과 요약 생성
echo ""
echo "실험 결과 디렉토리:"
ls -la results/

echo ""
echo "각 알고리즘별 최종 성능 요약:"
for algo in sac ppo td3 ddpg trpo; do
    result_dir="results/fetchpush-v3_$algo"
    if [ -d "$result_dir" ]; then
        echo "- $algo: $result_dir"
        if [ -f "$result_dir/multi_seed_results.json" ]; then
            echo "  멀티시드 결과 파일 존재"
        fi
    fi
done

echo ""
echo "🎉 FetchPush-v3 overnight 실험이 모두 완료되었습니다!"
echo "결과 분석을 위해 plot 명령어를 사용하세요:"
echo "python -m main plot --plot_type step_comparison --comparison_dirs results/fetchpush-v3_sac results/fetchpush-v3_ppo results/fetchpush-v3_td3 --labels SAC PPO TD3"