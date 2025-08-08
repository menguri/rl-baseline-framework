#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_simulations.sh  ─  record videos from trained checkpoints
# usage: ./config/run_simulations.sh HalfCheetah-v4 trpo 0 1 2
#        ./config/run_simulations.sh LunarLanderContinuous-v2 ddpg 0
# ──────────────────────────────────────────────────────────────

# GPU 환경에서 랜더링 시, 
export MUJOCO_GL=egl

set -e  # 오류 발생 시 즉시 중단
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"                # 프로젝트 루트로 이동

ENV_NAME="$1"                 # HalfCheetah-v4 ...
ALGO="$2"                     # ppo | trpo | ddpg
shift 2                       # 나머지는 seed 목록
SEEDS=("$@")                  # ex) (0 1 2)

if [[ -z "$ENV_NAME" || -z "$ALGO" || ${#SEEDS[@]} -eq 0 ]]; then
  echo "Usage: $0 <env_name> <algo> <seed0> [seed1] ..."
  exit 1
fi

# 결과 저장 폴더 (없으면 생성)
VIDEO_ROOT="videos/${ENV_NAME}_${ALGO}"
mkdir -p "$VIDEO_ROOT"

echo "▶ Recording ${#SEEDS[@]} seed(s) | env=${ENV_NAME} | algo=${ALGO}"
echo "  videos ➜ $VIDEO_ROOT"

for SEED in "${SEEDS[@]}"; do
  MODEL_PATH="results/${ENV_NAME,,}_${ALGO}/seed_${SEED}/best_model.pth"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "⚠️  checkpoint not found: $MODEL_PATH (skip)"
    continue
  fi

  OUT_DIR="${VIDEO_ROOT}/seed_${SEED}"
  mkdir -p "$OUT_DIR"

  echo "  • seed=$SEED  ➜  $OUT_DIR"
  python -m eval.simulate \
      --env "$ENV_NAME" \
      --model-path "$MODEL_PATH" \
      --algorithm "$ALGO" \
      --episodes 1 \
      --record-dir "$OUT_DIR"
done

echo "✅ done."
echo "▶ Videos saved to: $VIDEO_ROOT"