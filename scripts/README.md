# Scripts 사용 가이드

이 디렉토리에는 RL 프레임워크의 다양한 실험을 실행할 수 있는 스크립트들이 정리되어 있습니다.

## 📁 디렉토리 구조

```
scripts/
├── overnight/          # 환경별 밤새 실험 스크립트
├── tuning/            # 하이퍼파라미터 튜닝 스크립트
├── single/            # 단일 실험 스크립트
└── README.md          # 이 파일
```

## 🌙 Overnight 실험 스크립트

각 환경에서 모든 알고리즘(PPO, TRPO, SAC, TD3, DDPG, SQL)을 자동으로 실행합니다.

### 사용법
```bash
# CartPole 환경 (이산 행동공간)
chmod +x scripts/overnight/cartpole_overnight.sh
bash scripts/overnight/cartpole_overnight.sh

# LunarLander 환경 (연속 행동공간)
chmod +x scripts/overnight/lunarlander_overnight.sh
bash scripts/overnight/lunarlander_overnight.sh

# HalfCheetah 환경 (연속 행동공간)
chmod +x scripts/overnight/halfcheetah_overnight.sh
bash scripts/overnight/halfcheetah_overnight.sh
```

### 특징
- ✅ 자동 GPU 감지 및 설정
- ✅ wandb 로그인 상태 확인
- ✅ 실험 간 자동 대기 (10초)
- ✅ 결과 요약 출력
- ✅ 실패 시 에러 로깅

## 🎯 하이퍼파라미터 튜닝 스크립트

특정 알고리즘의 하이퍼파라미터를 체계적으로 튜닝할 수 있습니다.

### 알고리즘별 전용 스크립트

#### PPO 튜닝
```bash
chmod +x scripts/tuning/ppo_tuning.sh
bash scripts/tuning/ppo_tuning.sh cartpole
bash scripts/tuning/ppo_tuning.sh lunarlander
bash scripts/tuning/ppo_tuning.sh halfcheetah
```

**튜닝 파라미터:**
- learning_rates: [1e-4, 3e-4, 1e-3]
- clip_ratios: [0.1, 0.2, 0.3]
- gae_lambdas: [0.9, 0.95, 0.98]
- **총 27개 조합**

#### SAC 튜닝
```bash
chmod +x scripts/tuning/sac_tuning.sh
bash scripts/tuning/sac_tuning.sh lunarlander
bash scripts/tuning/sac_tuning.sh halfcheetah
```

**튜닝 파라미터:**
- learning_rates: [1e-4, 3e-4, 1e-3]
- batch_sizes: [128, 256, 512]
- tau_values: [0.001, 0.005, 0.01]
- **총 27개 조합**

#### DDPG 튜닝
```bash
chmod +x scripts/tuning/ddpg_tuning.sh
bash scripts/tuning/ddpg_tuning.sh lunarlander
bash scripts/tuning/ddpg_tuning.sh halfcheetah
```

**튜닝 파라미터:**
- lr_actors: [1e-5, 1e-4, 3e-4]
- lr_critics: [1e-4, 1e-3, 3e-3]
- tau_values: [0.001, 0.005, 0.01]
- noise_sigmas: [0.1, 0.2, 0.3]
- **총 81개 조합**

### 범용 튜닝 스크립트

모든 알고리즘에 사용할 수 있는 유연한 튜닝 스크립트입니다.

#### 기본 사용법
```bash
chmod +x scripts/tuning/generic_tuning.sh

# 기본 학습률 그리드 서치
bash scripts/tuning/generic_tuning.sh ppo cartpole
bash scripts/tuning/generic_tuning.sh sac lunarlander
```

#### 사용자 정의 파라미터 파일 사용
```bash
# 예시 파라미터 파일 확인
cat scripts/tuning/example_params.txt

# 사용자 정의 파라미터로 튜닝
bash scripts/tuning/generic_tuning.sh ppo cartpole scripts/tuning/example_params.txt
```

**사용자 정의 파라미터 파일 형식:**
```txt
# 주석은 #으로 시작
lr_actor=1e-4 clip_ratio=0.1 gae_lambda=0.9
lr_actor=3e-4 clip_ratio=0.2 gae_lambda=0.95
```

## 🔧 단일 실험 스크립트

개별 알고리즘 테스트를 위한 스크립트들입니다.

```bash
# 기존 스크립트들 (하위 호환성)
bash scripts/single/run_cartpole_ppo.sh
bash scripts/single/run_cartpole_trpo.sh
bash scripts/single/ddpg_halfcheetah.sh
```

## 📊 결과 확인

### 로컬 결과
```bash
# 결과 디렉토리 확인
ls results/

# 특정 실험 결과 확인
cat results/experiment_name/multi_seed_results.json
```

### wandb Dashboard
1. 실험 실행 후 터미널에 출력되는 wandb URL 확인
2. 또는 https://wandb.ai 에서 프로젝트 확인:
   - `rl-framework-cartpole`
   - `rl-framework-lunarlander`  
   - `rl-framework-mujoco-halfcheetah`

## 💡 사용 팁

### 1. 실험 전 체크리스트
```bash
# wandb 로그인 확인
wandb status

# GPU 확인
nvidia-smi

# 환경 설정
export PYTHONPATH=$(pwd)
```

### 2. 병렬 실험 실행
```bash
# 서로 다른 GPU에서 동시 실행
CUDA_VISIBLE_DEVICES=0 bash scripts/overnight/cartpole_overnight.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/overnight/lunarlander_overnight.sh &
```

### 3. 실험 모니터링
```bash
# 실행 중인 실험 확인
ps aux | grep python

# GPU 사용량 모니터링
watch nvidia-smi

# 로그 실시간 확인
tail -f nohup.out
```

### 4. 백그라운드 실행
```bash
# nohup으로 백그라운드 실행
nohup bash scripts/overnight/cartpole_overnight.sh > cartpole_log.txt 2>&1 &

# screen으로 세션 관리
screen -S rl_experiment
bash scripts/overnight/cartpole_overnight.sh
# Ctrl+A, D로 detach
# screen -r rl_experiment로 재연결
```

## ❗ 주의사항

1. **실험 시간**: Overnight 스크립트는 수 시간이 소요될 수 있습니다.
2. **GPU 메모리**: 대용량 환경(HalfCheetah 등)에서는 충분한 GPU 메모리가 필요합니다.
3. **파라미터 튜닝**: 튜닝 스크립트는 매우 많은 실험을 수행하므로 시간과 자원을 고려하세요.
4. **wandb 할당량**: 무료 계정은 실험 수에 제한이 있을 수 있습니다.

## 🔄 스크립트 수정

스크립트를 수정하여 다음을 조정할 수 있습니다:
- 실험 횟수 (`--seeds` 파라미터)
- GPU 번호 (`CUDA_VISIBLE_DEVICES`)
- 병렬 워커 수 (`--num_workers`)
- 실험 간 대기 시간 (`sleep` 값)

---

더 자세한 내용은 [CLAUDE.md](../CLAUDE.md)를 참조하세요.