# 🚀 Modular Reinforcement Learning Framework

PyTorch 기반의 모듈화된 강화학습 프레임워크입니다. 8개의 주요 강화학습 알고리즘을 CartPole, LunarLander, MuJoCo 환경에서 체계적으로 실험할 수 있습니다.

## 📋 구현된 알고리즘

### 🎯 On-Policy 알고리즘
- **PPO (Proximal Policy Optimization)** - Clipped objective, GAE 지원
- **TRPO (Trust Region Policy Optimization)** - Natural policy gradients, KL divergence constraints
- **A2C (Advantage Actor-Critic)** - Actor-Critic 아키텍처, entropy regularization
- **REINFORCE** - 순수 Monte Carlo policy gradient

### 🎯 Off-Policy 알고리즘
- **SAC (Soft Actor-Critic)** - Maximum entropy RL, automatic temperature tuning
- **DDPG (Deep Deterministic Policy Gradient)** - Deterministic policy gradient, target networks
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** - DDPG 개선, twin critics
- **SQL (Soft Q-Learning)** - Maximum entropy Q-learning, 연속/이산 행동공간 지원

## 🌍 지원 환경

| 환경 | 타입 | 상태 차원 | 행동 차원 | 설명 |
|------|------|-----------|-----------|------|
| **CartPole-v1** | 이산 | 4 | 2 | 카트-폴 균형 제어 |
| **LunarLanderContinuous-v3** | 연속 | 8 | 2 | 달 착륙선 제어 |
| **HalfCheetah-v4** | 연속 | 17 | 6 | MuJoCo 치타 로봇 |

## ⚡ 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd rl_framework

# 의존성 설치
pip install -r requirements.txt

# wandb 로그인 (선택적)
wandb login
```

### 2. 단일 실험 실행
```bash
# PPO로 CartPole 학습
python -m main train --config config/cartpole/ppo.yaml

# SAC로 LunarLander 학습
python -m main train --config config/lunarlander/sac.yaml

# TD3로 HalfCheetah 학습
python -m main train --config config/halfcheetah/td3.yaml
```

### 3. 멀티 시드 실험 실행
```bash
# 5개 시드로 안정적인 결과 획득
python -m main multi --config config/cartpole/ppo.yaml --seeds 0 1 2 3 4

# 병렬 실행 (3개 워커)
python -m main multi --config config/lunarlander/sac.yaml --seeds 0 1 2 3 4 --num_workers 3
```

## 🎬 스크립트 기반 실험

### 🌙 Overnight 실험 (환경별 모든 알고리즘 자동 실행)

#### 🎮 CartPole 환경 (이산 행동공간)
```bash
# 8개 알고리즘 순차 실행: PPO, TRPO, A2C, REINFORCE, SAC, TD3, DDPG, SQL
bash scripts/overnight/cartpole_overnight.sh
```
**환경 특징:**
- 🎯 **난이도**: 초급 (간단한 제어 문제)
- 🎮 **행동 공간**: 이산 (왼쪽/오른쪽)
- 📊 **목표 점수**: 475점 이상 (500점 만점)
- ⏱️ **에피소드 길이**: 최대 500스텝

**알고리즘별 성능 예측:**
- **🥇 최고 성능**: PPO, A2C (이산 행동 특화)
- **🥈 안정적 성능**: TRPO, REINFORCE
- **🥉 도전적**: SAC, SQL (이산 변환 버전)
- **⚠️ 주의**: DDPG, TD3 (연속→이산 변환, 성능 제한적)
- **⏰ 예상 시간**: 6-8시간 (빠른 수렴)

#### 🌙 LunarLander 환경 (연속 행동공간)
```bash
# 8개 알고리즘 순차 실행: PPO, TRPO, A2C, REINFORCE, DDPG, TD3, SAC, SQL
bash scripts/overnight/lunarlander_overnight.sh
```
**환경 특징:**
- 🚀 **난이도**: 중급 (2차원 연속 제어)
- 🎮 **행동 공간**: 연속 (메인 엔진, 좌우 엔진)
- 📊 **목표 점수**: 200점 이상 (안전 착륙)
- ⏱️ **에피소드 길이**: 최대 1000스텝

**알고리즘별 성능 예측:**
- **🥇 최고 성능**: SAC, TD3 (연속 제어 특화)
- **🥈 안정적 성능**: PPO, DDPG (정책/가치 기반)
- **🥉 일반적**: TRPO, A2C (범용 알고리즘)
- **🔬 실험적**: SQL (새로운 접근)
- **⚠️ 도전적**: REINFORCE (높은 분산)
- **⏰ 예상 시간**: 8-10시간 (중간 수렴 속도)

#### 🏃 HalfCheetah 환경 (고차원 연속 행동공간)  
```bash
# 8개 알고리즘 순차 실행: PPO, TRPO, A2C, REINFORCE, DDPG, TD3, SAC, SQL
bash scripts/overnight/halfcheetah_overnight.sh
```
**환경 특징:**
- 🎯 **난이도**: 고급 (6차원 고차원 연속 제어)
- 🎮 **행동 공간**: 연속 (6개 관절 토크 제어)
- 📊 **목표 점수**: 4000점 이상 (최대 속도 달성)
- ⏱️ **에피소드 길이**: 최대 1000스텝
- 🧠 **상태 차원**: 17차원 (위치, 속도, 각도 등)

**알고리즘별 성능 예측:**
- **🥇 최고 성능**: SAC, TD3 (고차원 연속 제어 특화)
- **🥈 강력한 성능**: PPO, TRPO (정책 기반 안정성)
- **🥉 일반적**: DDPG (기본적 연속 제어)
- **🔬 도전적**: A2C, SQL, REINFORCE (복잡한 환경 대응)
- **⏰ 예상 시간**: 12-15시간 (복잡한 물리 시뮬레이션)

**특징:**
- ✅ 자동 GPU 감지 및 설정
- ✅ wandb 로그인 상태 확인
- ✅ 알고리즘별 성공/실패 추적
- ✅ 10초 간격 자동 실행
- ✅ 결과 요약 출력

### 🔧 하이퍼파라미터 튜닝

#### 알고리즘별 전용 튜닝 스크립트

```bash
# PPO 하이퍼파라미터 튜닝 (27개 조합)
# 파라미터: learning_rate, clip_ratio, gae_lambda
bash scripts/tuning/ppo_tuning.sh cartpole

# SAC 하이퍼파라미터 튜닝 (27개 조합)
# 파라미터: learning_rate, batch_size, tau
bash scripts/tuning/sac_tuning.sh lunarlander

# DDPG 하이퍼파라미터 튜닝 (81개 조합)
# 파라미터: lr_actor, lr_critic, tau, ou_sigma
bash scripts/tuning/ddpg_tuning.sh halfcheetah

# TRPO 하이퍼파라미터 튜닝 (27개 조합)
# 파라미터: learning_rate, max_kl, damping
bash scripts/tuning/trpo_tuning.sh lunarlander

# TD3 하이퍼파라미터 튜닝 (27개 조합)
# 파라미터: lr_actor, policy_noise, policy_freq
bash scripts/tuning/td3_tuning.sh halfcheetah

# SQL 하이퍼파라미터 튜닝 (27개 조합)
# 파라미터: learning_rate, temperature, batch_size
bash scripts/tuning/sql_tuning.sh cartpole
```

#### 범용 튜닝 스크립트

```bash
# 기본 학습률 그리드 서치
bash scripts/tuning/generic_tuning.sh ppo cartpole

# 사용자 정의 파라미터 파일 사용
bash scripts/tuning/generic_tuning.sh ppo cartpole scripts/tuning/example_params.txt
```

**사용자 정의 파라미터 파일 예시:**
```txt
# scripts/tuning/custom_params.txt
lr_actor=1e-4 clip_ratio=0.1 gae_lambda=0.9
lr_actor=3e-4 clip_ratio=0.2 gae_lambda=0.95
lr_actor=1e-3 clip_ratio=0.3 gae_lambda=0.98
```

### 🎮 시뮬레이션 및 시각화

#### 학습된 모델 시뮬레이션
```bash
# 학습된 모델로 시뮬레이션 실행
bash scripts/run_simulations.sh
```

⚠️ **주의사항**: 시뮬레이션은 GUI가 필요합니다. **가상머신(VM)에서는 작동하지 않을 수 있으므로 로컬 환경에서 실행**하세요.

#### 결과 시각화
```bash
# 학습 곡선 플롯
python -m main plot --results_dir results/cartpole_ppo --plot_type learning_curves

# 스텝 기반 학습 곡선 (알고리즘 공정 비교)
python -m main plot --results_dir results/cartpole_ppo --plot_type step_learning_curves

# 여러 알고리즘 비교 (에피소드 기반)
python -m main plot --plot_type comparison \
  --comparison_dirs results/cartpole_ppo results/cartpole_sac \
  --labels PPO SAC

# 여러 알고리즘 비교 (스텝 기반 - 권장)
python -m main plot --plot_type step_comparison \
  --comparison_dirs results/cartpole_ppo results/cartpole_sac \
  --labels PPO SAC
```

## 📊 실험 모니터링 및 로깅

### wandb (Weights & Biases) 연동

#### 설정 방법
1. **계정 생성**: [wandb.ai](https://wandb.ai)에서 무료 계정 생성
2. **로그인**: 
   ```bash
   wandb login
   # API 키 입력 (wandb.ai/authorize에서 확인)
   ```
3. **상태 확인**:
   ```bash
   wandb status
   ```

#### wandb 활성화
config 파일에서 wandb 설정을 활성화하세요:

```yaml
# config/환경/알고리즘.yaml
logging:
  use_wandb: true
  wandb_project: "rl-framework-cartpole"  # 프로젝트 명
  wandb_entity: "your-username"           # 사용자명 또는 팀명
  enable_step_logging: true
  step_log_interval: 1000                 # 1000 스텝마다 로깅
```

#### wandb 대시보드 확인
- **실시간 모니터링**: 실험 실행 중 터미널에 표시되는 wandb URL 클릭
- **프로젝트별 접근**:
  - CartPole: `rl-framework-cartpole`
  - LunarLander: `rl-framework-lunarlander`
  - HalfCheetah: `rl-framework-mujoco-halfcheetah`

#### wandb에서 확인 가능한 메트릭
- **Episode Reward**: 에피소드별 보상
- **Step Reward**: 스텝별 보상 (알고리즘 공정 비교용)
- **Loss Values**: 정책 손실, 가치 손실, 엔트로피 손실
- **Algorithm Specific**: 알고리즘별 특수 메트릭 (KL divergence, alpha 값 등)
- **Evaluation Results**: 주기적 평가 결과

### 로컬 결과 확인

#### 결과 디렉토리 구조
```
results/
├── experiment_name/
│   ├── multi_seed_results.json    # 멀티 시드 요약
│   ├── seed_0/                    # 개별 시드 결과
│   │   ├── model.pth              # 학습된 모델
│   │   ├── metrics.json           # 상세 메트릭
│   │   └── config.yaml            # 사용된 설정
│   └── wandb/                     # wandb 로그
```

#### JSON 결과 확인
```bash
# 멀티 시드 요약 확인
cat results/cartpole_ppo/multi_seed_results.json | jq

# 특정 시드 메트릭 확인  
cat results/cartpole_ppo/seed_0/metrics.json | jq
```

## 🔧 고급 사용법

### 병렬 실험 실행
```bash
# 서로 다른 GPU에서 동시 실행
CUDA_VISIBLE_DEVICES=0 bash scripts/overnight/cartpole_overnight.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/overnight/lunarlander_overnight.sh &

# 백그라운드 실행
nohup bash scripts/overnight/halfcheetah_overnight.sh > halfcheetah_log.txt 2>&1 &
```

### 사용자 정의 설정
```bash
# 특정 시드로 실험
python -m main train --config config/cartpole/ppo.yaml --seed 1234

# GPU 지정
CUDA_VISIBLE_DEVICES=1 python -m main train --config config/lunarlander/sac.yaml

# CPU 강제 사용
python -m main train --config config/cartpole/ppo.yaml --device cpu
```

### Step 기반 로깅 (알고리즘 공정 비교)

**중요**: 모든 알고리즘에서 **동일한 `step_log_interval`**을 사용해야 공정한 비교가 가능합니다.

```yaml
# 모든 config 파일에서 동일하게 설정
logging:
  enable_step_logging: true
  step_log_interval: 1000  # 모든 알고리즘에서 동일해야 함
```

## 📁 프로젝트 구조

```
rl_framework/
├── algorithms/          # 8개 강화학습 알고리즘
│   ├── ppo.py          # Proximal Policy Optimization
│   ├── trpo.py         # Trust Region Policy Optimization  
│   ├── a2c.py          # Advantage Actor-Critic
│   ├── reinforce.py    # REINFORCE
│   ├── sac.py          # Soft Actor-Critic
│   ├── ddpg.py         # Deep Deterministic Policy Gradient
│   ├── td3.py          # Twin Delayed DDPG
│   ├── sql.py          # Soft Q-Learning
│   └── base.py         # 기본 클래스들
├── config/             # 환경별 설정 파일
│   ├── cartpole/       # CartPole 환경 설정
│   ├── lunarlander/    # LunarLander 환경 설정
│   └── halfcheetah/    # HalfCheetah 환경 설정
├── environments/       # 환경 래퍼들
├── networks/          # 신경망 아키텍처
├── scripts/           # 실험 스크립트들
│   ├── overnight/     # 환경별 모든 알고리즘 실행
│   ├── tuning/        # 하이퍼파라미터 튜닝
│   └── single/        # 단일 실험 스크립트
├── train/             # 학습 오케스트레이션
├── utils/             # 유틸리티 함수들
└── results/           # 실험 결과들
```

## 🎯 실험 추천 워크플로우

### 1. 빠른 테스트
```bash
# 단일 시드로 빠르게 테스트
python -m main train --config config/cartpole/ppo.yaml --seed 42
```

### 2. 안정적인 벤치마킹
```bash
# 멀티 시드로 안정적인 결과
python -m main multi --config config/cartpole/ppo.yaml --seeds 0 1 2 3 4
```

### 3. 알고리즘 비교
```bash
# 환경별 모든 알고리즘 비교
bash scripts/overnight/cartpole_overnight.sh
```

### 4. 하이퍼파라미터 최적화
```bash
# 최고 성능 알고리즘의 하이퍼파라미터 튜닝
bash scripts/tuning/ppo_tuning.sh cartpole
```

### 5. 결과 분석
```bash
# wandb dashboard 확인
# 로컬 시각화 생성
python -m main plot --plot_type step_comparison --comparison_dirs results/cartpole_ppo results/cartpole_sac --labels PPO SAC
```

## 🐛 문제 해결

### 자주 발생하는 문제들

#### GPU 메모리 부족
```bash
# 배치 크기 줄이기
# config 파일에서 batch_size 값을 낮추기 (예: 256 → 128)

# 또는 CPU 사용
python -m main train --config config/cartpole/ppo.yaml --device cpu
```

#### wandb 로그인 문제
```bash
# API 키 재설정
wandb login --relogin

# 오프라인 모드
export WANDB_MODE=offline
```

#### MuJoCo 환경 오류
```bash
# MuJoCo 라이센스 확인
# gymnasium[mujoco] 재설치
pip install gymnasium[mujoco] --upgrade
```

#### 시뮬레이션 GUI 문제
- **가상머신**: GUI가 지원되지 않을 수 있음 → **로컬 환경에서 실행**
- **헤드리스 서버**: X11 forwarding 또는 VNC 사용
- **Docker**: `--privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` 옵션 사용

### 로그 확인
```bash
# Python 실행 로그
tail -f nohup.out

# 시스템 리소스 모니터링
watch nvidia-smi    # GPU 사용량
htop               # CPU/Memory 사용량
```

## 📚 추가 문서

- **[CLAUDE.md](CLAUDE.md)**: 개발자를 위한 상세 가이드
- **[scripts/README.md](scripts/README.md)**: 스크립트 사용법 상세 안내

## 🤝 기여 및 문의

프로젝트에 대한 질문이나 기여는 GitHub Issues를 통해 남겨주세요.

---

**Happy Reinforcement Learning! 🎯**