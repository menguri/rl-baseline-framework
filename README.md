# Modular Reinforcement Learning Framework

모듈화된 강화학습 실험 프레임워크입니다. 다양한 환경과 알고리즘을 쉽게 추가하고 실험할 수 있습니다.

## 구조

```
rl_framework/
├── config/                     # 실험 설정 파일들
├── algorithms/                 # RL 알고리즘 구현
├── environments/              # 환경 래퍼 및 커스텀 환경
├── networks/                  # 신경망 아키텍처
├── policy/                    # 정책 네트워크
├── critic/                    # 가치 함수 네트워크
├── utils/                     # 유틸리티 함수들
├── train/                     # 학습 스크립트
├── test/                      # 평가 스크립트
├── plot/                      # 시각화 스크립트
├── results/                   # 실험 결과 저장
└── scripts/                   # 실행 스크립트
```

## 설치

```bash
pip install -r requirements.txt
```

## 실험 추적 (Experiment Tracking)

### Weights & Biases (wandb) 사용

1. **wandb 설치 및 로그인**
```bash
pip install wandb
wandb login
```

2. **설정 파일에서 wandb 활성화**
```yaml
# config/cartpole_ppo_wandb.yaml
logging:
  use_wandb: true
  wandb_project: "rl-framework-cartpole"
  wandb_entity: "your-username"  # 또는 팀명
```

3. **실험 실행**
```bash
python main.py train --config config/cartpole_ppo_wandb.yaml
```

### TensorBoard 사용
```bash
tensorboard --logdir results/cartpole_ppo/seed_0/tensorboard
```

## 사용법

### 1. 단일 실험 실행
```bash
python main.py train --config config/cartpole_ppo.yaml
```

### 2. 멀티 시드 실험 실행 (5개 시드 평균)
```bash
python main.py multi --config config/cartpole_ppo_wandb.yaml --seeds 0 1 2 3 4
```

### 3. 결과 시각화
```bash
python main.py plot --results_dir results/cartpole_ppo
```

### 4. 밤새 실험 실행 (PPO + TRPO)
```bash
# Linux/Mac
bash scripts/run_overnight_experiments.sh

# Windows
scripts/run_overnight_experiments.bat
```

## 밤새 실험 설정

### PPO 실험 (500 에피소드)
- **환경**: CartPole-v1
- **알고리즘**: PPO
- **시드**: 5개 (0, 1, 2, 3, 4)
- **에피소드**: 500개
- **예상 시간**: 약 2-3시간

### TRPO 실험 (500 에피소드)
- **환경**: CartPole-v1
- **알고리즘**: TRPO
- **시드**: 5개 (0, 1, 2, 3, 4)
- **에피소드**: 500개
- **예상 시간**: 약 3-4시간

## wandb에서 확인할 수 있는 메트릭

### 실시간 메트릭
- `episode/reward`: 각 에피소드 보상
- `episode/avg_reward_10`: 최근 10개 에피소드 평균 보상
- `episode/avg_reward_all`: 전체 평균 보상
- `episode/max_reward_so_far`: 지금까지 최고 보상
- `loss/policy`, `loss/value`, `loss/entropy`: 손실 함수들
- `evaluation/reward`: 평가 보상

### 멀티 시드 요약
- `summary/mean_reward`: 5개 시드 평균 최고 보상
- `summary/std_reward`: 5개 시드 표준편차
- `summary/success_rate`: 성공률

## 지원하는 알고리즘

- PPO (Proximal Policy Optimization)
- TRPO (Trust Region Policy Optimization)

## 지원하는 환경

- CartPole-v1
- MuJoCo 환경들 (HalfCheetah, Hopper, Walker2d, InvertedPendulum)

## 주요 기능

1. **모듈화**: 새로운 알고리즘이나 환경을 쉽게 추가
2. **설정 관리**: YAML 파일로 실험 설정 관리
3. **멀티 시드 실험**: 여러 시드로 실험하고 평균 성능 계산
4. **결과 관리**: 체계적인 로깅, 모델 저장, 시각화
5. **실험 추적**: wandb, TensorBoard 지원
6. **실시간 모니터링**: wandb에서 실시간으로 평균 성능 확인
7. **확장성**: 새로운 알고리즘과 환경을 쉽게 추가 가능

## 실험 추적 도구 비교

| 도구 | 장점 | 단점 | 추천도 |
|------|------|------|--------|
| **Weights & Biases** | 사용 쉬움, 강력한 시각화, 팀 협업 | 클라우드 기반, 유료 | ⭐⭐⭐⭐⭐ |
| **MLflow** | 오픈소스, 로컬 실행 | 설정 복잡, UI 단순 | ⭐⭐⭐⭐ |
| **TensorBoard** | 이미 포함됨, 로컬 실행 | 기능 제한적 | ⭐⭐⭐ |
| **Neptune.ai** | 깔끔한 UI, 강력한 기능 | 덜 알려짐 | ⭐⭐⭐⭐ |

## 빠른 시작

1. **wandb 로그인**
```bash
wandb login
```

2. **밤새 실험 실행**
```bash
# Windows
scripts/run_overnight_experiments.bat

# Linux/Mac
bash scripts/run_overnight_experiments.sh
```

3. **wandb에서 실시간 확인**
- 브라우저에서 wandb.ai 접속
- 프로젝트 "rl-framework-cartpole" 확인
- 실시간으로 PPO와 TRPO 성능 비교 가능 