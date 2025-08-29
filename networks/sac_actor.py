from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SACGaussianActor(nn.Module):
    """
    Squashed (tanh) Gaussian policy for SAC.
    - 입력:  state (B, state_dim)
    - 출력:  (sampled_action, log_prob, mean_action)  # sample()에서
              * sampled_action, mean_action은 [-1, 1] 범위
              * log_prob는 (B, 1)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20.0,   # std 하한: exp(-20) ≈ 2e-9 (과도한 탐색/발산 방지)
        log_std_max: float =  2.0     # std 상한: exp(2) ≈ 7.39 (너무 큰 노이즈 방지)
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # MLP backbone: ReLU 2층 (기본)
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # 가우시안의 평균(μ)과 로그표준편차(log σ) 예측
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환:
          mean:    (B, action_dim)
          log_std: (B, action_dim)  # [log_std_min, log_std_max]로 클램프
        """
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 학습 안정화 장치 : log_std 클램프
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        평가(inference)용: GUI/실험시 빠르게 액션만 얻고 싶을 때.
        deterministic=True면 tanh(mean), 아니면 sample().
        """
        if deterministic:
            mean, _ = self.forward(state)
            return torch.tanh(mean)
        # stochasitc
        a, _, _ = self.sample(state)
        return a

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        학습용: reparameterization + tanh 스쿼시 + 로그확률 보정
        반환:
          action:  (B, action_dim)  in [-1, 1]
          log_prob:(B, 1)
          mean:    (B, action_dim)  (tanh(μ), 결정론적 액션)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()                           # σ = exp(log σ)
        normal = Normal(mean, std)

        # Reparameterization trick: x_t = μ + σ ⊙ ε,  ε ~ N(0, I)
        x_t = normal.rsample()

        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        action = y_t

        # log π(a|s): change-of-variables로 tanh의 Jacobian 보정
        # 기본식: log_prob = log N(x_t; μ, σ) - Σ log(1 - tanh(x_t)^2)
        # 수치안정: log(1 - tanh(x)^2) = 2 * (log(1/2) - x - softplus(-2x))
        # (SAC 공식 구현에서 쓰는 트릭)
        log_prob = normal.log_prob(x_t)
        # 마지막 차원(action_dim) 합
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Stable tanh correction
        # 참고: 2 * (log(2) - x - softplus(-2x)) = log(1 - tanh(x)^2)
        correction = 2 * (math.log(2) - x_t - F.softplus(-2 * x_t))
        correction = correction.sum(dim=-1, keepdim=True)

        log_prob = log_prob - correction

        # 결정론적 출력(평가용): tanh(μ)
        mean_tanh = torch.tanh(mean)
        return action, log_prob, mean_tanh