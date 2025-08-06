import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class MLP(nn.Module):
    """다층 퍼셉트론 (Multi-Layer Perceptron)"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], activation='tanh'):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # 활성화 함수 선택
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 레이어 구성
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class MLPActorCritic(nn.Module):
    """MLP 기반 Actor-Critic 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], 
                 has_continuous_action_space=True, action_std_init=0.6):
        super(MLPActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        # 안전한 CUDA 체크
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except RuntimeError:
            self.device = torch.device('cpu')
        
        
        # Actor (정책) 네트워크
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                                        (self.action_dim,),
                                        action_std_init ** 2,
                                        dtype=torch.float32,
                                        device=self.device
                                    )
            
            self.actor = MLP(state_dim, action_dim, hidden_dims, activation='tanh')
        else:
            self.actor = MLP(state_dim, action_dim, hidden_dims, activation='tanh')
            # 마지막 레이어를 Softmax로 변경
            self.actor.network[-1] = nn.Sequential(
                self.actor.network[-1],
                nn.Softmax(dim=-1)
            )
        
        # Critic (가치 함수) 네트워크
        self.critic = MLP(state_dim, 1, hidden_dims, activation='tanh')
    
    def set_action_std(self, new_action_std):
        """연속 행동 공간에서 행동 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                                        (self.action_dim,),
                                        new_action_std ** 2,
                                        dtype=torch.float32,
                                    )
        else:
            print("WARNING: Calling set_action_std() on discrete action space policy")
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        
        """주어진 상태에서 행동을 선택합니다."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = Normal(action_mean, torch.sqrt(self.action_var))
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action = dist.sample()
        if self.has_continuous_action_space:
            action_logprob = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach(), action_logprob, state_val
    
    def evaluate(self, state, action, ignore_critic=False):
        """주어진 상태와 행동에 대한 평가를 수행합니다."""

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            dist = Normal(action_mean, torch.sqrt(self.action_var))

            # 단일 행동 환경을 위한 처리
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            # 🔧 중요: action shape flatten
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)  # [batch_size, 1] → [batch_size]

        # 로그 확률, 엔트로피, 가치 함수
        if self.has_continuous_action_space:
            action_logprobs = dist.log_prob(action).sum(dim=-1)
        else:
            action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()                       # shape: [batch_size]

        if ignore_critic:
            return action_logprobs, None, dist_entropy

        state_values = self.critic(state).squeeze(-1)       # shape: [batch_size]

        return action_logprobs, state_values, dist_entropy

    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다."""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            dist = Normal(action_mean, torch.sqrt(self.action_var))
            return dist.log_prob(action).sum(dim=-1)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            return dist.log_prob(action)
    
    def get_value(self, state):
        """주어진 상태의 가치를 계산합니다."""
        return self.critic(state)
