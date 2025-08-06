import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from torch.autograd import Variable

from .base import BaseOnPolicyAlgorithm
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from utils.buffer import RolloutBuffer


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    """켤레 기울기법 (Conjugate Gradient Method) 구현"""
    x = torch.zeros_like(b).to(b.device)
    r = b.clone().to(b.device)
    p = b.clone().to(b.device)
    rdotr = torch.dot(r, r)
    
    for i in range(nsteps):
        _Avp = Avp(p)
        safe_check(_Avp, f"CG_FVP_Avp_iter{i}")
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        safe_check(x, f"CG_x_iter{i}")
        safe_check(r, f"CG_r_iter{i}")
        
        if rdotr < residual_tol:
            break
    return x


def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """라인 서치 (Line Search) 구현"""
    fval = f(True).data
    
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def set_flat_params_to(model, flat_params):
    """평면화된 파라미터를 모델에 설정합니다."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_params_from(model):
    """모델의 파라미터를 평면화합니다."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def safe_check(tensor, name, step=None):
    if not torch.is_tensor(tensor):
        return
    if torch.isnan(tensor).any():
        print(f"[TRPO][{name}] NaN detected! Step: {step}")
    if torch.isinf(tensor).any():
        print(f"[TRPO][{name}] Inf detected! Step: {step}")
    max_abs = tensor.abs().max().item() if tensor.numel() > 0 else 0.0
    if max_abs > 1e6:
        print(f"[TRPO][{name}] Large value detected! max_abs={max_abs:.2e} Step: {step}")
    # print(f"[TRPO][{name}] tensor: {tensor}")


class TRPO(BaseOnPolicyAlgorithm):
    """Trust Region Policy Optimization (TRPO) 알고리즘"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 max_kl=0.01, damping=0.1, has_continuous_action_space=True, 
                 action_std_init=0.6, hidden_dims=[64, 64], device='cpu'):
        
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.has_continuous_action_space = has_continuous_action_space
        self.device = torch.device(device)

        # Actor 네트워크
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)

        # Critic 네트워크
        self.critic = CriticNetwork(        
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        
        # 옵티마이저 (only value function)
        self.value_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(lr)
        )
        
        # 이전 정책 (TRPO에서 필요)
        # Old Actor 네트워크
        self.actor_old = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Old Critic 네트워크
        self.critic_old = CriticNetwork(        
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict())
        
        # 버퍼
        self.buffer = RolloutBuffer(device)
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()
        
        # 연속 행동 공간에서의 행동 표준편차
        if has_continuous_action_space:
            self.action_std = action_std_init
    
    def set_action_std(self, new_action_std, device='cpu'):
        """연속 행동 공간에서 행동 표준편차를 설정합니다."""
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std, self.device)
            self.actor_old.set_action_std(new_action_std, self.device)
        else:
            print("WARNING: Calling set_action_std() on discrete action space actor")
    
    def select_action(self, state):
        """주어진 상태에서 행동을 선택합니다."""
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, action_logprob = self.actor_old.act(state)
            state_val = self.critic.evaluate(state)
        
        # 버퍼에 저장
        self.buffer.add(state, action, action_logprob, 0, state_val, False)
        
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()
    
    def update(self, batch=None):
        """TRPO 업데이트를 수행합니다."""
        if batch is None:
            # 버퍼에서 배치 가져오기
            batch = self.buffer.get_batch()
        
        states = batch['states']
        actions = batch['actions']
        old_logprobs = batch['logprobs']
        rewards = batch['rewards']
        state_values = batch['state_values']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # 이전 정책을 현재 정책으로 복사
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # TRPO 스텝 수행
        loss = self.trpo_step(states, actions, old_logprobs, advantages)
        
        # 가치 함수 업데이트
        for _ in range(80):
            state_values = self.critic.evaluate(states).squeeze(-1)       # shape: [batch_size]
            value_loss = self.mse_loss(state_values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # 버퍼 초기화
        self.buffer.clear()
        
        return {
            'policy_loss': loss.item(),
            'value_loss': value_loss.item()
        }
    
    def trpo_step(self, states, actions, old_logprobs, advantages):
        """TRPO의 핵심 스텝"""
        
        def get_loss():
            """손실 함수를 반환하는 함수"""
            logprobs, _ = self.actor.evaluate(states, actions)
            return -(logprobs * advantages.detach()).mean()
        
        def get_kl():
            """KL 발산을 계산하는 함수"""
            logprobs, _ = self.actor.evaluate(states, actions)
            return (old_logprobs - logprobs).mean()
        
        # 손실 계산
        loss = get_loss()
        safe_check(loss, "loss")
        grads = torch.autograd.grad(loss, self.actor.parameters())
        flat_grads = torch.cat([g.view(-1) for g in grads])
        safe_check(flat_grads, "flat_grads")
        loss_grad = flat_grads.data
        
        def Fvp(v):
            """Fisher Information Matrix F와 벡터 v의 곱 Fv를 계산"""
            kl = get_kl()
            kl = kl.mean()
            
            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            
            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
            safe_check(flat_grad_grad_kl, "FVP_flat_grad_grad_kl")
            
            return flat_grad_grad_kl + v * self.damping
        
        # 켤레 기울기법으로 스텝 방향 계산
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
        safe_check(stepdir, "stepdir")
        
        # 스텝 크기 계산
        fvp_val = Fvp(stepdir)
        safe_check(fvp_val, "FVP(stepdir)")
        shs = 0.5 * (stepdir * fvp_val).sum(0, keepdim=True)
        safe_check(shs, "shs")
        # 양정치 행렬 가정에 어긋나는 경우 업데이트 스킵(방어코드드)
        if (shs < 0).any():
            print('[TRPO] shs 음수 발생, 업데이트 스킵 및 파라미터 복구')
            return loss
        lm = torch.sqrt(shs / self.max_kl)
        safe_check(lm, "lm")
        fullstep = stepdir / lm[0]
        safe_check(fullstep, "fullstep")
        
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        
        # 라인 서치
        prev_params = get_flat_params_from(self.actor)
        safe_check(prev_params, "prev_params")
        success, new_params = linesearch(
            self.actor, lambda _: get_loss(), prev_params, fullstep,
            neggdotstepdir / lm[0]
        )
        safe_check(new_params, "new_params")
        # set_flat_params_to(self.actor, new_params)
        if success:
            set_flat_params_to(self.actor, new_params)
        else:
            print("❌ [DEBUG] linesearch 실패 → 업데이트 스킵")
        
        return loss
    
    def save(self, path):
        """모델을 저장합니다."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_old_state_dict': self.actor_old.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_old_state_dict': self.critic_old.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'action_std': self.action_std if self.has_continuous_action_space else None
        }, path)
    
    def load(self, path):
        """모델을 로드합니다."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_old.load_state_dict(checkpoint['actor_old_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_old.load_state_dict(checkpoint['critic_old_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if self.has_continuous_action_space and checkpoint['action_std'] is not None:
            self.action_std = checkpoint['action_std']
            self.set_action_std(self.action_std, self.device)
    
    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        self.actor.eval()
        self.actor_old.eval()
        self.critic.eval()
        self.critic_old.eval()
    
    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        self.actor.train()
        self.actor_old.train()
        self.critic.train()
        self.critic_old.train()
    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다."""
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.FloatTensor(action).to(self.device)
            return self.actor.get_action_logprob(state, action)
    
    def get_value(self, state):
        """주어진 상태의 가치를 계산합니다."""
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return self.critic.evaluate(state) 