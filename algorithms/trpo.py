import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from .base import BaseOnPolicyAlgorithm
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from utils.buffer import RolloutBuffer


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    """켤레 기울기법 (Conjugate Gradient Method) 구현
    Ax = b 를 푸는 방법! A는 Fisher Information Matrix (FIM)
    """
    x = torch.zeros_like(b).to(b.device) # 초기값 x = 0, Ax=b의 해를 담을 변수. b와 같은 크기로 초기화
    r = b.clone().to(b.device) # 잔차 r = b - Ax, 초기에는 r = b.  현재 해 x가 얼마나 부정확한지 나타냄
    p = b.clone().to(b.device) # conjugate direction, 초기에는 p = r. 탐색 방향. r과 유사하지만 A에 대해 conjugate함
    rdotr = torch.dot(r, r) # r과 r의 내적, 즉 ||r||^2. 잔차의 크기를 나타냄

    for i in range(nsteps):
        _Avp = Avp(p) # A * p 계산, A는 Fisher Information Matrix (FIM). FIM과 탐색방향 p의 곱
        safe_check(_Avp, f"CG_FVP_Avp_iter{i}")
        alpha = rdotr / torch.dot(p, _Avp) # 스텝 사이즈 alpha 계산, alpha = (r^T r) / (p^T A p). 얼마나 움직일지 결정
        x += alpha * p # x 업데이트, x = x + alpha * p. 해를 업데이트
        r -= alpha * _Avp # 잔차 r 업데이트, r = r - alpha * A p. 잔차를 줄임
        new_rdotr = torch.dot(r, r) # 새로운 잔차의 크기 계산. 얼마나 수렴했는지 평가
        betta = new_rdotr / rdotr # 다음 conjugate direction 계산을 위한 계수. 다음 탐색 방향을 계산하기 위한 가중치
        p = r + betta * p # conjugate direction 업데이트, p = r + beta * p.  새로운 탐색 방향
        rdotr = new_rdotr # rdotr 업데이트
        safe_check(x, f"CG_x_iter{i}")
        safe_check(r, f"CG_r_iter{i}")

        if rdotr < residual_tol: # 수렴 조건 체크, 잔차가 충분히 작으면 종료
            break
    return x # 최종 x 반환 (Ax = b의 해)


def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """라인 서치 (Line Search) 구현
    정책 업데이트 후 성능이 얼마나 향상되었는지 확인하고, 너무 많이 바뀌지 않도록 조절
    """
    fval = f(True).data # 현재 정책의 가치(손실)
    
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)): # 최대 max_backtracks번, 스텝 크기를 절반씩 줄여가면서
        xnew = x + stepfrac * fullstep # 새로운 정책 파라미터 (현재 파라미터 + 스텝 크기 * 업데이트 방향)
        set_flat_params_to(model, xnew) # 모델에 새로운 파라미터 설정
        newfval = f(True).data # 새로운 정책의 가치(손실)
        actual_improve = fval - newfval # 실제 성능 향상
        expected_improve = expected_improve_rate * stepfrac # 예상 성능 향상
        ratio = actual_improve / expected_improve # 실제 향상 / 예상 향상 비율
        
        if ratio.item() > accept_ratio and actual_improve.item() > 0: # 비율이 accept_ratio보다 크고, 실제 향상이 있으면
            return True, xnew # 성공, 새로운 파라미터 반환
    return False, x # 실패, 원래 파라미터 반환


def set_flat_params_to(model, flat_params):
    """평면화된 파라미터를 모델에 설정합니다.
    모델의 파라미터를 1차원 벡터로 만들어서 설정
    """
    prev_ind = 0 # 이전 인덱스
    for param in model.parameters(): # 모델의 각 파라미터에 대해
        flat_size = int(np.prod(list(param.size()))) # 파라미터의 크기 (평면화된 크기)
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size())) # 평면화된 파라미터 값을 복사
        prev_ind += flat_size # 인덱스 업데이트


def get_flat_params_from(model):
    """모델의 파라미터를 평면화합니다.
    모델의 모든 파라미터를 하나의 긴 벡터로 합침
    """
    params = []
    for param in model.parameters(): # 모델의 각 파라미터에 대해
        params.append(param.data.view(-1)) # 파라미터를 평면화하여 리스트에 추가
    return torch.cat(params) # 리스트를 하나의 벡터로 합쳐서 반환


def safe_check(tensor, name, step=None):
    """텐서에 NaN 또는 Inf 값이 있는지 확인하고, 너무 큰 값이 있는지 확인합니다."""
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
    """Trust Region Policy Optimization (TRPO) 알고리즘
    TRPO는 정책 경사 방법의 일종. 신뢰 영역(Trust Region) 안에서 정책을 업데이트하여 안정적인 학습을 보장
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 max_kl=0.01, damping=0.1, has_continuous_action_space=True, 
                 action_std_init=0.6, hidden_dims=[64, 64], critic_iters=80, device='cpu'):
        """ 초기화
        Args:
            state_dim (int): 상태 공간의 차원
            action_dim (int): 행동 공간의 차원
            lr (float): 학습률
            gamma (float): 감가율
            gae_lambda (float): GAE(Generalized Advantage Estimation)에 사용되는 lambda 값
            max_kl (float): KL 발산 제한 값 (신뢰 영역 크기)
            damping (float): Fisher Information Matrix에 더해지는 damping 값 (정칙화)
            has_continuous_action_space (bool): 연속 행동 공간인지 여부
            action_std_init (float): 초기 행동 표준 편차 (연속 행동 공간인 경우)
            hidden_dims (list): hidden layer 차원
            critic_iters (int): Critic 네트워크 업데이트 반복 횟수
            device (str): 학습에 사용할 장치 ('cpu' 또는 'cuda')
        """
        
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma # 감가율
        self.gae_lambda = gae_lambda # GAE에 사용되는 lambda 값
        self.max_kl = max_kl # KL 발산 제한 값 (신뢰 영역 크기)
        self.damping = damping # Fisher Information Matrix에 더해지는 damping 값 (정칙화)
        self.has_continuous_action_space = has_continuous_action_space # 연속 행동 공간인지 여부
        self.critic_update_steps = critic_iters  # 가치 함수 업데이트 스텝 수
        self.device = torch.device(device) # 학습에 사용할 장치

        # Actor 네트워크 (정책)
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)

        # Critic 네트워크 (가치 함수)
        self.critic = CriticNetwork(        
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        
        # 옵티마이저 (only value function).  정책망은 TRPO로 업데이트, 가치망은 Adam으로 업데이트
        self.value_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(lr)
        )
        
        # 이전 정책 (TRPO에서 필요).  TRPO는 현재 정책과 이전 정책의 KL 발산을 제한하므로, 이전 정책을 저장해두어야 함
        # Old Actor 네트워크
        self.actor_old = ActorNetwork(
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict()) # 초기화: 현재 정책으로

        # Old Critic 네트워크
        self.critic_old = CriticNetwork(        
            state_dim, action_dim, hidden_dims, 
            has_continuous_action_space, action_std_init
        ).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict()) # 초기화: 현재 가치함수로
        
        # 버퍼.  trajectory를 저장하는 버퍼
        self.buffer = RolloutBuffer(device)
        
        # 손실 함수.  MSE Loss (가치 함수 업데이트에 사용)
        self.mse_loss = nn.MSELoss()
        
        # 연속 행동 공간에서의 행동 표준편차
        if has_continuous_action_space:
            self.action_std = action_std_init
    
    def set_action_std(self, new_action_std, device='cpu'):
        """연속 행동 공간에서 행동 표준편차를 설정합니다.
        Args:
            new_action_std (float): 새로운 행동 표준 편차
            device (str): 사용할 장치
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std # 행동 표준 편차 업데이트
            self.actor.set_action_std(new_action_std, self.device) # Actor 네트워크에 적용
            self.actor_old.set_action_std(new_action_std, self.device) # Old Actor 네트워크에 적용
        else:
            print("WARNING: Calling set_action_std() on discrete action space actor")
    
    def select_action(self, state):
        """주어진 상태에서 행동을 선택합니다.
        Args:
            state (np.ndarray): 현재 상태
        Returns:
            action (np.ndarray or int): 선택된 행동
        """
        with torch.no_grad(): # gradient 계산 X
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) # 상태를 텐서로 변환
            action, action_logprob = self.actor_old.act(state) # 행동 선택 (이전 정책 사용)
            state_val = self.critic.evaluate(state) # 상태 가치 평가

        # 버퍼에 저장.  학습에 사용할 데이터를 버퍼에 저장
        self.buffer.add(state, action, action_logprob, 0, state_val, False)
        
        if self.has_continuous_action_space: # 연속 행동 공간인 경우
            return action.detach().cpu().numpy().flatten() # 행동을 numpy 배열로 변환하여 반환
        else: # 이산 행동 공간인 경우
            return action.item() # 행동을 int로 반환
    
    def update(self, batch=None):
        """TRPO 업데이트를 수행합니다.
        Args:
            batch (dict): 업데이트에 사용할 배치 데이터 (None인 경우 버퍼에서 가져옴)
        Returns:
            dict: 손실 값 (정책 손실, 가치 손실)
        """
        if batch is None: # 배치 데이터가 None인 경우
            # 버퍼에서 배치 가져오기
            batch = self.buffer.get_batch() # 버퍼에서 배치 데이터를 가져옴
        
        states = batch['states'] # 상태
        actions = batch['actions'] # 행동
        old_logprobs = batch['logprobs'] # 이전 정책의 로그 확률
        rewards = batch['rewards'] # 보상
        state_values = batch['state_values'] # 상태 가치
        advantages = batch['advantages'] # Advantage
        returns = batch['returns'] # Return
        
        # 이전 정책을 현재 정책으로 복사.  TRPO는 현재 정책과 이전 정책의 KL 발산을 제한하므로, 업데이트 전에 이전 정책을 현재 정책으로 복사
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # TRPO 스텝 수행.  정책 업데이트
        loss = self.trpo_step(states, actions, old_logprobs, advantages)
        
        # 가치 함수 업데이트.  가치 함수는 MSE Loss를 사용하여 업데이트
        for _ in range(self.critic_update_steps): # critic_update_steps번 반복
            state_values = self.critic.evaluate(states).squeeze(-1)       # shape: [batch_size] # 상태 가치 평가
            value_loss = self.mse_loss(state_values, returns) # MSE Loss 계산
            
            self.value_optimizer.zero_grad() # gradient 초기화
            value_loss.backward() # gradient 계산
            
            # ✅ Gradient Clipping 적용.  gradient explosion 방지
            # clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.value_optimizer.step() # 가치 함수 업데이트
        
        # 버퍼 초기화.  다음 에피소드를 위해 버퍼를 초기화
        self.buffer.clear()
        
        return {
            'policy_loss': loss.item(), # 정책 손실
            'value_loss': value_loss.item() # 가치 손실
        }
    
    def trpo_step(self, states, actions, old_logprobs, advantages):
        """TRPO의 핵심 스텝.  정책 업데이트를 수행
        Args:
            states (torch.Tensor): 상태
            actions (torch.Tensor): 행동
            old_logprobs (torch.Tensor): 이전 정책의 로그 확률
            advantages (torch.Tensor): Advantage
        Returns:
            torch.Tensor: 손실 값
        """
        
        def get_loss():
            """손실 함수를 반환하는 함수.  정책 손실 계산
            Returns:
                torch.Tensor: 손실 값
            """
            logprobs, _ = self.actor.evaluate(states, actions) # 현재 정책의 로그 확률 계산
            return -(logprobs * advantages.detach()).mean() # 정책 손실 반환 (Surrogate Loss)
        
        def get_kl():
            """KL 발산을 계산하는 함수.  현재 정책과 이전 정책의 KL 발산 계산
            Returns:
                torch.Tensor: KL 발산 값
            """
            logprobs, _ = self.actor.evaluate(states, actions) # 현재 정책의 로그 확률 계산
            return (old_logprobs - logprobs).mean() # KL 발산 값 반환
        
        # 손실 계산
        loss = get_loss() # 정책 손실 계산
        safe_check(loss, "loss")
        grads = torch.autograd.grad(loss, self.actor.parameters()) # 정책 손실에 대한 gradient 계산
        flat_grads = torch.cat([g.view(-1) for g in grads]) # gradient를 1차원 벡터로 변환
        safe_check(flat_grads, "flat_grads")
        loss_grad = flat_grads.data # gradient 값을 저장

        def Fvp(v):
            """Fisher Information Matrix F와 벡터 v의 곱 Fv를 계산
            Args:
                v (torch.Tensor): 벡터
            Returns:
                torch.Tensor: Fv 값
            """
            kl = get_kl() # KL 발산 계산
            kl = kl.mean() # KL 발산 평균

            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True) # KL 발산에 대한 gradient 계산 (create_graph=True: 2차 미분 계산을 위해)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads]) # gradient를 1차원 벡터로 변환

            kl_v = (flat_grad_kl * Variable(v)).sum() # KL 발산 gradient와 벡터 v의 내적
            grads = torch.autograd.grad(kl_v, self.actor.parameters()) # (KL 발산 gradient와 벡터 v의 내적)에 대한 gradient 계산
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data # gradient를 1차원 벡터로 변환
            safe_check(flat_grad_grad_kl, "FVP_flat_grad_grad_kl")
            
            return flat_grad_grad_kl + v * self.damping # Fisher vector product 계산 (Fv + damping * v)
        
        # 켤레 기울기법으로 스텝 방향 계산.  Ax = b를 푸는 과정 (A: Fisher Information Matrix, x: 스텝 방향, b: -gradient)
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10) # 켤레 기울기법을 사용하여 스텝 방향 계산
        safe_check(stepdir, "stepdir")
        
        # 스텝 크기 계산.  KL 발산 제한을 만족하는 최대 스텝 크기 계산
        fvp_val = Fvp(stepdir) # Fisher vector product 계산
        safe_check(fvp_val, "FVP(stepdir)")
        shs = 0.5 * (stepdir * fvp_val).sum(0, keepdim=True) # 스텝 크기 계산에 사용되는 값
        safe_check(shs, "shs")
        
        # 양정치 행렬 가정에 어긋나는 경우 업데이트 스킵(방어코드).  Fisher Information Matrix가 양정치 행렬이 아닌 경우 업데이트를 스킵
        if (shs < 0).any():
            print('[TRPO] shs 음수 발생, 업데이트 스킵 및 파라미터 복구')
            return loss
        lm = torch.sqrt(shs / self.max_kl) # 람다 계산 (스텝 크기 계산에 사용)
        safe_check(lm, "lm")
        fullstep = stepdir / lm[0] # full step 계산 (KL 발산 제한을 만족하는 스텝)
        safe_check(fullstep, "fullstep")
        
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True) # -g * stepdir 계산 (라인 서치에 사용)
        
        # 라인 서치.  정책 업데이트 후 성능이 얼마나 향상되었는지 확인하고, 너무 많이 바뀌지 않도록 조절
        prev_params = get_flat_params_from(self.actor) # 현재 정책 파라미터 저장
        safe_check(prev_params, "prev_params")
        success, new_params = linesearch(
            self.actor, lambda _: get_loss(), prev_params, fullstep,
            neggdotstepdir / lm[0]
        ) # 라인 서치 수행
        safe_check(new_params, "new_params")
        # set_flat_params_to(self.actor, new_params)
        if success: # 라인 서치 성공
            set_flat_params_to(self.actor, new_params) # 새로운 파라미터 적용
        else: # 라인 서치 실패
            print("❌ [DEBUG] linesearch 실패 → 업데이트 스킵")
        
        return loss # 손실 값 반환
    
    def save(self, path):
        """모델을 저장합니다.
        Args:
            path (str): 저장 경로
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(), # Actor 네트워크 파라미터
            'actor_old_state_dict': self.actor_old.state_dict(), # Old Actor 네트워크 파라미터
            'critic_state_dict': self.critic.state_dict(), # Critic 네트워크 파라미터
            'critic_old_state_dict': self.critic_old.state_dict(), # Old Critic 네트워크 파라미터
            'value_optimizer_state_dict': self.value_optimizer.state_dict(), # Value Optimizer 파라미터
            'action_std': self.action_std if self.has_continuous_action_space else None # 행동 표준 편차
        }, path)
    
    def load(self, path):
        """모델을 로드합니다.
        Args:
            path (str): 로드 경로
        """
        checkpoint = torch.load(path, map_location=self.device) # 체크포인트 로드
        self.actor.load_state_dict(checkpoint['actor_state_dict']) # Actor 네트워크 파라미터 로드
        self.actor_old.load_state_dict(checkpoint['actor_old_state_dict']) # Old Actor 네트워크 파라미터 로드
        self.critic.load_state_dict(checkpoint['critic_state_dict']) # Critic 네트워크 파라미터 로드
        self.critic_old.load_state_dict(checkpoint['critic_old_state_dict']) # Old Critic 네트워크 파라미터 로드
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict']) # Value Optimizer 파라미터 로드
        
        if self.has_continuous_action_space and checkpoint['action_std'] is not None: # 연속 행동 공간인 경우
            self.action_std = checkpoint['action_std'] # 행동 표준 편차 로드
            self.set_action_std(self.action_std, self.device) # 행동 표준 편차 설정
    
    def set_eval_mode(self):
        """평가 모드로 설정합니다."""
        self.actor.eval() # Actor 네트워크 평가 모드 설정
        self.actor_old.eval() # Old Actor 네트워크 평가 모드 설정
        self.critic.eval() # Critic 네트워크 평가 모드 설정
        self.critic_old.eval() # Old Critic 네트워크 평가 모드 설정
    
    def set_train_mode(self):
        """학습 모드로 설정합니다."""
        self.actor.train() # Actor 네트워크 학습 모드 설정
        self.actor_old.train() # Old Actor 네트워크 학습 모드 설정
        self.critic.train() # Critic 네트워크 학습 모드 설정
        self.critic_old.train() # Old Critic 네트워크 학습 모드 설정
    
    def get_action_logprob(self, state, action):
        """주어진 상태와 행동에 대한 로그 확률을 계산합니다.
        Args:
            state (np.ndarray): 상태
            action (np.ndarray): 행동
        Returns:
            torch.Tensor: 로그 확률
        """
        with torch.no_grad(): # gradient 계산 X
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) # 상태를 텐서로 변환
            action = torch.FloatTensor(action).to(self.device) # 행동을 텐서로 변환
            return self.actor.get_action_logprob(state, action) # 로그 확률 계산
    
    def get_value(self, state):
        """주어진 상태의 가치를 계산합니다.
        Args:
            state (np.ndarray): 상태
        Returns:
            torch.Tensor: 가치
        """
        with torch.no_grad(): # gradient 계산 X
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) # 상태를 텐서로 변환
            return self.critic.evaluate(state) # 가치 계산