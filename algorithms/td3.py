import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from .base import BaseOffPolicyAlgorithm
from networks.ddpg_actor import DDPGActor
from networks.ddpg_critic import DDPGCritic
from utils.buffer import ReplayBuffer

class TD3(BaseOffPolicyAlgorithm):
    def __init__(self, state_dim, action_dim, device='cpu',
                 hidden1=400, hidden2=300, init_w=3e-3,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=1000000, batch_size=64, policy_noise=0.2, 
                 noise_clip=0.5, exploration_std=0.1, policy_freq=2):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.stable_update_size = 10000
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_std = exploration_std

        # 1-Actor & 2-Critic
        self.actor = DDPGActor(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_first = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_first_optim = Adam(self.critic_first.parameters(), lr=lr_critic)
        self.critic_second = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_second_optim = Adam(self.critic_second.parameters(), lr=lr_critic)
        
        # 1-Target Actor & 2-Target Critic
        self.actor_target = DDPGActor(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_target_first = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_target_second = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self._init_target_networks()
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
        
        # Loss function
        self.mse_loss = nn.MSELoss()
        
        # Update count for policy frequency
        self.update_count = 0

    # Target network 초기화 로직을 별도 메서드로 분리
    def _init_target_networks(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_first.load_state_dict(self.critic_first.state_dict())
        self.critic_target_second.load_state_dict(self.critic_second.state_dict())

    def select_action(self, state, noise=True):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        if noise:
            # Add exploration noise 
            action += np.random.normal(0, self.exploration_std, size=self.action_dim) if noise else 0
        self.actor.train()
        return action

    def update(self):
        if len(self.buffer) < self.stable_update_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # Target policy smoothing for Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            target_actions = self.actor_target(next_states) + noise
            # target_actions = target_actions.clamp(-1, 1)
            q_first_next = self.critic_target_first(next_states, target_actions)
            q_second_next = self.critic_target_second(next_states, target_actions)
            q_next = torch.min(q_first_next, q_second_next)
            q_target = rewards + self.gamma * (1 - dones.float()) * q_next

        # Critic 1&2 update
        q_expected = self.critic_first(states, actions)
        critic_loss = self.mse_loss(q_expected, q_target)
        self.critic_first_optim.zero_grad()
        critic_loss.backward()
        self.critic_first_optim.step()
        q_expected_second = self.critic_second(states, actions)
        critic_loss_second = self.mse_loss(q_expected_second, q_target)     
        self.critic_second_optim.zero_grad()
        critic_loss_second.backward()
        self.critic_second_optim.step()
        
        # Actor & Target network update
        actor_loss = None  # 초기값 설정 필요
        if self.policy_freq == 0 or self.update_count % self.policy_freq == 0:
            actor_loss = -self.critic_first(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target_first, self.critic_first)
            self._soft_update(self.critic_target_second, self.critic_second)
        
        self.update_count += 1
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item() if actor_loss is not None else 0.0}

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic_first.state_dict(),
            'critic_second': self.critic_second.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target_first.state_dict(),
            'critic_target_second': self.critic_target_second.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_first_optim.state_dict(),
            'critic_second_optim': self.critic_second_optim.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        
        self.critic_first.load_state_dict(checkpoint['critic'])
        self.critic_target_first.load_state_dict(checkpoint['critic_target'])
        self.critic_first_optim.load_state_dict(checkpoint['critic_optim'])
        
        self.critic_second.load_state_dict(checkpoint['critic_second'])
        self.critic_target_second.load_state_dict(checkpoint['critic_target_second'])
        self.critic_second_optim.load_state_dict(checkpoint['critic_second_optim'])
        

    def set_eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_first.eval()
        self.critic_second.eval()
        self.critic_target_first.eval() 
        self.critic_target_second.eval() 

    def set_train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic_first.train()
        self.critic_second.train()
        self.critic_target_first.train() 
        self.critic_target_second.train() 