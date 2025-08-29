import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import math
from torch.optim import Adam
from torch.distributions import Normal

from .base import BaseOffPolicyAlgorithm
from networks.sac_actor import SACGaussianActor
from networks.ddpg_critic import DDPGCritic
from utils.buffer import ReplayBuffer


class SAC(BaseOffPolicyAlgorithm):
    def __init__(self, state_dim, action_dim, device='cpu',
                 hidden_dims=[256, 256], lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=256,
                 alpha=0.2, automatic_entropy_tuning=True, target_entropy=None,
                 stable_update_size=10000):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.stable_update_size = stable_update_size
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor Network
        self.actor = SACGaussianActor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)

        # Twin Critics (Q functions)
        self.critic1 = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        self.critic2 = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=lr_critic)

        # Target Critics
        self.critic1_target = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        self.critic2_target = DDPGCritic(state_dim, action_dim, hidden_dims[0], hidden_dims[1]).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Entropy Temperature
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, noise=False, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self):
        if len(self.buffer) < self.stable_update_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.sample(next_states)
            critic1_next_target = self.critic1_target(next_states, next_state_actions)
            critic2_next_target = self.critic2_target(next_states, next_state_actions)
            min_critic_next_target = torch.min(critic1_next_target, critic2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1 - dones.float()) * self.gamma * min_critic_next_target

        # Twin Q-functions update
        critic1 = self.critic1(states, actions)
        critic2 = self.critic2(states, actions)
        critic1_loss = self.mse_loss(critic1, next_q_value)
        critic2_loss = self.mse_loss(critic2, next_q_value)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # Policy update
        action, log_action, _ = self.actor.sample(states)
        critic1_pi = self.critic1(states, action)
        critic2_pi = self.critic2(states, action)
        min_critic_pi = torch.min(critic1_pi, critic2_pi)

        policy_loss = ((self.alpha * log_action) - min_critic_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Entropy temperature update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_action + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        # Soft update of target networks
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        return {
            'actor_loss': policy_loss.item(),
            'critic_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha_tlogs.item()
        }

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1_optim': self.critic1_optim.state_dict(),
            'critic2_optim': self.critic2_optim.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optim': self.alpha_optim.state_dict() if self.automatic_entropy_tuning else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim'])
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha']
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
            self.alpha = self.log_alpha.exp()

    def set_eval_mode(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()

    def set_train_mode(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic1_target.train()
        self.critic2_target.train()