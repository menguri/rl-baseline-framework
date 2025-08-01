import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

from .base import BaseOffPolicyAlgorithm
from networks.ddpg_actor import DDPGActor
from networks.ddpg_critic import DDPGCritic
from utils.buffer import ReplayBuffer
from utils.ou_noise import OrnsteinUhlenbeckProcess

class DDPG(BaseOffPolicyAlgorithm):
    def __init__(self, state_dim, action_dim, device='cpu',
                 hidden1=400, hidden2=300, init_w=3e-3,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001,
                 buffer_size=1000000, batch_size=64,
                 ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Actor & Critic
        self.actor = DDPGActor(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.actor_target = DDPGActor(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden1, hidden2, init_w).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)

        # OU Noise
        self.ou_noise = OrnsteinUhlenbeckProcess(size=action_dim, theta=ou_theta, mu=ou_mu, sigma=ou_sigma)

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            action += self.ou_noise.sample()
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones.float()) * q_next

        # Critic update
        q_expected = self.critic(states, actions)
        critic_loss = self.mse_loss(q_expected, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Target network soft update
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) + s_param.data * self.tau)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def reset_noise(self):
        self.ou_noise.reset_states()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])

    def set_eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train_mode(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train() 