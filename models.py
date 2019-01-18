import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    self.critic = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

  def forward(self, state):
    policy = Normal(self.actor(state), torch.ones(state.size(0), 1))
    value = self.critic(state).squeeze(dim=1)
    return policy, value


class Actor(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

  def forward(self, state):
    policy = self.actor(state)
    return policy


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.critic = nn.Sequential(nn.Linear(3 + 1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

  def forward(self, state, action):
    value = self.critic(torch.cat([state, action], dim=1))
    return value
