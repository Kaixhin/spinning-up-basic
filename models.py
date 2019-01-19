import copy
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


def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_rate):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_rate * target_param.data + (1 - polyak_rate) * param.data
