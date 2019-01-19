import copy
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

  def forward(self, state):
    policy = self.actor(state)
    return policy


class TanhNormal(Normal):
  def rsample(self):
    return torch.tanh(self.loc + self.scale * torch.randn_like(self.scale))

  def sample(self):
    return self.rsample().detach()

  def log_prob(self, value):
    return super().log_prob(torch.atan(value)) - torch.log(1 - value.pow(2) + 1e-6) 


class SoftActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))

  def forward(self, state):
    policy_mean, policy_log_std = self.actor(state).chunk(2, dim=1)
    policy = TanhNormal(policy_mean, policy_log_std.exp())
    return policy


class Critic(nn.Module):
  def __init__(self, state_action=False):
    super().__init__()
    self.state_action = state_action
    self.critic = nn.Sequential(nn.Linear(3 + (1 if state_action else 0), 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

  def forward(self, state, action=None):
    if self.state_action:
      value = self.critic(torch.cat([state, action], dim=1))
    else:
      value = self.critic(state)
    return value.squeeze(dim=1)


class ActorCritic(nn.Module):
  def __init__(self):
    super().__init__()
    self.actor = Actor()
    self.critic = Critic()
    self.policy_log_std = nn.Parameter(torch.tensor([[-0.5]]))

  def forward(self, state):
    policy = Normal(self.actor(state), self.policy_log_std.exp())
    value = self.critic(state)
    return policy, value


def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_rate):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_rate * target_param.data + (1 - polyak_rate) * param.data
