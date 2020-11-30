import copy
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform


class Actor(nn.Module):
  def __init__(self, observation_size, action_size, hidden_size, stochastic=True, layer_norm=False, initial_policy_log_std_dev=-0.):
    super().__init__()
    layers = [nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.policy = nn.Sequential(*layers)
    if stochastic:
      self.policy_log_std = nn.Parameter(torch.tensor([[initial_policy_log_std_dev]]))

  def forward(self, state):
    policy = self.policy(state)
    return policy


class SoftActor(nn.Module):
  def __init__(self, observation_size, action_size, hidden_size, min_action, max_action):
    super().__init__()
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    self.action_loc, self.action_scale = torch.tensor(max_action + min_action, dtype=torch.float32) / 2, torch.tensor(max_action - min_action, dtype=torch.float32) / 2    # Affine transformation from (-1, 1) to action space bounds
    layers = [nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2 * action_size)]
    self.policy = nn.Sequential(*layers)

  def forward(self, state):
    policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    policy = TransformedDistribution(Independent(Normal(policy_mean, policy_log_std.exp()), 1), [TanhTransform(), AffineTransform(loc=self.action_loc, scale=self.action_scale)])
    policy.mean_ = self.action_scale * torch.tanh(policy.base_dist.mean) + self.action_loc  # TODO: See if mean attr can be overwritten
    return policy


class Critic(nn.Module):
  def __init__(self, observation_size, action_size, hidden_size, state_action=False, layer_norm=False):
    super().__init__()
    self.state_action = state_action
    layers = [nn.Linear(observation_size + (action_size if state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.value = nn.Sequential(*layers)

  def forward(self, state, action=None):
    if self.state_action:
      value = self.value(torch.cat([state, action], dim=1))
    else:
      value = self.value(state)
    return value.squeeze(dim=1)


class ActorCritic(nn.Module):
  def __init__(self, observation_size, action_size, hidden_size, initial_policy_log_std_dev=-0.):
    super().__init__()
    self.actor = Actor(observation_size, action_size, hidden_size, stochastic=True, initial_policy_log_std_dev=initial_policy_log_std_dev)
    self.critic = Critic(observation_size, action_size, hidden_size)

  def forward(self, state):
    policy = Independent(Normal(self.actor(state), self.actor.policy_log_std.exp()), 1)
    value = self.critic(state)
    return policy, value


class DQN(nn.Module):
  def __init__(self, observation_size, hidden_size, num_actions=5):
    super().__init__()
    layers = [nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, num_actions)]
    self.dqn = nn.Sequential(*layers)

  def forward(self, state):
    values = self.dqn(state)
    return values


def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_factor):
  for param, target_param in zip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data
