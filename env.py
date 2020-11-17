import gym
import torch


class Env():
  def __init__(self):
    self._env = gym.make('Pendulum-v0')
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

  def reset(self):
    state = self._env.reset()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
  
  def step(self, action):
    state, reward, done, _ = self._env.step(action[0].detach().numpy())
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, done
