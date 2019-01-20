from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from env import Env
from models import Actor, Critic, create_target_network, update_target_network
from utils import plot


max_steps, update_start, update_interval, batch_size, discount, policy_delay, polyak_rate = 100000, 10000, 4, 128, 0.99, 2, 0.995
env = Env()
actor = Actor()
critic_1 = Critic(state_action=True)
critic_2 = Critic(state_action=True)
target_actor = create_target_network(actor)
target_critic_1 = create_target_network(critic_1)
target_critic_2 = create_target_network(critic_2)
actor_optimiser = optim.Adam(actor.parameters(), lr=1e-3)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=1e-3)
D = deque(maxlen=10000)


state, done, total_reward = env.reset(), False, 0
pbar = tqdm(range(1, max_steps + 1), unit_scale=1, smoothing=0)
for step in pbar:
  with torch.no_grad():
    if step < update_start:
      # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
      action = torch.tensor([[2 * random.random() - 1]])
    else:
      # Observe state s and select action a = clip(μ(s) + ε, a_low, a_high)
      action = torch.clamp(actor(state) + 0.1 * torch.randn(1, 1), min=-1, max=1)
    # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
    next_state, reward, done = env.step(action)
    total_reward += reward
    # Store (s, a, r, s', d) in replay buffer D
    D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32)})
    state = next_state
    # If s' is terminal, reset environment state
    if done:
      pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
      plot(step, total_reward, 'td3')
      state, total_reward = env.reset(), 0

  if step > update_start and step % update_interval == 0:
    # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
    batch = random.sample(D, batch_size)
    batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

    # Compute target actions with clipped noise (target policy smoothing)
    target_action = torch.clamp(target_actor(batch['next_state']) + torch.clamp(0.2 * torch.randn(1, 1), min=-0.5, max=0.5), min=-1, max=1)
    # Compute targets (clipped double Q-learning)
    y = batch['reward'] + discount * (1 - batch['done']) * torch.min(target_critic_1(batch['next_state'], target_action), target_critic_2(batch['next_state'], target_action))

    # Update Q-functions by one step of gradient descent
    value_loss = (critic_1(batch['state'], batch['action']) - y).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y).pow(2).mean()
    critics_optimiser.zero_grad()
    value_loss.backward()
    critics_optimiser.step()

    if step % (policy_delay * update_interval) == 0:
      # Update policy by one step of gradient ascent
      policy_loss = -critic_1(batch['state'], actor(batch['state'])).mean()
      actor_optimiser.zero_grad()
      policy_loss.backward()
      actor_optimiser.step()

      # Update target networks
      update_target_network(critic_1, target_critic_1, polyak_rate)
      update_target_network(critic_2, target_critic_2, polyak_rate)
      update_target_network(actor, target_actor, polyak_rate)
