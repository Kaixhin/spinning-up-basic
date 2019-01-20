from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from env import Env
from models import Critic, SoftActor, create_target_network, update_target_network
from utils import plot


max_steps, update_start, update_interval, batch_size, discount, alpha, polyak_rate = 100000, 5000, 4, 128, 0.99, 0.2, 0.995
env = Env()
actor = SoftActor()
critic_1 = Critic(state_action=True)
critic_2 = Critic(state_action=True)
value_critic = Critic()
target_value_critic = create_target_network(value_critic)
actor_optimiser = optim.Adam(actor.parameters())
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()))
value_critic_optimiser = optim.Adam(value_critic.parameters())
D = deque(maxlen=10000)


state, done, total_reward = env.reset(), False, 0
pbar = tqdm(range(1, max_steps + 1), unit_scale=1, smoothing=0)
for step in pbar:
  with torch.no_grad():
    if step < update_start:
      # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
      action = torch.tensor([[2 * random.random() - 1]])
    else:
      # Observe state s and select action a ~ μ(a|s)
      action = actor(state).sample()
    # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
    next_state, reward, done = env.step(action)
    total_reward += reward
    # Store (s, a, r, s', d) in replay buffer D
    D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32)})
    state = next_state
    # If s' is terminal, reset environment state
    if done:
      pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
      plot(step, total_reward, 'sac')
      state, total_reward = env.reset(), 0

  if step > update_start and step % update_interval == 0:
    # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
    batch = random.sample(D, batch_size)
    batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

    # Compute targets for Q and V functions
    y_q = batch['reward'] + discount * (1 - batch['done']) * target_value_critic(batch['next_state'])
    policy = actor(batch['state'])
    action = policy.rsample()  # a(s) is a sample from μ(·|s) which is differentiable wrt θ via the reparameterisation trick
    weighted_sample_entropy = (alpha * policy.log_prob(action)).sum(dim=1)
    y_v = torch.min(critic_1(batch['state'], action.detach()), critic_2(batch['state'], action.detach())) - weighted_sample_entropy.detach()

    # Update Q-functions by one step of gradient descent
    value_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
    critics_optimiser.zero_grad()
    value_loss.backward()
    critics_optimiser.step()

    # Update V-function by one step of gradient descent
    value_loss = (value_critic(batch['state']) - y_v).pow(2).mean()
    value_critic_optimiser.zero_grad()
    value_loss.backward()
    value_critic_optimiser.step()

    # Update policy by one step of gradient ascent
    policy_loss = -(critic_1(batch['state'], action) + weighted_sample_entropy).mean()
    actor_optimiser.zero_grad()
    policy_loss.backward()
    actor_optimiser.step()

    # Update target value network
    update_target_network(value_critic, target_value_critic, polyak_rate)
