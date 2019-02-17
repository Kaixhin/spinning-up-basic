from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from hyperparams import ACTION_NOISE, OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLICY_DELAY, POLYAK_FACTOR, REPLAY_SIZE, TARGET_ACTION_NOISE, TARGET_ACTION_NOISE_CLIP, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START
from env import Env
from models import Actor, Critic, create_target_network, update_target_network
from utils import plot


env = Env()
actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True)
critic_1 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
critic_2 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
target_actor = create_target_network(actor)
target_critic_1 = create_target_network(critic_1)
target_critic_2 = create_target_network(critic_2)
actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
D = deque(maxlen=REPLAY_SIZE)


def test(actor):
  with torch.no_grad():
    env = Env()
    state, done, total_reward = env.reset(), False, 0
    while not done:
      action = torch.clamp(actor(state), min=-1, max=1)  # Use purely exploitative policy at test time
      state, reward, done = env.step(action)
      total_reward += reward
    return total_reward


state, done = env.reset(), False
pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
for step in pbar:
  with torch.no_grad():
    if step < UPDATE_START:
      # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
      action = torch.tensor([[2 * random.random() - 1]])
    else:
      # Observe state s and select action a = clip(μ(s) + ε, a_low, a_high)
      action = torch.clamp(actor(state) + ACTION_NOISE * torch.randn(1, 1), min=-1, max=1)
    # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
    next_state, reward, done = env.step(action)
    # Store (s, a, r, s', d) in replay buffer D
    D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'next_state': next_state, 'done': torch.tensor([done], dtype=torch.float32)})
    state = next_state
    # If s' is terminal, reset environment state
    if done:
      state = env.reset()

  if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
    # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
    batch = random.sample(D, BATCH_SIZE)
    batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}

    # Compute target actions with clipped noise (target policy smoothing)
    target_action = torch.clamp(target_actor(batch['next_state']) + torch.clamp(TARGET_ACTION_NOISE * torch.randn(1, 1), min=-TARGET_ACTION_NOISE_CLIP, max=TARGET_ACTION_NOISE_CLIP), min=-1, max=1)
    # Compute targets (clipped double Q-learning)
    y = batch['reward'] + DISCOUNT * (1 - batch['done']) * torch.min(target_critic_1(batch['next_state'], target_action), target_critic_2(batch['next_state'], target_action))

    # Update Q-functions by one step of gradient descent
    value_loss = (critic_1(batch['state'], batch['action']) - y).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y).pow(2).mean()
    critics_optimiser.zero_grad()
    value_loss.backward()
    critics_optimiser.step()

    if step % (POLICY_DELAY * UPDATE_INTERVAL) == 0:
      # Update policy by one step of gradient ascent
      policy_loss = -critic_1(batch['state'], actor(batch['state'])).mean()
      actor_optimiser.zero_grad()
      policy_loss.backward()
      actor_optimiser.step()

      # Update target networks
      update_target_network(critic_1, target_critic_1, POLYAK_FACTOR)
      update_target_network(critic_2, target_critic_2, POLYAK_FACTOR)
      update_target_network(actor, target_actor, POLYAK_FACTOR)

  if step > UPDATE_START and step % TEST_INTERVAL == 0:
    actor.eval()
    total_reward = test(actor)
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'td3')
    actor.train()
