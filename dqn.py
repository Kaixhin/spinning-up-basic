from collections import deque
import random
import torch
from torch import optim
from tqdm import tqdm
from env import Env
from hyperparams import ACTION_DISCRETISATION, OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, EPSILON, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, REPLAY_SIZE, TARGET_UPDATE_INTERVAL, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START
from models import DQN, create_target_network
from utils import plot


env = Env()
agent = DQN(HIDDEN_SIZE, ACTION_DISCRETISATION)
target_agent = create_target_network(agent)
optimiser = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
D = deque(maxlen=REPLAY_SIZE)


def convert_discrete_to_continuous_action(action):
  return action.to(dtype=torch.float32) - ACTION_DISCRETISATION // 2


def test(agent):
  with torch.no_grad():
    env = Env()
    state, done, total_reward = env.reset(), False, 0
    while not done:
      action = agent(state).argmax(dim=1, keepdim=True)  # Use purely exploitative policy at test time
      state, reward, done = env.step(convert_discrete_to_continuous_action(action))
      total_reward += reward
    return total_reward


state, done = env.reset(), False
pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
for step in pbar:
  with torch.no_grad():
    if step < UPDATE_START:
      # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
      action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]])
    else:
      # Observe state s and select action a with an ε-greedy policy
      action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]]) if random.random() < EPSILON else agent(state).argmax(dim=1, keepdim=True)
    # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
    next_state, reward, done = env.step(convert_discrete_to_continuous_action(action))
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

    # Compute targets
    y = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_agent(batch['next_state']).max(dim=1)[0]

    # Update Q-function by one step of gradient descent
    value_loss = (agent(batch['state']).gather(1, batch['action']).squeeze(dim=1) - y).pow(2).mean()
    optimiser.zero_grad()
    value_loss.backward()
    optimiser.step()

  if step > UPDATE_START and step % TARGET_UPDATE_INTERVAL == 0:
    # Update target network
    target_agent = create_target_network(agent)

  if step > UPDATE_START and step % TEST_INTERVAL == 0:
    agent.eval()
    total_reward = test(agent)
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'dqn')
    agent.train()
