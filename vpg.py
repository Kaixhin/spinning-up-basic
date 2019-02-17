import torch
from torch import optim
from tqdm import tqdm
from env import Env
from hyperparams import ON_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, TRACE_DECAY
from models import ActorCritic
from utils import plot


env = Env()
agent = ActorCritic(HIDDEN_SIZE)
actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)


state, done, total_reward, D = env.reset(), False, 0, []
pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
for step in pbar:
  # Collect set of trajectories D by running policy π in the environment
  policy, value = agent(state)
  action = policy.sample()
  log_prob_action = policy.log_prob(action)
  next_state, reward, done = env.step(action)
  total_reward += reward
  D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'done': torch.tensor([done], dtype=torch.float32), 'log_prob_action': log_prob_action, 'value': value})
  state = next_state
  if done:
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'vpg')
    state, total_reward = env.reset(), 0

    if len(D) >= BATCH_SIZE:
      # Compute rewards-to-go R and advantage estimates based on the current value function V
      with torch.no_grad():
        reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
        for transition in reversed(D):
          reward_to_go = transition['reward'] + (1 - transition['done']) * (DISCOUNT * reward_to_go)
          transition['reward_to_go'] = reward_to_go
          td_error = transition['reward'] + (1 - transition['done']) * DISCOUNT * next_value - transition['value']
          advantage = td_error + (1 - transition['done']) * DISCOUNT * TRACE_DECAY * advantage
          transition['advantage'] = advantage
          next_value = transition['value']
      # Turn trajectories into a single batch for efficiency (valid for feedforward networks)
      trajectories = {k: torch.cat([trajectory[k] for trajectory in D], dim=0) for k in D[0].keys()}
      # Extra step: normalise advantages
      trajectories['advantage'] = (trajectories['advantage'] - trajectories['advantage'].mean()) / (trajectories['advantage'].std() + 1e-8)
      D = []

      # Estimate policy gradient and compute policy update
      policy_loss = -(trajectories['log_prob_action'].sum(dim=1) * trajectories['advantage']).mean()
      actor_optimiser.zero_grad()
      policy_loss.backward()
      actor_optimiser.step()

      # Fit value function by regression on mean-squared error
      value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
      critic_optimiser.zero_grad()
      value_loss.backward()
      critic_optimiser.step()
