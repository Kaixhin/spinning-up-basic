import torch
from torch import optim
from env import Env
from models import ActorCritic


epochs, batch_size, discount, trace_decay = 100, 16, 0.99, 0.97
agent = ActorCritic()
actor_optimiser = optim.Adam(agent.actor.parameters())
critic_optimiser = optim.Adam(agent.critic.parameters())
env = Env()


for epoch in range(epochs):
  total_reward = 0
  # Collect set of trajectories D by running policy π in the environment
  D = [[]] * batch_size
  for idx in range(batch_size):
    state, done = env.reset(), False
    while not done:
      policy, value = agent(state)
      action = policy.rsample()
      log_prob_action = policy.log_prob(action)
      next_state, reward, done = env.step(action)
      total_reward += reward
      D[idx].append({'state': state, 'action': action, 'reward': reward, 'log_prob_action': log_prob_action, 'value': value})
      state = next_state

  # Compute rewards-to-go R and advantage estimates based on the current value function V
  for idx in range(batch_size):
    reward_to_go, advantage, next_value = 0, 0, 0
    for transition in reversed(D[idx]):
      reward_to_go = transition['reward'] + discount * reward_to_go
      transition['reward_to_go'] = reward_to_go
      td_error = transition['reward'] + discount * (next_value - transition['value'].detach())
      advantage = advantage * discount * trace_decay + td_error
      transition['advantage'] = advantage
      next_value = transition['value'].detach()

  # Estimate policy gradient and compute policy update
  policy_loss = 0
  for idx in range(batch_size):
    for transition in D[idx]:
      policy_loss -= transition['log_prob_action'] * transition['advantage']
  actor_optimiser.zero_grad()
  policy_loss.backward()
  actor_optimiser.step()

  # Fit value function by regression on mean-squared error
  value_loss = 0
  for idx in range(batch_size):
    for transition in D[idx]:
      value_loss += (transition['value'] - transition['reward_to_go']).pow(2)
  critic_optimiser.zero_grad()
  value_loss.backward()
  critic_optimiser.step()

  print(epoch, total_reward / batch_size)
