import torch
from torch import optim
from tqdm import tqdm
from env import Env
from models import ActorCritic
from utils import plot


max_steps, batch_size, discount, trace_decay = 100000, 16, 0.99, 0.97
env = Env()
agent = ActorCritic()
actor_optimiser = optim.Adam(agent.actor.parameters())
critic_optimiser = optim.Adam(agent.critic.parameters())


step, pbar = 0, tqdm(total=max_steps, smoothing=0)
while step < max_steps:
  # Collect set of trajectories D by running policy π in the environment
  D = [[]] * batch_size
  for idx in range(batch_size):
    state, done, total_reward = env.reset(), False, 0
    while not done:
      policy, value = agent(state)
      action = policy.rsample()
      log_prob_action = policy.log_prob(action)
      next_state, reward, done = env.step(action)
      step += 1
      pbar.update(1)
      total_reward += reward
      D[idx].append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'log_prob_action': log_prob_action, 'old_log_prob_action': log_prob_action.detach(), 'value': value})
      state = next_state
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'ppo')

  # Compute rewards-to-go R and advantage estimates based on the current value function V
  for idx in range(batch_size):
    reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([[0.]]), torch.tensor([0.])
    for transition in reversed(D[idx]):
      reward_to_go = transition['reward'] + discount * reward_to_go
      transition['reward_to_go'] = reward_to_go
      td_error = transition['reward'] + discount * (next_value - transition['value'].detach())
      advantage = advantage * discount * trace_decay + td_error
      transition['advantage'] = advantage
      next_value = transition['value'].detach()
    # Extra step: turn trajectories into a batch for efficiency (valid for feedforward networks)
    D[idx] = {k: torch.cat([transition[k] for transition in D[idx]], dim=0) for k in D[idx][0].keys()}

  for epoch in range(5):
    # Recalculate outputs for subsequent iterations
    if epoch > 0:
      for trajectory in D:
        policy, trajectory['value'] = agent(trajectory['state'])
        trajectory['log_prob_action'] = policy.log_prob(trajectory['action'].detach())

    # Update the policy by maximising the PPO-Clip objective
    policy_loss = 0
    for trajectory in D:
      policy_ratio = (trajectory['log_prob_action'] - trajectory['old_log_prob_action']).exp()
      policy_loss -= torch.min((policy_ratio * trajectory['advantage']).sum(dim=1), (torch.clamp(policy_ratio, min=1 - 0.2, max=1 + 0.2) * trajectory['advantage']).sum(dim=1)).mean()
    actor_optimiser.zero_grad()
    (policy_loss / batch_size).backward()
    actor_optimiser.step()

    # Fit value function by regression on mean-squared error
    value_loss = 0
    for trajectory in D:
      value_loss += (trajectory['value'] - trajectory['reward_to_go']).pow(2).mean()
    critic_optimiser.zero_grad()
    (value_loss / batch_size).backward()
    critic_optimiser.step()

pbar.close()
