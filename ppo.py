import torch
from torch import optim
from tqdm import tqdm
from env import Env
from hyperparams import DISCOUNT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, PPO_CLIP_RATIO, PPO_EPOCHS, TRACE_DECAY
from hyperparams import ON_POLICY_BATCH_SIZE as BATCH_SIZE
from models import ActorCritic
from utils import plot


env = Env()
agent = ActorCritic(HIDDEN_SIZE)
actor_optimiser = optim.Adam(list(agent.actor.parameters()) + [agent.policy_log_std], lr=LEARNING_RATE)
critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)


step, pbar = 0, tqdm(total=MAX_STEPS, smoothing=0)
while step < MAX_STEPS:
  # Collect set of trajectories D by running policy π in the environment
  D = [[]] * BATCH_SIZE
  for idx in range(BATCH_SIZE):
    state, done, total_reward = env.reset(), False, 0
    while not done:
      policy, value = agent(state)
      action = policy.sample()
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
  for idx in range(BATCH_SIZE):
    reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([[0.]]), torch.tensor([0.])
    for transition in reversed(D[idx]):
      reward_to_go = transition['reward'] + DISCOUNT * reward_to_go
      transition['reward_to_go'] = reward_to_go
      td_error = transition['reward'] + DISCOUNT * next_value - transition['value'].detach()
      advantage = td_error + DISCOUNT * TRACE_DECAY * advantage
      transition['advantage'] = advantage
      next_value = transition['value'].detach()
    # Extra step: turn trajectories into a single batch for efficiency (valid for feedforward networks)
    D[idx] = {k: torch.cat([transition[k] for transition in D[idx]], dim=0) for k in D[idx][0].keys()}
  trajectories = {k: torch.cat([trajectory[k] for trajectory in D], dim=0) for k in D[0].keys()}

  for epoch in range(PPO_EPOCHS):
    # Recalculate outputs for subsequent iterations
    if epoch > 0:
      policy, trajectories['value'] = agent(trajectories['state'])
      trajectories['log_prob_action'] = policy.log_prob(trajectories['action'].detach())

    # Update the policy by maximising the PPO-Clip objective
    policy_ratio = (trajectories['log_prob_action'] - trajectories['old_log_prob_action']).exp()
    policy_loss = -torch.min((policy_ratio * trajectories['advantage']).sum(dim=1), (torch.clamp(policy_ratio, min=1 - PPO_CLIP_RATIO, max=1 + PPO_CLIP_RATIO) * trajectories['advantage']).sum(dim=1)).mean()
    actor_optimiser.zero_grad()
    policy_loss.backward()
    actor_optimiser.step()

    # Fit value function by regression on mean-squared error
    value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
    critic_optimiser.zero_grad()
    value_loss.backward()
    critic_optimiser.step()

pbar.close()
