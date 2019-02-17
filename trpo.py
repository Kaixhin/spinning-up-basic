from functools import partial
import torch
from torch import autograd, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
from env import Env
from hyperparams import BACKTRACK_COEFF, BACKTRACK_ITERS, ON_POLICY_BATCH_SIZE as BATCH_SIZE, CONJUGATE_GRADIENT_ITERS, DAMPING_COEFF, DISCOUNT, HIDDEN_SIZE, KL_LIMIT, LEARNING_RATE, MAX_STEPS, TRACE_DECAY
from models import ActorCritic
from utils import plot


env = Env()
agent = ActorCritic(HIDDEN_SIZE)
critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)


def hessian_vector_product(d_kl, x):
  g = parameters_to_vector(autograd.grad(d_kl, agent.actor.parameters(), create_graph=True))
  return parameters_to_vector(autograd.grad((g * x.detach()).sum(), agent.actor.parameters(), retain_graph=True)) + DAMPING_COEFF * x


def conjugate_gradient(Ax, b):
  x = torch.zeros_like(b)
  r = b - Ax(x)  # Residual
  p = r  # Conjugate vector
  r_dot_old = torch.dot(r, r)
  for _ in range(CONJUGATE_GRADIENT_ITERS):  # Run for a limited number of steps
    Ap = Ax(p)
    alpha = r_dot_old / (torch.dot(p, Ap) + 1e-8)
    x += alpha * p
    r -= alpha * Ap
    r_dot_new = torch.dot(r, r)
    p = r + (r_dot_new / r_dot_old) * p
    r_dot_old = r_dot_new
  return x


state, done, total_reward, D = env.reset(), False, 0, []
pbar = tqdm(range(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
for step in pbar:
  # Collect set of trajectories D by running policy π in the environment
  policy, value = agent(state)
  action = policy.sample()
  log_prob_action = policy.log_prob(action)
  next_state, reward, done = env.step(action)
  total_reward += reward
  D.append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'done': torch.tensor([done], dtype=torch.float32), 'log_prob_action': log_prob_action, 'old_log_prob_action': log_prob_action.detach(), 'old_policy_mean': policy.loc.detach(), 'old_policy_std': policy.scale.detach(), 'value': value})
  state = next_state
  if done:
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'trpo')
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

      # Estimate policy gradient
      policy = agent(trajectories['state'])[0]
      policy_ratio = (policy.log_prob(trajectories['action']).sum(dim=1) - trajectories['old_log_prob_action'].sum(dim=1)).exp()
      policy_loss = -(policy_ratio * trajectories['advantage']).mean()
      g = parameters_to_vector(autograd.grad(policy_loss, agent.actor.parameters(), retain_graph=True))

      # Use the conjugate gradient algorithm to compute x, where H is the Hessian of the sample average KL-divergence
      old_policy = Normal(trajectories['old_policy_mean'], trajectories['old_policy_std'])
      d_kl = kl_divergence(old_policy, policy).sum(dim=1).mean()
      Hx = partial(hessian_vector_product, d_kl)
      x = conjugate_gradient(Hx, g)  # Solve Hx = g for (step direction) x = inv(H)g

      # Update the policy by backtracking line search with the smallest value that improves the sample loss and satisfies the sample KL-divergence constraint
      alpha = torch.sqrt(2 * KL_LIMIT / (torch.dot(x, Hx(x)) + 1e-8)).item()  # Step size
      old_policy_loss = policy_loss.item()
      old_parameters = parameters_to_vector(agent.actor.parameters()).detach()
      with torch.no_grad():
        for j in range(BACKTRACK_ITERS):
          line_search_step = BACKTRACK_COEFF ** j
          vector_to_parameters(old_parameters - line_search_step * alpha * x, agent.actor.parameters())  # Gradient descent to minimise policy loss
          policy = agent(trajectories['state'])[0]
          policy_ratio = (policy.log_prob(trajectories['action']).sum(dim=1) - trajectories['old_log_prob_action'].sum(dim=1)).exp()
          policy_loss = -(policy_ratio * trajectories['advantage']).mean().item()
          d_kl = kl_divergence(old_policy, policy).sum(dim=1).mean().item()
          if policy_loss <= old_policy_loss and d_kl <= KL_LIMIT:
            break
          elif j == BACKTRACK_ITERS - 1:
            vector_to_parameters(old_parameters, agent.actor.parameters())

      # Fit value function by regression on mean-squared error
      value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
      critic_optimiser.zero_grad()
      value_loss.backward()
      critic_optimiser.step()
