from functools import partial
import torch
from torch import autograd, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
from env import Env
from hyperparams import BACKTRACK_COEFF, BACKTRACK_ITERS, CONJUGATE_GRADIENT_ITERS, DAMPING_COEFF, DISCOUNT, HIDDEN_SIZE, KL_LIMIT, LEARNING_RATE, MAX_STEPS, TRACE_DECAY
from hyperparams import ON_POLICY_BATCH_SIZE as BATCH_SIZE
from models import ActorCritic
from utils import plot


env = Env()
agent = ActorCritic(HIDDEN_SIZE)
critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)


def hessian_vector_product(d_kl, x):
  g = parameters_to_vector(autograd.grad(d_kl, agent.actor.parameters(), create_graph=True))
  return parameters_to_vector(autograd.grad((g * x.detach()).sum(), agent.actor.parameters(), allow_unused=True, retain_graph=True)) + DAMPING_COEFF * x


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
      D[idx].append({'state': state, 'action': action, 'reward': torch.tensor([reward]), 'old_log_prob_action': log_prob_action.detach(), 'old_policy_mean': policy.loc.detach(), 'old_policy_std': policy.scale.detach(), 'value': value})
      state = next_state
    pbar.set_description('Step: %i | Reward: %f' % (step, total_reward))
    plot(step, total_reward, 'trpo')

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

  # Estimate policy gradient
  policy = agent(transition['state'])[0]
  policy_ratio = (policy.log_prob(trajectories['action']) - trajectories['old_log_prob_action']).exp()
  policy_loss = -(policy_ratio * trajectories['advantage']).sum(dim=1).mean()
  g = parameters_to_vector(autograd.grad(policy_loss, agent.actor.parameters(), retain_graph=True)).detach()

  # Use the conjugate gradient algorithm to compute x, where H is the Hessian of the sample average KL-divergence
  old_policy = Normal(transition['old_policy_mean'], transition['old_policy_std'])
  d_kl = kl_divergence(old_policy, policy)
  Hx = partial(hessian_vector_product, d_kl)
  x = conjugate_gradient(Hx, g)  # Solve Hx = g for (step direction) x = inv(H)g

  # Update the policy by backtracking line search with the smallest value that improves the sample loss and satisfies the sample KL-divergence constraint
  alpha = torch.sqrt(2 * KL_LIMIT / (torch.dot(x, Hx(x)) + 1e-8)).item()  # Step size
  old_policy_loss = policy_loss.item()
  old_parameters = parameters_to_vector(agent.actor.parameters()).detach()
  with torch.no_grad():
    for j in range(BACKTRACK_ITERS):
      line_search_step = BACKTRACK_COEFF ** j
      vector_to_parameters(old_parameters - line_search_step * alpha * x, agent.actor.parameters())  # TODO: Check logic - should be plus but then fails?
      policy = agent(transition['state'])[0]
      policy_ratio = (policy.log_prob(trajectories['action']) - trajectories['old_log_prob_action']).exp()
      policy_loss = -(policy_ratio * trajectories['advantage']).sum(dim=1).mean().item()
      d_kl = kl_divergence(old_policy, policy).item()
      if policy_loss <= old_policy_loss and d_kl <= KL_LIMIT:
        break
      elif j == BACKTRACK_ITERS - 1:
        vector_to_parameters(old_parameters, agent.actor.parameters())

  # Fit value function by regression on mean-squared error
  value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
  critic_optimiser.zero_grad()
  value_loss.backward()
  critic_optimiser.step()

pbar.close()
