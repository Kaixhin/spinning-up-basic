import os
import matplotlib.pyplot as plt
from hyperparams import MAX_STEPS


steps, rewards = [], []


def plot(step, reward, title):
  steps.append(step)
  rewards.append(reward)
  plt.plot(steps, rewards, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.xlim((0, MAX_STEPS))
  plt.savefig(os.path.join('results', title + '.png'))
