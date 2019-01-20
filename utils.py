import os
import matplotlib.pyplot as plt


steps, rewards = [], []


def plot(step, reward, title):
  steps.append(step)
  rewards.append(reward)
  plt.plot(steps, rewards, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.ylim((-2000, 0))
  plt.savefig(os.path.join('results', title + '.png'))
