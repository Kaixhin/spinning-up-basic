import os
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import torch
from hyperparams import MAX_STEPS
from models import params_to_vec, vec_to_params


steps, rewards = [], []


def plot(step, reward, title):
  steps.append(step)
  rewards.append(reward)
  plt.plot(steps, rewards, 'b-')
  plt.title(title)
  plt.xlabel('Steps')
  plt.ylabel('Rewards')
  plt.xlim((0, MAX_STEPS))
  plt.ylim((-2000, 0))
  plt.savefig(os.path.join('results', title + '.png'))


# Sets MPI environment variables, initialises MPI and copies agent parameters to all processes
def setup_mpi(agent):
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  comm = MPI.COMM_WORLD
  param_vec = params_to_vec(agent, mode='params')
  comm.Bcast(param_vec, root=0)
  vec_to_params(param_vec, agent, mode='params')
  return comm
