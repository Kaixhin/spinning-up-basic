# spinning-up-basic

Basic versions of agents from [Spinning Up in Deep RL](https://spinningup.openai.com/) written in [PyTorch](https://pytorch.org/). Designed to run quickly on CPU on [`Pendulum-v0`](https://gym.openai.com/envs/Pendulum-v0/) from [OpenAI Gym](https://gym.openai.com/).

To see differences between algorithms, try running `diff -y <file1> <file2>`, e.g., `diff -y ddpg.py td3.py`.

For MPI versions of on-policy algorithms, see the [`mpi` branch](https://github.com/Kaixhin/spinning-up-basic/tree/mpi).

## Algorithms

- [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)/Advantage Actor-Critic (`vpg.py`)
- [Trust Region Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/trpo.html) (`trpo.py`)
- [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html) (`ppo.py`)
- [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) (`ddpg.py`)
- [Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html) (`td3.py`)
- [Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html) (`sac.py`)
- Deep Q-Network (`dqn.py`)

## Results

### Vanilla Policy Gradient/Advantage Actor-Critic

![VPG](results/vpg.png)

### Trust Region Policy Gradient

![TRPO](results/trpo.png)

### Proximal Policy Optimization

![PPO](results/ppo.png)

### Deep Deterministic Policy Gradient

![DDPG](results/ddpg.png)

### Twin Delayed DDPG

![TD3](results/td3.png)

### Soft Actor-Critic

![SAC](results/sac.png)

### Deep Q-Network

![DQN](results/dqn.png)

## Code Links

- [Spinning Up in Deep RL](https://github.com/openai/spinningup) (TensorFlow)
- [Fired Up in Deep RL](https://github.com/kashif/firedup) (PyTorch)
