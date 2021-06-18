import gym
import gym_cartpole_swingup
import numpy as np
import os.path as osp
import torch
import yaml
seed = 2
np.random.seed(seed)

from attrdict import AttrDict
from utils.misc import load_module

a_0 = np.random.uniform(0.0, 2 * np.pi)
u = np.random.uniform(0.0, 1.0)

pole_mass = np.random.uniform(0.01, 1.0)
cart_mass = np.random.uniform(0.1, 3.0)

env = gym.make('CartPoleSwingUp-v2')
env.seed(seed)
env.params.pole.mass = pole_mass
env.params.cart.mass = cart_mass

neuboots = True
# neuboots = False
print('=' * 100)
render = False
rollout_num = 1
warmup = 10
rollout_len = 200 + warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = AttrDict()
batch.xc = torch.tensor([])
batch.yc = torch.tensor([])

trajectories = []

for num in range(rollout_num):
    state = env.reset()
    w = 0
    print(state)
    print(a_0)
    print(u)
    for t in range(rollout_len):
        if num == 0 and t < warmup:
            w += np.random.normal(0.0, 1.0)
            action = np.sin(a_0 + u * w)
            batch.xc = torch.cat(
                [batch.xc,
                 torch.tensor(np.concatenate((state, [action]), axis=-1)[None, None, :],
                              dtype=torch.float,
                              device=device)],
                dim=-2)
            state, reward, done, _ = env.step(action)
            batch.yc = torch.cat(
                [batch.yc,
                 torch.tensor(np.concatenate((state, [reward]), axis=-1)[None, None, :],
                              dtype=torch.float,
                              device=device)],
                dim=-2)

            if done:
                state = env.reset()
                w = 0
        else:
            if neuboots:
                w += np.random.normal(0.0, 1.0)
                action = np.sin(a_0 + u * w)
                batch.xc = torch.cat(
                    [batch.xc,
                     torch.tensor(np.concatenate((state, [action]), axis=-1)[None, None, :],
                                  dtype=torch.float,
                                  device=device)],
                    dim=-2)
                state, reward, done, _ = env.step(action, batch=batch, neuboots=True)
                trajectories.append(np.concatenate([state, [reward]], axis=-1))
                batch.yc = torch.cat(
                    [batch.yc,
                     torch.tensor(np.concatenate((state, [reward]), axis=-1)[None, None, :],
                                  dtype=torch.float,
                                  device=device)],
                    dim=-2)

                if done:
                    state = env.reset()
                    w = 0
            else:
                # if render:
                #     env.render()

                w += np.random.normal(0.0, 1.0)
                action = np.sin(a_0 + u * w)
                state, reward, done, _ = env.step(action)
                trajectories.append(np.concatenate([state, [reward]], axis=-1))

                if done:
                    state = env.reset()
                    w = 0

print('pole mass:', env.params.pole.mass)
print('cart mass:', env.params.cart.mass)

if neuboots:
    np.save(f'/Users/minsublee/Desktop/limlab/Courses/Deep RL/mbmrl/'
            f'results/rollout/neubanp_{seed}_warm{warmup}.npy',
            np.array(trajectories))
else:
    np.save(f'/Users/minsublee/Desktop/limlab/Courses/Deep RL/mbmrl/'
            f'results/rollout/env_{seed}_warm{warmup}.npy',
            np.array(trajectories))
env.close()
