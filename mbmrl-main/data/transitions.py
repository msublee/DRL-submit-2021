import gym
import gym_cartpole_swingup
import numpy as np
import ray
import torch
ray.init()

from attrdict import AttrDict


@ray.remote
def rollout(env, rollout_num, rollout_len):
    x,y = [], []
    env.params.pole.mass = np.random.uniform(0.01, 1.0)
    env.params.cart.mass = np.random.uniform(0.1, 3.0)

    for num in range(rollout_num):
        state = env.reset()
        a_0 = np.random.uniform(0.0, 2 * np.pi)
        u = np.random.uniform(0.0, 1.0)
        w = 0
        for t in range(rollout_len):
            w += np.random.normal(0.0, 1.0)
            action = np.sin(a_0 + u * w)
            x.append(np.concatenate((state, [action]), axis=-1))
            state, reward, done, _ = env.step(action)
            y.append(np.concatenate((state, [reward]), axis=-1))

            if done:
                state = env.reset()
                a_0 = np.random.uniform(0.0, 2 * np.pi)
                u = np.random.uniform(0.0, 1.0)
                w = 0
    return x, y


class RaySampler:
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
    ):
        self.device = device
        self.env = gym.make('CartPoleSwingUp-v0')

    def sample(
            self,
            batch_size: int = 100,
            num_ctx: int = None,
            num_tar: int = None,
            max_num_points: int = 300,
            min_num_points: int = 64,
    ):
        rollout_num = 3
        rollout_len = 100
        tasks = [rollout.remote(self.env, rollout_num, rollout_len) for _ in range(batch_size)]
        data = torch.tensor(ray.get(tasks), dtype=torch.float, device=self.device)

        batch = AttrDict()
        idx = torch.randperm(rollout_num * rollout_len)
        batch.x, batch.y = data[:, 0, idx], data[:, 1, idx]

        num_ctx = num_ctx or torch.randint(min_num_points, max_num_points - min_num_points, size=[1]).item()
        num_tar = num_tar or torch.randint(min_num_points, max_num_points - num_ctx, size=[1]).item()

        batch.xc = batch.x[:, :num_ctx]
        batch.xt = batch.x[:, num_ctx: num_ctx + num_tar]
        batch.yc = batch.y[:, :num_ctx]
        batch.yt = batch.y[:, num_ctx: num_ctx + num_tar]

        # print(batch.xc.shape)
        # print(batch.xt.shape)
        # print(batch.yc.shape)
        # print(batch.yt.shape)
        return batch


class Sampler:
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
    ):
        self.device = device
        self.env = gym.make('CartPoleSwingUp-v0')

    def sample(
            self,
            batch_size: int = 100,
            num_ctx: int = None,
            num_tar: int = None,
            max_num_points: int = 300,
            min_num_points: int = 64,
    ):
        rollout_num = 3
        rollout_len = 100
        tasks = [rollout(self.env, rollout_num, rollout_len) for _ in range(batch_size)]
        data = torch.tensor(tasks, dtype=torch.float, device=self.device)

        batch = AttrDict()
        idx = torch.randperm(rollout_num * rollout_len)
        batch.x, batch.y = data[:, 0, idx], data[:, 1, idx]

        num_ctx = num_ctx or torch.randint(min_num_points, max_num_points - min_num_points, size=[1]).item()
        num_tar = num_tar or torch.randint(min_num_points, max_num_points - num_ctx, size=[1]).item()

        batch.xc = batch.x[:, :num_ctx]
        batch.xt = batch.x[:, num_ctx: num_ctx + num_tar]
        batch.yc = batch.y[:, :num_ctx]
        batch.yt = batch.y[:, num_ctx: num_ctx + num_tar]

        # print(batch.xc.shape)
        # print(batch.xt.shape)
        # print(batch.yc.shape)
        # print(batch.yt.shape)
        return batch


if __name__ == '__main__':
    import time
    start = time.time()
    sampler = Sampler()
    sampler.sample(batch_size=16)
    print(time.time() - start)
