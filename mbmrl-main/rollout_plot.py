import numpy as np
import matplotlib.pyplot as plt


color = ['blue', 'green', 'red']
title = ['Cart position', 'Cart velocity', 'Pole angle (cos)', 'Pole angle (sin)', 'Angle velocity', 'reward']
for i in range(6):
    for seed in range(3):
        warmup = 10
        neubanp_traj = np.load(f'/Users/minsublee/Desktop/limlab/Courses/Deep RL/mbmrl/'
                               f'results/rollout/neubanp_{seed}_warm{warmup}.npy',
                               allow_pickle=True)
        env_traj = np.load(f'/Users/minsublee/Desktop/limlab/Courses/Deep RL/mbmrl/'
                           f'results/rollout/env_{seed}_warm{warmup}.npy',
                           allow_pickle=True)

        plt.plot(env_traj[:, i], color=color[seed])
        plt.plot(neubanp_traj[:, i], color=color[seed], ls='--')

    plt.title(title[i])
    plt.savefig(f'/Users/minsublee/Desktop/limlab/Courses/Deep RL/mbmrl/results/rollout_plot/{title[i]}.png')
    plt.close()
