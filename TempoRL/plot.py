import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 1
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

FONT_SIZE=13

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    # shape = a.shape[:-1] + (a.shape[-1], window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def plot_curves(x_list, y_lists, xaxis, title, label):
    # plt.figure(figsize=(8,2))
    maxx = x_list[-1]
    minx = 0
    x = x_list
    y_mean = np.mean(y_lists, axis=0)
    y_std_error = np.std(y_lists, axis=0) / np.sqrt(y_lists.shape[0])
    y_upper = y_mean + y_std_error
    y_lower = y_mean - y_std_error

    # for (i, (x, y)) in enumerate(xy_list):
    #     color = COLORS[i]
    #     plt.scatter(x, y, s=2)
    x_filter, y_mean = window_func(x, y_mean, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_upper = window_func(x, y_upper, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    _, y_lower = window_func(x, y_lower, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes

    plt.plot(x_filter, y_mean, label=label)

    plt.fill_between(x_filter, y_lower, y_upper, alpha=0.25)
    # minx = 10000
    plt.xlim(minx, maxx)
    # plt.ylim(-210, -120)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel(xaxis, fontsize=FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.ylabel("Episode Rewards", fontsize=FONT_SIZE)
    # 横坐标指数表示
    plt.xscale('symlog')
    plt.tight_layout()

# read csv
# path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
# files = os.listdir(path)
# # 测出sample个数
# rew_1 = pd.read_csv(path + files[1])
# x_val = rew_1['Step'].values

# read txt
path = "/home/mqr/TempoRL-master/experiments/featurized_results/sparsemountain/tdqn/"
files = os.listdir(path)
rew_1 = np.loadtxt(path+files[0]+"/reward.txt")
x_val = np.loadtxt(path+files[0]+"/step.txt")
len_x = len(x_val)
# number of files
len_file = len(files)
rew = np.array(np.zeros(shape = (len_file, len_x)))
for file_i in range(len_file):
    rew[file_i] = pd.read_csv(path+files[file_i])['Value'].values

plot_curves(x_list=x_val, y_lists=rew, xaxis="train steps", title="MountainCar", label="average reward per episode")
plt.show()
