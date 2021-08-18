import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

FONT_SIZE=13


def plot_curves(xy_list, xaxis, title, label):
    # plt.figure(figsize=(8,2))
    maxx = xy_list[0][-1]
    minx = 0
    x, y_lists = xy_list
    y_mean = np.mean(y_lists, axis=0)
    y_std_error = np.std(y_lists, axis=0) / np.sqrt(y_lists.shape[0])
    y_upper = y_mean + y_std_error
    y_lower = y_mean - y_std_error
    #for (i, (x, y)) in enumerate(xy_list):
        #color = COLORS[i]
        # plt.scatter(x, y, s=2)
    #x_filter, y_mean = window_func(x, y_mean, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    #_, y_upper = window_func(x, y_upper, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    #_, y_lower = window_func(x, y_lower, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    plt.plot(x, y_mean, label=label)
    plt.fill_between(x, y_lower, y_upper, alpha=0.25)
    plt.xlim(minx, maxx)
    plt.title(title, fontsize=FONT_SIZE)
    plt.xlabel(xaxis, fontsize=FONT_SIZE)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.ylabel("Episode Rewards", fontsize=FONT_SIZE)
    plt.tight_layout()
