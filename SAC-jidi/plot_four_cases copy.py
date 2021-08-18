import numpy as np
import matplotlib
import os
#matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

#from baselines.bench.monitor import load_results

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

FONT_SIZE=13

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y

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

def plot_results(dirs, num_timesteps, xaxis, task_name, label, nsteps=2048, if_sac=False):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)
        ts = ts[ts.l.cumsum() <= num_timesteps]
        tslist.append(ts)
    xy_list = [ts2xy(ts, xaxis) for ts in tslist]
    # process the network
    y_lists = []
    x_list = xy_list[0][0]
    for x, y in xy_list:
        if if_sac:
            print(x)
            print(y)
            print(len(x))
            print(len(y))
        x_new, y_new = generate_data(x, y, nsteps=nsteps, t_mean=100, if_sac=if_sac)
        y_lists.append(y_new)
    y_lists = np.array(y_lists)
    plot_curves([x_new, y_lists], xaxis, task_name, label)

def generate_data(x, y, nsteps, t_mean, if_sac):
    y = np.array(y)
    x_new = []
    y_new = []
    select_ids = []
    start_id = 1
    end_id = start_id + nsteps - 1
    while end_id < x[-1]:
        if if_sac:
            print(x)
            print(start_id)
            print(end_id)
        #select_id = np.where((x >= start_id) & (x < end_id))[0][-1]
        select_id = np.where((x > start_id) & (x <= end_id))[0]
        if len(select_id) != 0:
            select_id = select_id[-1]
        else:
            #if len(select_ids) == 0:
            #    select_id = 0
            #else:
            select_id = select_ids[-1]
        x_new.append(end_id)
        select_ids.append(select_id)
        start_id = end_id + 1
        end_id = start_id + nsteps - 1
    for ids in select_ids:
        prev_ids = max(0, ids - t_mean + 1)
        y_new.append(y[prev_ids:ids+1].mean())
    return np.array(x_new), np.array(y_new)

# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files
RANDOM_SEEDS = [123, 153, 223, 253, 323]

DPP_LABEL='DAIM'
RIN_LABEL='DAIM ' + r'$(\lambda^{ex}=0)$'
REX_LABEL='DAIM ' + r'$(\lambda^{ex}=0.01)$'
PPO_LABEL='PPO'
LIRPG_LABEL='LIRPG'
SAC_LABEL='SAC'


def generate_dir_lists(base_algo, freq, env_name):
    """
    generate the dir lists
    """
    dir_lists = ['./{}-logs/logs_{}_reward_delay_{}_seed_{}/{}/'.format(base_algo, base_algo, freq, seed, env_name) for seed in RANDOM_SEEDS]
    return dir_lists

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs = '*', default=['./logs_ppo/Walker2d-v2/'])
    parser.add_argument('--num_timesteps', type=int, default=int(1e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'Walker2d-v2')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
   
    # ------------------------------------------
    """
    this part is for the plots of the hopper
    """
    plt.figure(figsize=(17, 16))

    env_name = 'Hopper-v2'
    plt_name = 'hopper'
    # create the folder to save path
    plot_dir_path = 'plot_results/{}'.format(env_name)
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path, exist_ok=True)

    plt.subplot(4, 4, 1)
    delay=1
    task_name = '{} (Standard)'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 2)
    delay=10
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 3)
    delay=20
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    plt.subplot(4, 4, 4)
    delay=40
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    """
    this part is for the plots of the Walker2d
    """
    env_name = 'Walker2d-v2'
    plt_name = 'walker2d'
    # create the folder to save path
    plot_dir_path = 'plot_results/{}'.format(env_name)
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path, exist_ok=True)
    
    plt.subplot(4, 4, 5)
    delay=1
    task_name = '{} (Standard)'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 6)
    delay=10
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL, if_sac=True)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 7)
    delay=20
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    plt.subplot(4, 4, 8)
    delay=40
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    #plt.legend(loc='lower right', prop={'size': FONT_SIZE})
    
    """
    this part is for the plots of the ant
    """
    env_name = 'Ant-v2'
    plt_name = 'ant'
    # create the folder to save path
    plot_dir_path = 'plot_results/{}'.format(env_name)
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path, exist_ok=True)
    
    plt.subplot(4, 4, 9)
    delay=1
    task_name = '{} (Standard)'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 10)
    delay=10
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 11)
    delay=20
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    plt.subplot(4, 4, 12)
    delay=40
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    """
    this part is for the plots of the halfcheetah
    """
    env_name = 'Humanoid-v2'
    plt_name = 'humanoid'
    # create the folder to save path
    plot_dir_path = 'plot_results/{}'.format(env_name)
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path, exist_ok=True)
    
    plt.subplot(4, 4, 13)
    delay=1
    task_name = '{} (Standard)'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 14)
    delay=10
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)

    plt.subplot(4, 4, 15)
    delay=20
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    plt.subplot(4, 4, 16)
    delay=40
    task_name = '{} (Delay = {})'.format(env_name, delay)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=DPP_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_in_only', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=RIN_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('dpp_r_ex_0.01', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=REX_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('ppo', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=PPO_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('lirpg', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=LIRPG_LABEL)
    plot_results([os.path.abspath(dir) for dir in generate_dir_lists('sac', delay, env_name)], args.num_timesteps, args.xaxis, task_name, label=SAC_LABEL, nsteps=1000, if_sac=True)
    
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=False, ncol=5)
    plt.legend(loc='lower right', prop={'size': FONT_SIZE})
    plt.savefig('mujoco_result_44.pdf'.format(plot_dir_path, plt_name, delay))

if __name__ == '__main__':
    main()
