import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import ceil
from constants import ENV_NAMES

import seaborn # sets some style parameters automatically

COLORS = [(57, 106, 177), (218, 124, 48)]

def switch_to_outer_plot(fig):
    ax0 = fig.add_subplot(111, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    return ax0

def ema(data_in, smoothing=0):
    data_out = np.zeros_like(data_in)
    curr = np.nan

    for i in range(len(data_in)):
        x = data_in[i]
        if np.isnan(curr):
            curr = x
        else:
            curr = (1 - smoothing) * x + smoothing * curr

        data_out[i] = curr

    return data_out

def plot_data_mean_std(ax, data_y, color_idx=0, data_x=None, x_scale=1, smoothing=0, first_valid=0, label=None):
    color = COLORS[color_idx]
    hexcolor = '#%02x%02x%02x' % color

    data_y = data_y[:,first_valid:]
    nx, num_datapoint = np.shape(data_y)

    if smoothing > 0:
        for i in range(nx):
            data_y[i,...] = ema(data_y[i,...], smoothing)

    if data_x is None:
        data_x = (np.array(range(num_datapoint)) + first_valid) * x_scale

    data_mean = np.mean(data_y, axis=0)
    data_std = np.std(data_y, axis=0, ddof=1)

    ax.plot(data_x, data_mean, color=hexcolor, label=label, linestyle='solid', alpha=1, rasterized=True)
    ax.fill_between(data_x, data_mean - data_std, data_mean + data_std, color=hexcolor, alpha=.25, linewidth=0.0, rasterized=True)

def read_csv(filename, key_name):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        key_index = -1

        values = []

        for line_num, row in enumerate(csv_reader):
            row = [x.lower() for x in row]
            if line_num == 0:
                idxs = [i for i, val in enumerate(row) if val == key_name]
                key_index = idxs[0]
            else:
                values.append(row[key_index])

    return np.array(values, dtype=np.float32)

def plot_values(ax, all_values, title=None, max_x=0, label=None, **kwargs):
    if max_x > 0:
        all_values = all_values[...,:max_x]

    if ax is not None:
        plot_data_mean_std(ax, all_values, label=label, **kwargs)
        ax.set_title(title)

    return all_values

def plot_experiment(run_directory_prefix, titles=None, suffixes=[''], normalization_ranges=None, key_name='eprewmean', **kwargs):
    run_folders = [f'{run_directory_prefix}{x}' for x in range(3)]

    num_envs = len(ENV_NAMES)
    will_normalize_and_reduce = normalization_ranges is not None

    if will_normalize_and_reduce:
        num_visible_plots = 1
        f, axarr = plt.subplots()
    else:
        num_visible_plots = num_envs
        dimx = dimy = ceil(np.sqrt(num_visible_plots))
        f, axarr = plt.subplots(dimx, dimy, sharex=True)

    for suffix_idx, suffix in enumerate(suffixes):
        all_values = []
        game_weights = [1] * num_envs

        for env_idx in range(num_envs):
            env_name = ENV_NAMES[env_idx]
            label = suffix if env_idx == 0 else None # only label the first graph to avoid legend duplicates
            print(f'loading results from {env_name}...')

            if num_visible_plots == 1:
                ax = axarr
            else:
                dimy = len(axarr[0])
                ax = axarr[env_idx // dimy][env_idx % dimy]

            csv_files = [f"results/{resid}/progress-{env_name}{'-' if len(suffix) > 0 else ''}{suffix}.csv" for resid in run_folders]
            curr_ax = None if will_normalize_and_reduce else ax

            raw_data = np.array([read_csv(file, key_name) for file in csv_files])
            values = plot_values(curr_ax, raw_data, title=env_name, color_idx=suffix_idx, label=label, **kwargs)

            if will_normalize_and_reduce:
                game_range = normalization_ranges[env_name]
                game_min = game_range[0]
                game_max = game_range[1]
                game_delta = game_max - game_min
                sub_values = game_weights[env_idx] * (np.array(values) - game_min) / (game_delta)
                all_values.append(sub_values)

        if will_normalize_and_reduce:
            normalized_data = np.sum(all_values, axis=0)
            normalized_data = normalized_data / np.sum(game_weights)
            title = 'Mean Normalized Score'
            plot_values(ax, normalized_data, title=None, color_idx=suffix_idx, label=suffix, **kwargs)

    if len(suffixes) > 1:
        if num_visible_plots == 1:
            ax.legend(loc='lower right')
        else:
            f.legend(loc='lower right', bbox_to_anchor=(.5, 0, .5, 1))

    return f, axarr