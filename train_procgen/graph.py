from graph_util import plot_experiment, switch_to_outer_plot
from constants import ENV_NAMES, NAME_TO_CASE, EASY_GAME_RANGES, HARD_GAME_RANGES

import matplotlib
import matplotlib.pyplot as plt

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution_mode', type=str, default='easy', help="Environment distribution_mode ('easy' or 'hard')")
    parser.add_argument('--normalize_and_reduce', dest='normalize_and_reduce', action='store_true')
    parser.add_argument('--restrict_training_set', dest='restrict_training_set', action='store_true')
    parser.add_argument('--save', dest='save', action='store_true')
    args = parser.parse_args()

    run_directory_prefix = main_pcg_sample_entry(args.distribution_mode, args.normalize_and_reduce, args.restrict_training_set)

    plt.tight_layout()

    if args.save:
        suffix = '-mean' if args.normalize_and_reduce else ''
        plt.savefig(f'results/{run_directory_prefix}{suffix}.pdf')
    else:
        plt.show()

def main_pcg_sample_entry(distribution_mode, normalize_and_reduce, restrict_training_set):
    params = {
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 24,
        'legend.fontsize': 18,
        'figure.figsize': [9, 9]
    }
    matplotlib.rcParams.update(params)

    kwargs = {'smoothing': .9}

    if distribution_mode == 'easy':
        kwargs['x_scale'] = 1 * 256 * 64 / 1e6 # num_workers * num_steps_per_rollout * num_envs_per_worker / graph_scaling
        num_train_levels = 200
        normalization_ranges = EASY_GAME_RANGES
    elif distribution_mode == 'hard':
        kwargs['x_scale'] = 4 * 256 * 64 / 1e6 # num_workers * num_steps_per_rollout * num_envs_per_worker / graph_scaling
        num_train_levels = 500
        normalization_ranges = HARD_GAME_RANGES
    else:
        assert False, "specify distribution_mode as 'easy' or 'hard'"

    y_label = 'Score'
    x_label = 'Timesteps (M)'

    run_directory_prefix = f"{distribution_mode}-{num_train_levels if restrict_training_set else 'all'}"
    kwargs['run_directory_prefix'] = f"{run_directory_prefix}-run"

    # We throw out the first few datapoints to give the episodic reward buffers time to fill up
    # Otherwise, there could be a short-episode bias
    kwargs['first_valid'] = 10

    if restrict_training_set:
        kwargs['suffixes'] = ['train', 'test']

    if normalize_and_reduce:
        kwargs['normalization_ranges'] = normalization_ranges
        y_label = 'Mean Normalized Score'

    fig, axarr = plot_experiment(**kwargs)

    if normalize_and_reduce:
        axarr.set_xlabel(x_label, labelpad=20)
        axarr.set_ylabel(y_label, labelpad=20)
    else:
        ax0 = switch_to_outer_plot(fig)
        ax0.set_xlabel(x_label, labelpad=40)
        ax0.set_ylabel(y_label, labelpad=35)

    return run_directory_prefix

if __name__ == '__main__':
    main()