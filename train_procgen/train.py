import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
from .alternate_ppo2 import alt_ppo2

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, args, is_test_worker=False, log_dir='./tmp/procgen', comm=None, alternate_ppo=False, do_eval=False, eval_num_envs=None, eval_env_name=None, eval_num_levels=None, eval_start_level=None, eval_distribution_mode=None, do_test=False, test_num_envs=None, test_env_name=None, test_num_levels=None, test_start_level=None, test_distribution_mode=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    eval_env = None
    if do_eval:
        eval_env = ProcgenEnv(num_envs=eval_num_envs, env_name=eval_env_name, num_levels=eval_num_levels, start_level=eval_start_level, distribution_mode=eval_distribution_mode)
        eval_env = VecExtractDictObs(eval_env, "rgb")

        eval_env = VecMonitor(
            venv=eval_env, filename=None, keep_buf=100,
        )

        eval_env = VecNormalize(venv=eval_env, ob=False)

    test_env = None
    if do_test:
        test_env = ProcgenEnv(num_envs=test_num_envs, env_name=test_env_name, num_levels=test_num_levels, start_level=test_start_level, distribution_mode=test_distribution_mode)
        test_env = VecExtractDictObs(test_env, "rgb")

        test_env = VecMonitor(
            venv=test_env, filename=None, keep_buf=100,
        )

        test_env = VecNormalize(venv=test_env, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    if alternate_ppo:
        alt_ppo2.learn(
            env=venv,
            eval_env=eval_env,
            test_env=test_env,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=1,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            args=args
        )
    else:
        ppo2.learn(
            env=venv,
            eval_env=eval_env,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=1,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            args=args
        )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--start_level', type=int, default=1000)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000)
    parser.add_argument('--alternate_ppo', action='store_true')

    # For evaluation (validation set)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--eval_num_envs', type=int, default=64)
    parser.add_argument('--eval_env_name', type=str, default='fruitbot')
    parser.add_argument('--eval_num_levels', type=int, default=500)
    parser.add_argument('--eval_start_level', type=int, default=500)
    parser.add_argument('--eval_distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])

    # For evaluation (test set)
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_num_envs', type=int, default=64)
    parser.add_argument('--test_env_name', type=str, default='fruitbot')
    parser.add_argument('--test_num_levels', type=int, default=500)
    parser.add_argument('--test_start_level', type=int, default=0)
    parser.add_argument('--test_distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])

    # For data augmentation
    parser.add_argument('--do_aug', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    # Autoaugment
    parser.add_argument('--autoaugment', action='store_true')
    # 0:ImageNetPolicy 1:CIFAR10Policy 2:SVHNPolicy
    parser.add_argument('--autoaug_policy_idx', type=int, default=0, choices=[0, 1, 2])
    # Augmix
    parser.add_argument('--augmix', action='store_true')
    parser.add_argument('--mixture_width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
    parser.add_argument('--mixture_depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug_severity', default=1, type=int, help='Severity of base augmentation operators')
    parser.add_argument('--aug_prob_coeff', default=1., type=float, help='Probability distribution coefficients')

    args = parser.parse_args()

    args.do_aug = args.augmix or args.autoaugment

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(args.env_name,
            args.num_envs,
            args.distribution_mode,
            args.num_levels,
            args.start_level,
            args.timesteps_per_proc,
            args,
            is_test_worker=is_test_worker,
            log_dir=args.log_dir,
            comm=comm,
            alternate_ppo=args.alternate_ppo,
            do_eval=args.do_eval,
            eval_num_envs=args.eval_num_envs,
            eval_env_name=args.eval_env_name,
            eval_num_levels=args.eval_num_levels,
            eval_start_level=args.eval_start_level,
            eval_distribution_mode=args.eval_distribution_mode,
            do_test=args.do_test,
            test_num_envs=args.test_num_envs,
            test_env_name=args.test_env_name,
            test_num_levels=args.test_num_levels,
            test_start_level=args.test_start_level,
            test_distribution_mode=args.eval_distribution_mode
            )

if __name__ == '__main__':
    main()
