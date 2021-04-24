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
import os
from baselines.common import set_global_seeds
from baselines.common.policies import build_policy

def eval_fn(load_path, args, env_name='fruitbot', distribution_mode='easy', num_levels=500, start_level=500, log_dir='./tmp/procgen', comm=None, num_trials=3, gui=False):

    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    vf_coef = 0.5
    max_grad_norm = 0.5

    mpi_rank_weight = 1
    log_interval = 1
    seed=None

    log_comm = comm.Split(0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info(f"evaluating")

    set_global_seeds(seed)

    policy = build_policy(venv, conv_fn)

    # Get the nb of env
    nenvs = venv.num_envs
    # Get state_space and action_space
    ob_space = venv.observation_space
    ac_space = venv.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    from .alternate_ppo2.model import Model
    model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if os.path.isfile(load_path):
        alt_ppo2.eval(
            network=conv_fn,
            nsteps=nsteps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            lam=lam,
            log_interval=log_interval,
            nminibatches=nminibatches,
            noptepochs=ppo_epochs,
            load_path=load_path,
            mpi_rank_weight=mpi_rank_weight,
            comm=comm,
            clip_vf=use_vf_clipping,
            lr=learning_rate,
            cliprange=clip_range,
            policy=policy,
            nenvs=nenvs,
            ob_space=ob_space,
            ac_space=ac_space,
            nbatch=nbatch,
            nbatch_train=nbatch_train,
            model_fn=model_fn,
            model=model,
            num_trials=num_trials,
            num_levels=num_levels,
            start_level=start_level,
            gui=gui,
            args=args
        )
    elif os.path.isdir(load_path):
        for file in os.listdir(load_path):
            log_comm = comm.Split(0, 0)
            format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
            logger.configure(comm=log_comm, dir=log_dir+'/'+file, format_strs=format_strs)
            alt_ppo2.eval(
                network=conv_fn,
                nsteps=nsteps,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                gamma=gamma,
                lam=lam,
                log_interval=log_interval,
                nminibatches=nminibatches,
                noptepochs=ppo_epochs,
                load_path=load_path+'/'+file,
                mpi_rank_weight=mpi_rank_weight,
                comm=comm,
                clip_vf=use_vf_clipping,
                lr=learning_rate,
                cliprange=clip_range,
                policy=policy,
                nenvs=nenvs,
                ob_space=ob_space,
                ac_space=ac_space,
                nbatch=nbatch,
                nbatch_train=nbatch_train,
                model_fn=model_fn,
                model=model,
                num_trials=num_trials,
                num_levels=num_levels,
                start_level=start_level,
                gui=gui,
                args=args
            )
    else:
        print('Model path does not exist.')
    return

def main():
    parser = argparse.ArgumentParser(description='Process procgen evaluation arguments.')
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/eval')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--start_level', type=int, default=500)
    parser.add_argument('--num_trials', type=int, default=3)
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    eval_fn(args.load_model,
        log_dir=args.log_dir,
        env_name=args.env_name,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        num_trials=args.num_trials,
        comm=comm,
        gui=args.gui,
        args=args
       )

if __name__ == '__main__':
    main()
