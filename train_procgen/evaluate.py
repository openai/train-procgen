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

def eval_fn(load_path, env_name='fruitbot', num_envs=64, distribution_mode='easy', num_levels=500, start_level=500, log_dir='./tmp/procgen', comm=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 1
    log_comm = comm.Split(0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
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

    logger.info("evaluating")
    
    if os.path.isfile(load_path):
        alt_ppo2.eval(
            network=conv_fn,
            eval_env=venv,
            nsteps=nsteps,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gamma=gamma,
            lam=lam,
            log_interval=1,
            nminibatches=nminibatches,
            noptepochs=ppo_epochs,
            load_path=load_path,
            update_fn=None,
            init_fn=None,
            mpi_rank_weight=mpi_rank_weight,
            comm=comm,
            clip_vf=use_vf_clipping,
            lr=learning_rate,
            cliprange=clip_range,
        )
        
    return 
# def model_avg_loss(env, model):
    
# def chkpt_metric(path, model, env):
#     model.load(path)
#     loss = model_loss(env, model)
#     eprewmean = model_eprewmean(env, model)
#     return loss, eprewmean

# def optimal_chkpt(path, model, env):
#     best_loss = float("inf")
#     best_chkpt = None
#     for chkpt in os.listdir(path):
#         model.load(path + chkpt)
#         curr_loss, _ = chkpt_metric(path, model, env)
#         if curr_loss < best_loss:
#             best_loss = curr_loss
#             best_chkpt = path + chkpt
#     return best_chkpt

def main():
    parser = argparse.ArgumentParser(description='Process procgen evaluation arguments.')
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='./logs/eval')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--start_level', type=int, default=500)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD 

    if os.path.isdir(args.load_model):
        for chkpt in os.listdir(args.load_model):
            eval_fn(args.load_model + "/" + chkpt, env_name=args.env_name, num_envs=args.num_envs, distribution_mode=args.distribution_mode, num_levels=args.num_levels, start_level=args.start_level, log_dir=args.log_dir + "/" + chkpt, comm=comm)
    else:
        eval_fn(args.load_model,
            log_dir=args.log_dir,
            env_name=args.env_name,
            num_envs=args.num_envs,
            distribution_mode=args.distribution_mode,
            num_levels=args.num_levels,
            start_level=args.start_level,
            comm=comm,
           )

if __name__ == '__main__':
    main()
