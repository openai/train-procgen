from gym3 import types_np, ViewerWrapper
from procgen import ProcgenEnv, ProcgenGym3Env
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
import argparse
from tqdm import tqdm
# from train import cnn_small

def eval_fn(num_levels, use_model, load_path, gui):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 1
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    mpi_rank_weight = 1

    venv = ProcgenEnv(num_envs=1, env_name='fruitbot', num_levels=1, start_level=0, distribution_mode='easy')
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32])

    if use_model:
        model = ppo2.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=0,
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
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            load_path=load_path
        )

    avg_reward = 0

    for num_level in tqdm(range(num_levels)):
        if gui:
            env = ViewerWrapper(ProcgenGym3Env(num=1, env_name="fruitbot", num_levels=1, start_level=num_level, distribution_mode='easy', render_mode="rgb_array"), info_key='rgb')
        else:
            env = ProcgenGym3Env(num=1, env_name="fruitbot", num_levels=1, start_level=num_level, distribution_mode='easy')
        _, obs, _ = env.observe()
        step = 0
        total_reward = 0
        while True:
            if not use_model:
                env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
            else:
                actions, _, _, _ = model.step(obs['rgb'])
                env.act(actions)
            rew, obs, first = env.observe()
            total_reward += rew
            if step > 0 and first:
                # print(f"step {step} reward {rew} first {first}")
                break
            step += 1
        # print(f"Level: {num_level} reward: {total_reward}")
        avg_reward += total_reward
    print(f"Avg reward: {avg_reward / num_levels}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--load_path', type=str, default='models/model_first_50')
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()

    eval_fn(args.num_levels, args.use_model, args.load_path, args.gui)

if __name__ == '__main__':
    main()