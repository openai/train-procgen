import os
import time
import numpy as np
import os.path as osp
import joblib
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
import tensorflow as tf
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from train_procgen.tma_ppo2.synchronous_runner import SynchronousRunner


def constfn(val):
    def f(_):
        return val
    return f

def get_model(network, env, ob_space, ac_space, nbatch_train, nenvs=1, vf_coef=0.5, ent_coef=0.0, model_fn=None,
              max_grad_norm=0.5,  mpi_rank_weight=1, comm=None,nsteps=2048, **network_kwargs):
    policy = build_policy(env, network, **network_kwargs)
    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from train_procgen.tma_ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)
    return model

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None,
            Teacher=None, nb_test_episodes=50, max_ep_len=2000, logger_dir=None, reset_frequency=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    With some modifications were made to make it use an ACL teacher and a test env.

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    if logger_dir is not None:
        logger.configure(dir=logger_dir)

    # Get the nb of env
    nenvs = 1 #env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    model = get_model(network, env, ob_space, ac_space, nbatch_train, nenvs=nenvs, vf_coef=vf_coef, ent_coef=ent_coef,
                      model_fn=model_fn, max_grad_norm=max_grad_norm,  mpi_rank_weight=mpi_rank_weight, comm=comm,
                      nsteps=nsteps, **network_kwargs)
    value_estimator_fn = lambda states: np.array([model.value(state) for state in states])
    Teacher.set_value_estimator(value_estimator_fn) # TODO : Check

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    # runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    runner = SynchronousRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, max_ep_len=max_ep_len,
                               Teacher=Teacher)

    epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    def test_agent(n, state=None):
        # global eval_env
        ep_returns = []
        ep_lens = []
        for j in range(n):
            if Teacher: Teacher.set_test_env_params(eval_env.get_raw_env())
            (unscaled_o, o), r, d, ep_ret, ep_len = eval_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                if state is not None:
                    actions, _, state, _ = model.step(o, S=state, M=np.array([d]))
                else:
                    actions, _, _, _ = model.step(o)
                o, r, d, infos = eval_env.step(actions)
                unscaled_reward = infos[0]["original_reward"][0]
                ep_ret += unscaled_reward
                ep_len += 1
            if Teacher: Teacher.record_test_episode(ep_ret, ep_len)
            ep_returns.append(ep_ret)
            ep_lens.append(ep_len)
        return ep_returns, ep_lens

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(Teacher) #pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprewmax', safemax([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                ep_returns, ep_lens = test_agent(nb_test_episodes, states)
                logger.logkv('eval_eprewmean', safemean(ep_returns))
                logger.logkv('eval_eprewmax', safemax(ep_returns))
                logger.logkv('eval_eplenmean', safemean(ep_lens))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

            # Save env
            try:
                joblib.dump({
                    #'env': env, 'test_env': eval_env,
                    'ob_rms': env.ob_rms, 'ret_rms': env.ret_rms},
                            osp.join(logger.get_dir(), 'vars.pkl'))
            except Exception as err:
                logger.log('Warning: could not pickle state_dict.\n{}'.format(err))
            if Teacher: Teacher.dump(logger.get_dir() + '/env_params_save.pkl')

        #### RESET ####
        if reset_frequency is not None and (update*nbatch) % reset_frequency == 0:
            print("Reset student.")
            model.reset()

    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
def safemax(xs):
    return np.nan if len(xs) == 0 else np.max(xs)



