import numpy as np

class SynchronousRunner():
    """
    Synchronous runner to interact with environment and sample trajectories (created for TeachMyAgent).

    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, max_ep_len, Teacher):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.nb_total_steps = 0
        self.batch_ob_shape = (1 * nsteps,) + env.observation_space.shape
        self.obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

        if Teacher:
            Teacher.set_env_params(env.env)
        o = env.reset()
        if Teacher:
            Teacher.record_train_task_initial_state(o[0])
        self.prev_unscaled_obs = o[0]
        self.obs = np.array(o)
        self.states = model.initial_state
        self.dones = [False]

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

        # ACL utils
        self.max_ep_len = max_ep_len
        self.ep_len = 0
        self.ep_ret = 0
        self.raw_ep_ret = 0

    def run(self, Teacher):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.nb_total_steps += 1

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            a = actions
            # o, r, d, i = self.env.step(a)
            step_res = self.env.step(a)
            o, r, d, i  = step_res
            unscaled_reward = r[0]
            unscaled_o = o
            self.ep_len += 1
            self.ep_ret += r[0]
            self.raw_ep_ret += unscaled_reward
            Teacher.record_train_step(self.prev_unscaled_obs, a[0], unscaled_reward, unscaled_o, d[0])

            if d or self.ep_len == self.max_ep_len:
                if Teacher:
                    success = False if 'success' not in i else i["success"]
                    Teacher.record_train_episode(self.raw_ep_ret, self.ep_len, success)
                    Teacher.set_env_params(self.env.env)
                o = self.env.reset()
                unscaled_o = o[0]
                Teacher.record_train_task_initial_state(unscaled_o)
                epinfos.append({
                    "r": self.raw_ep_ret,
                    "l": self.ep_len
                })
                self.ep_len = 0
                self.ep_ret = 0
                self.raw_ep_ret = 0

            self.obs, self.prev_unscaled_obs, rewards, self.dones, infos = o, unscaled_o, r, d, i
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


