from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
import numpy as np

class VecProcgen(VecEnvWrapper):
    def __init__(self, venv, level_sampler=None):
        super(VecProcgen, self).__init__(venv)
        self.level_sampler = level_sampler

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, 'venv'):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.level_sampler:
            seeds = np.zeros(self.venv.num_envs, dtype=np.int32)
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample('sequential')
                seeds[e] = seed
                self.venv.seed(seed,e)

        obs = self.venv.reset()

        if self.level_sampler:
            return obs, seeds
        else:
            return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # print(f"stepping {info[0]['level_seed']}, done: {done}")

        # reset environment here
        if self.level_sampler:
            for e in done.nonzero()[0]:
                seed = self.level_sampler.sample()
                self.venv.seed(seed, e) # seed resets the corresponding level

            # NB: This reset call propagates upwards through all VecEnvWrappers
            obs = self.raw_venv.observe()['rgb'] # Note reset does not reset game instances, but only returns latest observations

        return obs, reward, done, info