import numpy as np
from gym.spaces import Box
from train_procgen.teachers.algos.AbstractTeacher import AbstractTeacher

class RandomTeacher(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub):
        '''
            Random teacher sampling tasks uniformly random over the task space.
        '''
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        self.random_task_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.random_task_generator.seed(self.seed)

    def sample_task(self):
        return self.random_task_generator.sample()

    def non_exploratory_task_sampling(self):
        return {"task": self.sample_task(),
                "infos": {
                    "bk_index": -1,
                    "task_infos": None}
                }