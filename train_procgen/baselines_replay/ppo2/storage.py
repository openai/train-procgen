import numpy as np

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 split_ratio=0.05):
        self.obs = np.zeros((num_steps + 1, num_processes, *obs_shape))
        self.rewards = np.zeros((num_steps, num_processes, 1))
        self.value_preds = np.zeros((num_steps + 1, num_processes, 1))
        self.returns = np.zeros((num_steps + 1, num_processes, 1))
        self.action_log_dist = np.zeros((num_steps, num_processes, action_space.n))
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = np.zeros((num_steps, num_processes, action_shape))
        self.masks = np.ones((num_steps + 1, num_processes, 1))

        self.level_seeds = np.zeros((num_steps, num_processes, 1), dtype=np.int32)

        self.num_steps = num_steps
        self.step = 0
        
        self.split_ratio = split_ratio

    def insert(self, obs, actions, action_log_dist,
               value_preds, rewards, masks, level_seeds=None):
        if len(rewards.shape) == 3: rewards = rewards.squeeze(2)
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_dist[self.step] = action_log_dist.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if level_seeds is not None:
            self.level_seeds[self.step] = level_seeds.copy()

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step +
                                        1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step +
                                                            1] * gae
            self.returns[step] = gae + self.value_preds[step]