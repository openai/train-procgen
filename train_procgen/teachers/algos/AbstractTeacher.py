import numpy as np
class AbstractTeacher(object):
    '''
        Base class for ACL methods.

        This will be used to sample tasks for the DeepRL student given a task space provided at the beginning of training.
    '''
    def __init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed=None, **args):
        '''
            Creates an ACL teacher.

            Args:
                mins: Lower bounds of task space
                max: Upper bounds of task space
                env_reward_lb: Minimum return possible of the environment (used only if `scale_reward` is activated on the `TeacherController`)
                env_reward_ub: maximum return possible of the environment (used only if `scale_reward` is activated on the `TeacherController`)
                seed: Seed
                **args: Additional kwargs specific to the ACL method
        '''
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        self.random_state = np.random.RandomState(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        # Student's value estimator
        self.value_estimator = lambda state: None

        # If reward should be normalized
        self.env_reward_lb = env_reward_lb
        self.env_reward_ub = env_reward_ub

        # Book-keeping logs
        self.bk = {}

    def create_dist_from_bounds(self, mins, maxs, subspace):
        '''
            Create a gaussian distribution from bounds (either over the whole space or only a subspace if `subspace == True`)

            Args:
                mins: Lower bounds of task space
                max: Upper bounds of task space
                subspace (bool): Whether the gaussian distribution should be over a subspace
                                (with mean randomly sampled and std equal to 10% of each dimension) or spread over the whole
                                task space
        '''
        if subspace:
            mean = np.array([self.random_state.uniform(min, max) for min, max in zip(mins, maxs)])
            variance = [(abs(max - min) * 0.1) ** 2 for min, max in zip(mins, maxs)] # std = 10 % of each dimension
        else:
            mean = np.array([np.mean([min, max]) for min, max in zip(mins, maxs)])
            variance = [(abs(max - min) / 4)**2 for min, max in zip(mins, maxs)] # std = 0.25 * range => ~95.5% of samples are between the bounds
        variance = [1e-6 if v == 0 else v for v in variance]  # avoid errors with null variance
        covariance = np.diag(variance)

        return mean, covariance

    def get_or_create_dist(self, dist_dict, mins, maxs, subspace=False):
        '''
            Get distribution if `dist_dict` is not None else create a new one (Gaussian).

            Args:
                dist_dict: Dictionary containing a gaussian distribution
                mins: Lower bounds of task space
                max: Upper bounds of task space
                subspace (bool): Whether the gaussian distribution should be over a subspace
                          (with mean randomly sampled and std equal to 10% of each dimension) or spread over the whole
                          task space
        '''
        if dist_dict is not None:
            dist_mean = dist_dict["mean"]
            dist_variance = dist_dict["variance"]
        else:
            dist_mean, dist_variance = self.create_dist_from_bounds(mins, maxs, subspace)
        return dist_mean, dist_variance

    def rescale_task(self, task, original_space=(0, 1)):
        '''
            Maps a task from the n-dimensional task space towards a n-dimensional [0, 1] space.

            Args:
                task: Task that has to be mapped
                original_space: Target space bounds
        '''
        return np.array([np.interp(task[i], original_space, (self.mins[i], self.maxs[i]))
                         for i in range(len(self.mins))])

    def inverse_rescale_task(self, task, original_space=(0, 1)):
        '''
            Maps a task from a n-dimensional [0, 1] space towards the n-dimensional task space.

            Args:
                task: Task that has to be mapped
                original_space: Source space bounds
        '''
        return np.array([np.interp(task[i], (self.mins[i], self.maxs[i]), original_space)
                         for i in range(len(self.mins))])

    def record_initial_state(self, task, state):
        '''
            Record initial state of the environment given a task.
        '''
        pass

    def episodic_update(self, task, reward, is_success):
        '''
            Get the episodic reward and binary success reward of a task.
        '''
        pass

    def step_update(self, state, action, reward, next_state, done):
        '''
            Get step-related information.
        '''
        pass

    def sample_task(self):
        '''
            Sample a new task.
        '''
        pass

    def non_exploratory_task_sampling(self):
        '''
            Sample a task without exploration (used to visualize the curriculum)
        '''
        return {"task": self.sample_task(), "infos": None}

    def is_non_exploratory_task_sampling_available(self):
        '''
            Whether the method above can be called.
        '''
        return True

    def dump(self, dump_dict):
        '''
            Save the teacher.

            Args:
                dump_dict: Dictionary storing what must be saved.
        '''
        dump_dict.update(self.bk)
        return dump_dict