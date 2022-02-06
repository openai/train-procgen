import numpy as np
import pickle
import copy

from train_procgen.teachers.algos.random_teacher import RandomTeacher
from train_procgen.teachers.utils.dimensions_shuffler import DimensionsShuffler
from collections import OrderedDict

# Utils functions to convert task vector into dictionary (or the opposite)
def param_vec_to_param_dict(param_env_bounds, param):
    '''
        Convert a task vector into a dictionary.

        Args:
            param_env_bounds: Dictionary containing bounds of each dimension
            param: Task vector
        Returns:
            Task as a dictionary
    '''
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if type(bounds[0]) is list:
            nb_dims = len(bounds)
            param_dict[name] = param[cpt:cpt+nb_dims]
            cpt += nb_dims
        else:
            if len(bounds) == 2:
                param_dict[name] = param[cpt]
                cpt += 1
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                nb_dims = bounds[2]
                param_dict[name] = param[cpt:cpt+nb_dims]
                cpt += nb_dims

    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):
    '''
        Convert a task dictionary into a vector.

        Args:
            param_env_bounds: Dictionary containing bounds of each dimension
            param_dict: Task dictionary
        Returns:
            Task vector
    '''
    param_vec = []
    for name, bounds in param_env_bounds.items():
        if isinstance(param_dict[name], list) or isinstance(param_dict[name], np.ndarray):
            param_vec.extend(param_dict[name])
        else:
            param_vec.append(param_dict[name])

    return np.array(param_vec)

# Class controlling the interactions between ACL methods and DeepRL students
class TeacherController(object):
    '''
        Control the interactions between the ACL method and the DeepRL student.
    '''
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, seed=None, test_set=None,
                 keep_periodical_task_samples=None, shuffle_dimensions=False, scale_reward=False, **teacher_params):
        '''
            Create a TeacherController to make and ACL method interact with a DeepRL student.
            Pass this object to your DeepRL student and use the methods below to as for new tasks and record trajectories.

            Args:
                teacher (str): Teacher's name
                nb_test_episodes: Number of episodes in the test set (used if a new test set must be generated)
                param_env_bounds (dict): Bounds of the task space
                seed: Seed for the teacher and the test set generation (if needed)
                test_set: Test set's name if an existing test set must be used.
                          Saved test set are in the `TeachMyAgent/teachers/test_sets` folder.
                          Just specify the filename without the path (and without the .pkl extension).
                keep_periodical_task_samples: How frequently the teacher must be asked to sample 100 (non exploratory) tasks.
                                              This is then used to visualize the curriculum.
                shuffle_dimensions (bool): Whether the task space the ACL method uses should be cut into hypercubes and shuffled.
                                    If set to True, the ACL teacher interacts with  the shuffled task space.
                                    Tasks are then mapped towards the real task space using a DimensionsShuffler.
                scale_reward (bool): Whether rewards should be scaled to a [0, 1] interval
                teacher_params: Additional kwargs for the ACL method.
        '''
        self.teacher = teacher
        self.nb_test_episodes = nb_test_episodes
        self.test_set = test_set
        self.test_ep_counter = 0
        self.train_step_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)
        self.keep_periodical_task_samples = keep_periodical_task_samples
        self.scale_reward = scale_reward

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if type(bounds[0]) is list:
                try:
                    # Define min / max for each dim
                    for dim in bounds:
                        mins.append(dim[0])
                        maxs.append(dim[1])
                except:
                    print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                    exit(1)
            else:
                if len(bounds) == 2:
                    mins.append(bounds[0])
                    maxs.append(bounds[1])
                elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                    mins.extend([bounds[0]] * bounds[2])
                    maxs.extend([bounds[1]] * bounds[2])
                else:
                    print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                    exit(1)
        self.task_dim = len(mins)

        # If `shuffle_dimensions` is set to True, the ACL teacher interacts with tasks in the shuffled task space.
        # Tasks are then mapped towards the real task space.
        # This uses a DimensionsShuffler.
        if shuffle_dimensions:
            self.dimensions_shuffler = DimensionsShuffler(mins, maxs, seed=seed)
            if "initial_dist" in teacher_params:
                teacher_params["initial_dist"]["mean"] = self.dimensions_shuffler.inverse_interpolate_task(
                    teacher_params["initial_dist"]["mean"])
            if "target_dist" in teacher_params:
                    teacher_params["target_dist"]["mean"] = self.dimensions_shuffler.inverse_interpolate_task(
                        teacher_params["target_dist"]["mean"])
        else:
            self.dimensions_shuffler = None

        # setup tasks generator
        if teacher == 'Random':
            self.task_generator = RandomTeacher(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'RIAC':
            self.task_generator = RIAC(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'ALP-GMM':
            self.task_generator = ALPGMM(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'ADR':
            self.task_generator = ADR(mins, maxs, seed=seed, scale_reward=scale_reward, **teacher_params)
        elif teacher == 'Self-Paced':
            self.task_generator = SelfPacedTeacher(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'GoalGAN':
            self.task_generator = GoalGAN(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'Setter-Solver':
            self.task_generator = SetterSolver(mins, maxs, seed=seed, **teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        # Generate test set
        ## Use evenly distributed tasks for StumpTracks
        ## Use uniform sampling otherwise
        ## Or load a saved test set
        test_param_vec = None
        if test_set is None:
            if self.task_dim == 2 and "stump_height" in param_env_bounds and "obstacle_spacing" in param_env_bounds: # StumpTracks
                print("Using random test set for two fixed dimensions.")
                # select <nb_test_episodes> parameters choosen uniformly in the task space
                nb_steps = int(nb_test_episodes ** (1 / self.task_dim))
                d1 = np.linspace(mins[0], maxs[0], nb_steps, endpoint=True)
                d2 = np.linspace(mins[1], maxs[1], nb_steps, endpoint=True)
                test_param_vec = np.transpose([np.tile(d1, len(d2)), np.repeat(d2, len(d1))])  # cartesian product
            else:
                print("Using random test set.")
                test_random_state = np.random.RandomState(
                    31)  # Seed a new random generator not impacting the global one to always get the same test set
                test_param_vec = test_random_state.uniform(mins, maxs, size=(nb_test_episodes, self.task_dim))
        else:
            test_param_vec = np.array(pickle.load(open("TeachMyAgent/teachers/test_sets/"+test_set+".pkl", "rb")))
            self.nb_test_episodes = len(test_param_vec)
            print('fixed set of {} tasks loaded'.format(len(test_param_vec)))
        test_param_dicts = [param_vec_to_param_dict(param_env_bounds, vec) for vec in test_param_vec]
        self.test_env_list = test_param_dicts

        # Data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []
        self.periodical_task_samples = []
        self.periodical_task_infos = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def _get_last_task(self):
        params = self.env_params_train[-1]
        if self.dimensions_shuffler is not None:
            params = self.dimensions_shuffler.last_raw_task
        return params

    def set_value_estimator(self, estimator):
        '''
            Give the DeepRL value estimator to the teacher.
        '''
        self.task_generator.value_estimator = estimator

    def record_train_task_initial_state(self, initial_state):
        '''
            Record the initial state of the lastly sampled task.
        '''
        self.task_generator.record_initial_state(self._get_last_task(), initial_state)

    def record_train_step(self, state, action, reward, next_state, done):
        '''
            Record a step for the last task.
        '''
        self.train_step_counter += 1
        self.task_generator.step_update(state, action, reward, next_state, done)
        # Monitor curriculum
        if self.keep_periodical_task_samples is not None \
                and self.train_step_counter % self.keep_periodical_task_samples == 0:
            tasks = []
            infos = []
            if self.task_generator.is_non_exploratory_task_sampling_available():
                for i in range(100):
                    task_and_infos = self.task_generator.non_exploratory_task_sampling()
                    tasks.append(task_and_infos["task"])
                    infos.append(task_and_infos["infos"])
            self.periodical_task_samples.append(np.array(tasks))
            self.periodical_task_infos.append(np.array(infos))

    def record_train_episode(self, ep_reward, ep_len, is_success=False):
        '''
            Record the episode associated to the last task.

                ep_reward: Return
                ep_len: Number of steps done
                is_success: Binary reward
        '''
        self.env_train_rewards.append(ep_reward)
        self.env_train_len.append(ep_len)
        if self.scale_reward:
            ep_reward = np.interp(ep_reward,
                                  (self.task_generator.env_reward_lb, self.task_generator.env_reward_ub),
                                  (0, 1))
            self.env_train_norm_rewards.append(ep_reward)
        self.task_generator.episodic_update(self._get_last_task(), ep_reward, is_success)

    def record_test_episode(self, reward, ep_len):
        '''
            Record the episode for the last test task sampled.
        '''
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def dump(self, filename):
        '''
            Save teacher and all book-keeped information.
        '''
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len,
                         'env_param_bounds': list(self.param_env_bounds.items()),
                         'periodical_samples': self.periodical_task_samples,
                         'periodical_infos': self.periodical_task_infos}
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_env_params(self, env):
        '''
            Sample a task and set the PCG of the environment with it.
        '''
        params = copy.copy(self.task_generator.sample_task())
        if self.dimensions_shuffler is not None:
            params = self.dimensions_shuffler.interpolate_task(params)
        self.env_params_train.append(params)
        if len(params) > 0:
            assert type(params[0]) == np.float32
            param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
            # get the nr of sections to use
            sections_to_take = round(param_dict['nr_sections'])
            # round them and take what's needed
            normalized = list(map(round, param_dict['level_seeds']))[0:sections_to_take]
            env.set_environment([normalized])
        return params

    def set_test_env_params(self, test_env):
        '''
            Sample a test task from the test set and set the PCG of the test environment with it.
        '''
        self.test_ep_counter += 1
        test_param_dict = self.test_env_list[self.test_ep_counter - 1]

        if self.test_set == "hexagon_test_set":
            # removing legacy parameters from test_set, don't pay attention
            legacy = ['tunnel_height', 'gap_width', 'step_height', 'step_number']
            keys = test_param_dict.keys()
            for env_param in legacy:
                if env_param in keys:
                    del test_param_dict[env_param]

        test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
        if len(test_param_vec) > 0:
            self.env_params_test.append(test_param_vec)
            test_env.set_environment(**test_param_dict)

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0
        return test_param_dict