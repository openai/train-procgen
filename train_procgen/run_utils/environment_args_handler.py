from collections import OrderedDict
from gym import spaces
import numpy as np
import gym
from train_procgen.run_utils.abstract_args_handler import AbstractArgsHandler
from typing import List


class EnvironmentArgsHandler(AbstractArgsHandler):
 
    @staticmethod
    def set_parser_arguments(parser):
        '''
            Declaration of arguments for each environment and embodiment.
        '''
        parser.add_argument('--env', type=str, default="coinrun")

        ##### Coinrun Env #####
        # Provide the seeds for the levels
        parser.add_argument('--level_seeds', type=int, nargs='+', help="select a seed sequence used for level generation")


    @classmethod
    def get_object_from_arguments(cls, args):
        '''
            Create an environment given arguments.

            Returns:
                env_f (function creating the environment),
                param_env_bounds (bounds of the task space controlling PCG),
                initial_dist (Distribution of easy tasks to start with),
                target_dist (Target task distribution)
        '''
        param_env_bounds = OrderedDict()
        # For teachers using an initial distribution of easy tasks
        initial_dist = None
        # For teachers using a target distribution of tasks
        target_dist = None

        if args.env == "coinrun":
            # not actually used, only here for the validation code
            args.env_reward_lb = -100
            args.env_reward_ub = 100

            # how many sections to be used, procgen was using a number between 1 and 6
            param_env_bounds["nr_sections"] = [1, 6] 
            # seed generation params
            param_env_bounds["level_seeds"] = [1, 32768, 6] 

            # expose the set_environment method via monkey patching
            def _set_environment(self, nr_sections: int, level_seeds: List[float]):
                # get the nr of sections to use
                sections_to_take = round(nr_sections)
                # round them and take what's needed
                normalized = list(map(round, level_seeds))[0:sections_to_take]
                self.unwrapped.env.env.set_environment([normalized])

            def make_env(level_seeds):
                env = gym.make('procgen:procgen-coinrun-v0')
                # bind the set_env method to the instance
                env.set_environment = _set_environment.__get__(env)
                if level_seeds is not None:
                    env.set_environment(len(level_seeds), level_seeds)
                return env

            env =  make_env(args.level_seeds)

        else:
            print("Using an unknown env with no parameters...")
            args.env_reward_lb = -100
            args.env_reward_ub = 100
            env_f = lambda: gym.make(args.env)
            # raise Exception("No such an environment !")

        return env, param_env_bounds, initial_dist, target_dist
