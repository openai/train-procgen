import numpy as np
import random

class DimensionsShuffler():
    def __init__(self, mins, maxs, cuttings=4, seed=21):
        '''
            Object evenly cutting a task space into hypercubes and shuffling them.

            Args:
                mins: Lower bounds of task space
                max: Upper bounds of task space
                cuttings: How many cuttings should be done on each dimension
                seed: Seed of the random shuffler
        '''
        self.rnd_state = np.random.RandomState(seed)
        self.nb_dims = len(mins)
        self.dims_dicts = []
        self.last_raw_task = None
        self.last_interpolated_task = None
        for i in range(len(mins)):
            current_min = mins[i]
            current_max = maxs[i]
            region_size = abs(current_max-current_min) / cuttings
            current_dim_dict = {
                "original": [(current_min + j*region_size, current_min + (j+1)*region_size) for j in range(cuttings)],
                "shuffled": [(current_min + j*region_size, current_min + (j+1)*region_size) for j in range(cuttings)]
            }

            self.rnd_state.shuffle(current_dim_dict["shuffled"])
            self.dims_dicts.append(current_dim_dict)

    def interpolate_task(self, task):
        '''
            Maps a task from the original task space towards the shuffled one.
        '''
        new_task = []
        for i in range(self.nb_dims):
            try:
                region_index = next(idx for idx,v in enumerate(self.dims_dicts[i]["original"]) if v[0] <= task[i] <= v[1])
            except StopIteration as err:
                raise Exception("Unable to find the index of the {} dimension of task {}."
                                .format(i, task)) from err

            new_value = np.interp(task[i],
                                  self.dims_dicts[i]["original"][region_index],
                                  self.dims_dicts[i]["shuffled"][region_index])
            new_task.append(new_value)
        self.last_raw_task = task
        self.last_interpolated_task = np.array(new_task, dtype=np.float32)
        return self.last_interpolated_task

    def inverse_interpolate_task(self, task):
        '''
            Maps a task from the shuffled task space towards the original one.
        '''
        new_task = []
        for i in range(self.nb_dims):
            try:
                region_index = next(idx for idx,v in enumerate(self.dims_dicts[i]["shuffled"]) if v[0] <= task[i] <= v[1])
            except StopIteration as err:
                raise Exception("Unable to find the index of the {} dimension of task {}."
                                .format(i, task)) from err

            new_value = np.interp(task[i],
                                  self.dims_dicts[i]["shuffled"][region_index],
                                  self.dims_dicts[i]["original"][region_index])
            new_task.append(new_value)
        return np.array(new_task, dtype=np.float32)