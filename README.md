**Status:** Archive (code is provided as-is, no updates expected)

# Leveraging Procedural Generation to Benchmark Reinforcement Learning

#### [[Blog Post]](https://openai.com/blog/procgen-benchmark/) [[Paper]](https://arxiv.org/abs/1912.01588)

This is code for training agents for some of the experiments in [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://cdn.openai.com/procgen.pdf) [(citation)](#citation).  The code for the environments is in the [Procgen Benchmark](https://github.com/openai/procgen) repo.

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/openai/train-procgen.git
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
pip install -e train-procgen
```

## Try it out

Train an agent using PPO on the environment StarPilot:

```
python -m train_procgen.train --env_name starpilot
```

Train an agent using PPO on the environment StarPilot using the easy difficulty:

```
python -m train_procgen.train --env_name starpilot --distribution_mode easy
```

Run parallel training using MPI:

```
mpiexec -np 8 python -m train_procgen.train --env_name starpilot
```

Train an agent on a fixed set of N levels:

```
python -m train_procgen.train --env_name starpilot --num_levels N
```

Train an agent on the same 500 levels used in the paper:

```
python -m train_procgen.train --env_name starpilot --num_levels 500
```

Train an agent on a different set of 500 levels:

```
python -m train_procgen.train --env_name starpilot --num_levels 500 --start_level 1000
```

Run simultaneous training and testing using MPI. 1 in every 4 workers will be test workers, and the rest will be training workers.

```
mpiexec -np 8 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 4
```

Train an agent using PPO on a level in Jumper that requires hard exploration

```
python -m train_procgen.train --env_name jumper --distribution_mode exploration
```

Train an agent using PPO on a variant of CaveFlyer that requires memory

```
python -m train_procgen.train --env_name caveflyer --distribution_mode memory
```

View training options:

```
python -m train_procgen.train --help
```

# Citation

Please cite using the following bibtex entry:

```
@article{cobbe2019procgen,
  title={Leveraging Procedural Generation to Benchmark Reinforcement Learning},
  author={Cobbe, Karl and Hesse, Christopher and Hilton, Jacob and Schulman, John},
  journal={arXiv preprint arXiv:1912.01588},
  year={2019}
}
```
