# training procgen with ACL: step by step setup guide

### Create the venv
1. create a virtual environment with python 3.7: `conda create --name trainprocgen-tma python=3.7`
2. activate it: `conda conda activate trainprocgen-tma`

### Install train-procgen libs
3. clone the train-procgen repo https://github.com/meln1k/train-procgen/
4. checkout the branch `nm/ACL_training`
5. update the environment from the environment.yaml file: `conda env update --name trainprocgen-tma --file envirinment.yml`

### Install custom procgen with set_environment
6. clone the Nikita's procgen repo https://github.com/meln1k/procgen
7. go inside and install the dependencies from the environment.yaml file: `conda env update --name trainprocgen-tma --file envirinment.yml`
8. install the openai baselines `pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip`
9. check that build procgen works: `python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"`
10. install the custom procgen locally: `pip install --editable .`

### Install TMA dependencies
11. clone the Nikita's TMA fork https://github.com/meln1k/TeachMyAgent
12. go to the TMA folder and install the depenencies: `pip install --editable .`

### Install train-procgen and running it
13. go back to the `train-procgen` repository
14. install the train-procgen package locally: `pip install --editable .` 
15. run the training: `python -m train_procgen.train --env_name coinrun --distribution_mode easy`
