# th_rl
Pytorch-based package for multi-agent reinforcement learning in an iterated prisonner dilemma setting.

## Installation
- create python venv 
```
python -m venv power_markets
power_markets\scripts\activate
```

- install in local editable mode:
```
git clone https://github.com/nikitcha/th_rl
pip install -e th_rl
```

- install as module from Github
```
pip install -U git+https://github.com/nikitcha/th_rl.git
```

- if you want to install just dependencies:
```
pip install -r th_rl/requirements.txt
```

## Usage - training
- Create a set of training configs and store them somewhere, i.e. /some_path/configs
    -   Configs should follow the structured laid out in 'example_config.json'
- Run training with the follwing command:
    - This would run each config 20 times
    - Result will be stored under /some_path/runs


```
python main.py --cdir=/some_path/configs --runs=20  
```

Or if installed:

```
python -m th_rl.main.py --cdir=/some_path/configs --runs=20  
```

## Usage - plot results
- To plot a single trajectory:
```
python utils.py --dir=/some_path/runs/qtable_001/1 --fun=plot_experiment
```

- To plot the mean of all [20] runs:
```
python utils.py --dir=/some_path/runs/qtable_001 --fun=plot_mean_result
```

Or if installed:

```
python -m th_rl.utils --dir=/some_path/runs/qtable_001 --fun=plot_mean_result
```