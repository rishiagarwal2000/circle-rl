# Contextual character animation using RL

## Getting started

### Installation requirements
1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim)
```
$ git clone https://github.com/facebookresearch/habitat-sim
$ conda create -n habitat python=3.9 cmake=3.14.0
$ cd habitat-sim
$ python setup.py develop --bullet
```

2. Install [fairmotion](https://github.com/facebookresearch/fairmotion)
```
$ git clone https://github.com/facebookresearch/fairmotion.git
$ cd fairmotion/
$ pip install -e .
```

3. Install other deps
```
pip install torch
pip install wandb
```