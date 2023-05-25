# Contextual character animation using RL

## Getting started

### Installation requirements
1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim)
```
$ git clone https://github.com/facebookresearch/habitat-sim
$ conda create -n habitat python=3.9 cmake=3.14.0
$ conda activate habitat
$ cd habitat-sim
$ python setup.py develop --bullet
```

2. Install [fairmotion](https://github.com/facebookresearch/fairmotion)
```
$ git clone https://github.com/facebookresearch/fairmotion.git
$ cd fairmotion/
$ pip install -e . (might need to remove torch version in setup.py, as the version is not compatible with newer python versions)
```

3. Install other deps
```
pip install pybullet
pip install wandb
```