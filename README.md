# Federated Reinforcement Learning

## Introduction

The goal of this project is to simulate an environment for **Federated Learning** and experiment different algorithms using **Reinforcement Learning** for better aggregation or to select in a better way the next participants.

## Installation

1. First, download the repository
```
git clone https://github.com/bourbonut/reinforcedFL.git 
```

2. Then **create your own environment** with a version of `python >= 3.8.5`.
For instance, with `conda` :
```shell
conda create -n reinforcedFL python=3.8.5
conda activate reinforcedFL
```

3. Install `pytorch`
For example, with `conda` :
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

4. Then install other packages :
```shell
pip install -r requirements.txt
```

## Usage

### Configuration tree
Three configurations files must be created to run a simulation. It is recommended to create the following tree:
```
.
├── ascii_sequential_execution.py
├── ...
└── configurations
    ├── environment
    │   ├── env20.json
    │   ├── ...
    │   └── env100.json
    ├── distribution
    │   ├── iid.json
    │   ├── ...
    │   └── noniid.json
    └── model
        ├── fedavg.json
        ├── ...
        └── evaluator.json
```

### Example of configuration files

- In an `environment` file :
```
TODO
```
- In an `distribution` file :
```
TODO
```
- In an `model` file :
```
TODO
```

### Run a simulation

Run the following command :
```shell
python ascii_sequential_execution.py <path_env_conf> <path_distrb_file> <path_model_file>
```
Add the flag `--gpu` to run on GPU and the flag `--refresh` to refresh the distribution of data if you need.

For instance :
```shell
python ascii_sequential_execution.py ./configurations/environment/env20.json ./configurations/distribution/iid.json ./configurations/model/fedavg.json --gpu
```
