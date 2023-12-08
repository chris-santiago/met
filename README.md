# MET - PyTorch

This repo reproduces the MET (Masked Encoding for Tabular Data) framework for self-supervised learning with tabular data.

*Authors: Kushal Majmundar, Sachin Goyal, Praneeth Netrapalli, Prateek Jain*

*Reference: Kushal Majmundar, Sachin Goyal, Praneeth Netrapalli, Prateek Jain, "MET: Masked Encoding for Tabular Data," Neural Information Processing Systems (NeurIPS), 2022.*

Original paper: https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf

Original repo: https://github.com/google-research/met

## Install

Clone this repository, create a new Conda environment and 

```bash
git clone https://github.com/chris-santiago/met.git
conda env create -f environment.yml
cd met
pip install -e .
```

## Use

### Prerequisites

#### Task

This project uses [Task](https://taskfile.dev/) as a task runner. Though the underlying Python
commands can be executed without it, we recommend [installing Task](https://taskfile.dev/installation/)
for ease of use. Details located in `Taskfile.yml`.

#### Current commands

```bash
> task -l
task: Available tasks for this project:
* check-config:       Check Hydra configuration
* compare:            Compare using linear baselines
* train:              Train a model
* wandb:              Login to Weights & Biases
```

#### PDM

This project was built using [this cookiecutter](https://github.com/chris-santiago/cookie) and is
setup to use [PDM](https://pdm.fming.dev/latest/) for dependency management, though it's not required
for package installation.

#### Hydra

This project uses [Hydra](https://hydra.cc/docs/intro/) for managing configuration CLI arguments. See `met/conf` for full
configuration details.

#### Weights and Biases

This project is set up to log experiment results with [Weights and Biases](https://wandb.ai/). It
expects an API key within a `.env` file in the root directory:

```toml
WANDB_KEY=<my-super-secret-key>
```

Users can configure different logger(s) within the `conf/trainer/default.yaml` file.
