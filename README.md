# Triton autotune with Reinforcement Learning

## Installation
Create a conda environment where the packages will be installed.
```
conda create --name triton-rl python=3.9
conda activate triton-rl
```
Then, in the root directory of this repository, run:
```
pip install -e .;
```

## Usage Examples

### Config
in `cfgs/` directory there are serveral examples 
- `sim`: Use the precollected benchmark results from A100 or not
- `random_sample`: During training whether the observations are in order or not
- `datasizes`: Possible observations of M N K (currently only support square matrix)
- `repeat`: Repetition time (in ms) for `do_bench_cudagraph()`
- `search_space`: Possible values for these configs
    - `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_stages`, `num_warps`

### Code
run training code
- `python3 train.py --config cfgs/{config_name}.yaml`
- experiment results will be saved in `exp/` folder

run evaluation code
- `python3 eval.py --exp_folder exp/{exp_folder}`
- best model for square matrix on A100
## Ackowledgements

Parts of this code are adapted from [Triton Documentations](https://triton-lang.org/main/getting-started/tutorials/).
