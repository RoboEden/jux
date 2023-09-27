# JUX
JUX is a <ins>J</ins>ax-accelerated game core for L<ins>ux</ins> AI Challenge Season 2, aimed to maximize game environment throughput for reinforcement learning (RL) training.

## Installation

### Install JAX
JAX is a main dependency of JUX, and must be installed by user manually.
```sh
pip install --upgrade "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You can test whether jax is installed successfully by running the following command.
```sh
python -c "import jax.numpy as jnp; \
    a = jnp.convolve(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5])); \
    print(a); \
    print(a.device());"
# You shall get something like this:
# [0.  1.  2.5 4.  1.5]
# gpu:0
```

### Install JUX
Finally, upgrade your pip and install JUX.
```sh
pip install --upgrade pip
pip install juxai-s2
```

## Usage
See [tutorial.ipynb](tutorial.ipynb) for a quick start. JUX is guaranteed to implement the same game logic as `luxai_s2==3.0.0`, if players' input actions are valid. When players' input actions are invalid, JUX and LuxAI-S2 may process them differently.

## Performance
JUX maps all game logic to array operators in JAX so that we can harvest the computational power of modern GPUs and support tons of environments running in parallel. We benchmarked JUX on several different GPUs, and increased the throughput by hundreds to thousands of times, compared with the original single-thread Python implementation.

LuxAI_S2 is a game with a dynamic number of units, making it hard to be accelerated by JAX, because `jax.jit()` only supports arrays with static shapes. As a workaround, we allocate a large buffer with static length to store units. The buffer length (`buf_cfg.MAX_N_UNITS`) greatly affects the performance. Theoretically, no player can build more than 1500 units under current game configs, so `MAX_N_UNITS=1500` is a safe choice. However, we found that no player builds more than 200 units by watching game replays, so `MAX_N_UNITS=200` is a practical choice.

### Relative Throughput
Here, we report the relative throughput over the original Python implementation (`luxai_s2==1.1.3`), on several different GPUs with different `MAX_N_UNITS` settings. The original single-thread Python implementation running on an 8255C CPU serves as the baseline. We can observe that the throughput is proportional to GPU memory bandwidth because the game logic is mostly memory-bound, not compute-bound. Byte-access operators take a large portion of the game logic in JUX implementation.

| Relative Throughput  |                    |            |           |           |           |           |           |            |
|:-------------------- | ------------------:| ----------:| ---------:| ---------:| ---------:| ---------:| ---------:| ----------:|
| GPU                  | GPU Mem. Bandwidth | Batch Size | UNITS=100 | UNITS=200 | UNITS=400 | UNITS=600 | UNITS=800 | UNITS=1000 |
| A100-SXM4-40GB       |          1555 GB/s |        20k |     1166x |      985x |      748x |      598x |      508x |       437x |
| Tesla V100-SXM2-32GB |           900 GB/s |        20k |      783x |      647x |      480x |      375x |      317x |       269x |
| Tesla T4             |           320 GB/s |        10k |      263x |      217x |      160x |      125x |      105x |        89x |
| GTX 1660 Ti          |           288 GB/s |         3k |      218x |      178x |      130x |      103x |       84x |        71x |


|                                           | Batch Size | Relative Throughput |
|:----------------------------------------- | ----------:| -------------------:|
| Intel® Core™ i7-12700	 	                |          1 |               2.12x |
| Intel® Xeon® Platinum 8255C CPU @ 2.50GHz	|          1 |               1.00x |
| Intel® Xeon® Gold 6133 @ 2.50GHz	        |          1 |               0.89x |



### Absolute Throughput
We also report absolute throughput in steps per second here.
| Throughput (steps/s) |            |           |           |           |           |           |            |
|:-------------------- | ----------:| ---------:| ---------:| ---------:| ---------:| ---------:| ----------:|
| GPU                  | Batch Size | UNITS=100 | UNITS=200 | UNITS=400 | UNITS=600 | UNITS=800 | UNITS=1000 |
| A100-SXM4-40GB       |        20k |      363k |      307k |      233k |      186k |      158k |       136k |
| Tesla V100-SXM2-32GB |        20k |      244k |      202k |      149k |      117k |       99k |        84k |
| Tesla T4             |        10k |       82k |       68k |       50k |       39k |       33k |        28k |
| GTX 1660 Ti          |         3k |       68k |       56k |       40k |       32k |       26k |        22k |

|                                           | Batch Size | Throughput (steps/s) |
|:----------------------------------------- | ----------:| --------------------:|
| Intel® Core™ i7-12700	 	                |          1 |                0.66k |
| Intel® Xeon® Platinum 8255C CPU @ 2.50GHz	|          1 |                0.31k |
| Intel® Xeon® Gold 6133 @ 2.50GHz	        |          1 |                0.28k |

## Contributing
If you find any bugs or have any suggestions, please feel free to open an issue or a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a developing environment.
