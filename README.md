# TODO: write readme
# JUX Demo

JUX is a jax-accelerated lux engine. This is a toy demo of how to accelerated lux by jax. The goal of this demo is answer following two questions:
1. how to use jax to accelerate lux.
2. how much speedup we can get.

## How to accelerate lux by jax?
Jax is a array computation library. To accelerate lux, we need to convert all data structures in game state into arrays and all the game logic into array operations, which has a very limited programming flexibility. Converting normal program statements, such as if-else and for-loop, into array operations is not trivial, and always involves some kind of workarounds.

This demo show how to do such conversion by implementing unit movement and collision detection logic. The movement and collision detection is relatively simple, but it is enough for the purpose of demonstration, as it takes about 100 lines of code in `luxai2022/env.py`.
1. movement is relatively simple, but still involves if-else and switch logic.
    - if-else logic: not all units move. Only units with move action will move.
    - switch logic: units move in different directions. According to the direction, we need to update position differently.
2. collision detection is much more complicated. It involves nearly all python control flow statements, and also involves a special data structure, the `dict`.
    - `dict` usage in collision: Units in the same position are found by inserting units into a `dict` with their position as key.
    - complicated live/dead decision: The collision results depend on not only the unit type but also whether the unit is moving or not.

## How much speedup we can get?

In general, native python code is about 10-100x slower than native cpp code (CPU), and cpp code is about 10-100x slower than native cuda code. 

| language        | approximate<br> speed | our goal              |
|:--------------- |:--------------------- |:--------------------- |
| python          | 1x                    |                       |
| single-core cpp | 10-100x               | <-- jax on cpu backend|
| multi-core cpp  | 100-1000x             |                       |
| cuda            | 100-10000x            | <-- jax on gpu backend|

### Expectation
The speedup we can get from jax depends on the complexity of the game logic, but we expect following results:
1. __CPU__ backend: jax is faster than single-core cpp, but slower than multi-core cpp, which means __10-100x__ speedup.
    - If restrict to single-core, jax is slower than single-core cpp, because of the speed loss from substituting normal control flow by array operation workarounds.
    - However, jax array operation could utilize multiple cores, so it is expected to be faster than single-core cpp, but generally slower than multi-core native cpp implementation.
2. __GPU__ backend: comparable with native cuda code, which means __100-10000x__ speedup. Such acceleration is expected to be observed only when env batch size is large.
    - jax is expected to be comparable with native cuda code but slower than it, still because of the array operation workarounds.
    - We are confident in achieving 100x speedup on GPU given large batch size, and will strive for speedup of 300x or more.

There are two kinds of speedup are expected:
1. __Speedup for an individual game__ (batch_size = 1): 
    - such speedup greatly relies on the number of agents in a single game. 
    - If there thousands of agents in a single game, the speedup is expected to be very large.
    - If there are only a few agents in a single game, the speedup is expected to be small or even no speed up.
    - For an individual game, jax with CPU backend is expected to achieve higher speedup than jax with GPU backend.
2. __Speedup for a batch of games__:
    - such speedup greatly relies on the batch size.
    - If the batch size is large, the speedup is expected to be very large.
    - The batch size should be as large as possible within the GPU/CPU memory limit.
    - When the batch size is large enough, jax with GPU backend is expected to achieve higher speedup than jax with CPU backend.

### Speedup we observed in the demo

|                      |                  |   batch=1 |         |  batch=10 |         |  batch=1k |         |  batch=10k |         |
|----------------------|------------------|----------:|--------:|----------:|--------:|----------:|--------:|-----------:|--------:|
|                      |                  | time/step | speedup | time/step | speedup | time/step | speedup |  time/step | speedup |
| update_unit_location | native python    |    552 us |      1x |   1213 us |      1x |   1582 us |      1x | 1568.11 us |      1x |
|                      | jax cpu with jit |    7.5 us |     73x |    3.8 us |    319x |  0.651 us |  2.4k x |    1.57 us |    1k x |
|                      | jax gpu with jit |   32.1 us |     17x |    3.7 us |    327x |  0.087 us | 18.1k x |   0.085 us | 18.4k x |
|                      |                  |           |         |           |         |           |         |            |         |
| resolve_collision    | native python    |   34.6 us |    1.0x |  80.05 us |    1.0x | 107.64 us | 1x      |   105.0 us |    1.0x |
|                      | jax cpu with jit |   81.5 us |   0.42x |  81.59 us |   0.98x |  81.97 us | 1.3x    |   85.92 us |   1.22x |
|                      | jax gpu with jit |   55.0 us |   0.63x |   6.15 us |   13.1x |   0.92 us | 116.6x  |    0.83 us |  127.1x |


Our test platform:

|             | #core | FLOPS       |
| ----------- | -----:|------------:|
| Intel 8255C | 32    | 56G FLOP/s  |
| Nvida T4    | 2560  | 8.1T FLOP/s |


# Discussion

1. what is the expected development timeline?
    - When will the Lux AI Challenge Season 2 main challenge starts?
    - Do we plan to merge our code into `Lux-Design-2022` before the main challenge starts?

2. About current game rule:
    - The players initially granted 200-600 water and metal.
    - On map, there are usually less than 100 metals. 
        - It may be not very reasonable. According to my experience in RTS games, major part of player resources should be collected from map.
    - Given the current resource amount, a player can build no more than 100 robots. 
    - However, in-game parallel greatly replies on the number of robots. If there are only 100 robots, the speedup for individual games may be not obvious.

# Run the demo

There are three files:
 - `jux_demo.py`: a demo implementation of unit movement and collision detection logic in lux.
 - `native_pythin.py`: the native python implementation of unit movement and collision detection logic.
 - `time.py`: time the execution time of the two implementations.


## requirements
Install requirements.
```console
pip install -r requirements.txt
```

## check correctness
There is a test script in `native_pythin.py` to check whether the jax implementation is correct.
```console
python native_pythin.py
```

## time it!
To time the execution time of the two implementations, run `time.py`.
```console
python time.py --env-num 1000
```